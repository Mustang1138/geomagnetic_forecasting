"""
API route integration tests for the Aurora Forecast web application.

Uses FastAPI's TestClient to exercise all four route modules without starting
a real HTTP server.  The data-loading and forecast layers are replaced with
lightweight mocks so no trained model artefacts or live DSCOVR data are
required.

Coverage:
    GET /api/predictions  — valid model keys, invalid model key → 400
    GET /api/snapshot     — valid index, out-of-range → 400, negative → 422
    GET /api/models       — response shape and required fields
    GET /api/forecast     — successful result, DSCOVR failure → 503
"""

import numpy as np
import pandas as pd
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

_N = 10
_VALID_MODELS = ("rf", "lr", "ls", "gr", "pe")


# ── Synthetic fixtures ────────────────────────────────────────────────────────

def _mock_df(n: int = _N) -> pd.DataFrame:
    """Return a minimal aligned predictions DataFrame matching the real schema."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "datetime":    pd.date_range("2023-01-01", periods=n, freq="6h"),
        "auroral_lat": np.full(n, 63.0),
        "storm_class": ["q"] * n,
        "true":        rng.random(n).round(5),
        "rf":          rng.random(n).round(5),
        "lr":          rng.random(n).round(5),
        "ls":          rng.random(n).round(5),
        "gr":          rng.random(n).round(5),
        "pe":          rng.random(n).round(5),
    })


def _mock_metrics() -> list[dict]:
    return [
        {"key": "rf", "model": "random_forest",     "label": "Random Forest",    "rmse": 0.05, "mae": 0.04, "r2": 0.80},
        {"key": "lr", "model": "linear_regression",  "label": "Linear Regression","rmse": 0.07, "mae": 0.05, "r2": 0.70},
        {"key": "ls", "model": "lstm",               "label": "LSTM",             "rmse": 0.04, "mae": 0.03, "r2": 0.85},
        {"key": "gr", "model": "gru",                "label": "GRU",              "rmse": 0.04, "mae": 0.03, "r2": 0.84},
        {"key": "pe", "model": "persistence",        "label": "Persistence",      "rmse": 0.10, "mae": 0.08, "r2": 0.50},
    ]


def _mock_forecast_result() -> dict:
    steps = 4
    payload = {"ssi": [0.1] * steps, "auroral_lat": [65.0] * steps, "storm_class": ["q"] * steps}
    return {
        "generated_at": "2025-01-01 00:00 UTC",
        "steps": steps,
        "step_hours": 6,
        "dscovr_conditions_used": True,
        "timestamps": [f"2025-01-0{i + 1} 00:00" for i in range(steps)],
        "models": {k: payload for k in _VALID_MODELS},
    }


# ── GET /api/predictions ──────────────────────────────────────────────────────

def test_predictions_default_model_is_rf():
    with patch("src.api.routes.predictions.load_aligned_predictions", return_value=_mock_df()):
        response = client.get("/api/predictions")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "rf"


def test_predictions_response_shape():
    with patch("src.api.routes.predictions.load_aligned_predictions", return_value=_mock_df()):
        body = client.get("/api/predictions").json()
    assert body["n"] == _N
    assert len(body["dt"]) == _N
    assert len(body["true"]) == _N
    assert len(body["pred"]) == _N
    assert set(body["all"].keys()) == set(_VALID_MODELS)


def test_predictions_each_valid_model():
    df = _mock_df()
    for model in _VALID_MODELS:
        with patch("src.api.routes.predictions.load_aligned_predictions", return_value=df):
            response = client.get(f"/api/predictions?model={model}")
        assert response.status_code == 200, f"Model '{model}' returned {response.status_code}"
        assert response.json()["model"] == model


def test_predictions_invalid_model_returns_400():
    with patch("src.api.routes.predictions.load_aligned_predictions", return_value=_mock_df()):
        response = client.get("/api/predictions?model=xx")
    assert response.status_code == 400


# ── GET /api/snapshot ─────────────────────────────────────────────────────────

def test_snapshot_valid_index_returns_200():
    with patch("src.api.routes.snapshot.load_aligned_predictions", return_value=_mock_df()):
        response = client.get("/api/snapshot?idx=0")
    assert response.status_code == 200


def test_snapshot_response_keys():
    with patch("src.api.routes.snapshot.load_aligned_predictions", return_value=_mock_df()):
        body = client.get("/api/snapshot?idx=0").json()
    assert set(body["models"].keys()) == set(_VALID_MODELS)
    for field in ("idx", "dt", "lat", "cl", "true", "models"):
        assert field in body, f"Missing field '{field}' in snapshot response"


def test_snapshot_last_valid_index():
    with patch("src.api.routes.snapshot.load_aligned_predictions", return_value=_mock_df()):
        response = client.get(f"/api/snapshot?idx={_N - 1}")
    assert response.status_code == 200


def test_snapshot_out_of_range_returns_400():
    with patch("src.api.routes.snapshot.load_aligned_predictions", return_value=_mock_df()):
        response = client.get(f"/api/snapshot?idx={_N + 100}")
    assert response.status_code == 400


def test_snapshot_negative_index_rejected_by_schema():
    """FastAPI Query(ge=0) should reject negative indices with a 422 validation error."""
    with patch("src.api.routes.snapshot.load_aligned_predictions", return_value=_mock_df()):
        response = client.get("/api/snapshot?idx=-1")
    assert response.status_code == 422


# ── GET /api/models ───────────────────────────────────────────────────────────

def test_models_returns_list_of_five():
    with patch("src.api.routes.models.get_metrics", return_value=_mock_metrics()):
        body = client.get("/api/models").json()
    assert isinstance(body, list)
    assert len(body) == 5


def test_models_required_fields_present():
    with patch("src.api.routes.models.get_metrics", return_value=_mock_metrics()):
        body = client.get("/api/models").json()
    for entry in body:
        for field in ("key", "label", "rmse", "mae", "r2"):
            assert field in entry, f"Missing field '{field}' in /api/models entry"


def test_models_all_keys_present():
    with patch("src.api.routes.models.get_metrics", return_value=_mock_metrics()):
        body = client.get("/api/models").json()
    keys = {entry["key"] for entry in body}
    assert keys == set(_VALID_MODELS)


# ── GET /api/forecast ─────────────────────────────────────────────────────────

def test_forecast_success():
    with patch("src.api.routes.forecast_route.generate_forecast", return_value=_mock_forecast_result()):
        response = client.get("/api/forecast")
    assert response.status_code == 200
    body = response.json()
    assert "models" in body
    assert set(body["models"].keys()) == set(_VALID_MODELS)
    assert body["steps"] == 4


def test_forecast_returns_503_when_dscovr_unavailable():
    """generate_forecast() returns None when DSCOVR feeds cannot be fetched."""
    with patch("src.api.routes.forecast_route.generate_forecast", return_value=None):
        response = client.get("/api/forecast")
    assert response.status_code == 503


def test_forecast_model_payload_shape():
    """Each model's payload should contain ssi, auroral_lat, and storm_class arrays."""
    with patch("src.api.routes.forecast_route.generate_forecast", return_value=_mock_forecast_result()):
        body = client.get("/api/forecast").json()
    for model_key in _VALID_MODELS:
        payload = body["models"][model_key]
        assert "ssi" in payload
        assert "auroral_lat" in payload
        assert "storm_class" in payload
        assert len(payload["ssi"]) == body["steps"]
