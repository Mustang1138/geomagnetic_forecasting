"""
Unit tests for the autoregressive forecasting engine (src/api/forecast.py).

Individual components are tested using mocked scalers, lightweight model
instances, and synthetic seed windows so no trained artefacts or live DSCOVR
data are required.

Coverage:
    _generate_timestamps      — length, format, 6-hour spacing
    _inverse_scale_predictions — output length, clipping to [0, 1]
    _derive_forecast_metadata  — lengths, latitude bounds, single-char class
    _run_autoregressive_loop   — output length, window rolling behaviour
    _forecast_sklearn          — output length, correct call count
    _forecast_temporal         — LSTM and GRU output length
    _forecast_persistence      — output length, constant-value property
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.api.forecast import (
    FORECAST_STEPS,
    STEP_HOURS,
    _derive_forecast_metadata,
    _forecast_persistence,
    _forecast_sklearn,
    _forecast_temporal,
    _generate_timestamps,
    _inverse_scale_predictions,
    _run_autoregressive_loop,
)
from src.models.temporal_model import GRURegressor, LSTMRegressor

_N_FEATURES = 5
_SEQ_LEN = 4
_N_STEPS = 6


# ── Helpers ───────────────────────────────────────────────────────────────────

def _seed(seq_len: int = _SEQ_LEN, n_features: int = _N_FEATURES) -> np.ndarray:
    return np.random.default_rng(0).standard_normal((seq_len, n_features)).astype(np.float32)


def _future(n_steps: int = _N_STEPS, n_features: int = _N_FEATURES) -> np.ndarray:
    return np.random.default_rng(1).standard_normal((n_steps, n_features)).astype(np.float32)


# ── _generate_timestamps ──────────────────────────────────────────────────────

def test_generate_timestamps_correct_length():
    assert len(_generate_timestamps(FORECAST_STEPS)) == FORECAST_STEPS


def test_generate_timestamps_parseable_format():
    for stamp in _generate_timestamps(4):
        pd.to_datetime(stamp, format="%Y-%m-%d %H:%M")  # raises if format is wrong


def test_generate_timestamps_6h_spacing():
    """Consecutive timestamps should be exactly STEP_HOURS apart."""
    dts = pd.to_datetime(_generate_timestamps(4), format="%Y-%m-%d %H:%M")
    gaps = dts[1:] - dts[:-1]
    assert all(g == pd.Timedelta(hours=STEP_HOURS) for g in gaps)


def test_generate_timestamps_starts_at_6h_boundary():
    """First timestamp should fall on a 6-hour UTC boundary (00, 06, 12, or 18)."""
    first = pd.to_datetime(_generate_timestamps(1)[0], format="%Y-%m-%d %H:%M")
    assert first.hour % STEP_HOURS == 0


# ── _inverse_scale_predictions ────────────────────────────────────────────────

def test_inverse_scale_clips_below_zero():
    scaler = MagicMock()
    scaler.inverse_transform.return_value = np.array([[-0.1], [0.5], [1.2]])
    result = _inverse_scale_predictions([-0.1, 0.5, 1.2], scaler)
    assert all(0.0 <= v <= 1.0 for v in result), f"Values outside [0, 1]: {result}"


def test_inverse_scale_output_length():
    scaler = MagicMock()
    scaler.inverse_transform.side_effect = lambda x: x
    assert len(_inverse_scale_predictions([0.1, 0.2, 0.3], scaler)) == 3


def test_inverse_scale_identity_passthrough():
    """Values already in [0, 1] should pass through unchanged (within rounding)."""
    scaler = MagicMock()
    preds = [0.0, 0.25, 0.5, 0.75, 1.0]
    scaler.inverse_transform.return_value = np.array(preds).reshape(-1, 1)
    result = _inverse_scale_predictions(preds, scaler)
    for v, expected in zip(result, preds):
        assert abs(v - expected) < 1e-4


# ── _derive_forecast_metadata ─────────────────────────────────────────────────

def test_derive_metadata_lengths():
    lats, classes = _derive_forecast_metadata([0.1, 0.3, 0.6, 0.8])
    assert len(lats) == 4
    assert len(classes) == 4


def test_derive_metadata_latitude_within_bounds():
    lats, _ = _derive_forecast_metadata([0.0, 0.5, 1.0])
    assert all(45.0 <= lat <= 67.0 for lat in lats), f"Latitude out of [45, 67]: {lats}"


def test_derive_metadata_storm_class_single_char():
    _, classes = _derive_forecast_metadata([0.05, 0.2, 0.4, 0.6, 0.9])
    assert all(len(c) == 1 for c in classes), f"Storm class not single char: {classes}"


def test_derive_metadata_extreme_ssi_minimum_latitude():
    """Extreme SSI (1.0) should produce the minimum auroral latitude (45°)."""
    lats, _ = _derive_forecast_metadata([1.0])
    assert lats[0] == pytest.approx(45.0, abs=0.01)


def test_derive_metadata_quiet_ssi_maximum_latitude():
    """Quiet SSI (0.0) should produce the maximum auroral latitude (67°)."""
    lats, _ = _derive_forecast_metadata([0.0])
    assert lats[0] == pytest.approx(67.0, abs=0.01)


# ── _run_autoregressive_loop ──────────────────────────────────────────────────

def test_autoregressive_loop_output_length():
    preds = _run_autoregressive_loop(lambda w: 0.1, _seed(), _future(), _N_STEPS)
    assert len(preds) == _N_STEPS


def test_autoregressive_loop_window_rolls_forward():
    """After each step, the oldest row is dropped and the next future condition appended."""
    seen_last_rows = []

    def track(window):
        seen_last_rows.append(window[0, -1, :].copy())
        return 0.0

    future = _future(_N_STEPS)
    _run_autoregressive_loop(track, _seed(), future, _N_STEPS)

    # At step i (i >= 1), the last row in the window should be future_conditions[i-1].
    for i in range(1, _N_STEPS):
        np.testing.assert_array_equal(seen_last_rows[i], future[i - 1])


# ── _forecast_sklearn ─────────────────────────────────────────────────────────

def test_forecast_sklearn_output_length():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.2])
    preds = _forecast_sklearn(mock_model, _seed(), _future(), _N_STEPS)
    assert len(preds) == _N_STEPS


def test_forecast_sklearn_calls_predict_once_per_step():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.3])
    _forecast_sklearn(mock_model, _seed(), _future(), _N_STEPS)
    assert mock_model.predict.call_count == _N_STEPS


# ── _forecast_temporal ────────────────────────────────────────────────────────

def test_forecast_temporal_lstm_output_length():
    model = LSTMRegressor(n_features=_N_FEATURES, hidden_size=8, num_layers=1)
    model.eval()
    assert len(_forecast_temporal(model, _seed(), _future(), _N_STEPS)) == _N_STEPS


def test_forecast_temporal_gru_output_length():
    model = GRURegressor(n_features=_N_FEATURES, hidden_size=8, num_layers=1)
    model.eval()
    assert len(_forecast_temporal(model, _seed(), _future(), _N_STEPS)) == _N_STEPS


def test_forecast_temporal_returns_floats():
    model = LSTMRegressor(n_features=_N_FEATURES, hidden_size=4, num_layers=1)
    model.eval()
    preds = _forecast_temporal(model, _seed(), _future(), _N_STEPS)
    assert all(isinstance(p, float) for p in preds)


# ── _forecast_persistence ─────────────────────────────────────────────────────

def _persistence_scalers():
    """Return mock scaler_X and scaler_y for persistence tests."""
    scaler_X = MagicMock()
    # Physical values: bt=5, bz_gsm=-5, speed=450, density=8, dst=-50
    scaler_X.inverse_transform.return_value = np.array([[5.0, -5.0, 450.0, 8.0, -50.0]])

    scaler_y = MagicMock()
    scaler_y.transform.return_value = np.array([[0.25]])
    return scaler_X, scaler_y


def test_forecast_persistence_output_length():
    scaler_X, scaler_y = _persistence_scalers()
    preds = _forecast_persistence(_seed(), scaler_X, scaler_y, _N_STEPS)
    assert len(preds) == _N_STEPS


def test_forecast_persistence_constant_value():
    """All persistence predictions should be the same value (current SSI repeated)."""
    scaler_X, scaler_y = _persistence_scalers()
    preds = _forecast_persistence(_seed(), scaler_X, scaler_y, _N_STEPS)
    assert all(p == preds[0] for p in preds), f"Persistence forecast is not constant: {preds}"
