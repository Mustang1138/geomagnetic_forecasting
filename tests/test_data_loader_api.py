"""
Unit tests for the API data loading layer (src/api/data_loader.py).

Tests verify that _load_pred reads the correct filename conventions for
baseline vs temporal models, that load_aligned_predictions correctly applies
TEMPORAL_OFFSET and DOWNSAMPLE_STEP to align five model prediction series
onto a common time axis, and that get_metrics correctly parses the
consolidated metrics CSV and maps internal model names to API keys.

The lru_cache on load_aligned_predictions is cleared between tests to
prevent one test's mock from leaking into the next.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from src.api.data_loader import (
    TEMPORAL_OFFSET,
    _load_pred,
    get_metrics,
    load_aligned_predictions,
)

_N = 100  # total rows in synthetic meta / baseline CSVs
_N_TEMPORAL = _N - TEMPORAL_OFFSET  # temporal CSVs are shorter by the offset


# ── CSV fixture writers ───────────────────────────────────────────────────────

def _write_pred_csv(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    pd.DataFrame({"y_true": rng.random(n), "y_pred": rng.random(n)}).to_csv(path, index=False)


def _write_meta_csv(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "datetime":             pd.date_range("2022-01-01", periods=n, freq="6h"),
        "auroral_latitude_deg": np.full(n, 63.0),
        "storm_severity_class": ["quiet"] * n,
    }).to_csv(path, index=False)


def _write_metrics_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "model": ["random_forest", "linear_regression", "lstm", "gru", "persistence"],
        "rmse": [0.050, 0.070, 0.040, 0.041, 0.100],
        "mae": [0.040, 0.050, 0.030, 0.031, 0.080],
        "r2": [0.800, 0.700, 0.850, 0.840, 0.500],
    }).to_csv(path, index=False)


@pytest.fixture
def dirs(tmp_path):
    """Full synthetic outputs + processed directory tree."""
    processed = tmp_path / "processed"
    outputs = tmp_path / "outputs"

    _write_meta_csv(processed / "test_meta.csv", _N)

    baselines_pred = outputs / "baselines" / "predictions"
    temporal_pred = outputs / "temporal" / "predictions"
    _write_pred_csv(temporal_pred / "lstm_predictions.csv", _N_TEMPORAL)
    _write_pred_csv(temporal_pred / "gru_predictions.csv", _N_TEMPORAL)
    _write_pred_csv(baselines_pred / "random_forest_test_predictions.csv", _N)
    _write_pred_csv(baselines_pred / "linear_regression_test_predictions.csv", _N)
    _write_metrics_csv(outputs / "metrics_all_models.csv")

    return processed, outputs


# ── _load_pred ────────────────────────────────────────────────────────────────

def test_load_pred_baseline_reads_correct_file(tmp_path):
    """Baseline predictions use a 'test_' filename prefix."""
    pred_dir = tmp_path / "outputs" / "baselines" / "predictions"
    _write_pred_csv(pred_dir / "random_forest_test_predictions.csv", 50)
    df = _load_pred(tmp_path / "outputs", "baselines", "random_forest")
    assert list(df.columns) == ["y_true", "y_pred"]
    assert len(df) == 50


def test_load_pred_temporal_reads_correct_file(tmp_path):
    """Temporal predictions use no 'test_' prefix."""
    pred_dir = tmp_path / "outputs" / "temporal" / "predictions"
    _write_pred_csv(pred_dir / "lstm_predictions.csv", 40)
    df = _load_pred(tmp_path / "outputs", "temporal", "lstm")
    assert list(df.columns) == ["y_true", "y_pred"]
    assert len(df) == 40


def test_load_pred_index_is_reset(tmp_path):
    """Returned index should be 0-based regardless of CSV row numbers."""
    pred_dir = tmp_path / "outputs" / "baselines" / "predictions"
    _write_pred_csv(pred_dir / "random_forest_test_predictions.csv", 20)
    df = _load_pred(tmp_path / "outputs", "baselines", "random_forest")
    assert list(df.index) == list(range(20))


# ── load_aligned_predictions ──────────────────────────────────────────────────

def test_aligned_predictions_expected_columns(dirs):
    processed, outputs = dirs
    load_aligned_predictions.cache_clear()
    with patch("src.api.data_loader._get_paths", return_value=(processed, outputs)):
        df = load_aligned_predictions()
    expected = {"datetime", "auroral_lat", "storm_class", "true", "rf", "lr", "ls", "gr", "pe"}
    assert expected.issubset(df.columns), f"Missing columns: {expected - set(df.columns)}"


def test_aligned_predictions_no_nan_in_numeric_columns(dirs):
    """
    The persistence column (pe) is derived via shift(1), making the first row
    NaN; the iloc[1::DOWNSAMPLE_STEP] slice discards that row, so no NaN should
    remain in any numeric column of the returned DataFrame.
    """
    processed, outputs = dirs
    load_aligned_predictions.cache_clear()
    with patch("src.api.data_loader._get_paths", return_value=(processed, outputs)):
        df = load_aligned_predictions()
    for col in ("true", "rf", "lr", "ls", "gr", "pe"):
        n_nan = df[col].isna().sum()
        assert n_nan == 0, f"Unexpected NaN values in column '{col}': {n_nan}"


def test_aligned_predictions_positive_row_count(dirs):
    processed, outputs = dirs
    load_aligned_predictions.cache_clear()
    with patch("src.api.data_loader._get_paths", return_value=(processed, outputs)):
        df = load_aligned_predictions()
    assert len(df) > 0


def test_aligned_predictions_downsampled(dirs):
    """
    The returned DataFrame should contain fewer rows than the raw temporal
    series, confirming that DOWNSAMPLE_STEP is applied.
    """
    processed, outputs = dirs
    load_aligned_predictions.cache_clear()
    with patch("src.api.data_loader._get_paths", return_value=(processed, outputs)):
        df = load_aligned_predictions()
    assert len(df) < _N_TEMPORAL


def test_aligned_predictions_datetime_is_monotonic(dirs):
    """Datetime column should be in ascending order after alignment."""
    processed, outputs = dirs
    load_aligned_predictions.cache_clear()
    with patch("src.api.data_loader._get_paths", return_value=(processed, outputs)):
        df = load_aligned_predictions()
    assert df["datetime"].is_monotonic_increasing


# ── get_metrics ───────────────────────────────────────────────────────────────

def test_get_metrics_returns_five_entries(dirs):
    processed, outputs = dirs
    with patch("src.api.data_loader._get_paths", return_value=(processed, outputs)):
        metrics = get_metrics()
    assert len(metrics) == 5


def test_get_metrics_required_fields(dirs):
    processed, outputs = dirs
    with patch("src.api.data_loader._get_paths", return_value=(processed, outputs)):
        metrics = get_metrics()
    for entry in metrics:
        for field in ("key", "model", "label", "rmse", "mae", "r2"):
            assert field in entry, f"Missing field '{field}' in metrics entry"


def test_get_metrics_api_keys(dirs):
    """Internal model names should be mapped to the short API keys used by the frontend."""
    processed, outputs = dirs
    with patch("src.api.data_loader._get_paths", return_value=(processed, outputs)):
        metrics = get_metrics()
    keys = {m["key"] for m in metrics}
    assert keys == {"rf", "lr", "ls", "gr", "pe"}


def test_get_metrics_values_are_finite(dirs):
    processed, outputs = dirs
    with patch("src.api.data_loader._get_paths", return_value=(processed, outputs)):
        metrics = get_metrics()
    for entry in metrics:
        for field in ("rmse", "mae", "r2"):
            assert np.isfinite(entry[field]), f"Non-finite {field} for model '{entry['key']}'"
