"""
Unit tests for the real-time DSCOVR data pipeline (src/preprocessing/realtime_pipeline.py).

Individual pipeline stages are tested with synthetic DataFrames and mocked
HTTP calls so no network access or trained scalers are required.

Coverage:
    _parse_dscovr_feed    — normal parse, empty feed, headers-only, sort order
    merge_dscovr_feeds    — normal merge, None/empty feed, no timestamp overlap
    resample_to_6hourly   — output columns, dst placeholder, row reduction
    apply_physical_limits — clipping, unknown column ignored, empty config
    _tile_to_length       — exact multiple, non-multiple, already-long array, values
    scale_features        — output shape and type
    build_seed_window     — success path, DSCOVR failure, insufficient resampled rows
"""

import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from src.preprocessing.realtime_pipeline import (
    _parse_dscovr_feed,
    _tile_to_length,
    apply_physical_limits,
    build_seed_window,
    merge_dscovr_feeds,
    resample_to_6hourly,
    scale_features,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mag_df(n: int = 30) -> pd.DataFrame:
    """Synthetic hourly magnetometer DataFrame."""
    return pd.DataFrame({
        "time_tag": pd.date_range("2024-01-01", periods=n, freq="1h"),
        "bt":       np.full(n, 5.0),
        "bz_gsm":   np.full(n, -3.0),
    })


def _plasma_df(n: int = 30) -> pd.DataFrame:
    """Synthetic hourly plasma DataFrame."""
    return pd.DataFrame({
        "time_tag": pd.date_range("2024-01-01", periods=n, freq="1h"),
        "speed":    np.full(n, 450.0),
        "density":  np.full(n, 6.0),
    })


def _config() -> dict:
    return {
        "data": {
            "urls": {"dscovr": {"mag": "http://fake/mag", "plasma": "http://fake/plasma"}}
        },
        "physical_limits": {
            "bt":      [0.0,   100.0],
            "bz_gsm":  [-50.0,  50.0],
            "speed":   [200.0, 1200.0],
            "density": [0.1,   200.0],
        },
    }


# ── _parse_dscovr_feed ────────────────────────────────────────────────────────

def test_parse_dscovr_feed_normal():
    raw = [
        ["time_tag", "bt", "bz_gsm"],
        ["2024-01-01 00:00:00", "5.2", "-2.1"],
        ["2024-01-01 01:00:00", "4.8", "-1.5"],
    ]
    df = _parse_dscovr_feed(raw)
    assert df is not None
    assert len(df) == 2
    assert "bt" in df.columns
    assert pd.api.types.is_float_dtype(df["bt"])


def test_parse_dscovr_feed_empty_returns_none():
    assert _parse_dscovr_feed([]) is None


def test_parse_dscovr_feed_headers_only_returns_none():
    assert _parse_dscovr_feed([["time_tag", "bt"]]) is None


def test_parse_dscovr_feed_sorted_ascending_by_time():
    raw = [
        ["time_tag", "bt"],
        ["2024-01-01 02:00:00", "4.0"],
        ["2024-01-01 00:00:00", "5.0"],
        ["2024-01-01 01:00:00", "6.0"],
    ]
    df = _parse_dscovr_feed(raw)
    times = df["time_tag"].tolist()
    assert times == sorted(times), "Feed rows should be sorted by time_tag ascending"


# ── merge_dscovr_feeds ────────────────────────────────────────────────────────

def test_merge_dscovr_feeds_produces_all_columns():
    merged = merge_dscovr_feeds(_mag_df(), _plasma_df())
    assert merged is not None
    for col in ("bt", "bz_gsm", "speed", "density"):
        assert col in merged.columns


def test_merge_dscovr_feeds_none_mag_returns_none():
    assert merge_dscovr_feeds(None, _plasma_df()) is None


def test_merge_dscovr_feeds_none_plasma_returns_none():
    assert merge_dscovr_feeds(_mag_df(), None) is None


def test_merge_dscovr_feeds_empty_mag_returns_none():
    assert merge_dscovr_feeds(pd.DataFrame(), _plasma_df()) is None


def test_merge_dscovr_feeds_no_timestamp_overlap_returns_none():
    mag = _mag_df(5)
    plasma = _plasma_df(5)
    plasma["time_tag"] = plasma["time_tag"] + pd.Timedelta(days=365)
    assert merge_dscovr_feeds(mag, plasma) is None


# ── resample_to_6hourly ───────────────────────────────────────────────────────

def test_resample_to_6hourly_columns():
    merged = merge_dscovr_feeds(_mag_df(), _plasma_df())
    resampled = resample_to_6hourly(merged)
    assert "datetime" in resampled.columns
    assert "dst" in resampled.columns


def test_resample_to_6hourly_dst_placeholder_is_zero():
    """Dst is not available from DSCOVR and should be set to 0.0."""
    merged = merge_dscovr_feeds(_mag_df(), _plasma_df())
    resampled = resample_to_6hourly(merged)
    assert (resampled["dst"] == 0.0).all()


def test_resample_to_6hourly_reduces_row_count():
    """30 hourly rows should produce fewer than 30 6-hourly rows."""
    merged = merge_dscovr_feeds(_mag_df(30), _plasma_df(30))
    resampled = resample_to_6hourly(merged)
    assert len(resampled) < 30


# ── apply_physical_limits ─────────────────────────────────────────────────────

def test_apply_physical_limits_clips_above_max():
    df = pd.DataFrame({"bt": [200.0], "speed": [2000.0]})
    limits = {"physical_limits": {"bt": [0.0, 100.0], "speed": [200.0, 1200.0]}}
    result = apply_physical_limits(df, limits)
    assert result["bt"].iloc[0] <= 100.0
    assert result["speed"].iloc[0] <= 1200.0


def test_apply_physical_limits_clips_below_min():
    df = pd.DataFrame({"bz_gsm": [-100.0]})
    result = apply_physical_limits(df, {"physical_limits": {"bz_gsm": [-50.0, 50.0]}})
    assert result["bz_gsm"].iloc[0] >= -50.0


def test_apply_physical_limits_ignores_missing_columns():
    df = pd.DataFrame({"bt": [5.0]})
    result = apply_physical_limits(df, {"physical_limits": {"bz_gsm": [-50.0, 50.0]}})
    assert "bz_gsm" not in result.columns


def test_apply_physical_limits_empty_config_is_noop():
    df = pd.DataFrame({"bt": [5.0, 200.0]})
    result = apply_physical_limits(df, {})
    pd.testing.assert_frame_equal(result, df)


# ── _tile_to_length ───────────────────────────────────────────────────────────

def test_tile_exact_multiple():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert _tile_to_length(arr, 4).shape == (4, 2)


def test_tile_non_multiple():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert _tile_to_length(arr, 5).shape == (5, 2)


def test_tile_target_shorter_than_source():
    arr = np.ones((10, 3))
    assert _tile_to_length(arr, 3).shape == (3, 3)


def test_tile_values_cycle_correctly():
    arr = np.array([[1.0], [2.0]])
    result = _tile_to_length(arr, 5)
    np.testing.assert_array_equal(result[:, 0], [1.0, 2.0, 1.0, 2.0, 1.0])


# ── scale_features ────────────────────────────────────────────────────────────

def test_scale_features_returns_ndarray_correct_shape():
    df = pd.DataFrame({
        "bt": [5.0], "bz_gsm": [-3.0], "speed": [450.0], "density": [6.0], "dst": [0.0]
    })
    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((1, 5))
    result = scale_features(df, scaler)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 5)


# ── build_seed_window ─────────────────────────────────────────────────────────

def test_build_seed_window_success(tmp_path):
    """Returns (seed, future) with correct shapes when feeds are available."""
    scaler = MagicMock()
    scaler.transform.side_effect = lambda x: np.zeros_like(x)

    with patch("src.preprocessing.realtime_pipeline.fetch_dscovr_feeds",
               return_value=(_mag_df(30), _plasma_df(30))), \
         patch("src.preprocessing.realtime_pipeline.load_pickle", return_value=scaler):
        result = build_seed_window(
            sequence_length=4,
            forecast_steps=4,
            processed_dir=tmp_path,
            config=_config(),
        )

    assert result is not None
    seed, future = result
    assert seed.shape[0] == 4
    assert future.shape[0] == 4
    assert seed.shape[1] == 5  # N_FEATURES


def test_build_seed_window_returns_none_when_feeds_unavailable(tmp_path):
    with patch("src.preprocessing.realtime_pipeline.fetch_dscovr_feeds",
               return_value=(None, None)):
        result = build_seed_window(
            sequence_length=4,
            forecast_steps=4,
            processed_dir=tmp_path,
            config=_config(),
        )
    assert result is None


def test_build_seed_window_returns_none_when_insufficient_rows(tmp_path):
    """Returns None when resampled data is shorter than sequence_length."""
    # 5 hourly rows → ~1 6-hourly row after resampling; sequence_length=50 will fail.
    with patch("src.preprocessing.realtime_pipeline.fetch_dscovr_feeds",
               return_value=(_mag_df(5), _plasma_df(5))):
        result = build_seed_window(
            sequence_length=50,
            forecast_steps=28,
            processed_dir=tmp_path,
            config=_config(),
        )
    assert result is None


def test_build_seed_window_tiles_future_when_short(tmp_path):
    """When resampled rows < forecast_steps, future_conditions is tiled to fill the gap."""
    scaler = MagicMock()
    scaler.transform.side_effect = lambda x: np.zeros_like(x)

    with patch("src.preprocessing.realtime_pipeline.fetch_dscovr_feeds",
               return_value=(_mag_df(30), _plasma_df(30))), \
         patch("src.preprocessing.realtime_pipeline.load_pickle", return_value=scaler):
        result = build_seed_window(
            sequence_length=2,
            forecast_steps=100,  # more steps than available 6-hourly rows
            processed_dir=tmp_path,
            config=_config(),
        )

    assert result is not None
    _, future = result
    assert future.shape[0] == 100
