"""Fetches live DSCOVR observations and returns a scaled seed window for model inference."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.preprocessing.preprocess import FEATURE_COLS
from src.utils import load_pickle

logger = logging.getLogger(__name__)


def _fetch_json(url: str, timeout: int = 30) -> Optional[list]:
    """Fetch a JSON list from a NOAA SWPC endpoint."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        logger.error("Failed to fetch %s: %s", url, exc)
        return None


def _parse_dscovr_feed(data: list) -> Optional[pd.DataFrame]:
    """Parse a NOAA SWPC JSON feed into a DataFrame."""
    if not data or len(data) < 2:
        logger.error("DSCOVR feed is empty or malformed.")
        return None

    # Feed format: first element is column headers; subsequent elements are data rows.
    df = pd.DataFrame(data[1:], columns=data[0])
    df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce")
    df = df.dropna(subset=["time_tag"])

    non_ts_cols = [c for c in df.columns if c != "time_tag"]
    df[non_ts_cols] = df[non_ts_cols].apply(pd.to_numeric, errors="coerce")

    return df.sort_values("time_tag").reset_index(drop=True)


def fetch_dscovr_feeds(config: dict) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Fetch the DSCOVR magnetometer and plasma feeds from NOAA SWPC."""
    urls = config["data"]["urls"]["dscovr"]
    feeds = {}
    for key in ("mag", "plasma"):
        raw = _fetch_json(urls[key])
        feeds[key] = _parse_dscovr_feed(raw) if raw is not None else None

    return feeds["mag"], feeds["plasma"]


def merge_dscovr_feeds(
        mag_df: pd.DataFrame,
        plasma_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Merge magnetometer and plasma feeds on their timestamps using an inner join."""
    if mag_df is None or mag_df.empty or plasma_df is None or plasma_df.empty:
        logger.error("One or both DSCOVR feeds are unavailable for merging.")
        return None

    merged = pd.merge(
        mag_df[["time_tag", "bt", "bz_gsm"]],
        plasma_df[["time_tag", "speed", "density"]],
        on="time_tag",
        how="inner",
    )

    if merged.empty:
        logger.error("Merged DSCOVR DataFrame is empty — no overlapping timestamps.")
        return None

    return merged.sort_values("time_tag").reset_index(drop=True)


def resample_to_6hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a DSCOVR DataFrame to 6-hourly averages."""
    # Resampled to 6-hourly to match the training data cadence.
    resampled = (
        df.set_index("time_tag")
        .resample("6h")
        .mean()
        .dropna(how="all")
        .reset_index()
        .rename(columns={"time_tag": "datetime"})
    )

    # Dst is not available from DSCOVR — set to zero as a neutral placeholder.
    resampled["dst"] = 0.0

    return resampled


def apply_physical_limits(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Clip feature values to the physical bounds defined in config.yaml."""
    for col, (low, high) in config.get("physical_limits", {}).items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)
    return df


def scale_features(df: pd.DataFrame, scaler) -> np.ndarray:
    """Apply the fitted feature scaler to a DataFrame."""
    return scaler.transform(df[FEATURE_COLS].values)


def _tile_to_length(arr: np.ndarray, target_length: int) -> np.ndarray:
    """Tile a 2-D array along axis 0 until it reaches ``target_length`` rows."""
    repeats = -(-target_length // len(arr))  # ceiling division
    return np.tile(arr, (repeats, 1))[:target_length]


def build_seed_window(
        sequence_length: int,
        forecast_steps: int,
        processed_dir: Path,
        config: dict,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Fetch, process, and scale the seed window and future conditions.

    Orchestrates the full real-time pipeline: fetch DSCOVR feeds, merge,
    resample to 6-hourly, clip to physical bounds, scale, and split into
    a seed window and future-conditions array.
    """
    mag_df, plasma_df = fetch_dscovr_feeds(config)

    if mag_df is None or plasma_df is None:
        logger.error("Cannot build seed window — DSCOVR feeds unavailable.")
        return None

    merged = merge_dscovr_feeds(mag_df, plasma_df)
    if merged is None:
        return None

    resampled = resample_to_6hourly(merged)

    if len(resampled) < sequence_length:
        logger.error(
            "Insufficient DSCOVR data for seed window: need %d rows, got %d.",
            sequence_length,
            len(resampled),
        )
        return None

    resampled = apply_physical_limits(resampled, config)

    scaler_X = load_pickle(processed_dir / "scaler_X.pkl")
    scaled_all = scale_features(resampled, scaler_X)

    seed = scaled_all[-sequence_length:]

    if len(scaled_all) >= forecast_steps:
        future_conditions = scaled_all[-forecast_steps:]
    else:
        logger.warning(
            "Only %d 6-hourly rows available; tiling to fill %d forecast steps.",
            len(scaled_all),
            forecast_steps,
        )
        future_conditions = _tile_to_length(scaled_all, forecast_steps)

    logger.info(
        "Seed window built: %d timesteps. Future conditions: %d steps (%d features).",
        seed.shape[0],
        future_conditions.shape[0],
        seed.shape[1],
    )

    return seed, future_conditions
