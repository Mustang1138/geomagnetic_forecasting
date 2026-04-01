"""
Real-time data pipeline for live geomagnetic storm forecasting.

Fetches the latest DSCOVR solar wind observations from NOAA SWPC,
merges the magnetometer and plasma feeds, resamples to 6-hourly averages
to match the training data cadence, and returns a scaled seed window
ready for model inference.

This module is intentionally stateless — all functions are pure
transformations with no side effects, making them straightforward to
test and reuse (Martin, 2008).

References:
    - Cristoforetti et al. (2022) — real-time vs historical data streams
    - Papitashvili and King (2020) — OMNI2 data documentation
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.utils import load_pickle

logger = logging.getLogger(__name__)

# Feature order must match DataPreprocessor.FEATURE_COLS exactly so that
# the fitted scaler_X transforms the correct columns in the correct order.
FEATURE_COLS = ["bt", "bz_gsm", "speed", "density", "dst"]


# DSCOVR fetching

def _fetch_json(url: str, timeout: int = 30) -> Optional[list]:
    """Fetch a JSON list from a NOAA SWPC endpoint.

    Args:
        url: The endpoint URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        The parsed JSON list, or ``None`` if the request fails.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        logger.error("Failed to fetch %s: %s", url, exc)
        return None


def _parse_dscovr_feed(data: list) -> Optional[pd.DataFrame]:
    """Parse a NOAA SWPC JSON feed into a DataFrame.

    The feed format is a list whose first element contains column headers
    and whose subsequent elements contain data rows.

    Args:
        data: Raw JSON list from the NOAA SWPC endpoint.

    Returns:
        A parsed DataFrame with a ``time_tag`` datetime column, or ``None``
        if parsing fails.
    """
    if not data or len(data) < 2:
        logger.error("DSCOVR feed is empty or malformed.")
        return None

    df = pd.DataFrame(data[1:], columns=data[0])
    df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce")
    df = df.dropna(subset=["time_tag"])

    # Coerce all non-timestamp columns to numeric, replacing invalid
    # sentinel values with NaN.
    non_ts_cols = [c for c in df.columns if c != "time_tag"]
    df[non_ts_cols] = df[non_ts_cols].apply(pd.to_numeric, errors="coerce")

    return df.sort_values("time_tag").reset_index(drop=True)


def fetch_dscovr_feeds(config: dict) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Fetch the DSCOVR magnetometer and plasma feeds from NOAA SWPC.

    Args:
        config: The full project configuration dictionary.

    Returns:
        A tuple ``(mag_df, plasma_df)``.  Either element may be ``None``
        if its feed could not be fetched or parsed.
    """
    urls = config["data"]["urls"]["dscovr"]
    feeds = {}
    for key in ("mag", "plasma"):
        raw = _fetch_json(urls[key])
        feeds[key] = _parse_dscovr_feed(raw) if raw is not None else None

    return feeds["mag"], feeds["plasma"]


# Merging and resampling

def merge_dscovr_feeds(
        mag_df: pd.DataFrame,
        plasma_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Merge magnetometer and plasma feeds on their timestamps.

    Performs an inner join so only timesteps present in both feeds are
    retained, avoiding NaN-filled rows that would degrade model input
    quality.

    Args:
        mag_df: Parsed magnetometer DataFrame containing ``bt`` and ``bz_gsm``.
        plasma_df: Parsed plasma DataFrame containing ``speed`` and ``density``.

    Returns:
        A merged DataFrame indexed by ``time_tag``, or ``None`` if either
        input is empty.
    """
    if mag_df is None or mag_df.empty or plasma_df is None or plasma_df.empty:
        logger.error("One or both DSCOVR feeds are unavailable for merging.")
        return None

    # Select required columns — the NOAA SWPC plasma feed uses plain
    # "speed" and "density" column names (no "proton_" prefix).
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
    """Resample a 1-minute DSCOVR DataFrame to 6-hourly averages.

    Averaging over 6-hour windows matches the cadence used during training
    and reduces noise in the real-time feed (Cristoforetti et al., 2022).

    Since DSCOVR does not provide a Dst index, the ``dst`` column is filled
    with zeros after resampling.  This is a deliberate simplification —
    real-time Dst estimates are available from other sources but are out of
    scope here.  The effect is that the forecast will under-predict SSI
    slightly during active periods, which is a conservative and safe bias
    for an operational system.

    Args:
        df: Merged DSCOVR DataFrame with a ``time_tag`` column.

    Returns:
        A resampled DataFrame with one row per 6-hour window and a
        ``dst`` column set to zero.
    """
    resampled = (
        df.set_index("time_tag")
        .resample("6h")
        .mean()
        .dropna(how="all")
        .reset_index()
        .rename(columns={"time_tag": "datetime"})
    )

    # Dst is not available from DSCOVR — set to zero as a neutral placeholder.
    # This means the model relies on the remaining four features for the seed,
    # which is conservative but avoids fabricating a geomagnetic index value.
    resampled["dst"] = 0.0

    return resampled


# Physical clipping

def apply_physical_limits(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Clip feature values to the physical bounds defined in config.yaml.

    Mirrors the outlier handling in ``DataPreprocessor._remove_physical_outliers``
    but uses clipping rather than row removal so that the seed window always
    retains the required number of rows.

    Args:
        df: DataFrame containing the feature columns.
        config: The full project configuration dictionary.

    Returns:
        DataFrame with feature values clipped to physical bounds.
    """
    for col, (low, high) in config.get("physical_limits", {}).items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)

    return df


# Scaling

def scale_features(df: pd.DataFrame, scaler) -> np.ndarray:
    """Apply the fitted feature scaler to a DataFrame.

    Args:
        df: DataFrame containing exactly the columns in ``FEATURE_COLS``,
            in the correct order.
        scaler: A fitted scikit-learn ``StandardScaler``.

    Returns:
        A scaled NumPy array of shape ``(n_rows, n_features)``.
    """
    return scaler.transform(df[FEATURE_COLS].values)


# Public entry point

def _tile_to_length(arr: np.ndarray, target_length: int) -> np.ndarray:
    """Tile a 2-D array along axis 0 until it reaches ``target_length`` rows.

    Used to fill the future-conditions window when fewer than
    ``target_length`` 6-hourly rows are available from DSCOVR.

    Args:
        arr: Source array of shape ``(n, features)`` where ``n > 0``.
        target_length: Desired number of rows in the output.

    Returns:
        Array of shape ``(target_length, features)``.
    """
    repeats = -(-target_length // len(arr))  # ceiling division
    return np.tile(arr, (repeats, 1))[:target_length]


def build_seed_window(
        sequence_length: int,
        forecast_steps: int,
        processed_dir: Path,
        config: dict,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Fetch, process, and scale the seed window and future conditions.

    Orchestrates the full real-time pipeline:
        1. Fetch DSCOVR mag and plasma feeds.
        2. Merge on timestamp.
        3. Resample to 6-hourly averages.
        4. Clip to physical bounds.
        5. Scale using the fitted ``scaler_X``.
        6. Split into a seed window (most recent ``sequence_length`` rows)
           and a future-conditions array (``forecast_steps`` rows drawn
           from the full resampled window).

    The future-conditions array drives the rolling input at each forecast
    step, replacing the frozen-conditions assumption with real observed
    solar wind variability from the past 7 days.  If fewer than
    ``forecast_steps`` rows are available after resampling, the available
    rows are tiled cyclically to fill the window.

    Args:
        sequence_length: Number of timesteps for the model seed window.
            Must match the ``sequence_length`` used during training.
        forecast_steps: Number of future-condition rows required
            (one per forecast step).
        processed_dir: Directory containing the fitted ``scaler_X.pkl``.
        config: The full project configuration dictionary.

    Returns:
        A tuple ``(seed, future_conditions)`` where:
            - ``seed`` has shape ``(sequence_length, n_features)``
            - ``future_conditions`` has shape ``(forecast_steps, n_features)``
        Returns ``None`` if the pipeline fails at any step.
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

    # Seed: the most recent sequence_length rows prime the model state.
    seed = scaled_all[-sequence_length:]

    # Future conditions: use the full resampled window as rolling inputs
    # for each forecast step, tiling if necessary to reach forecast_steps rows.
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
