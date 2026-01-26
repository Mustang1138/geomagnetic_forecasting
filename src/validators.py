"""
Data validation utilities for geomagnetic forecasting project.
"""

import logging
from typing import Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)

# Expected schema for OMNI2 data after parsing.
# Explicit schema validation ensures consistency between experiments
# and prevents silent downstream failures (Martin, 2008).
REQUIRED_COLUMNS = {"datetime", "bz_gsm", "speed", "density", "dst"}

# Physical bounds are derived from empirical observations of the
# near-Earth solar wind and geomagnetic response.
# Values outside these ranges typically indicate sensor errors,
# data corruption, or upstream processing artefacts rather than
# true physical extremes (Liemohn et al., 2021).

# NOTE:
# These limits are intentionally broader than those used in preprocessing.
# Validators report potential anomalies; preprocessing applies stricter,
# experiment-specific bounds defined in config.yaml.
PHYSICAL_LIMITS = {
    "bz_gsm": (-100.0, 100.0),  # nT
    "speed": (200.0, 2000.0),  # km/s
    "density": (0.0, 100.0),  # particles / cm^3
    "dst": (-500.0, 100.0),  # nT
}


def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"OMNI2 schema validation failed: missing columns {missing}")


def check_missing_data(df: pd.DataFrame) -> Dict[str, Any]:
    overall_pct = (df.isnull().sum().sum() / df.size) * 100

    per_column = (
        df.isnull()
        .mean()
        .mul(100)
        .round(2)
        .to_dict()
    )

    return {
        "overall_percentage": round(overall_pct, 2),
        "by_column": per_column,
    }


def check_date_continuity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify temporal gaps in hourly OMNI2 data.

    Continuous sampling is a key assumption for many time series
    forecasting methods, including LSTMs
    (Box et al., 2015; Cerqueira et al., 2020).
    """

    # Temporal gaps are reported rather than automatically filled.
    # Imputation strategies can introduce bias in downstream models
    # and are therefore deferred to the modelling stage, where
    # assumptions can be explicitly controlled
    # (Cerqueira et al., 2020; Liemohn et al., 2021).

    df_sorted = df.sort_values("datetime")
    deltas = df_sorted["datetime"].diff().dropna()

    hours = pd.Series(
        deltas.dt.total_seconds() / 3600.0,
        index=deltas.index,
    )

    gaps = hours[hours > 1.0]

    return {
        "num_gaps": int(len(gaps)),
        "max_gap_hours": float(gaps.max()) if not gaps.empty else 0.0,
        "gap_fraction": round(len(gaps) / len(df_sorted), 4),
    }


def check_physical_outliers(df: pd.DataFrame) -> Dict[str, Any]:
    outliers: Dict[str, Dict[str, float]] = {}

    for col, (low, high) in PHYSICAL_LIMITS.items():
        if col not in df.columns:
            continue

        series = df[col].dropna()
        mask = (series < low) | (series > high)

        outliers[col] = {
            "count": int(mask.sum()),
            "percentage": round((mask.sum() / len(series)) * 100, 3)
            if len(series) else 0.0,
        }

    return outliers


def validate_omni_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Primary OMNI2 validation entry point.
    """
    if df.empty:
        raise ValueError("OMNI2 validation failed: DataFrame is empty")

    validate_schema(df)

    return {
        "total_records": int(len(df)),
        "missing_data": check_missing_data(df),
        "date_continuity": check_date_continuity(df),
        "physical_outliers": check_physical_outliers(df),
    }


def validate_preprocessed_data(summary: Dict[str, Any]) -> None:
    """
    Validate preprocessed data meets minimum requirements for ML training.

    Enforcing minimum dataset sizes reduces the risk of unstable training
    dynamics and unreliable evaluation metrics
    (Breiman, 2001; Liemohn et al., 2021).
    """

    # Minimum sample requirements for reliable ML training
    MIN_TRAIN_SAMPLES = 1000
    MIN_VAL_SAMPLES = 200
    MIN_TEST_SAMPLES = 200

    required_keys = {
        "train_samples",
        "val_samples",
        "test_samples",
        "sequence_length",
        "n_features",
        "feature_names",
        "target_name",
    }

    missing = required_keys - summary.keys()
    if missing:
        raise ValueError(f"Preprocessing summary missing keys: {missing}")

    if summary["train_samples"] < MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"Insufficient training samples: {summary['train_samples']}"
        )

    if summary["val_samples"] < MIN_VAL_SAMPLES:
        raise ValueError(
            f"Insufficient validation samples: {summary['val_samples']}"
        )

    if summary["test_samples"] < MIN_TEST_SAMPLES:
        raise ValueError(
            f"Insufficient test samples: {summary['test_samples']}"
        )

    if summary["sequence_length"] <= 0:
        raise ValueError("Sequence length must be > 0")

    if summary["n_features"] != len(summary["feature_names"]):
        raise ValueError("Feature count mismatch")

    logger.info("✓ Preprocessed data validation passed")
