"""Schema, continuity, and physical-bounds validators for OMNI2 input data."""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"datetime", "bt", "bz_gsm", "speed", "density", "dst"}

# Broader than the preprocessing bounds in config.yaml — validators report
# anomalies rather than enforcing experiment-specific filtering.
PHYSICAL_LIMITS = {
    "bt": (0.0, 50.0),       # nT
    "bz_gsm": (-100.0, 100.0),  # nT
    "speed": (200.0, 2000.0),   # km/s
    "density": (0.0, 100.0),    # particles / cm³
    "dst": (-500.0, 100.0),     # nT
}


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if any required columns are absent."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"OMNI2 schema validation failed: missing columns {missing}")


def check_missing_data(df: pd.DataFrame) -> dict[str, Any]:
    """Return overall and per-column missing-value percentages."""
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


def check_date_continuity(df: pd.DataFrame) -> dict[str, Any]:
    """Identify temporal gaps in hourly OMNI2 data."""
    df_sorted = df.sort_values("datetime")
    deltas = df_sorted["datetime"].diff().dropna()

    hours = pd.Series(
        deltas.dt.total_seconds() / 3600.0,
        index=deltas.index,
    )

    # Gaps are reported rather than filled; imputation is deferred to the modelling stage.
    gaps = hours[hours > 1.0]

    return {
        "num_gaps": int(len(gaps)),
        "max_gap_hours": float(gaps.max()) if not gaps.empty else 0.0,
        "gap_fraction": round(len(gaps) / len(df_sorted), 4),
    }


def check_physical_outliers(df: pd.DataFrame) -> dict[str, Any]:
    """Return per-column counts and percentages of values outside physical bounds."""
    outliers: dict[str, dict[str, float]] = {}

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


def validate_omni_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Run all OMNI2 validation checks and return a summary dict."""
    if df.empty:
        raise ValueError("OMNI2 validation failed: DataFrame is empty")

    validate_schema(df)

    return {
        "total_records": int(len(df)),
        "missing_data": check_missing_data(df),
        "date_continuity": check_date_continuity(df),
        "physical_outliers": check_physical_outliers(df),
    }


def validate_preprocessed_data(summary: dict[str, Any]) -> None:
    """Raise ``ValueError`` if the preprocessing summary fails minimum-size checks."""
    min_train_samples = 1000
    min_val_samples = 200
    min_test_samples = 200

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

    if summary["train_samples"] < min_train_samples:
        raise ValueError(
            f"Insufficient training samples: {summary['train_samples']}"
        )

    if summary["val_samples"] < min_val_samples:
        raise ValueError(
            f"Insufficient validation samples: {summary['val_samples']}"
        )

    if summary["test_samples"] < min_test_samples:
        raise ValueError(
            f"Insufficient test samples: {summary['test_samples']}"
        )

    if summary["sequence_length"] <= 0:
        raise ValueError("Sequence length must be > 0")

    if summary["n_features"] != len(summary["feature_names"]):
        raise ValueError("Feature count mismatch")

    logger.info("Preprocessed data validation passed.")
