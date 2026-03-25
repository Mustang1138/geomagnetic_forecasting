"""Shared data-loading utilities for the Aurora Forecast API."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.utils import load_config

# Rows to skip at the start so baseline indices align with the temporal models,
# which consume a warm-up window before producing their first prediction.
TEMPORAL_OFFSET = 6
DOWNSAMPLE_STEP = 6  # keep every Nth row to reduce API payload size


# Internal helpers

def _project_root() -> Path:
    """Walk up the directory tree until config.yaml is found."""
    current = Path(__file__).resolve().parent
    for _ in range(6):
        if (current / "config.yaml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("config.yaml not found – check project structure.")


def _get_paths() -> tuple[Path, Path]:
    root = _project_root()
    cfg = load_config(str(root / "config.yaml"))
    return root / cfg["data"]["processed_dir"], root / "outputs"


def _load_pred(outputs_dir: Path, subdir: str, model_key: str) -> pd.DataFrame:
    """Load a predictions CSV and return only the y_true / y_pred columns."""
    # Baseline CSVs include a 'test_' prefix in the filename; temporal ones don't
    prefix = "test_" if subdir == "baselines" else ""
    path = outputs_dir / subdir / "predictions" / f"{model_key}_{prefix}predictions.csv"
    return pd.read_csv(path)[["y_true", "y_pred"]].reset_index(drop=True)


# Public API

@lru_cache(maxsize=1)
def load_aligned_predictions() -> pd.DataFrame:
    """
    Load and align all model predictions onto a common time axis.

    LSTM and GRU produce fewer rows than the baselines due to their
    sequence warm-up.  Baseline rows are sliced to match, then the
    whole dataset is downsampled to 6-hour steps.
    """
    processed_dir, outputs_dir = _get_paths()

    baseline = pd.read_csv(processed_dir / "test_baseline.csv", parse_dates=["datetime"])
    lstm_df = _load_pred(outputs_dir, "temporal", "lstm")
    gru_df = _load_pred(outputs_dir, "temporal", "gru")
    rf_df = _load_pred(outputs_dir, "baselines", "random_forest")
    lr_df = _load_pred(outputs_dir, "baselines", "linear_regression")

    n = len(lstm_df)
    sl = slice(TEMPORAL_OFFSET, TEMPORAL_OFFSET + n)

    base = baseline.iloc[sl].reset_index(drop=True)
    rf = rf_df.iloc[sl].reset_index(drop=True)
    lr = lr_df.iloc[sl].reset_index(drop=True)

    aligned = pd.DataFrame({
        "datetime": base["datetime"],
        "auroral_lat": base["auroral_latitude_deg"].round(3),
        "storm_class": base["storm_severity_class"].str[0],
        "true": rf["y_true"].round(5),
        "rf": rf["y_pred"].round(5),
        "lr": lr["y_pred"].round(5),
        "ls": lstm_df["y_pred"].round(5),
        "gr": gru_df["y_pred"].round(5),
        "pe": rf["y_true"].shift(1).round(5),  # persistence: previous timestep's observed value
    })

    # Row 0 has a NaN persistence value; drop it then downsample
    return aligned.iloc[1::DOWNSAMPLE_STEP].reset_index(drop=True)


def get_metrics() -> list[dict]:
    """Return per-model evaluation metrics from the consolidated CSV."""
    _, outputs_dir = _get_paths()
    df = pd.read_csv(outputs_dir / "metrics_all_models.csv")

    KEY_MAP = {"random_forest": "rf", "linear_regression": "lr", "lstm": "ls", "gru": "gr", "persistence": "pe"}
    LABEL_MAP = {"random_forest": "Random Forest", "linear_regression": "Linear Regression",
                 "lstm": "LSTM", "gru": "GRU", "persistence": "Persistence"}

    return [
        {
            "key": KEY_MAP.get(row["model"], row["model"]),
            "model": row["model"],
            "label": LABEL_MAP.get(row["model"], row["model"]),
            "rmse": round(float(row["rmse"]), 6),
            "mae": round(float(row["mae"]), 6),
            "r2": round(float(row["r2"]), 6),
        }
        for _, row in df.iterrows()
    ]
