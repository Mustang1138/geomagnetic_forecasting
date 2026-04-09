"""Shared data-loading utilities for the Aurora Forecast API."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.utils import find_project_root, load_config

# Temporal models consume a warm-up sequence before emitting their first
# prediction, shifting their series forward relative to the baselines.
TEMPORAL_OFFSET = 6

DOWNSAMPLE_STEP = 6  # keep every Nth row to reduce API payload size


def _get_paths() -> tuple[Path, Path]:
    """Resolve the processed-data and outputs directories from project root."""
    root = find_project_root(Path(__file__).resolve().parent)
    cfg = load_config(str(root / "config.yaml"))
    return root / cfg["data"]["processed_dir"], root / "outputs"


def _load_pred(outputs_dir: Path, subdir: str, model_key: str) -> pd.DataFrame:
    """Load a predictions CSV and return only the y_true / y_pred columns."""
    # Baseline CSVs include a 'test_' prefix in the filename; temporal ones do not.
    prefix = "test_" if subdir == "baselines" else ""
    path = outputs_dir / subdir / "predictions" / f"{model_key}_{prefix}predictions.csv"
    return pd.read_csv(path)[["y_true", "y_pred"]].reset_index(drop=True)


@lru_cache(maxsize=1)
def load_aligned_predictions() -> pd.DataFrame:
    """Load and align all model predictions onto a common time axis."""
    processed_dir, outputs_dir = _get_paths()

    # Physical metadata is loaded from the pre-scaling snapshot so that
    # auroral_latitude_deg and storm_severity_class reflect the true SSI in
    # [0, 1] rather than the standardised training target.
    meta = pd.read_csv(processed_dir / "test_meta.csv", parse_dates=["datetime"])

    lstm_df = _load_pred(outputs_dir, "temporal", "lstm")
    gru_df = _load_pred(outputs_dir, "temporal", "gru")
    rf_df = _load_pred(outputs_dir, "baselines", "random_forest")
    lr_df = _load_pred(outputs_dir, "baselines", "linear_regression")

    n = len(lstm_df)
    sl = slice(TEMPORAL_OFFSET, TEMPORAL_OFFSET + n)

    base = meta.iloc[sl].reset_index(drop=True)
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
        "pe": rf["y_true"].shift(1).round(5),
    })

    return aligned.iloc[1::DOWNSAMPLE_STEP].reset_index(drop=True)


def get_metrics() -> list[dict]:
    """Return per-model evaluation metrics from the consolidated CSV."""
    _, outputs_dir = _get_paths()
    df = pd.read_csv(outputs_dir / "metrics_all_models.csv")

    _MODEL_META: dict[str, tuple[str, str]] = {
        "random_forest":     ("rf", "Random Forest"),
        "linear_regression": ("lr", "Linear Regression"),
        "lstm":              ("ls", "LSTM"),
        "gru":               ("gr", "GRU"),
        "persistence":       ("pe", "Persistence"),
    }

    rows = []
    for _, row in df.iterrows():
        name = row["model"]
        key, label = _MODEL_META.get(name, (name, name))
        rows.append({
            "key":   key,
            "model": name,
            "label": label,
            "rmse":  round(float(row["rmse"]), 6),
            "mae":   round(float(row["mae"]), 6),
            "r2":    round(float(row["r2"]), 6),
        })
    return rows
