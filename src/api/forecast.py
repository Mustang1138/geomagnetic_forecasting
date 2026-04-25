"""Autoregressive forecasting engine for live geomagnetic storm prediction."""

import logging
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.features.derived_features import (
    assign_storm_severity_class,
    compute_storm_severity_index,
    estimate_auroral_latitude,
)
from src.models.rf_quantile import predict_with_ci
from src.models.temporal_model import GRURegressor, LSTMRegressor
from src.preprocessing.preprocess import FEATURE_COLS
from src.preprocessing.realtime_pipeline import build_seed_window
from src.utils import find_project_root, load_config, load_pickle

logger = logging.getLogger(__name__)

FORECAST_STEPS = 28  # 28 × 6 hours = 7 days
STEP_HOURS = 6


def _resolve_data_dirs() -> tuple[Path, Path]:
    """Resolve the processed-data and outputs directories from project root."""
    root = find_project_root(Path(__file__).resolve().parent)
    config = load_config(str(root / "config.yaml"))
    return root / config["data"]["processed_dir"], root / "outputs"


def _load_temporal_model(
        model_class: type,
        checkpoint_path: Path,
        n_features: int,
        config: dict,
        model_key: str,
) -> torch.nn.Module:
    """Instantiate and load weights for a temporal model.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        model_key: Config key (``"lstm"`` or ``"gru"``) used to read architecture hyperparameters.

    Returns:
        The model in evaluation mode on CPU.
    """
    model_cfg = config["models"][model_key]
    model = model_class(
        n_features=n_features,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
    )

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model


def _run_autoregressive_loop(
        predict_fn,
        seed_window: np.ndarray,
        future_conditions: np.ndarray,
        n_steps: int,
) -> list[float]:
    """Run the autoregressive forecast loop for a single model."""
    window = seed_window.copy()
    predictions = []

    for step in range(n_steps):
        scaled_pred = predict_fn(window[np.newaxis, :, :])
        predictions.append(float(scaled_pred))
        # Frozen-conditions assumption: the most recent DSCOVR observation is
        # repeated as input for all future steps. Standard practice in
        # operational space weather forecasting.
        window = np.vstack([window[1:], future_conditions[step]])

    return predictions


def _forecast_sklearn(
        model,
        seed_window: np.ndarray,
        future_conditions: np.ndarray,
        n_steps: int,
) -> list[float]:
    """Run the autoregressive loop for a scikit-learn tabular model."""
    current_row = seed_window[-1].reshape(1, -1)
    predictions = []

    for step in range(n_steps):
        pred = model.predict(current_row)[0]
        predictions.append(float(pred))
        current_row = future_conditions[step].reshape(1, -1)

    return predictions


def _forecast_rf_with_ci(
        rf_model,
        seed_window: np.ndarray,
        future_conditions: np.ndarray,
        n_steps: int,
        quantiles: tuple[float, float] = (0.05, 0.95),
) -> tuple[list[float], list[float], list[float]]:
    """Run the autoregressive loop for RF and return per-step 90 % quantile bounds.

    Returns ``(mean_predictions, lower_bounds, upper_bounds)``. The mean
    predictions are identical to those from ``_forecast_sklearn`` for the same
    RF model, ensuring the point-estimate series is unchanged.
    """
    current_row = seed_window[-1].reshape(1, -1)
    means: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []

    for step in range(n_steps):
        mean_pred, lower, upper = predict_with_ci(rf_model, current_row, quantiles)
        means.append(float(mean_pred[0]))
        lowers.append(float(lower[0]))
        uppers.append(float(upper[0]))
        current_row = future_conditions[step].reshape(1, -1)

    return means, lowers, uppers


def _forecast_temporal(
        model: torch.nn.Module,
        seed_window: np.ndarray,
        future_conditions: np.ndarray,
        n_steps: int,
) -> list[float]:
    """Run the autoregressive loop for a PyTorch temporal model."""

    def predict_fn(window: np.ndarray) -> float:
        with torch.no_grad():
            return model(torch.tensor(window, dtype=torch.float32)).item()

    return _run_autoregressive_loop(predict_fn, seed_window, future_conditions, n_steps)


def _forecast_persistence(
        seed_window: np.ndarray,
        scaler_X,
        scaler_y,
        n_steps: int,
) -> list[float]:
    """Generate a persistence forecast by carrying forward the current SSI."""
    last_row_physical = scaler_X.inverse_transform(seed_window[-1].reshape(1, -1))
    physical_df = compute_storm_severity_index(
        pd.DataFrame(last_row_physical, columns=FEATURE_COLS)
    )
    current_ssi = float(physical_df["storm_severity_index"].iloc[0])

    # Re-scale SSI to the same space as the other models before inverse transformation.
    scaled_ssi = float(scaler_y.transform([[current_ssi]])[0][0])

    return [scaled_ssi] * n_steps


def _derive_forecast_metadata(ssi_values: list[float]) -> tuple[list[float], list[str]]:
    """Compute auroral latitude and storm class from predicted SSI values."""
    df = assign_storm_severity_class(
        estimate_auroral_latitude(
            pd.DataFrame({"storm_severity_index": ssi_values})
        )
    )

    auroral_lats = df["auroral_latitude_deg"].round(3).tolist()

    # storm_severity_class is a pandas Categorical; take only the first character
    # to match the existing API convention.
    storm_classes = [str(c)[0] for c in df["storm_severity_class"].tolist()]

    return auroral_lats, storm_classes


def _generate_timestamps(n_steps: int) -> list[str]:
    """Generate ISO-format timestamps for each forecast step, starting at the next 6-hour boundary."""
    now = datetime.utcnow()
    hours_ahead = STEP_HOURS - (now.hour % STEP_HOURS)
    start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours_ahead)

    return [
        (start + timedelta(hours=i * STEP_HOURS)).strftime("%Y-%m-%d %H:%M")
        for i in range(n_steps)
    ]


@lru_cache(maxsize=1)
def _cached_config() -> dict:
    """Return the project configuration, cached after first load."""
    return load_config()


def _inverse_scale_predictions(
        scaled_preds: list[float],
        scaler_y,
) -> list[float]:
    """Inverse-scale a list of SSI predictions to physical units, clipped to [0, 1]."""
    ssi_values = (
        scaler_y.inverse_transform(np.array(scaled_preds).reshape(-1, 1))
        .flatten()
        .tolist()
    )
    return [round(max(0.0, min(1.0, v)), 5) for v in ssi_values]


def generate_forecast() -> Optional[dict]:
    """Generate a 7-day, 6-hourly forecast for all five models."""
    config = _cached_config()
    processed_dir, outputs_dir = _resolve_data_dirs()

    lstm_seq_len = config["models"]["lstm"]["sequence_length"]
    gru_seq_len = config["models"]["gru"]["sequence_length"]
    max_seq_len = max(lstm_seq_len, gru_seq_len)

    logger.info("Building seed window (length=%d) …", max_seq_len)
    result = build_seed_window(max_seq_len, FORECAST_STEPS, processed_dir, config)

    if result is None:
        logger.error("Forecast aborted — seed window unavailable.")
        return None

    seed_window, future_conditions = result
    n_features = seed_window.shape[1]

    scaler_X = load_pickle(processed_dir / "scaler_X.pkl")
    scaler_y = load_pickle(processed_dir / "scaler_y.pkl")

    lstm_model = _load_temporal_model(
        LSTMRegressor,
        outputs_dir / "temporal" / "models" / "lstm_best.pt",
        n_features, config, "lstm",
    )
    gru_model = _load_temporal_model(
        GRURegressor,
        outputs_dir / "temporal" / "models" / "gru_best.pt",
        n_features, config, "gru",
    )

    rf_model = load_pickle(outputs_dir / "baselines" / "models" / "random_forest.pkl")
    rf_mean, rf_lower, rf_upper = _forecast_rf_with_ci(
        rf_model, seed_window, future_conditions, FORECAST_STEPS,
    )

    raw_forecasts = {
        "rf": rf_mean,
        "lr": _forecast_sklearn(
            load_pickle(outputs_dir / "baselines" / "models" / "linear_regression.pkl"),
            seed_window, future_conditions, FORECAST_STEPS,
        ),
        "ls": _forecast_temporal(lstm_model, seed_window[-lstm_seq_len:], future_conditions, FORECAST_STEPS),
        "gr": _forecast_temporal(gru_model, seed_window[-gru_seq_len:], future_conditions, FORECAST_STEPS),
        "pe": _forecast_persistence(seed_window, scaler_X, scaler_y, FORECAST_STEPS),
    }

    rf_ci_bounds = {
        "lower": _inverse_scale_predictions(rf_lower, scaler_y),
        "upper": _inverse_scale_predictions(rf_upper, scaler_y),
    }

    models_output = {}
    for model_key, scaled_preds in raw_forecasts.items():
        ssi_values = _inverse_scale_predictions(scaled_preds, scaler_y)
        auroral_lats, storm_classes = _derive_forecast_metadata(ssi_values)
        entry = {
            "ssi": ssi_values,
            "auroral_lat": auroral_lats,
            "storm_class": storm_classes,
        }
        if model_key == "rf":
            entry["ssi_lower"] = rf_ci_bounds["lower"]
            entry["ssi_upper"] = rf_ci_bounds["upper"]
            entry["ci_level"] = 0.90
        models_output[model_key] = entry

    logger.info("Forecast generated successfully (%d steps).", FORECAST_STEPS)

    return {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "steps": FORECAST_STEPS,
        "step_hours": STEP_HOURS,
        "dscovr_conditions_used": True,
        "timestamps": _generate_timestamps(FORECAST_STEPS),
        "models": models_output,
    }
