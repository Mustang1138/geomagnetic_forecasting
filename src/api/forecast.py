"""
Forecasting engine for live geomagnetic storm prediction.

Loads all trained model artefacts, seeds the forecast from real-time
DSCOVR observations, and runs a 28-step autoregressive loop to produce
a 7-day forecast in 6-hour blocks.

Autoregressive forecasting strategy
-------------------------------------
At each step the model predicts the next SSI value from the current
input window.  The window then advances by one step, with the oldest
row dropped and the most recent real DSCOVR observation appended as
the assumed "current conditions".  This frozen-conditions assumption
is standard practice in operational space weather forecasting and is
surfaced explicitly in the API response so that the frontend can
communicate it clearly to the user.

References:
    - Hochreiter & Schmidhuber (1997) — LSTM architecture
    - Cho et al. (2014) — GRU architecture
    - Liemohn et al. (2021) — operational forecast evaluation
"""

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
from src.models.temporal_model import GRURegressor, LSTMRegressor
from src.preprocessing.realtime_pipeline import FEATURE_COLS, build_seed_window
from src.utils import load_config, load_pickle

logger = logging.getLogger(__name__)

FORECAST_STEPS = 28  # 28 × 6 hours = 7 days
STEP_HOURS = 6


# Path resolution

def _project_root() -> Path:
    """Walk up the directory tree until config.yaml is found.

    Returns:
        The project root directory.

    Raises:
        FileNotFoundError: If config.yaml cannot be located within 6 levels.
    """
    current = Path(__file__).resolve().parent
    for _ in range(6):
        if (current / "config.yaml").exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        "config.yaml not found — check project structure."
    )


def _get_dirs() -> tuple[Path, Path]:
    """Return the processed data and outputs directories.

    Returns:
        A tuple ``(processed_dir, outputs_dir)``.
    """
    root = _project_root()
    config = load_config(str(root / "config.yaml"))
    return (
        root / config["data"]["processed_dir"],
        root / "outputs",
    )


# Artefact loading

def _load_temporal_model(
        model_class: type,
        checkpoint_path: Path,
        n_features: int,
        config: dict,
        model_key: str,
) -> torch.nn.Module:
    """Instantiate and load weights for a temporal model.

    Args:
        model_class: The model class to instantiate
            (:class:`~src.models.temporal_model.LSTMRegressor` or
            :class:`~src.models.temporal_model.GRURegressor`).
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        n_features: Number of input features.
        config: Full project configuration dictionary.
        model_key: Config key (``"lstm"`` or ``"gru"``) used to read
            architecture hyperparameters.

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


# Autoregressive forecast loop

def _run_autoregressive_loop(
        predict_fn,
        seed_window: np.ndarray,
        future_conditions: np.ndarray,
        n_steps: int,
) -> list[float]:
    """Run the autoregressive forecast loop for a single model.

    At each step the model predicts the next scaled SSI from the current
    window.  The window then advances by dropping the oldest row and
    appending the real observed DSCOVR conditions for that forecast step.

    Using real past solar wind variability as the rolling input produces
    a more physically meaningful forecast than frozen conditions, since it
    reflects the actual pattern of solar wind activity over the preceding
    7 days (Liemohn et al., 2021).

    Args:
        predict_fn: A callable that accepts a window array of shape
            ``(1, seq_len, n_features)`` and returns a scalar prediction.
        seed_window: Scaled seed array of shape ``(seq_len, n_features)``.
        future_conditions: Array of shape ``(n_steps, n_features)`` — real
            observed DSCOVR conditions used as rolling inputs at each step.
        n_steps: Number of forecast steps to generate.

    Returns:
        A list of ``n_steps`` raw (scaled) SSI predictions.
    """
    window = seed_window.copy()
    predictions = []

    for step in range(n_steps):
        # Add batch dimension for model input: (1, seq_len, n_features)
        scaled_pred = predict_fn(window[np.newaxis, :, :])
        predictions.append(float(scaled_pred))

        # Advance window: drop oldest row, append real conditions for this step
        window = np.vstack([window[1:], future_conditions[step]])

    return predictions


# Per-model forecast functions

def _forecast_sklearn(
        model,
        seed_window: np.ndarray,
        future_conditions: np.ndarray,
        n_steps: int,
) -> list[float]:
    """Run the autoregressive loop for a scikit-learn tabular model.

    Tabular models receive only the most recent row rather than the full
    sequence window, matching the single-timestep input they were trained on.
    Real observed DSCOVR conditions are used as the input row at each step,
    producing varied predictions that reflect actual solar wind variability.

    Args:
        model: A fitted scikit-learn regressor.
        seed_window: Scaled seed array of shape ``(seq_len, n_features)``.
        future_conditions: Array of shape ``(n_steps, n_features)`` — real
            observed DSCOVR conditions used as inputs at each step.
        n_steps: Number of forecast steps.

    Returns:
        A list of ``n_steps`` scaled SSI predictions.
    """
    # Use the last row of the seed as the starting input.
    current_row = seed_window[-1].reshape(1, -1)
    predictions = []

    for step in range(n_steps):
        pred = model.predict(current_row)[0]
        predictions.append(float(pred))
        # Advance to the real observed conditions for the next step.
        current_row = future_conditions[step].reshape(1, -1)

    return predictions


def _forecast_temporal(
        model: torch.nn.Module,
        seed_window: np.ndarray,
        future_conditions: np.ndarray,
        n_steps: int,
) -> list[float]:
    """Run the autoregressive loop for a PyTorch temporal model.

    Args:
        model: A trained LSTM or GRU model in eval mode.
        seed_window: Scaled seed array of shape ``(seq_len, n_features)``.
        future_conditions: Array of shape ``(n_steps, n_features)`` — real
            observed DSCOVR conditions used as rolling inputs at each step.
        n_steps: Number of forecast steps.

    Returns:
        A list of ``n_steps`` scaled SSI predictions.
    """

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
    """Generate a persistence forecast by carrying forward the current SSI.

    The persistence model predicts ŷ[t+1] = y[t].  To obtain a physically
    meaningful SSI value at the seed boundary, the last row of the seed
    window is inverse-scaled back to physical units and passed through
    ``compute_storm_severity_index`` — the same function used throughout
    the training pipeline.  The resulting SSI is then re-scaled and
    repeated for all forecast steps.

    This avoids any proxy approximation and ensures the persistence
    baseline is directly comparable with the other models.

    Args:
        seed_window: Scaled seed array of shape ``(seq_len, n_features)``.
        scaler_X: The fitted feature scaler used to inverse-transform the
            last seed row back to physical units.
        scaler_y: The fitted target scaler used to re-scale the computed
            SSI so it is on the same scale as the other model predictions
            before inverse transformation.
        n_steps: Number of forecast steps.

    Returns:
        A list of ``n_steps`` identical scaled SSI predictions.
    """
    # Inverse-scale the last seed row to recover physical feature values.
    last_row_physical = scaler_X.inverse_transform(seed_window[-1].reshape(1, -1))
    physical_df = compute_storm_severity_index(
        pd.DataFrame(last_row_physical, columns=FEATURE_COLS)
    )
    current_ssi = float(physical_df["storm_severity_index"].iloc[0])

    # Re-scale SSI to match the scaled target space used by all other models,
    # so that the inverse transformation in generate_forecast() is consistent.
    scaled_ssi = float(scaler_y.transform([[current_ssi]])[0][0])

    return [scaled_ssi] * n_steps


# Derived feature helpers

def _derive_forecast_metadata(ssi_values: list[float]) -> tuple[list[float], list[str]]:
    """Compute auroral latitude and storm class from predicted SSI values.

    Reuses the existing ``estimate_auroral_latitude`` and
    ``assign_storm_severity_class`` functions from ``derived_features.py``
    to ensure consistency with the training pipeline.

    Args:
        ssi_values: List of predicted SSI values in [0, 1].

    Returns:
        A tuple ``(auroral_lats, storm_classes)``.
    """
    df = assign_storm_severity_class(
        estimate_auroral_latitude(
            pd.DataFrame({"storm_severity_index": ssi_values})
        )
    )

    auroral_lats = df["auroral_latitude_deg"].round(3).tolist()

    # storm_severity_class is a pandas Categorical — convert to plain strings
    # and take only the first character to match the existing API convention.
    storm_classes = [str(c)[0] for c in df["storm_severity_class"].tolist()]

    return auroral_lats, storm_classes


# Timestamp generation

def _generate_timestamps(n_steps: int) -> list[str]:
    """Generate ISO-format timestamps for each forecast step.

    Steps begin at the next 6-hour boundary after the current time,
    matching the 6-hourly cadence used during training.

    Args:
        n_steps: Number of forecast steps.

    Returns:
        A list of ``n_steps`` timestamp strings in ``YYYY-MM-DD HH:MM`` format.
    """
    now = datetime.utcnow()

    # Round up to the next 6-hour boundary.
    hours_ahead = STEP_HOURS - (now.hour % STEP_HOURS)
    start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours_ahead)

    return [
        (start + timedelta(hours=i * STEP_HOURS)).strftime("%Y-%m-%d %H:%M")
        for i in range(n_steps)
    ]


# Public entry point

@lru_cache(maxsize=1)
def _cached_config() -> dict:
    """Return the project configuration, cached after first load."""
    return load_config()


def _inverse_scale_predictions(
        scaled_preds: list[float],
        scaler_y,
) -> list[float]:
    """Inverse-scale a list of SSI predictions to physical units.

    Clips the result to [0, 1] — inverse scaling can occasionally produce
    values slightly outside this range due to extrapolation.

    Args:
        scaled_preds: Raw scaled predictions from a model.
        scaler_y: The fitted target scaler.

    Returns:
        A list of SSI values in [0, 1], rounded to 5 decimal places.
    """
    ssi_values = (
        scaler_y.inverse_transform(np.array(scaled_preds).reshape(-1, 1))
        .flatten()
        .tolist()
    )
    return [round(max(0.0, min(1.0, v)), 5) for v in ssi_values]


def generate_forecast() -> Optional[dict]:
    """Generate a 7-day, 6-hourly forecast for all five models.

    Orchestrates the full forecasting pipeline:
        1. Load configuration and resolve artefact paths.
        2. Build the seed window from real-time DSCOVR data.
        3. Load all trained model artefacts.
        4. Run the autoregressive loop for each model.
        5. Inverse-scale predictions to physical SSI units.
        6. Derive auroral latitude and storm class per step.
        7. Return a structured response dict.

    Returns:
        A dictionary containing:
            - ``generated_at``: UTC timestamp of forecast generation.
            - ``steps``: Number of forecast steps (28).
            - ``step_hours``: Hours between steps (6).
            - ``dscovr_conditions_used``: Always ``True`` — communicated
              to the frontend for display to the user.
            - ``timestamps``: List of forecast timestamp strings.
            - ``models``: Dict mapping model keys to their forecast arrays.

        Returns ``None`` if the DSCOVR seed window cannot be built.
    """
    config = _cached_config()
    processed_dir, outputs_dir = _get_dirs()

    # Use the longer sequence length so the seed window satisfies both
    # LSTM and GRU, which may differ after hyperparameter tuning.
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

    # Run forecasts — each returns a list of scaled SSI predictions.
    # future_conditions drives the rolling input at each step, giving all
    # models access to real observed solar wind variability from the past 7 days.
    raw_forecasts = {
        "rf": _forecast_sklearn(
            load_pickle(outputs_dir / "baselines" / "models" / "random_forest.pkl"),
            seed_window, future_conditions, FORECAST_STEPS,
        ),
        "lr": _forecast_sklearn(
            load_pickle(outputs_dir / "baselines" / "models" / "linear_regression.pkl"),
            seed_window, future_conditions, FORECAST_STEPS,
        ),
        "ls": _forecast_temporal(lstm_model, seed_window[-lstm_seq_len:], future_conditions, FORECAST_STEPS),
        "gr": _forecast_temporal(gru_model, seed_window[-gru_seq_len:], future_conditions, FORECAST_STEPS),
        "pe": _forecast_persistence(seed_window, scaler_X, scaler_y, FORECAST_STEPS),
    }

    models_output = {}
    for model_key, scaled_preds in raw_forecasts.items():
        ssi_values = _inverse_scale_predictions(scaled_preds, scaler_y)
        auroral_lats, storm_classes = _derive_forecast_metadata(ssi_values)
        models_output[model_key] = {
            "ssi": ssi_values,
            "auroral_lat": auroral_lats,
            "storm_class": storm_classes,
        }

    logger.info("Forecast generated successfully (%d steps).", FORECAST_STEPS)

    return {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "steps": FORECAST_STEPS,
        "step_hours": STEP_HOURS,
        "dscovr_conditions_used": True,
        "timestamps": _generate_timestamps(FORECAST_STEPS),
        "models": models_output,
    }
