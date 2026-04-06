"""
Hyperparameter tuning script for temporal sequence models.

Performs a full grid search over candidate hyperparameter combinations for
both the LSTM and GRU models.  For each combination the script:

    1. Rebuilds sequence arrays for both models at the candidate sequence_length.
    2. Trains the model using the shared training loop from train_utils.
    3. Records the best validation loss achieved.

Test-set inference is intentionally excluded from the tuning loop to keep
the test set truly held-out.  A single final training run on the winning
configuration (with test inference) should be performed manually afterwards
by updating ``config.yaml`` and re-running the individual training scripts.

Sequence length search values are expressed in 6-hourly time steps,
consistent with the 6-hourly resampling applied to training data by
preprocess.py.  The correspondence between search values and lookback
windows is:

    sequence_length=12  →  12 × 6 h =  3 days
    sequence_length=24  →  24 × 6 h =  6 days
    sequence_length=48  →  48 × 6 h = 12 days

Results are saved to ``outputs/tuning/tuning_results.csv`` and printed as
a ranked summary on completion.

Usage:
    python -m src.models.training.tune

References:
    - Bergstra & Bengio (2012) — random vs grid search for hyperparameters
    - Cerqueira et al. (2020) — time-series evaluation best practices
"""

import copy
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.temporal_model import GRURegressor, LSTMRegressor
from src.models.training.train_utils import (
    _load_split,
    build_dataloader,
    run_epoch,
    run_validation,
)
from src.preprocessing.preprocess import DataPreprocessor
from src.utils import load_config, setup_logging

logger = setup_logging()

# Search space

# Sequence lengths are expressed in 6-hourly time steps, matching the
# resampled cadence of the training data.  Values correspond to lookback
# windows of 3, 6, and 12 days respectively.
SEARCH_SPACE: dict[str, list[Any]] = {
    "sequence_length": [12, 24, 48],
    "num_layers": [1, 2],
    "hidden_size": [64, 128],
    "learning_rate": [0.001, 0.0005],
    "patience": [8, 15],
}

# Models to tune — maps a label to its model class.
MODELS: dict[str, type[nn.Module]] = {
    "lstm": LSTMRegressor,
    "gru": GRURegressor,
}


# Result container

@dataclass
class TuningResult:
    """Stores the outcome of a single hyperparameter combination.

    Attributes:
        model_name: ``"lstm"`` or ``"gru"``.
        sequence_length: Sequence window length used (in 6-hourly steps).
        num_layers: Number of stacked recurrent layers.
        hidden_size: Hidden state dimensionality.
        learning_rate: Adam optimiser learning rate.
        patience: Early-stopping patience in epochs.
        best_val_loss: Best validation loss achieved during training.
        epochs_trained: Number of epochs completed before stopping.
        duration_seconds: Wall-clock training time in seconds.
    """

    model_name: str
    sequence_length: int
    num_layers: int
    hidden_size: int
    learning_rate: float
    patience: int
    best_val_loss: float
    epochs_trained: int
    duration_seconds: float


# Sequence rebuilding

def rebuild_sequences(
        sequence_length: int,
        config: dict[str, Any],
        data_dir: Path,
) -> None:
    """Rebuild per-model ``.npy`` sequence arrays for a given window length.

    Generates separate arrays for LSTM and GRU at the candidate
    sequence_length, saving them with model-specific filename suffixes
    (``X_train_lstm.npy``, ``X_train_gru.npy``, etc.).  This ensures the
    tuning loop exercises both models against identically windowed data.

    Modifies a copy of the config in memory so that ``DataPreprocessor``
    uses the candidate ``sequence_length`` without altering ``config.yaml``
    on disk.

    Args:
        sequence_length: The candidate window length to build sequences for.
        config: The full project configuration dictionary.
        data_dir: Directory containing the baseline CSV splits and to which
            the new ``.npy`` arrays will be written.
    """
    logger.info("Rebuilding sequences with sequence_length=%d …", sequence_length)

    # Deep-copy so we never mutate the live config dict.
    patched_config = copy.deepcopy(config)
    patched_config["models"]["lstm"]["sequence_length"] = sequence_length
    patched_config["models"]["gru"]["sequence_length"] = sequence_length

    preprocessor = DataPreprocessor.__new__(DataPreprocessor)
    preprocessor.config = patched_config
    preprocessor.scaler_X = _load_scaler(data_dir / "scaler_X.pkl")
    preprocessor.scaler_y = _load_scaler(data_dir / "scaler_y.pkl")

    # Rebuild sequences for each split from the frozen 6-hourly baseline CSVs.
    for split in ("train", "val", "test"):
        df = pd.read_csv(data_dir / f"{split}_baseline.csv")
        X_seq, y_seq = preprocessor._make_sequences(df, sequence_length)

        # Save with per-model suffixes so train_lstm.py and train_gru.py
        # each load the correct arrays after tuning completes.
        for model_key in ("lstm", "gru"):
            np.save(data_dir / f"X_{split}_{model_key}.npy", X_seq)
            np.save(data_dir / f"y_{split}_{model_key}.npy", y_seq)

    logger.info("Sequences rebuilt successfully.")


def _load_scaler(path: Path) -> Any:
    """Load a fitted scikit-learn scaler from disk.

    Args:
        path: Path to the pickled scaler file.

    Returns:
        The deserialised scaler object.

    Raises:
        FileNotFoundError: If the scaler file does not exist.
    """
    import pickle

    if not path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {path}. "
            "Run the preprocessing pipeline before tuning."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


# Single combination training run

def train_combination(
        model_name: str,
        model_class: type[nn.Module],
        combo: dict[str, Any],
        data_dir: Path,
        output_dir: Path,
        base_config: dict[str, Any],
) -> TuningResult:
    """Train a single hyperparameter combination and return the result.

    Mirrors the training loop in :class:`~src.models.training.train_utils.Trainer`
    but omits test inference and final checkpointing to keep tuning fast.
    Only the best validation loss is recorded.

    Args:
        model_name: ``"lstm"`` or ``"gru"``.
        model_class: The recurrent model class to instantiate.
        combo: Dictionary of hyperparameter values for this combination.
        data_dir: Directory containing preprocessed ``.npy`` arrays.
        output_dir: Directory for temporary tuning checkpoints.
        base_config: Full project configuration dictionary.

    Returns:
        A :class:`TuningResult` summarising the outcome.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model-specific prefixed arrays so both models always train on
    # windows of the candidate sequence_length.
    X_train, y_train = _load_split(data_dir, "train", prefix=model_name)
    X_val, y_val = _load_split(data_dir, "val", prefix=model_name)

    train_loader = build_dataloader(X_train, y_train, batch_size=32)
    val_loader = build_dataloader(X_val, y_val, batch_size=256)

    input_size = X_train.shape[2]

    model = model_class(
        n_features=input_size,
        hidden_size=combo["hidden_size"],
        num_layers=combo["num_layers"],
    ).to(device)

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=combo["learning_rate"])

    best_val_loss = float("inf")
    patience_counter = 0
    epochs_trained = 0
    max_epochs = base_config["models"][model_name]["epochs"]

    # Tuning checkpoints go to a temporary directory so they do not
    # overwrite the production model weights.
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}_tune_best.pt"

    start = time.perf_counter()

    for epoch in range(max_epochs):
        run_epoch(model, train_loader, criterion, optimiser, device)
        val_loss = run_validation(model, val_loader, criterion, device)
        epochs_trained += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= combo["patience"]:
                break

    duration = time.perf_counter() - start

    return TuningResult(
        model_name=model_name,
        sequence_length=combo["sequence_length"],
        num_layers=combo["num_layers"],
        hidden_size=combo["hidden_size"],
        learning_rate=combo["learning_rate"],
        patience=combo["patience"],
        best_val_loss=best_val_loss,
        epochs_trained=epochs_trained,
        duration_seconds=round(duration, 1),
    )


# Grid search

def run_grid_search(
        data_dir: Path,
        output_dir: Path,
        config: dict[str, Any],
) -> list[TuningResult]:
    """Run the full grid search across all combinations and both models.

    Rebuilds sequence arrays only when ``sequence_length`` changes from the
    previous combination, avoiding redundant preprocessing work.

    Args:
        data_dir: Directory containing preprocessed baseline CSVs and
            ``.npy`` arrays.
        output_dir: Directory for tuning outputs.
        config: Full project configuration dictionary.

    Returns:
        A list of :class:`TuningResult` objects, one per
        (model, combination) pair.
    """
    param_names = list(SEARCH_SPACE.keys())
    param_values = list(SEARCH_SPACE.values())
    combinations = [
        dict(zip(param_names, values))
        for values in itertools.product(*param_values)
    ]

    total_runs = len(combinations) * len(MODELS)
    logger.info(
        "Starting grid search: %d combinations × %d models = %d total runs.",
        len(combinations), len(MODELS), total_runs,
    )

    results: list[TuningResult] = []
    current_seq_len: int | None = None
    run_number = 0

    for combo in combinations:
        # Rebuild sequences only when sequence_length changes.
        if combo["sequence_length"] != current_seq_len:
            rebuild_sequences(combo["sequence_length"], config, data_dir)
            current_seq_len = combo["sequence_length"]

        for model_name, model_class in MODELS.items():
            run_number += 1
            logger.info(
                "Run %d/%d — %s | seq_len=%d  layers=%d  "
                "hidden=%d  lr=%s  patience=%d",
                run_number, total_runs,
                model_name.upper(),
                combo["sequence_length"],
                combo["num_layers"],
                combo["hidden_size"],
                combo["learning_rate"],
                combo["patience"],
            )

            result = train_combination(
                model_name=model_name,
                model_class=model_class,
                combo=combo,
                data_dir=data_dir,
                output_dir=output_dir,
                base_config=config,
            )

            results.append(result)
            logger.info(
                "  → best val loss: %.6f  (%d epochs, %.1fs)",
                result.best_val_loss,
                result.epochs_trained,
                result.duration_seconds,
            )

    return results


# Reporting

def save_and_report(results: list[TuningResult], output_dir: Path) -> None:
    """Save results to CSV and print a ranked summary per model.

    Args:
        results: All tuning results from the grid search.
        output_dir: Directory to which ``tuning_results.csv`` is written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "model": r.model_name,
            "sequence_length": r.sequence_length,
            "num_layers": r.num_layers,
            "hidden_size": r.hidden_size,
            "learning_rate": r.learning_rate,
            "patience": r.patience,
            "best_val_loss": round(r.best_val_loss, 8),
            "epochs_trained": r.epochs_trained,
            "duration_seconds": r.duration_seconds,
        }
        for r in results
    ]

    df = pd.DataFrame(rows).sort_values(
        ["model", "best_val_loss"]
    ).reset_index(drop=True)

    csv_path = output_dir / "tuning_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Tuning results saved → %s", csv_path)

    # Print ranked summary per model.
    for model_name in MODELS:
        subset = df[df["model"] == model_name].reset_index(drop=True)
        print(f"\n{'=' * 60}")
        print(f"  {model_name.upper()} — ranked by validation loss")
        print(f"{'=' * 60}")
        print(subset.to_string(index=False))

    # Print the best config per model for copy-paste into config.yaml.
    print(f"\n{'=' * 60}")
    print("  RECOMMENDED CONFIG.YAML UPDATES")
    print("  (sequence_length values are in 6-hourly steps)")
    print(f"{'=' * 60}")

    for model_name in MODELS:
        subset = df[df["model"] == model_name]
        best = subset.iloc[0]
        print(f"\n  {model_name}:")
        print(f"    sequence_length: {int(best['sequence_length'])}")
        print(f"    num_layers:      {int(best['num_layers'])}")
        print(f"    hidden_size:     {int(best['hidden_size'])}")
        print(f"    learning_rate:   {best['learning_rate']}")
        print(f"    patience:        {int(best['patience'])}")


# Restore original sequences

def restore_original_sequences(
        config: dict[str, Any],
        data_dir: Path,
) -> None:
    """Restore per-model ``.npy`` arrays to the lengths in ``config.yaml``.

    Called after the grid search completes so that the on-disk arrays
    match the sequence lengths configured for the final training runs.

    Args:
        config: Full project configuration dictionary (unmodified).
        data_dir: Directory containing baseline CSV splits and ``.npy`` files.
    """
    lstm_seq = config["models"]["lstm"]["sequence_length"]
    gru_seq = config["models"]["gru"]["sequence_length"]

    # Restore each model's arrays independently so each reflects the
    # sequence_length that will be used in the final training run.
    for model_key, seq_len in (("lstm", lstm_seq), ("gru", gru_seq)):
        logger.info(
            "Restoring %s sequence arrays (sequence_length=%d) …",
            model_key.upper(), seq_len,
        )
        preprocessor = DataPreprocessor.__new__(DataPreprocessor)
        preprocessor.config = config
        preprocessor.scaler_X = _load_scaler(data_dir / "scaler_X.pkl")
        preprocessor.scaler_y = _load_scaler(data_dir / "scaler_y.pkl")

        for split in ("train", "val", "test"):
            df = pd.read_csv(data_dir / f"{split}_baseline.csv")
            X_seq, y_seq = preprocessor._make_sequences(df, seq_len)
            np.save(data_dir / f"X_{split}_{model_key}.npy", X_seq)
            np.save(data_dir / f"y_{split}_{model_key}.npy", y_seq)

    logger.info("Original sequences restored.")


# Entry point

def main() -> None:
    """Run the full hyperparameter grid search."""
    config = load_config()

    data_dir = Path(config["data"]["processed_dir"])
    output_dir = Path("outputs/tuning")

    results = run_grid_search(
        data_dir=data_dir,
        output_dir=output_dir,
        config=config,
    )

    save_and_report(results, output_dir)

    # Restore the correct per-model arrays so nothing downstream breaks.
    restore_original_sequences(config, data_dir)

    logger.info("Tuning complete.")


if __name__ == "__main__":
    main()
