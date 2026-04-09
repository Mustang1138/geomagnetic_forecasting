"""Grid search over sequence length and architecture hyperparameters for LSTM and GRU."""

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
from src.utils import load_config, load_pickle, setup_logging

logger = setup_logging()

# Sequence lengths are expressed in 6-hourly time steps, matching the resampled
# cadence of the training data (12 → 3 days, 24 → 6 days, 48 → 12 days).
SEARCH_SPACE: dict[str, list[Any]] = {
    "sequence_length": [12, 24, 48],
    "num_layers": [1, 2],
    "hidden_size": [64, 128],
    "learning_rate": [0.001, 0.0005],
    "patience": [8, 15],
}

MODELS: dict[str, type[nn.Module]] = {
    "lstm": LSTMRegressor,
    "gru": GRURegressor,
}


@dataclass
class TuningResult:
    """Outcome of a single hyperparameter combination."""

    model_name: str
    sequence_length: int
    num_layers: int
    hidden_size: int
    learning_rate: float
    patience: int
    best_val_loss: float
    epochs_trained: int
    duration_seconds: float


def rebuild_sequences(
        sequence_length: int,
        config: dict[str, Any],
        data_dir: Path,
) -> None:
    """Rebuild per-model ``.npy`` sequence arrays for a candidate window length.

    Writes model-specific suffixed files (e.g. ``X_train_lstm.npy``) without
    modifying ``config.yaml`` on disc.
    """
    logger.info("Rebuilding sequences with sequence_length=%d …", sequence_length)

    # Deep-copy so we never mutate the live config dict.
    patched_config = copy.deepcopy(config)
    patched_config["models"]["lstm"]["sequence_length"] = sequence_length
    patched_config["models"]["gru"]["sequence_length"] = sequence_length

    preprocessor = DataPreprocessor.__new__(DataPreprocessor)
    preprocessor.config = patched_config
    preprocessor.scaler_X = load_pickle(data_dir / "scaler_X.pkl")
    preprocessor.scaler_y = load_pickle(data_dir / "scaler_y.pkl")

    for split in ("train", "val", "test"):
        df = pd.read_csv(data_dir / f"{split}_baseline.csv")
        X_seq, y_seq = preprocessor._make_sequences(df, sequence_length)

        for model_key in ("lstm", "gru"):
            np.save(data_dir / f"X_{split}_{model_key}.npy", X_seq)
            np.save(data_dir / f"y_{split}_{model_key}.npy", y_seq)

    logger.info("Sequences rebuilt successfully.")


def train_combination(
        model_name: str,
        model_class: type[nn.Module],
        combo: dict[str, Any],
        data_dir: Path,
        output_dir: Path,
        base_config: dict[str, Any],
) -> TuningResult:
    """Train one hyperparameter combination and return its validation result."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Tuning checkpoints go to a temporary directory to avoid overwriting production weights.
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

    # Test-set inference is omitted here to keep the test set truly held-out.
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


def run_grid_search(
        data_dir: Path,
        output_dir: Path,
        config: dict[str, Any],
) -> list[TuningResult]:
    """Run the full grid search across all combinations and both models.

    Sequences are only rebuilt when ``sequence_length`` changes between combinations.
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


def save_and_report(results: list[TuningResult], output_dir: Path) -> None:
    """Save tuning results to CSV and print a ranked summary per model."""
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

    for model_name in MODELS:
        subset = df[df["model"] == model_name].reset_index(drop=True)
        print(f"\n{'=' * 60}")
        print(f"  {model_name.upper()} — ranked by validation loss")
        print(f"{'=' * 60}")
        print(subset.to_string(index=False))

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


def restore_original_sequences(
        config: dict[str, Any],
        data_dir: Path,
) -> None:
    """Restore per-model ``.npy`` arrays to the sequence lengths in ``config.yaml``.

    Called after the grid search so that on-disc arrays match the lengths
    configured for the final training runs.
    """
    lstm_seq = config["models"]["lstm"]["sequence_length"]
    gru_seq = config["models"]["gru"]["sequence_length"]

    for model_key, seq_len in (("lstm", lstm_seq), ("gru", gru_seq)):
        logger.info(
            "Restoring %s sequence arrays (sequence_length=%d) …",
            model_key.upper(), seq_len,
        )
        preprocessor = DataPreprocessor.__new__(DataPreprocessor)
        preprocessor.config = config
        preprocessor.scaler_X = load_pickle(data_dir / "scaler_X.pkl")
        preprocessor.scaler_y = load_pickle(data_dir / "scaler_y.pkl")

        for split in ("train", "val", "test"):
            df = pd.read_csv(data_dir / f"{split}_baseline.csv")
            X_seq, y_seq = preprocessor._make_sequences(df, seq_len)
            np.save(data_dir / f"X_{split}_{model_key}.npy", X_seq)
            np.save(data_dir / f"y_{split}_{model_key}.npy", y_seq)

    logger.info("Original sequences restored.")


def main() -> None:
    """Run the full hyperparameter grid search and restore original sequence arrays."""
    config = load_config()

    data_dir = Path(config["data"]["processed_dir"])
    output_dir = Path("outputs/tuning")

    results = run_grid_search(
        data_dir=data_dir,
        output_dir=output_dir,
        config=config,
    )

    save_and_report(results, output_dir)

    restore_original_sequences(config, data_dir)

    logger.info("Tuning complete.")


if __name__ == "__main__":
    main()
