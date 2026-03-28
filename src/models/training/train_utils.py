"""
Shared training utilities for temporal sequence models.

Centralising the training loop, validation loop, and early stopping logic
here eliminates duplication between the LSTM and GRU training scripts and
makes each component independently testable (Martin, 2008).
"""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# Configuration dataclass

@dataclass
class TrainingConfig:
    """Hyperparameters and path settings for a single training run.

    Attributes:
        model_name: Identifier used for log messages and output filenames
            (e.g. ``"lstm"`` or ``"gru"``).
        hidden_size: Number of units in each recurrent layer.
        num_layers: Number of stacked recurrent layers.
        dropout: Dropout probability applied between recurrent layers.
        learning_rate: Adam optimiser learning rate.
        batch_size: Mini-batch size for the training DataLoader.
        epochs: Maximum number of training epochs.
        patience: Number of epochs without validation improvement before
            early stopping is triggered.
        data_dir: Directory containing the preprocessed ``.npy`` arrays
            and ``scaler_y.pkl``.
        output_dir: Directory to which the best model checkpoint and
            prediction CSV are written.
    """

    model_name: str
    hidden_size: int
    num_layers: int
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    patience: int = 16
    data_dir: Path = field(default_factory=lambda: Path("data/processed"))
    output_dir: Path = field(default_factory=lambda: Path("outputs/temporal"))


# Data loading helpers

def _load_split(
        data_dir: Path,
        split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a preprocessed split from disk.

    Supports both the ``.npy`` format produced by ``preprocess.py`` and the
    ``.npz`` archive format produced by ``sequence_datasets.py``.

    Args:
        data_dir: Directory containing the preprocessed arrays.
        split: One of ``"train"``, ``"val"``, or ``"test"``.

    Returns:
        A tuple ``(X, y)`` of NumPy arrays.

    Raises:
        FileNotFoundError: If neither the ``.npy`` nor the ``.npz`` files
            exist for the requested split.
    """
    npy_x = data_dir / f"X_{split}.npy"
    npy_y = data_dir / f"y_{split}.npy"

    if npy_x.exists() and npy_y.exists():
        return np.load(npy_x), np.load(npy_y)

    npz = data_dir / f"{split}.npz"
    if npz.exists():
        with np.load(npz) as archive:
            return archive["X"], archive["y"]

    raise FileNotFoundError(
        f"No preprocessed data found for split '{split}' in {data_dir}. "
        "Run the preprocessing pipeline first."
    )


def build_dataloader(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
        pin_memory: bool = True,
) -> DataLoader:
    """Wrap NumPy arrays in a PyTorch DataLoader.

    Shuffling is disabled by default to preserve temporal ordering, which
    is critical for time-series evaluation (Cerqueira et al., 2020).

    Args:
        X: Input sequences of shape ``(N, seq_len, n_features)``.
        y: Target values of shape ``(N,)`` or ``(N, 1)``.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle samples each epoch.
        pin_memory: Whether to pin memory for faster GPU transfer.

    Returns:
        A configured :class:`~torch.utils.data.DataLoader`.
    """
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).squeeze(-1),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )


# Training and validation loops

def run_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimiser: torch.optim.Optimizer,
        device: torch.device,
) -> float:
    """Run a single training epoch.

    Args:
        model: The model to train.
        loader: DataLoader providing mini-batches.
        criterion: Loss function.
        optimiser: Parameter update rule.
        device: Compute device.

    Returns:
        Mean training loss over all mini-batches.
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimiser.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimiser.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def run_validation(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
) -> float:
    """Evaluate the model on a validation set without updating weights.

    Args:
        model: The model to evaluate.
        loader: DataLoader providing validation mini-batches.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Mean validation loss over all mini-batches.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()

    return total_loss / len(loader)


# Inference and output helpers

def run_inference(
        model: nn.Module,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scaler_y_path: Path,
        device: torch.device,
) -> pd.DataFrame:
    """Generate test-set predictions and inverse-scale them.

    Args:
        model: Trained model in eval mode.
        X_test: Test input sequences.
        y_test: Scaled test targets.
        scaler_y_path: Path to the fitted ``scaler_y.pkl``.
        device: Compute device.

    Returns:
        A DataFrame with columns ``y_true`` and ``y_pred`` in physical
        (unscaled) SSI units.
    """
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()

    with open(scaler_y_path, "rb") as fh:
        scaler_y = pickle.load(fh)

    y_true_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

    return pd.DataFrame({"y_true": y_true_inv, "y_pred": y_pred_inv})


# Main trainer

class Trainer:
    """Orchestrates training, early stopping, and inference for a recurrent model.

    Designed to be model-agnostic — any :class:`~torch.nn.Module` whose
    ``forward`` method accepts a tensor of shape
    ``(batch, seq_len, n_features)`` and returns a tensor of shape
    ``(batch,)`` is compatible.

    Args:
        model_class: The recurrent model class to instantiate
            (e.g. :class:`~src.models.temporal_model.LSTMRegressor`).
        cfg: Hyperparameters and path settings for this run.
    """

    def __init__(self, model_class: Type[nn.Module], cfg: TrainingConfig):
        self._model_class = model_class
        self._cfg = cfg
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("Trainer initialised on device: %s", self._device)

    def run(self) -> nn.Module:
        """Execute the full training pipeline.

        Loads data, trains with early stopping, runs test inference, saves
        predictions, and returns the best model moved to CPU.

        Returns:
            The trained model with the best validation weights, on CPU.
        """
        cfg = self._cfg

        # Load data
        X_train, y_train = _load_split(cfg.data_dir, "train")
        X_val, y_val = _load_split(cfg.data_dir, "val")

        train_loader = build_dataloader(X_train, y_train, cfg.batch_size)
        val_loader = build_dataloader(X_val, y_val, batch_size=256)

        # Initialise model
        input_size = X_train.shape[2]

        model = self._model_class(
            n_features=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
        ).to(self._device)

        criterion = nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        # Output paths
        model_dir = cfg.output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = model_dir / f"{cfg.model_name}_best.pt"

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(cfg.epochs):
            train_loss = run_epoch(
                model, train_loader, criterion, optimiser, self._device
            )
            val_loss = run_validation(
                model, val_loader, criterion, self._device
            )

            logger.info(
                "Epoch %d/%d — train: %.6f  val: %.6f",
                epoch + 1,
                cfg.epochs,
                train_loss,
                val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
                logger.info("Checkpoint saved (val loss: %.6f).", val_loss)
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    logger.info(
                        "Early stopping triggered after %d epochs.", epoch + 1
                    )
                    break

        # Test inference
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=self._device)
        )

        scaler_path = cfg.data_dir / "scaler_y.pkl"

        try:
            X_test, y_test = _load_split(cfg.data_dir, "test")
        except FileNotFoundError:
            logger.warning(
                "Test data not found — skipping inference for %s.",
                cfg.model_name,
            )
            return model.cpu()

        if not scaler_path.exists():
            logger.warning(
                "scaler_y.pkl not found — skipping inference for %s.",
                cfg.model_name,
            )
            return model.cpu()

        pred_df = run_inference(
            model, X_test, y_test, scaler_path, self._device
        )

        pred_dir = cfg.output_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_path = pred_dir / f"{cfg.model_name}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        logger.info(
            "%s training complete. Predictions saved to %s.",
            cfg.model_name.upper(),
            pred_path,
        )

        # Return to CPU so callers can run inference without device
        # mismatch (e.g. in unit tests running on CPU-only machines).
        return model.cpu()
