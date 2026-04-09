"""Shared training loop, validation loop, and early-stopping logic for recurrent models."""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters and path settings for a single training run."""

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
    # Falls back to unprefixed filenames if the prefixed file is not found.
    sequence_file_prefix: str = ""


def _load_split(
        data_dir: Path,
        split: str,
        prefix: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Load a preprocessed split from disc, trying prefixed files before legacy fallbacks.

    Parameters
    ----------
    data_dir
        Directory containing the preprocessed arrays.
    split
        One of ``"train"``, ``"val"``, or ``"test"``.
    prefix
        Optional model-specific suffix (e.g. ``"lstm"`` or ``"gru"``);
        prefixed files are tried first.

    Returns
    -------
    tuple
        ``(X, y)`` NumPy arrays for the requested split.
    """
    if prefix:
        npy_x = data_dir / f"X_{split}_{prefix}.npy"
        npy_y = data_dir / f"y_{split}_{prefix}.npy"
        if npy_x.exists() and npy_y.exists():
            return np.load(npy_x), np.load(npy_y)

    npy_x = data_dir / f"X_{split}.npy"
    npy_y = data_dir / f"y_{split}.npy"
    if npy_x.exists() and npy_y.exists():
        return np.load(npy_x), np.load(npy_y)

    npz = data_dir / f"{split}.npz"
    if npz.exists():
        with np.load(npz) as archive:
            return archive["X"], archive["y"]

    raise FileNotFoundError(
        f"No preprocessed data found for split '{split}' "
        f"(prefix='{prefix}') in {data_dir}. "
        "Run the preprocessing pipeline first."
    )


def build_dataloader(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = False,  # disabled by default to preserve temporal ordering
        pin_memory: bool = True,
) -> DataLoader:
    """Wrap NumPy arrays in a PyTorch DataLoader."""
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


def run_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimiser: torch.optim.Optimizer,
        device: torch.device,
) -> float:
    """Run one training epoch and return the mean loss."""
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
    """Return mean validation loss without updating model weights."""
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


def run_inference(
        model: nn.Module,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scaler_y_path: Path,
        device: torch.device,
) -> pd.DataFrame:
    """Generate test-set predictions and inverse-scale them to physical SSI units.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``y_true`` and ``y_pred`` in unscaled SSI units.
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


class Trainer:
    """Orchestrates training, early stopping, and test inference for a recurrent model.

    Any ``nn.Module`` whose ``forward`` method accepts ``(batch, seq_len, n_features)``
    and returns ``(batch,)`` is compatible.
    """

    def __init__(self, model_class: type[nn.Module], cfg: TrainingConfig):
        self._model_class = model_class
        self._cfg = cfg
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("Trainer initialised on device: %s", self._device)

    def run(self) -> nn.Module:
        """Execute the full training pipeline and return the best model on CPU."""
        cfg = self._cfg

        X_train, y_train = _load_split(cfg.data_dir, "train", cfg.sequence_file_prefix)
        X_val, y_val = _load_split(cfg.data_dir, "val", cfg.sequence_file_prefix)

        train_loader = build_dataloader(X_train, y_train, cfg.batch_size)
        val_loader = build_dataloader(X_val, y_val, batch_size=256)

        input_size = X_train.shape[2]

        model = self._model_class(
            n_features=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
        ).to(self._device)

        criterion = nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        model_dir = cfg.output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = model_dir / f"{cfg.model_name}_best.pt"

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
                epoch + 1, cfg.epochs, train_loss, val_loss,
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

        model.load_state_dict(
            torch.load(checkpoint_path, map_location=self._device)
        )

        scaler_path = cfg.data_dir / "scaler_y.pkl"

        try:
            X_test, y_test = _load_split(cfg.data_dir, "test", cfg.sequence_file_prefix)
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

        return model.cpu()
