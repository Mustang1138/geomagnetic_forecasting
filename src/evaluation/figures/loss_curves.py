"""LSTM and GRU training/validation loss curves (Figure F.1).

Retrains both models from the saved sequence arrays to recover per-epoch loss
history, which the production training script does not persist. Uses the
canonical ``LSTMRegressor`` / ``GRURegressor`` from ``src.models.temporal_model``
with ``dropout=0.2`` to match Table 4.1 of the dissertation.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.figures._common import DATA_DIR, GREEN, ORANGE, OUT_DIR, PURPLE, RED
from src.models.temporal_model import GRURegressor, LSTMRegressor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _load_split(split: str, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(DATA_DIR / f"X_{split}_{prefix}.npy")
    y = np.load(DATA_DIR / f"y_{split}_{prefix}.npy")
    return X, y


def _train_with_history(
        model_class: type,
        model_kwargs: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        lr: float,
        patience: int,
        max_epochs: int = 120,
        batch_size: int = 32,
) -> tuple[list[float], list[float], int]:
    """Train ``model_class`` and return ``(train_history, val_history, best_epoch)``."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(**model_kwargs).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    def _loader(X: np.ndarray, y: np.ndarray, shuf: bool = False) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).squeeze(-1),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuf)

    train_loader = _loader(X_train, y_train)
    val_loader = _loader(X_val, y_val)

    train_hist, val_hist = [], []
    best_val, patience_ctr, best_epoch = float("inf"), 0, 0

    for epoch in range(max_epochs):
        model.train()
        t_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                v_loss += criterion(model(xb), yb).item()
        v_loss /= len(val_loader)

        train_hist.append(t_loss)
        val_hist.append(v_loss)

        if v_loss < best_val:
            best_val, patience_ctr, best_epoch = v_loss, 0, epoch
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                log.info("Early stop at epoch %d (best epoch %d)",
                         epoch + 1, best_epoch + 1)
                break

    return train_hist, val_hist, best_epoch


def fig_loss_curves(output_path: Path) -> None:
    """Train LSTM and GRU and plot their train/val loss curves side by side."""
    log.info("Training LSTM for loss curves …")
    X_tr_l, y_tr_l = _load_split("train", "lstm")
    X_va_l, y_va_l = _load_split("val", "lstm")
    n_feat = X_tr_l.shape[2]
    lstm_train, lstm_val, lstm_best = _train_with_history(
        LSTMRegressor,
        dict(n_features=n_feat, hidden_size=64, num_layers=2, dropout=0.2),
        X_tr_l, y_tr_l, X_va_l, y_va_l,
        lr=0.0005, patience=15,
    )

    log.info("Training GRU for loss curves …")
    X_tr_g, y_tr_g = _load_split("train", "gru")
    X_va_g, y_va_g = _load_split("val", "gru")
    gru_train, gru_val, gru_best = _train_with_history(
        GRURegressor,
        dict(n_features=n_feat, hidden_size=64, num_layers=1, dropout=0.2),
        X_tr_g, y_tr_g, X_va_g, y_va_g,
        lr=0.005, patience=15,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    for ax, name, t_hist, v_hist, best_ep, colour in [
        (axes[0], "LSTM", lstm_train, lstm_val, lstm_best, RED),
        (axes[1], "GRU", gru_train, gru_val, gru_best, PURPLE),
    ]:
        epochs = np.arange(1, len(t_hist) + 1)
        ax.plot(epochs, t_hist, color=colour, linewidth=1.5, label="Train loss")
        ax.plot(epochs, v_hist, color=ORANGE, linewidth=1.5, linestyle="--",
                label="Val loss")
        ax.axvline(best_ep + 1, color=GREEN, linewidth=1.1, linestyle=":",
                   label=f"Best checkpoint (ep {best_ep + 1})")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("MSE loss", fontsize=10)
        ax.set_title(f"{name} training history", fontsize=10, pad=6)
        ax.legend(fontsize=8)

    fig.suptitle(
        "LSTM and GRU training and validation loss curves\n"
        "(Adam optimiser, MSE loss, early stopping with patience 15)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", output_path)


if __name__ == "__main__":
    fig_loss_curves(OUT_DIR / "fig_loss_curves.png")
