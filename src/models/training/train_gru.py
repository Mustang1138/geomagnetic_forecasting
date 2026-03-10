"""
GRU training script for geomagnetic storm severity forecasting.

Implements a Gated Recurrent Unit (GRU) architecture for
sequence-to-one regression.

GRUs provide a computationally lighter alternative to LSTMs
while retaining gated temporal memory mechanisms
(Cho et al., 2014).

Design principles identical to LSTM training.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils import load_config, setup_logging

logger = setup_logging()


# Model Definition
class GRURegressor(nn.Module):
    """
    Multi-layer GRU regressor for SSI forecasting.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        # Squeeze to (N,) - consistent shape for training loss and inference
        return self.fc(out).squeeze(-1)


# Training Function
def train_gru(data_dir="data/processed", num_epochs=None, batch_size=None):

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_dir = Path(data_dir)

    # Flexible loading for .npy or .npz formats
    try:
        X_train = np.load(data_dir / "X_train.npy")
        y_train = np.load(data_dir / "y_train.npy")
    except FileNotFoundError:
        with np.load(data_dir / "train.npz") as data:
            X_train = data['X']
            y_train = data['y']

    try:
        X_val = np.load(data_dir / "X_val.npy")
        y_val = np.load(data_dir / "y_val.npy")
    except FileNotFoundError:
        with np.load(data_dir / "val.npz") as data:
            X_val = data['X']
            y_val = data['y']

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )

    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    if batch_size is None:
        batch_size = config["models"]["gru"]["batch_size"]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    input_size = X_train.shape[2]

    model = GRURegressor(
        input_size=input_size,
        hidden_size=config["models"]["gru"]["hidden_size"],
        num_layers=config["models"]["gru"]["num_layers"],
        dropout=config["models"]["gru"]["dropout"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["models"]["gru"]["learning_rate"],
    )

    model_dir = Path("outputs/temporal/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience = 8
    patience_counter = 0

    if num_epochs is None:
        num_epochs = config["models"]["gru"]["epochs"]

    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        logger.info(
            f"Epoch {epoch + 1}: "
            f"Train={train_loss:.6f}, "
            f"Val={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_dir / "gru_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    # Test inference (only if test data exists)
    test_npy_exists = (
                              data_dir /
                              "X_test.npy").exists() and (
                              data_dir /
                              "y_test.npy").exists()
    test_npz_exists = (data_dir / "test.npz").exists()
    scaler_pkl_exists = (data_dir / "scaler_y.pkl").exists()
    if (test_npy_exists or test_npz_exists) and scaler_pkl_exists:
        model.load_state_dict(torch.load(model_dir / "gru_best.pt"))
        model.eval()

        if test_npy_exists:
            X_test = np.load(data_dir / "X_test.npy")
            y_test = np.load(data_dir / "y_test.npy")
        elif test_npz_exists:
            with np.load(data_dir / "test.npz") as data:
                X_test = data['X']
                y_test = data['y']

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        with torch.no_grad():
            preds = model(X_test_tensor).cpu().numpy()

        with open(data_dir / "scaler_y.pkl", "rb") as f:
            scaler_y = pickle.load(f)

        y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        preds_inv = scaler_y.inverse_transform(preds.reshape(-1, 1))

        pred_dir = Path("outputs/temporal/predictions")
        pred_dir.mkdir(parents=True, exist_ok=True)

        df_preds = pd.DataFrame({
            "y_true": y_test_inv.flatten(),
            "y_pred": preds_inv.flatten(),
        })

        df_preds.to_csv(pred_dir / "gru_predictions.csv", index=False)

        logger.info("GRU training and inference complete.")
    else:
        logger.info(
            "GRU training complete (skipping inference due to missing test data).")

    # Move model back to CPU before returning so callers (e.g. tests)
    # can run inference with plain CPU tensors without a device mismatch.
    model = model.cpu()
    return model


if __name__ == "__main__":
    train_gru()
