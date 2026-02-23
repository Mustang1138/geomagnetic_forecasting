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


class GRURegressor(nn.Module):
    """
    Sequence-to-one GRU regression model.
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
        return self.fc(out)


def train_gru(data_dir="data/processed"):
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_dir = Path(data_dir)

    # Load arrays
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")

    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")

    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )

    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["models"]["lstm"]["batch_size"],
        shuffle=False,
    )

    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    input_size = X_train.shape[2]

    model = GRURegressor(
        input_size=input_size,
        hidden_size=config["models"]["lstm"]["hidden_size"],
        num_layers=config["models"]["lstm"]["num_layers"],
        dropout=config["models"]["lstm"]["dropout"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["models"]["lstm"]["learning_rate"],
    )

    best_val_loss = float("inf")
    patience = 8
    patience_counter = 0

    for epoch in range(config["models"]["lstm"]["epochs"]):

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
            f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, "
            f"Val Loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_dir = Path("outputs/temporal/models")
            model_dir.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), model_dir / "gru_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    # Test inference
    model.load_state_dict(torch.load(model_dir / "gru_best.pt"))
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X_test_tensor).cpu().numpy()

    with open(data_dir / "scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)

    y_test_inv = scaler_y.inverse_transform(y_test)
    preds_inv = scaler_y.inverse_transform(preds)

    pred_dir = Path("outputs/temporal/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    df_preds = pd.DataFrame(
        {
            "y_true": y_test_inv.flatten(),
            "y_pred": preds_inv.flatten(),
        }
    )

    df_preds.to_csv(pred_dir / "gru_predictions.csv", index=False)

    logger.info("GRU training and inference complete.")

    return model


if __name__ == "__main__":
    train_gru()
