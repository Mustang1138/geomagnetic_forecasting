"""
Minimal training loop for GRU-based SSI forecasting.

Mirrors train_lstm.py but uses GRU cells for direct comparison.

References:
- Cho et al. (2014) - GRU architecture
- Paszke et al. (2019) - PyTorch framework
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.sequence_datasets import load_sequence_dataset
from src.data.torch_datasets import SSITimeSeriesDataset
from src.models.temporal_model import GRURegressor
from src.utils import setup_logging

logger = setup_logging()


def train_gru(
        data_dir: Path,
        batch_size: int = 64,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        device: str | None = None,
):
    """
    Train a minimal GRU model for SSI forecasting.

    Identical to train_lstm() except uses GRURegressor for fair
    comparison between LSTM and GRU architectures (Cho et al., 2014).

    Parameters
    ----------
    data_dir : Path
        Directory containing train.npz and val.npz sequence datasets.
    batch_size : int, optional
        Batch size for training. Default is 64.
    num_epochs : int, optional
        Number of training epochs. Default is 10.
    learning_rate : float, optional
        Adam optimiser learning rate. Default is 1e-3.
    device : str or None, optional
        "cpu" or "cuda". If None, auto-selects GPU if available.

    Returns
    -------
    GRURegressor
        Trained model (on CPU).
    """

    # Device setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load sequence datasets
    X_train, y_train, _ = load_sequence_dataset(data_dir / "train.npz")
    X_val, y_val, _ = load_sequence_dataset(data_dir / "val.npz")

    train_ds = SSITimeSeriesDataset(X_train, y_train)
    val_ds = SSITimeSeriesDataset(X_val, y_val)

    # Create data loaders
    # Training: shuffle for better gradients, drop_last for consistent batches
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Validation: no shuffle (temporal order), use all samples
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Model, loss, optimiser
    n_features = X_train.shape[2]

    model = GRURegressor(
        n_features=n_features,
        hidden_size=64,  # Matches LSTM for fair comparison
        num_layers=1,
    ).to(device)

    criterion = nn.MSELoss()  # MSE loss for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, num_epochs + 1):

        # Training phase
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()  # Clear gradients
            y_pred = model(X_batch)  # Forward pass
            loss = criterion(y_pred, y_batch)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters

            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():  # No gradient computation
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_losses.append(loss.item())

        # Logging
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f}"
        )

    # Return model on CPU for saving/inference
    model = model.to("cpu")
    return model
