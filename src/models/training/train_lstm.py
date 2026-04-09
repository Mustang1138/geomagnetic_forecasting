"""Entry point for training the LSTM storm severity regressor."""

from pathlib import Path

import torch.nn as nn

from src.models.temporal_model import LSTMRegressor
from src.models.training.train_utils import Trainer, TrainingConfig
from src.utils import load_config, setup_logging

logger = setup_logging()


def train_lstm(
        data_dir: str = "data/processed",
        output_dir: str = "outputs/temporal",
        num_epochs: int | None = None,
        batch_size: int | None = None,
) -> nn.Module:
    """Train the LSTM regressor using configuration from ``config.yaml``.

    Parameters
    ----------
    num_epochs
        Overrides the epoch count from config; used in testing.
    batch_size
        Overrides the batch size from config; used in testing.

    Returns
    -------
    nn.Module
        Trained LSTM with the best validation weights, on CPU.
    """
    config = load_config()
    lstm_cfg = config["models"]["lstm"]

    cfg = TrainingConfig(
        model_name="lstm",
        hidden_size=lstm_cfg["hidden_size"],
        num_layers=lstm_cfg["num_layers"],
        dropout=lstm_cfg["dropout"],
        learning_rate=lstm_cfg["learning_rate"],
        batch_size=batch_size if batch_size is not None else lstm_cfg["batch_size"],
        epochs=num_epochs if num_epochs is not None else lstm_cfg["epochs"],
        patience=lstm_cfg["patience"],
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        sequence_file_prefix="lstm",
    )

    return Trainer(model_class=LSTMRegressor, cfg=cfg).run()


if __name__ == "__main__":
    train_lstm()
