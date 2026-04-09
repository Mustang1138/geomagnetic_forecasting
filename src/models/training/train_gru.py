"""Entry point for training the GRU storm severity regressor."""

from pathlib import Path

import torch.nn as nn

from src.models.temporal_model import GRURegressor
from src.models.training.train_utils import Trainer, TrainingConfig
from src.utils import load_config, setup_logging

logger = setup_logging()


def train_gru(
        data_dir: str = "data/processed",
        output_dir: str = "outputs/temporal",
        num_epochs: int | None = None,
        batch_size: int | None = None,
) -> nn.Module:
    """Train the GRU regressor using configuration from ``config.yaml``.

    Parameters
    ----------
    num_epochs
        Overrides the epoch count from config; used in testing.
    batch_size
        Overrides the batch size from config; used in testing.

    Returns
    -------
    nn.Module
        Trained GRU with the best validation weights, on CPU.
    """
    config = load_config()
    gru_cfg = config["models"]["gru"]

    cfg = TrainingConfig(
        model_name="gru",
        hidden_size=gru_cfg["hidden_size"],
        num_layers=gru_cfg["num_layers"],
        dropout=gru_cfg["dropout"],
        learning_rate=gru_cfg["learning_rate"],
        batch_size=batch_size if batch_size is not None else gru_cfg["batch_size"],
        epochs=num_epochs if num_epochs is not None else gru_cfg["epochs"],
        patience=gru_cfg["patience"],
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        sequence_file_prefix="gru",
    )

    return Trainer(model_class=GRURegressor, cfg=cfg).run()


if __name__ == "__main__":
    train_gru()
