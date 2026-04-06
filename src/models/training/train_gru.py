"""
GRU training entry point for geomagnetic storm severity forecasting.

Delegates all training logic to :class:`~src.models.training.train_utils.Trainer`,
keeping this script focused solely on GRU-specific configuration.

References:
    - Cho et al. (2014) — GRU architecture
    - Cerqueira et al. (2020) — time-series evaluation best practices
"""

from pathlib import Path

from src.models.temporal_model import GRURegressor
from src.models.training.train_utils import Trainer, TrainingConfig
from src.utils import load_config, setup_logging

logger = setup_logging()


def train_gru(
        data_dir: str = "data/processed",
        output_dir: str = "outputs/temporal",
        num_epochs: int | None = None,
        batch_size: int | None = None,
) -> None:
    """Train the GRU regressor using configuration from ``config.yaml``.

    Args:
        data_dir: Directory containing preprocessed ``.npy`` arrays and
            ``scaler_y.pkl``.
        output_dir: Directory to which the model checkpoint and prediction
            CSV are written.
        num_epochs: Override the epoch count from config (used in testing).
        batch_size: Override the batch size from config (used in testing).
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
        # Load GRU-specific sequence arrays (X_train_gru.npy etc.) generated
        # by preprocess.py using the GRU sequence_length from config.yaml.
        data_prefix="gru",
    )

    trainer = Trainer(model_class=GRURegressor, cfg=cfg)
    trainer.run()


if __name__ == "__main__":
    train_gru()
