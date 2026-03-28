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
) -> None:
    """Train the GRU regressor using configuration from ``config.yaml``.

    Args:
        data_dir: Directory containing preprocessed ``.npy`` arrays and
            ``scaler_y.pkl``.
        output_dir: Directory to which the model checkpoint and prediction
            CSV are written.
    """
    config = load_config()
    gru_cfg = config["models"]["gru"]

    cfg = TrainingConfig(
        model_name="gru",
        hidden_size=gru_cfg["hidden_size"],
        num_layers=gru_cfg["num_layers"],
        dropout=gru_cfg["dropout"],
        learning_rate=gru_cfg["learning_rate"],
        batch_size=gru_cfg["batch_size"],
        epochs=gru_cfg["epochs"],
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
    )

    trainer = Trainer(model_class=GRURegressor, cfg=cfg)
    trainer.run()


if __name__ == "__main__":
    train_gru()