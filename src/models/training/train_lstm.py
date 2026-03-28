"""
LSTM training entry point for geomagnetic storm severity forecasting.

Delegates all training logic to :class:`~src.models.training.train_utils.Trainer`,
keeping this script focused solely on LSTM-specific configuration.

References:
    - Hochreiter & Schmidhuber (1997) — LSTM architecture
    - Cerqueira et al. (2020) — time-series evaluation best practices
"""

from pathlib import Path

from src.models.temporal_model import LSTMRegressor
from src.models.training.train_utils import Trainer, TrainingConfig
from src.utils import load_config, setup_logging

logger = setup_logging()


def train_lstm(
        data_dir: str = "data/processed",
        output_dir: str = "outputs/temporal",
) -> None:
    """Train the LSTM regressor using configuration from ``config.yaml``.

    Args:
        data_dir: Directory containing preprocessed ``.npy`` arrays and
            ``scaler_y.pkl``.
        output_dir: Directory to which the model checkpoint and prediction
            CSV are written.
    """
    config = load_config()
    lstm_cfg = config["models"]["lstm"]

    cfg = TrainingConfig(
        model_name="lstm",
        hidden_size=lstm_cfg["hidden_size"],
        num_layers=lstm_cfg["num_layers"],
        dropout=lstm_cfg["dropout"],
        learning_rate=lstm_cfg["learning_rate"],
        batch_size=lstm_cfg["batch_size"],
        epochs=lstm_cfg["epochs"],
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
    )

    trainer = Trainer(model_class=LSTMRegressor, cfg=cfg)
    trainer.run()


if __name__ == "__main__":
    train_lstm()