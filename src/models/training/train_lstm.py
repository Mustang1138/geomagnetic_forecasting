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
        num_epochs: int | None = None,
        batch_size: int | None = None,
) -> None:
    """Train the LSTM regressor using configuration from ``config.yaml``.

    Args:
        data_dir: Directory containing preprocessed ``.npy`` arrays and
            ``scaler_y.pkl``.
        output_dir: Directory to which the model checkpoint and prediction
            CSV are written.
        num_epochs: Override the epoch count from config (used in testing).
        batch_size: Override the batch size from config (used in testing).
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
        # Load LSTM-specific sequence arrays (X_train_lstm.npy etc.) generated
        # by preprocess.py using the LSTM sequence_length from config.yaml.
        data_prefix="lstm",
    )

    trainer = Trainer(model_class=LSTMRegressor, cfg=cfg)
    trainer.run()


if __name__ == "__main__":
    train_lstm()
