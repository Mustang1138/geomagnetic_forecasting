import numpy as np
import torch

from src.data.sequence_datasets import save_sequence_dataset
from src.models.training.train_lstm import train_lstm


def _make_fake_sequence_dataset(path, n=32, seq_len=5, n_features=3):
    X = np.random.randn(n, seq_len, n_features).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)
    ts = np.arange(n)

    save_sequence_dataset(path, X, y, ts)


def test_train_lstm_runs(tmp_path):
    """
    Integration test for LSTM training loop.

    Verifies that:
    - Training executes without error
    - Returned model is callable
    - Forward pass produces correct output shape
    """

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    _make_fake_sequence_dataset(data_dir / "train.npz")
    _make_fake_sequence_dataset(data_dir / "val.npz")

    model = train_lstm(
        data_dir=data_dir,
        num_epochs=1,
        batch_size=8,
    )

    # Sanity check: forward pass
    X = torch.randn(4, 5, 3)
    y_pred = model(X)

    assert y_pred.shape == (4,)
