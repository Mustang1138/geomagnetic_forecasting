import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.torch_datasets import SSITimeSeriesDataset


def make_dummy_data(
        n_samples: int = 10,
        seq_len: int = 24,
        n_features: int = 3,
):
    X = np.random.randn(n_samples, seq_len, n_features)
    y = np.random.randn(n_samples)
    return X, y


def test_dataset_length():
    X, y = make_dummy_data(n_samples=12)
    dataset = SSITimeSeriesDataset(X, y)
    assert len(dataset) == 12


def test_dataset_item_shapes():
    X, y = make_dummy_data(seq_len=48, n_features=5)
    dataset = SSITimeSeriesDataset(X, y)

    X_seq, y_target = dataset[0]

    assert X_seq.shape == (48, 5)
    assert y_target.shape == ()


def test_dataset_tensor_types():
    X, y = make_dummy_data()
    dataset = SSITimeSeriesDataset(X, y)

    X_seq, y_target = dataset[0]

    assert isinstance(X_seq, torch.Tensor)
    assert isinstance(y_target, torch.Tensor)
    assert X_seq.dtype == torch.float32
    assert y_target.dtype == torch.float32


def test_dataset_works_with_dataloader():
    X, y = make_dummy_data(n_samples=20)
    dataset = SSITimeSeriesDataset(X, y)

    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    X_batch, y_batch = next(iter(loader))

    assert X_batch.shape == (4, X.shape[1], X.shape[2])
    assert y_batch.shape == (4,)


def test_invalid_shapes_raise():
    X = np.random.randn(10, 24)  # wrong shape
    y = np.random.randn(10)

    try:
        SSITimeSeriesDataset(X, y)
        assert False, "Expected ValueError for invalid X shape"
    except ValueError:
        pass
