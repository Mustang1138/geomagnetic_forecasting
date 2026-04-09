"""PyTorch Dataset wrappers for precomputed SSI sequence arrays."""

import numpy as np
import torch
from torch.utils.data import Dataset


class SSITimeSeriesDataset(Dataset):
    """PyTorch Dataset for SSI sequence-to-one forecasting.

    Wraps precomputed, scaled, chronologically split sequence arrays.
    Each sample is an input sequence of shape (seq_len, n_features) paired
    with a scalar SSI target.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be NumPy arrays")

        if X.ndim != 3:
            raise ValueError("X must have shape (N, seq_len, n_features)")

        if y.ndim != 1:
            raise ValueError("y must have shape (N,)")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the input sequence and scalar target at *idx*."""
        return self.X[idx], self.y[idx]
