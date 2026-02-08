"""
PyTorch Dataset utilities for SSI time series forecasting.

This module provides Dataset wrappers around precomputed, frozen
sequence arrays produced by the preprocessing pipeline.

Design principles:
- Read-only access to preprocessed data
- No windowing, scaling, or feature engineering
- Deterministic and reproducible
- Explicit shape and dtype validation
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SSITimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for SSI sequence-to-one forecasting.

    Each sample consists of:
    - X: sequence of shape (seq_len, n_features)
    - y: scalar target value (SSI at forecast horizon)

    The dataset assumes inputs are already:
    - Chronologically split
    - Scaled
    - Windowed
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialise the dataset.

        Parameters
        ----------
        X : np.ndarray
            Input sequences of shape (N, seq_len, n_features).
        y : np.ndarray
            Target values of shape (N,).

        Raises
        ------
        ValueError
            If shapes are incompatible or inputs are not NumPy arrays.
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be NumPy arrays")

        if X.ndim != 3:
            raise ValueError("X must have shape (N, seq_len, n_features)")

        if y.ndim != 1:
            raise ValueError("y must have shape (N,)")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        # Convert to torch tensors (float32 for PyTorch compatibility)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        """Return number of samples."""
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        X_seq : torch.Tensor
            Input sequence of shape (seq_len, n_features).
        y_target : torch.Tensor
            Scalar SSI target.
        """
        return self.X[idx], self.y[idx]
