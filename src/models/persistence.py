"""
Persistence (naïve) baseline for time series forecasting.

This baseline predicts the next timestep value as the current value:
    y_hat[t+1] = y[t]

It serves as a lower-bound benchmark for temporal models.
"""

from typing import Tuple

import numpy as np


def persistence_forecast(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate persistence (t → t+1) predictions.

    Parameters
    ----------
    y : np.ndarray
        Target time series of shape (T,).

    Returns
    -------
    y_true : np.ndarray
        Ground-truth values from t=1..T-1.
    y_pred : np.ndarray
        Persistence predictions from t=0..T-2.

    Raises
    ------
    ValueError
        If input has fewer than 2 timesteps.
    """
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")

    if len(y) < 2:
        raise ValueError("Persistence baseline requires at least 2 timesteps")

    y_pred = y[:-1]
    y_true = y[1:]

    return y_true, y_pred
