"""Persistence baseline that predicts each timestep as the value of the previous one."""

import numpy as np


def persistence_forecast(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ground-truth and persistence predictions aligned by one timestep.

    Parameters
    ----------
    y
        1-D target time series of length T.

    Returns
    -------
    tuple
        ``(y_true, y_pred)`` where ``y_true`` spans t=1…T-1 and
        ``y_pred`` spans t=0…T-2 (i.e. the previous value as the forecast).
    """
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")

    if len(y) < 2:
        raise ValueError("Persistence baseline requires at least 2 timesteps")

    return y[1:], y[:-1]
