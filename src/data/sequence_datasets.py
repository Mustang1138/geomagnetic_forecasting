"""
Sequence dataset construction utilities.

This module provides deterministic, leak-free construction of
windowed sequence datasets for temporal models (e.g. persistence,
linear-on-window, LSTM, GRU).

Design principles:
- Strict temporal causality (targets are always in the future)
- No data leakage across time or splits
- Stateless, testable core logic
- Compatible with frozen preprocessing outputs
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_sequence_dataset(
        X: pd.DataFrame,
        y: pd.Series,
        window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a windowed sequence dataset.

    For each time step t, the input sequence consists of the previous
    `window` observations up to and including t, and the target is
    the value at t+1.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix, time-ordered.
    y : pd.Series
        Target vector, time-ordered and aligned with X.
    window : int
        Number of past time steps to include in each input sequence.

    Returns
    -------
    X_seq : np.ndarray
        Array of shape (N - window, window, F)
    y_seq : np.ndarray
        Array of shape (N - window)
    timestamps : np.ndarray
        Target timestamps corresponding to y_seq (t+1)

    Raises
    ------
    ValueError
        If inputs are misaligned or insufficient length.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series")

    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    if window < 1:
        raise ValueError("window must be >= 1")

    if len(X) <= window:
        raise ValueError(
            f"Insufficient rows ({len(X)}) for window length {window}"
        )

    X_values = X.values
    y_values = y.values
    timestamps = y.index.values

    X_seq = []
    y_seq = []
    ts_seq = []

    # Last usable t is len(X) - 2 (because target is t+1)
    for t in range(window - 1, len(X) - 1):
        X_seq.append(X_values[t - window + 1: t + 1])
        y_seq.append(y_values[t + 1])
        ts_seq.append(timestamps[t + 1])

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        np.asarray(ts_seq),
    )


def save_sequence_dataset(
        output_path: Path,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        timestamps: np.ndarray,
):
    """
    Save a sequence dataset to disk as a compressed NumPy archive.

    Parameters
    ----------
    output_path : Path
        Path to the .npz file.
    X_seq : np.ndarray
        Input sequences.
    y_seq : np.ndarray
        Targets.
    timestamps : np.ndarray
        Target timestamps.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X=X_seq,
        y=y_seq,
        timestamps=timestamps,
    )


def load_sequence_dataset(
        input_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a sequence dataset from a compressed NumPy archive.

    Parameters
    ----------
    input_path : Path
        Path to the .npz file.

    Returns
    -------
    X_seq, y_seq, timestamps
    """
    data = np.load(input_path)
    return data["X"], data["y"], data["timestamps"]


def build_and_save_all_splits(
        processed_dir: Path,
        output_dir: Path,
        window: int,
):
    """
    Build and save sequence datasets for all data splits.

    Expects the following files under processed_dir:
    - X_train.csv, y_train.csv
    - X_val.csv, y_val.csv
    - X_test.csv, y_test.csv

    Parameters
    ----------
    processed_dir : Path
        Directory containing frozen preprocessing outputs.
    output_dir : Path
        Directory to save sequence datasets.
    window : int
        Sequence window length.
    """
    for split in ["train", "val", "test"]:
        X = pd.read_csv(
            processed_dir / f"X_{split}.csv",
            index_col=0,
            parse_dates=True,
        )
        y = pd.read_csv(
            processed_dir / f"y_{split}.csv",
            index_col=0,
            parse_dates=True,
        ).iloc[:, 0]

        X_seq, y_seq, ts = build_sequence_dataset(X, y, window)

        save_sequence_dataset(
            output_dir / f"w{window:02d}" / f"{split}.npz",
            X_seq,
            y_seq,
            ts,
        )
