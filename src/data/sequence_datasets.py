"""Deterministic, leak-free windowed sequence dataset construction for temporal models."""

from pathlib import Path

import numpy as np
import pandas as pd


def build_sequence_dataset(
        X: pd.DataFrame,
        y: pd.Series,
        window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a windowed sequence dataset from aligned feature and target arrays.

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
        Shape (N - window, window, F).
    y_seq : np.ndarray
        Shape (N - window,).
    timestamps : np.ndarray
        Target timestamps corresponding to y_seq (t+1).
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
    """Save a sequence dataset to disk as a compressed NumPy archive."""
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
    """Load a sequence dataset from a compressed NumPy archive."""
    data = np.load(input_path)
    return data["X"], data["y"], data["timestamps"]


def build_and_save_all_splits(
        processed_dir: Path,
        output_dir: Path,
        window: int,
):
    """Build and save sequence datasets for train, val, and test splits."""
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
