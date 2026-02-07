from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.sequence_datasets import (
    build_sequence_dataset,
    save_sequence_dataset,
    load_sequence_dataset,
)


def _make_dummy_data(n_rows=10, n_features=3):
    """
    Create a simple deterministic time series dataset.
    """
    index = pd.date_range("2020-01-01", periods=n_rows, freq="h")
    X = pd.DataFrame(
        np.arange(n_rows * n_features).reshape(n_rows, n_features),
        index=index,
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(
        np.arange(n_rows),
        index=index,
        name="target",
    )
    return X, y


def test_build_sequence_dataset_shapes():
    X, y = _make_dummy_data(n_rows=10, n_features=2)
    window = 3

    X_seq, y_seq, ts = build_sequence_dataset(X, y, window)

    # N - window samples
    assert X_seq.shape == (7, 3, 2)
    assert y_seq.shape == (7,)
    assert ts.shape == (7,)


def test_build_sequence_dataset_temporal_alignment():
    """
    Ensure that:
    - Each input window ends at time t
    - Target corresponds to t+1
    """
    X, y = _make_dummy_data(n_rows=6, n_features=1)
    window = 2

    X_seq, y_seq, ts = build_sequence_dataset(X, y, window)

    # First sample:
    # X window should be rows [0, 1]
    # y target should be row 2
    np.testing.assert_array_equal(X_seq[0, :, 0], [0, 1])
    assert y_seq[0] == 2
    assert ts[0] == y.index[2]


def test_build_sequence_dataset_rejects_misaligned_inputs():
    X, y = _make_dummy_data(n_rows=5)
    y = y.iloc[:-1]  # misalign lengths

    with pytest.raises(ValueError):
        build_sequence_dataset(X, y, window=2)


def test_build_sequence_dataset_rejects_short_series():
    X, y = _make_dummy_data(n_rows=3)

    with pytest.raises(ValueError):
        build_sequence_dataset(X, y, window=3)


def test_build_sequence_dataset_rejects_invalid_window():
    X, y = _make_dummy_data(n_rows=10)

    with pytest.raises(ValueError):
        build_sequence_dataset(X, y, window=0)


def test_save_and_load_sequence_dataset(tmp_path: Path):
    X, y = _make_dummy_data(n_rows=8, n_features=2)
    window = 3

    X_seq, y_seq, ts = build_sequence_dataset(X, y, window)

    path = tmp_path / "seq.npz"
    save_sequence_dataset(path, X_seq, y_seq, ts)

    X_loaded, y_loaded, ts_loaded = load_sequence_dataset(path)

    np.testing.assert_array_equal(X_loaded, X_seq)
    np.testing.assert_array_equal(y_loaded, y_seq)
    np.testing.assert_array_equal(ts_loaded, ts)
