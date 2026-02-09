"""
Tests for temporal LSTM model definitions.

These tests validate the structural and interface correctness of the
LSTMRegressor model, ensuring it adheres to the expected input/output
contracts for sequence-based SSI forecasting.

The tests intentionally avoid training or optimisation logic and focus
solely on forward-pass behaviour and tensor shape correctness.
"""

import pytest
import torch

from src.models.temporal_model import LSTMRegressor


def test_lstm_regressor_instantiation():
    """
    Test that the LSTMRegressor can be instantiated with valid parameters.
    """
    model = LSTMRegressor(
        n_features=3,
        hidden_size=16,
        num_layers=1,
    )

    assert isinstance(model, torch.nn.Module)
    assert model.n_features == 3
    assert model.hidden_size == 16
    assert model.num_layers == 1


def test_lstm_forward_pass_shape():
    """
    Test that the forward pass returns the correct output shape.

    Expected:
        Input shape:  (batch, seq_len, n_features)
        Output shape: (batch,)
    """
    batch_size = 8
    seq_len = 12
    n_features = 3

    model = LSTMRegressor(
        n_features=n_features,
        hidden_size=32,
    )

    x = torch.randn(batch_size, seq_len, n_features)
    y = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (batch_size,)


def test_lstm_forward_pass_different_batch_sizes():
    """
    Test that the model supports varying batch sizes.
    """
    n_features = 5
    seq_len = 10

    model = LSTMRegressor(n_features=n_features)

    for batch_size in [1, 4, 16]:
        x = torch.randn(batch_size, seq_len, n_features)
        y = model(x)
        assert y.shape == (batch_size,)


def test_lstm_forward_pass_different_sequence_lengths():
    """
    Test that the model supports varying sequence lengths.
    """
    batch_size = 4
    n_features = 3

    model = LSTMRegressor(n_features=n_features)

    for seq_len in [2, 8, 24]:
        x = torch.randn(batch_size, seq_len, n_features)
        y = model(x)
        assert y.shape == (batch_size,)


def test_lstm_raises_on_invalid_input_shape():
    """
    Test that the model fails clearly when given malformed input.
    """
    model = LSTMRegressor(n_features=3)

    # Missing sequence dimension
    bad_input = torch.randn(8, 3)

    with pytest.raises(RuntimeError):
        model(bad_input)
