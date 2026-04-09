"""LSTM and GRU regressors for sequence-to-one storm severity forecasting."""

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """LSTM-based regressor that maps an input sequence to a single SSI value."""

    def __init__(
            self,
            n_features: int,
            hidden_size: int = 64,
            num_layers: int = 1,
    ):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # batch_first=True: input shape is (batch, seq_len, features).
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted SSI values of shape ``(batch,)`` from input sequences."""
        if x.ndim != 3:
            raise RuntimeError(
                f"Expected 3D input (batch, seq_len, features), got shape {x.shape}"
            )
        if x.shape[2] != self.n_features:
            raise RuntimeError(
                f"Expected {self.n_features} features, got {x.shape[2]}"
            )

        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        return self.fc(h_last).squeeze(-1)


class GRURegressor(nn.Module):
    """GRU-based regressor that maps an input sequence to a single SSI value."""

    def __init__(
            self,
            n_features: int,
            hidden_size: int = 64,
            num_layers: int = 1,
    ):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # batch_first=True: input shape is (batch, seq_len, features).
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted SSI values of shape ``(batch,)`` from input sequences."""
        if x.ndim != 3:
            raise RuntimeError(
                f"Expected 3D input (batch, seq_len, features), got shape {x.shape}"
            )
        if x.shape[2] != self.n_features:
            raise RuntimeError(
                f"Expected {self.n_features} features, got {x.shape[2]}"
            )

        _, h_n = self.gru(x)
        h_last = h_n[-1]
        return self.fc(h_last).squeeze(-1)
