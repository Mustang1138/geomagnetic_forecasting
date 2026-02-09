"""
Temporal sequence models for geomagnetic storm forecasting.

Implements LSTM and GRU architectures for sequence-to-one SSI prediction.

References:
- Hochreiter and Schmidhuber (1997) - LSTM architecture
- Cho et al. (2014) - GRU architecture
- Abduallah et al. (2022) - LSTM for Dst prediction
"""

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """
    LSTM-based regressor for sequence-to-one SSI forecasting.

    Architecture:
    - LSTM layer(s) process temporal sequence
    - Fully connected layer maps final hidden state to SSI prediction

    Uses final hidden state from LSTM to predict next time step,
    following standard sequence-to-one forecasting (Hochreiter and
    Schmidhuber, 1997; Abduallah et al., 2022).
    """

    def __init__(
            self,
            n_features: int,
            hidden_size: int = 64,
            num_layers: int = 1,
    ):
        """
        Initialise LSTM regressor.

        Parameters
        ----------
        n_features : int
            Number of input features per time step.
        hidden_size : int, optional
            Dimensionality of LSTM hidden state. Default is 64.
        num_layers : int, optional
            Number of stacked LSTM layers. Default is 1.
        """
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM processes sequences with batch-first convention
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # FC layer maps final hidden state to scalar prediction
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM regressor.

        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (batch_size, seq_len, n_features).

        Returns
        -------
        torch.Tensor
            Predicted SSI values of shape (batch_size,).
        """
        # Validate input shape
        if x.ndim != 3:
            raise RuntimeError(
                f"Expected 3D input (batch, seq_len, features), got shape {x.shape}"
            )
        if x.shape[2] != self.n_features:
            raise RuntimeError(
                f"Expected {self.n_features} features, got {x.shape[2]}"
            )

        # Process sequence through LSTM
        # h_n[-1] is final hidden state from top layer
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]

        # Map to scalar prediction and remove trailing dimension
        y = self.fc(h_last)
        return y.squeeze(-1)


class GRURegressor(nn.Module):
    """
    GRU-based regressor for sequence-to-one SSI forecasting.

    Simplified variant of LSTM with fewer parameters. Often performs
    comparably whilst training faster (Cho et al., 2014).

    Architecture identical to LSTMRegressor but uses GRU cells,
    which lack separate cell state.
    """

    def __init__(
            self,
            n_features: int,
            hidden_size: int = 64,
            num_layers: int = 1,
    ):
        """
        Initialise GRU regressor.

        Parameters
        ----------
        n_features : int
            Number of input features per time step.
        hidden_size : int, optional
            Dimensionality of GRU hidden state. Default is 64.
        num_layers : int, optional
            Number of stacked GRU layers. Default is 1.
        """
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU processes sequences with batch-first convention
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # FC layer maps final hidden state to scalar prediction
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GRU regressor.

        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (batch_size, seq_len, n_features).

        Returns
        -------
        torch.Tensor
            Predicted SSI values of shape (batch_size,).
        """
        # Validate input shape
        if x.ndim != 3:
            raise RuntimeError(
                f"Expected 3D input (batch, seq_len, features), got shape {x.shape}"
            )
        if x.shape[2] != self.n_features:
            raise RuntimeError(
                f"Expected {self.n_features} features, got {x.shape[2]}"
            )

        # Process sequence through GRU (no cell state unlike LSTM)
        # h_n[-1] is final hidden state from top layer
        _, h_n = self.gru(x)
        h_last = h_n[-1]

        # Map to scalar prediction and remove trailing dimension
        y = self.fc(h_last)
        return y.squeeze(-1)
