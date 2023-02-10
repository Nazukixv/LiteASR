import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        i_dim: int,
        h_units: int,
        dropout_rate: float,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.fc1 = nn.Linear(i_dim, h_units)
        self.fc2 = nn.Linear(h_units, i_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))
