import torch.nn as nn


class PositionwiseFeedForward(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))
