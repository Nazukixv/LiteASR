"""RNN decoder."""

from typing import Tuple, List

import torch
from torch import Tensor
import torch.nn as nn


class RNNDecoder(nn.Module):

    def __init__(
        self,
        i_dim: int,
        h_dim: int,
        h_units: int,
        n_layer: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.embed = nn.Embedding(i_dim, h_dim)
        self.dropout_embed = nn.Dropout(dropout_rate)
        self.dec_layers = nn.ModuleList(
            [nn.LSTMCell(h_dim, h_units)]
            + [nn.LSTMCell(h_units, h_units) for _ in range(1, n_layer)]
        )
        self.dropout_dec = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(n_layer)]
        )
        self.h_units = h_units
        self.n_layer = n_layer

    def init_state(self, y) -> Tuple[List[Tensor], List[Tensor]]:
        batch = y.size(0)
        device = y.device
        h_states = [
            torch.zeros(batch, self.h_units).to(device=device)
            for _ in range(self.n_layer)
        ]
        c_states = [
            torch.zeros(batch, self.h_units).to(device=device)
            for _ in range(self.n_layer)
        ]
        return h_states, c_states

    def rnn_forward(
        self,
        yi,
        hi: List[Tensor],
        ci: List[Tensor],
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        hj, cj = [], []
        h = hi[0]  # initialization just to pass TorchScript check
        for n, (layer,
                dropout) in enumerate(zip(self.dec_layers, self.dropout_dec)):
            if n == 0:
                h, c = layer(yi, (hi[n], ci[n]))
                hj.append(h)
                cj.append(c)
                h = dropout(h)
            else:
                h, c = layer(h, (hi[n], ci[n]))
                hj.append(h)
                cj.append(c)
                h = dropout(h)
        return h, hj, cj

    def forward(self, y) -> Tensor:
        # embedding
        y = self.dropout_embed(self.embed(y))

        # LSTM
        ht, ct = self.init_state(y)
        h_dec_t = []
        for t in range(y.size(1)):
            h_out, ht, ct = self.rnn_forward(y[:, t, :], ht, ct)
            h_dec_t.append(h_out)

        # compose final output
        h_dec = torch.stack(h_dec_t, dim=1)

        return h_dec
