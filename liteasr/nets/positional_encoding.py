import math

import torch
from torch import Tensor
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        h_dim: int,
        dropout_rate: float,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        if h_dim % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding "
                "with odd dim (got dim={:d})".format(h_dim)
            )
        self.h_dim = h_dim
        self.scale = math.sqrt(h_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.max_len = max_len
        pe = self.init_pe()
        self.register_buffer("pe", pe)

    def init_pe(self) -> Tensor:
        pe = torch.zeros(self.max_len, self.h_dim)
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.h_dim, 2).float()
            * -(math.log(10000.0) / self.h_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def extend_pe(self, x) -> None:
        """Extend current positional encoding.

        :param x: `Tensor` with size (B x L x D)
        :type x: Tensor
        """
        self.max_len = x.size(1)
        self.pe = self.init_pe()

    def forward(self, x) -> Tensor:
        if self.pe.size(1) < x.size(1):
            self.extend_pe(x)
        if self.pe.dtype != x.dtype or str(self.pe.device) != str(x.device):
            self.pe = self.pe.to(device=x.device, dtype=x.dtype)
        x = x * self.scale + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x


class RelativePositionalEncoding(PositionalEncoding):

    def __init__(
        self,
        h_dim: int,
        dropout_rate: float,
        max_len: int = 5000,
    ) -> None:
        super().__init__(h_dim, dropout_rate, max_len)

    def forward(self, x):
        if self.pe.size(1) < x.size(1):
            self.extend_pe(x)
        if self.pe.dtype != x.dtype or str(self.pe.device) != str(x.device):
            self.pe = self.pe.to(device=x.device, dtype=x.dtype)
        x = x * self.scale
        pos_emb = self.pe[:, :x.size(1)]
        return self.dropout(x), self.dropout(pos_emb)
