"""Module CTC."""

import torch.nn as nn
import torch.nn.functional as F


class CTC(nn.Module):
    """Module CTC.

    :param int i_dim: Input dimension
    :param int o_dim: Output dimension

    """

    def __init__(
        self,
        i_dim: int,
        o_dim: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.ctc_lo = nn.Linear(i_dim, o_dim)
        self.dropout_rate = dropout_rate

    def forward(self, xs):
        xs_hat = self.ctc_lo(F.dropout(xs, p=self.dropout_rate))
        return xs_hat
