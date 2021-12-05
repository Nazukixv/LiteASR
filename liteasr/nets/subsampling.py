"""Subsampling module."""

from typing import Optional

from torch import Tensor
import torch.nn as nn


class Conv2DLayer(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Module accepts an input tensor with size (B, T1, I_DIM),
    then subsamples it into tensor with size (B, T2, O_DIM)
    in which T2 is about T1 / 4.

    :param i_dim: Dimension of input feature
    :type i_dim: int
    :param o_dim: Dimension of output feature
    :type o_dim: int
    :param dropout_rate: Dropout rate
    :type o_dim: float
    """

    def __init__(
        self,
        i_dim: int,
        o_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, o_dim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(o_dim, o_dim, 3, 2),
            nn.ReLU(),
        )
        # after convolution, tensor size should be (b, o_dim, t, f_dim)
        f_dim = (i_dim - 3) // 2 + 1
        f_dim = (f_dim - 3) // 2 + 1
        self.out = nn.Linear(o_dim * f_dim, o_dim)

    def forward(self, x) -> Tensor:
        x = x.unsqueeze(1)  # (b, t, i_dim) -> (b, 1, t, i_dim)
        x = self.conv(x)  # (b, 1, t, i_dim) -> (b, o_dim, t/4, f_dim)
        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        x = self.out(x)
        return x
