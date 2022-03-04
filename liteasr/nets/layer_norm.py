"""Layer normalization."""

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.LayerNorm):

    def __init__(self, nout: int, dim=-1):
        super().__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x) -> Tensor:
        if self.dim == -1:
            return F.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )
        else:
            return F.layer_norm(
                x.transpose(1, -1),
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            ).transpose(1, -1)
