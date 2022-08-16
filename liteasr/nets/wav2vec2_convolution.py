from typing import List, Tuple

import torch.nn as nn

from liteasr.nets.layer_norm import LayerNorm


class ConvolutionBlock(nn.Module):

    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel: int,
        stride: int,
        conv_bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv1d(n_in, n_out, kernel, stride, bias=conv_bias)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = LayerNorm(n_out, dim=-2)
        self.gelu = nn.GELU()
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        x = self.dropout(self.conv(x))
        x = self.layer_norm(x)
        x = self.gelu(x)
        return x


class Convolution(nn.Module):

    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        conv_bias: bool = False,
    ):
        super().__init__()
        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, params in enumerate(conv_layers):
            dim, kernel, stride = params
            self.conv_layers.append(
                ConvolutionBlock(
                    in_d,
                    dim,
                    kernel,
                    stride,
                    conv_bias=conv_bias,
                    dropout=dropout,
                )
            )
            in_d = dim

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, time) -> (batch, 1, time)

        for conv in self.conv_layers:
            x = conv(x)

        return x
