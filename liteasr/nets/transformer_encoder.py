"""Transformer encoder."""

from typing import Optional

from torch import Tensor
import torch.nn as nn

from liteasr.nets.attention import MultiHeadAttention
from liteasr.nets.feed_forward import PositionwiseFeedForward
from liteasr.nets.positional_encoding import PositionalEncoding
from liteasr.nets.subsampling import Conv2DLayer
from liteasr.nets.transformer_layer import EncoderLayer
from liteasr.nets.transformer_layer import LayerNorm


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        i_dim: int,
        h_dim: int,
        ff_dim: int,
        n_head: int,
        n_layer: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.embed = Conv2DLayer(i_dim, h_dim, dropout_rate)
        self.pe = PositionalEncoding(h_dim, dropout_rate)
        self.enc_layers = nn.ModuleList(
            [
                EncoderLayer(
                    size=h_dim,
                    self_attn=MultiHeadAttention(n_head, h_dim, dropout_rate),
                    feed_forward=PositionwiseFeedForward(
                        h_dim, ff_dim, dropout_rate
                    ),
                    dropout_rate=dropout_rate,
                ) for _ in range(n_layer)
            ]
        )
        self.after_norm = LayerNorm(h_dim)

    def forward(self, x, mask: Optional[Tensor] = None):
        """Forward function of Transformer encoder.

        :param x: Tensor with shape (batch, time, feature)
        :type x: Tensor
        """
        if mask is not None:
            assert mask.size() == x.size()[:2]
        x = self.embed(x)
        x = self.pe(x)
        if mask is not None:
            mask = mask[:, :-2:2][:, :-2:2]  # convolution simulation
            b, d = mask.size()
            mask = mask.view(b, 1, 1, d)
        for n, layer in enumerate(self.enc_layers):
            x = layer(x, mask=mask)
        x = self.after_norm(x)
        return x
