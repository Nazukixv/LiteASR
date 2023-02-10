"""Parallel decoder."""

import torch.nn as nn

from liteasr.nets.attention import MultiHeadAttention
from liteasr.nets.feed_forward import PositionwiseFeedForward
from liteasr.nets.transformer_layer import DecoderLayer
from liteasr.nets.transformer_layer import LayerNorm


class ParallelDecoder(nn.Module):

    def __init__(
        self,
        i_dim: int,
        h_dim: int,
        ff_dim: int,
        n_head: int,
        n_layer: int,
        dropout_rate: float,
        self_attn_dropout_rate: float,
        src_attn_dropout_rate: float,
        ff_dropout_rate: float,
    ) -> None:
        super().__init__()
        # self.pe = PositionalEncoding(h_dim, pos_dropout_rate)
        self.dec_layers = nn.ModuleList(
            [
                DecoderLayer(
                    size=h_dim,
                    self_attn=MultiHeadAttention(
                        n_head=n_head,
                        i_dim=h_dim,
                        dropout_rate=self_attn_dropout_rate,
                    ),
                    src_attn=MultiHeadAttention(
                        n_head=n_head,
                        i_dim=h_dim,
                        dropout_rate=src_attn_dropout_rate,
                    ),
                    feed_forward=PositionwiseFeedForward(
                        i_dim=h_dim,
                        h_units=ff_dim,
                        dropout_rate=ff_dropout_rate,
                    ),
                    dropout_rate=dropout_rate,
                ) for _ in range(n_layer)
            ]
        )
        self.after_norm = LayerNorm(h_dim)
        self.linear_out = nn.Linear(h_dim, i_dim)

    def forward(self, y, memory, memory_mask):
        # y = self.pe(y)
        if memory_mask is not None:
            memory_mask = memory_mask[:, :-2:2][:, :-2:2]
            assert memory_mask.shape == (memory.shape[0], memory.shape[1])
            b, d = memory_mask.size()
            memory_mask = memory_mask.view(b, 1, 1, d)

        for layer in self.dec_layers:
            y = layer(y, None, memory, memory_mask)
        y = self.after_norm(y)
        y = self.linear_out(y)

        return y
