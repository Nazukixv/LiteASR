"""Transformer decoder."""

import torch
import torch.nn as nn

from liteasr.nets.attention import MultiHeadAttention
from liteasr.nets.feed_forward import PositionwiseFeedForward
from liteasr.nets.positional_encoding import PositionalEncoding
from liteasr.nets.transformer_layer import DecoderLayer
from liteasr.nets.transformer_layer import LayerNorm


class TransformerDecoder(nn.Module):

    def __init__(
        self,
        i_dim: int,
        h_dim: int,
        ff_dim: int,
        n_head: int,
        n_layer: int,
        dropout_rate: float,
        pos_dropout_rate: float,
        self_attn_dropout_rate: float,
        src_attn_dropout_rate: float,
        ff_dropout_rate: float,
        arch: str,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(i_dim, h_dim)
        self.pe = PositionalEncoding(h_dim, dropout_rate=pos_dropout_rate)
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

    def forward_one_step(self, y, mask, memory, memory_mask, cache):
        y = self.embed(y)
        y = self.pe(y)
        new_cache = []
        for i, layer in enumerate(self.dec_layers):
            c = None if cache is None else cache[i]
            y = layer(y, mask, memory, memory_mask, cache=c)
            new_cache.append(y)
        y = self.after_norm(y[:, -1])
        y = torch.log_softmax(self.linear_out(y), dim=-1)
        return y, new_cache

    def forward(
        self,
        y,
        mask,
        memory,
        memory_mask,
    ):
        y = self.embed(y)
        y = self.pe(y)
        if mask is not None:
            assert mask.shape == (y.shape[0], y.shape[1], y.shape[1])
            mask = mask.unsqueeze(1)
        if memory_mask is not None:
            memory_mask = memory_mask[:, :-2:2][:, :-2:2]
            assert memory_mask.shape == (memory.shape[0], memory.shape[1])
            b, d = memory_mask.size()
            memory_mask = memory_mask.view(b, 1, 1, d)

        for layer in self.dec_layers:
            y = layer(y, mask, memory, memory_mask)
        y = self.after_norm(y)
        y = self.linear_out(y)

        return y
