from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from liteasr.nets.layer_norm import LayerNorm


class EncoderLayer(nn.Module):

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.self_attn_norm = LayerNorm(size)
        self.feed_forward_norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def mha(
        self,
        x,
        mask: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
    ) -> Tensor:
        """Multi-head Attention + ResNet sublayer."""

        residual = x
        x = self.self_attn_norm(x) if self.normalize_before else x

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = mask[:, -1:, :] if mask is not None else mask

        x = self.dropout(self.self_attn(x_q, x, x, mask)) + residual
        x = self.self_attn_norm(x) if not self.normalize_before else x

        return x

    def ff(self, x) -> Tensor:
        """Feed-forward + ResNet sublayer."""

        residual = x
        x = self.feed_forward_norm(x) if self.normalize_before else x
        x = self.dropout(self.feed_forward(x)) + residual
        x = self.feed_forward_norm(x) if not self.normalize_before else x

        return x

    def forward(
        self,
        x,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None
    ):
        x = self.mha(x, mask=mask, cache=cache)
        x = self.ff(x)

        # concatenate the cache
        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x


class RelativeEncoderLayer(EncoderLayer):

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        super().__init__(
            size,
            self_attn,
            feed_forward,
            dropout_rate,
            normalize_before,
            concat_after,
        )

    def mha(
        self,
        x,
        pos_emb,
        mask: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        residual = x
        x = self.self_attn_norm(x) if self.normalize_before else x

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = mask[:, -1:, :] if mask is not None else mask

        x = self.dropout(self.self_attn(x_q, x, x, pos_emb, mask)) + residual
        x = self.self_attn_norm(x) if not self.normalize_before else x

        return x, pos_emb

    def forward(
        self,
        x_pos_emb: Tuple[Tensor, Tensor],
        mask: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
    ):
        x, pos_emb = x_pos_emb[0], x_pos_emb[1]

        x, pos_emb = self.mha(x, pos_emb, mask=mask, cache=cache)
        x = self.ff(x)

        # concatenate the cache
        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, pos_emb
