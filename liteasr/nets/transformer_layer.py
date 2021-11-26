from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.LayerNorm):

    def __init__(self, nout, dim=-1):
        super().__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x: torch.Tensor):
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

    def forward(
        self,
        x,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None
    ):
        # self attention + ResNet
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

        # feed-forward + ResNet
        residual = x
        x = self.feed_forward_norm(x) if self.normalize_before else x
        x = self.dropout(self.feed_forward(x)) + residual
        x = self.feed_forward_norm(x) if not self.normalize_before else x

        # concatenate the cache
        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x
