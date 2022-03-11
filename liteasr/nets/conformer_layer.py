from typing import Optional, Tuple

import torch
from torch import Tensor

from liteasr.nets.layer_norm import LayerNorm
from liteasr.nets.transformer_layer import EncoderLayer as LegacyEncoderLayer


class EncoderLayer(LegacyEncoderLayer):

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv,
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
        self.feed_forward_macaron = feed_forward_macaron
        self.conv = conv
        self.feed_forward_macaron_norm = LayerNorm(size)
        self.conv_norm = LayerNorm(size)
        self.final_norm = LayerNorm(size)
        self.feed_forward_scale = 0.5

    def ff_macaron(self, x) -> Tensor:
        """Feed-forward macaron + ResNet sublayer."""

        residual = x
        x = self.feed_forward_macaron_norm(x) if self.normalize_before else x
        x = residual + self.feed_forward_scale * self.dropout(
            self.feed_forward_macaron(x)
        )
        x = self.feed_forward_macaron_norm(
            x
        ) if not self.normalize_before else x

        return x

    def convolution(self, x) -> Tensor:
        """Convolution + ResNet sublayer."""

        residual = x
        x = self.conv_norm(x) if self.normalize_before else x
        x = residual + self.dropout(self.conv(x))
        x = self.conv_norm(x) if not self.normalize_before else x
        return x

    def ff(self, x) -> Tensor:
        """Feed-forward + ResNet sublayer."""

        residual = x
        x = self.feed_forward_norm(x) if self.normalize_before else x
        x = residual + self.feed_forward_scale * self.dropout(
            self.feed_forward(x)
        )
        x = self.feed_forward_norm(x) if not self.normalize_before else x

        return x

    def forward(
        self,
        x,
        mask: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
    ):
        x = self.ff_macaron(x)
        x = self.mha(x, mask=mask, cache=cache)
        x = self.convolution(x)
        x = self.ff(x)
        x = self.final_norm(x)
        if cache is not None:
            x = torch.cat([cache, x], dim=1)
        return x


class RelativeEncoderLayer(EncoderLayer):

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        super().__init__(
            size,
            self_attn,
            feed_forward,
            feed_forward_macaron,
            conv,
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
        x_pos_emb,
        mask: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
    ):
        x, pos_emb = x_pos_emb[0], x_pos_emb[1]

        x = self.ff_macaron(x)
        x, pos_emb = self.mha(x, pos_emb, mask=mask, cache=cache)
        x = self.convolution(x)
        x = self.ff(x)
        x = self.final_norm(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, pos_emb
