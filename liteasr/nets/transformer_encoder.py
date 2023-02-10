"""Transformer encoder."""

from typing import Optional

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from liteasr.nets.attention import MultiHeadAttention
from liteasr.nets.attention import RelativeMultiHeadAttention
from liteasr.nets.conformer_convolution import Convolution
from liteasr.nets.conformer_layer import EncoderLayer as ConformerEncoderLayer
from liteasr.nets.conformer_layer import (
    RelativeEncoderLayer as RelativeConformerEncoderLayer,
)
from liteasr.nets.feed_forward import PositionwiseFeedForward
from liteasr.nets.positional_encoding import PositionalEncoding
from liteasr.nets.positional_encoding import RelativePositionalEncoding
from liteasr.nets.subsampling import Conv2DLayer
from liteasr.nets.swish import Swish
from liteasr.nets.transformer_layer import EncoderLayer as TransformerEncoderLayer
from liteasr.nets.transformer_layer import LayerNorm
from liteasr.nets.transformer_layer import (
    RelativeEncoderLayer as RelativeTransformerEncoderLayer,
)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        use_rel: bool,
        i_dim: int,
        h_dim: int,
        ff_dim: int,
        n_head: int,
        n_layer: int,
        dropout_rate: float,
        pos_dropout_rate: float,
        attn_dropout_rate: float,
        ff_dropout_rate: float,
        activation: str,
        arch: str,
    ) -> None:
        super().__init__()
        self.embed = Conv2DLayer(i_dim, h_dim, dropout_rate)

        pe = RelativePositionalEncoding if use_rel else PositionalEncoding
        mha = RelativeMultiHeadAttention if use_rel else MultiHeadAttention

        self.pe = pe(h_dim, dropout_rate=pos_dropout_rate)

        if arch == "transformer":
            tfm_layer = (
                RelativeTransformerEncoderLayer if use_rel else TransformerEncoderLayer
            )
            self.enc_layers = nn.ModuleList(
                [
                    tfm_layer(
                        size=h_dim,
                        self_attn=mha(n_head, h_dim, attn_dropout_rate),
                        feed_forward=PositionwiseFeedForward(
                            h_dim,
                            ff_dim,
                            dropout_rate=ff_dropout_rate,
                        ),
                        dropout_rate=dropout_rate,
                    )
                    for _ in range(n_layer)
                ]
            )
        elif arch == "conformer":
            cfm_layer = (
                RelativeConformerEncoderLayer if use_rel else ConformerEncoderLayer
            )

            if activation == "relu":
                activation_f = nn.ReLU()
            elif activation == "swish":
                activation_f = Swish()

            self.enc_layers = nn.ModuleList(
                [
                    cfm_layer(
                        size=h_dim,
                        self_attn=mha(n_head, h_dim, attn_dropout_rate),
                        feed_forward=PositionwiseFeedForward(
                            h_dim,
                            ff_dim,
                            dropout_rate=ff_dropout_rate,
                            activation=activation_f,
                        ),
                        feed_forward_macaron=PositionwiseFeedForward(
                            h_dim,
                            ff_dim,
                            dropout_rate=ff_dropout_rate,
                            activation=activation_f,
                        ),
                        conv=Convolution(h_dim, 15, activation=activation_f),
                        dropout_rate=dropout_rate,
                    )
                    for _ in range(n_layer)
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

        if isinstance(x, tuple):
            x = x[0]
        x = self.after_norm(x)
        return x


class Wav2Vec2TansformerEncoder(nn.Module):
    def __init__(
        self,
        i_dim: int,
        h_dim: int,
        ff_dim: int,
        n_head: int,
        n_layer: int,
        dropout_rate: float,
        attn_dropout_rate: float,
        ff_dropout_rate: float,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.embed = nn.Conv1d(
            in_channels=i_dim,  # cfg.encoder_embed_dim
            out_channels=i_dim,  # cfg.encoder_embed_dim
            kernel_size=128,  # cfg.conv_pos
            padding=128 // 2,  # cfg.conv_pos // 2
            groups=16,  # cfg.conv_pos_groups
        )
        self.gelu = nn.GELU()
        self.embed_norm = LayerNorm(i_dim)

        # assert arch == "transformer", f"{arch}"
        self.enc_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=i_dim,
                    self_attn=MultiHeadAttention(
                        n_head,
                        h_dim,
                        attn_dropout_rate,
                    ),
                    feed_forward=PositionwiseFeedForward(
                        h_dim,
                        ff_dim,
                        dropout_rate=ff_dropout_rate,
                    ),
                    dropout_rate=dropout_rate,
                )
                for _ in range(n_layer)
            ]
        )

    def extract_features(self, x, mask: Optional[Tensor] = None):
        residual = x
        x = self.embed(x.transpose(1, 2))
        x = x[:, :, :-1]
        x = residual + self.gelu(x).transpose(1, 2)
        x = self.embed_norm(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return x

    def forward(self, x, mask: Optional[Tensor] = None):
        x = self.extract_features(x, mask)

        x = x.transpose(0, 1)  # B x T x C -> T x B x C
        for n, layer in enumerate(self.enc_layers):
            x = layer(x, mask=mask)
        x = x.transpose(0, 1)  # T x B x C -> B x T x C

        return x
