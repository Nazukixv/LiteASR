"""Paraformer."""

from dataclasses import dataclass
from dataclasses import field
import logging
from typing import Optional

from omegaconf import II
from omegaconf import MISSING
import torch
from torch import Tensor
import torch.nn as nn

from liteasr.config import LiteasrDataclass
from liteasr.models import LiteasrModel
from liteasr.models import register_model
from liteasr.nets.paraformer import GlancingSampler
from liteasr.nets.paraformer import ParallelDecoder
from liteasr.nets.paraformer import Predictor
from liteasr.nets.positional_encoding import PositionalEncoding
from liteasr.nets.transformer_encoder import TransformerEncoder
from liteasr.utils.mask import padding_mask

logger = logging.getLogger(__name__)


@dataclass
class ParaformerConfig(LiteasrDataclass):
    name: Optional[str] = field(default="Paraformer")

    # Paraformer
    dropout_rate: float = field(default=0.0)

    # encoder
    use_rel: bool = field(default=True)
    input_dim: int = field(default=MISSING)
    enc_dim: int = field(default=256)
    enc_ff_dim: int = field(default=2048)
    enc_attn_heads: int = field(default=4)
    enc_dropout_rate: float = II("model.dropout_rate")
    enc_pos_dropout_rate: float = II("model.enc_dropout_rate")
    enc_attn_dropout_rate: float = II("model.enc_dropout_rate")
    enc_ff_dropout_rate: float = II("model.enc_dropout_rate")
    enc_layers: int = field(default=12)
    activation: str = field(default="swish")

    # sampler
    sample_ratio: float = field(default=0.75)

    # decoder
    vocab_size: int = field(default=MISSING)
    dec_dim: int = field(default=256)
    dec_ff_dim: int = field(default=2048)
    dec_attn_heads: int = field(default=4)
    dec_dropout_rate: float = II("model.dropout_rate")
    dec_self_attn_dropout_rate: float = II("model.dec_dropout_rate")
    dec_src_attn_dropout_rate: float = II("model.dec_dropout_rate")
    dec_ff_dropout_rate: float = II("model.dec_dropout_rate")
    dec_layers: int = field(default=6)

    # positional encoding
    pos_dropout_rate: float = II("model.dec_dropout_rate")


@register_model("Paraformer", dataclass=ParaformerConfig)
class Paraformer(LiteasrModel):
    def __init__(self, cfg: ParaformerConfig, task=None):
        super().__init__()

        self.encoder = TransformerEncoder(
            use_rel=cfg.use_rel,
            i_dim=cfg.input_dim,
            h_dim=cfg.enc_dim,
            ff_dim=cfg.enc_ff_dim,
            n_head=cfg.enc_attn_heads,
            n_layer=cfg.enc_layers,
            dropout_rate=cfg.enc_dropout_rate,
            pos_dropout_rate=cfg.enc_pos_dropout_rate,
            attn_dropout_rate=cfg.enc_attn_dropout_rate,
            ff_dropout_rate=cfg.enc_ff_dropout_rate,
            activation=cfg.activation,
            arch="conformer",
        )
        self.decoder = ParallelDecoder(
            i_dim=cfg.vocab_size,
            h_dim=cfg.dec_dim,
            ff_dim=cfg.dec_ff_dim,
            n_head=cfg.dec_attn_heads,
            n_layer=cfg.dec_layers,
            dropout_rate=cfg.dec_dropout_rate,
            self_attn_dropout_rate=cfg.dec_self_attn_dropout_rate,
            src_attn_dropout_rate=cfg.dec_ff_dropout_rate,
            ff_dropout_rate=cfg.dec_ff_dropout_rate,
        )
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dec_dim)
        self.pe = PositionalEncoding(cfg.dec_dim, cfg.pos_dropout_rate)
        self.predictor = Predictor(cfg.enc_dim)
        self.sampler = GlancingSampler(cfg.sample_ratio)

        self.ignore = -1
        self.blank = 0
        self.eos = cfg.vocab_size - 1

    def forward(self, xs, xlens, ys, ylens):
        xs_in, xs_mask, ys_in, ys_mask = self._preprocess(xs, xlens, ys, ylens)

        hs_enc = self.encoder(xs_in, xs_mask)
        sampled_xlens = self.get_pred_len(xlens)
        hs_cif, sum_alpha = self.predictor(hs_enc, sampled_xlens, ylens)
        embed_ys = self.pe(self.embed(ys_in))

        # first pass (w/o grads)
        with torch.no_grad():
            hs_hat = self.decoder(hs_cif, memory=hs_enc, memory_mask=xs_mask)
            ys_hat = torch.argmax(hs_hat, dim=-1)
            ys_hat = ys_hat.masked_fill(ys_mask, self.eos)

        # second pass (w grads)
        hs_mix = self.sampler(hs_cif, embed_ys, ys_in, ys_hat, ylens)
        hs_attn = self.decoder(hs_mix, memory=hs_enc, memory_mask=xs_mask)

        return hs_attn, sum_alpha

    def inference(self, x):
        h = self.encoder(x)  # (1, time, 83) -> (1, frame, 256)
        h_cif, _ = self.predictor(h)
        h_attn = self.decoder(h_cif, memory=h, memory_mask=None)
        hyp = torch.argmax(h_attn, dim=-1)
        return hyp[0].tolist()

    def get_pred_len(self, xlens) -> Tensor:
        pred_len = ((xlens - 1) // 2 - 1) // 2
        return pred_len

    def get_target(self, ys, ylens) -> Tensor:
        return ys

    def get_target_len(self, ylens) -> Tensor:
        target_len = ylens
        return target_len

    def _preprocess(self, xs, xlens, ys, ylens):
        # xs_in
        xs_in = xs

        # xs_mask
        xs_mask = padding_mask(xlens)

        # ys_in
        ys_in = ys.masked_fill(ys == self.ignore, self.eos)

        # ys_mask
        ys_mask = padding_mask(ylens)

        return xs_in, xs_mask, ys_in, ys_mask

    @classmethod
    def build_model(cls, cfg: ParaformerConfig, task=None):
        cfg.input_dim = task.feat_dim
        cfg.vocab_size = task.vocab_size
        return cls(cfg, task)
