"""Transducer."""

from dataclasses import dataclass
from dataclasses import field

from omegaconf import II
import torch
import torch.nn as nn

from liteasr.config import LiteasrDataclass
from liteasr.models import LiteasrModel
from liteasr.models import register_model
from liteasr.nets.rnn_decoder import RNNDecoder
from liteasr.nets.transformer_encoder import TransformerEncoder
from liteasr.utils.mask import padding_mask


@dataclass
class TransducerConfig(LiteasrDataclass):
    # transducer
    joint_dim: int = field(default=768)
    dropout_rate: float = field(default=0.0)

    # encoder
    input_dim: int = field(default=0)
    enc_dim: int = field(default=256)
    enc_ff_dim: int = field(default=2048)
    enc_attn_heads: int = field(default=4)
    enc_dropout_rate: float = II("model.dropout_rate")
    enc_layers: int = field(default=4)
    activation: str = field(default="relu")

    # decoder
    vocab_size: int = field(default=0)
    dec_dim: int = field(default=256)
    dec_units: int = field(default=2048)
    dec_dropout_rate: float = II("model.dropout_rate")
    dec_layers: int = field(default=2)


@register_model("transducer", dataclass=TransducerConfig)
class Transducer(LiteasrModel):

    def __init__(self, cfg: TransducerConfig, task=None):
        super().__init__()
        self.encoder = TransformerEncoder(
            i_dim=cfg.input_dim,
            h_dim=cfg.enc_dim,
            ff_dim=cfg.enc_ff_dim,
            n_head=cfg.enc_attn_heads,
            n_layer=cfg.enc_layers,
            dropout_rate=cfg.enc_dropout_rate,
        )
        self.decoder = RNNDecoder(
            vocab_size=cfg.vocab_size,
            h_dim=cfg.dec_dim,
            h_units=cfg.dec_units,
            n_layer=cfg.dec_layers,
            dropout_rate=cfg.dec_dropout_rate,
        )
        self.lin_enc = nn.Linear(cfg.enc_dim, cfg.joint_dim)
        self.lin_dec = nn.Linear(cfg.dec_units, cfg.joint_dim, bias=False)
        self.lin_jnt = nn.Linear(cfg.joint_dim, cfg.vocab_size)
        self.joint_activation = nn.Tanh()
        self.padding_id = 0

    def forward(self, x, y):
        """Forward function of Transducer.

        :param x: (batch, time, feat)
        :type x: Tensor
        :param y: (batch, length)
        :type y: Tensor
        :return: (batch, frame, length, vocab)
        :rtype: Tensor
        """

        xs, ys, xs_mask, ys_mask = self._input_prep(x, y)
        h_enc = self.encoder(xs, mask=xs_mask).unsqueeze(2)
        h_dec = self.decoder(ys).unsqueeze(1)
        h_jnt = self.joint(h_enc, h_dec)
        return h_jnt

    def get_pred_len(self, xlens):
        pred_len = torch.tensor(xlens)
        pred_len = ((pred_len - 1) // 2 - 1) // 2
        return pred_len

    def get_target_len(self, ylens):
        target_len = torch.tensor(ylens)
        return target_len

    def joint(self, h_enc, h_dec):
        h_enc = self.lin_enc(h_enc)
        h_dec = self.lin_dec(h_dec)
        h_jnt = self.lin_jnt(self.joint_activation(h_enc + h_dec))
        return h_jnt

    def _input_prep(self, x, y):
        xlens = (x != self.padding_id).sum(-1).bool().sum(-1).tolist()
        xs_mask = padding_mask(xlens).to(device=x.device)

        ylens = ((y != self.padding_id).sum(-1)
                 + 1).tolist()  # add <blank> in front
        ys_mask = padding_mask(ylens).to(device=y.device)

        blank = y.new(y.size(0), 1).fill_(0)
        blank_y = torch.cat((blank, y), dim=-1)

        return x, blank_y, xs_mask, ys_mask

    @classmethod
    def build_model(cls, cfg, task=None):
        return cls(cfg, task)
