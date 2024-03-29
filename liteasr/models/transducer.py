"""Transducer."""

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import List, Optional

from omegaconf import II
from omegaconf import MISSING
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from liteasr.config import LiteasrDataclass
from liteasr.models import LiteasrModel
from liteasr.models import register_model
from liteasr.nets.initialization import lecun_normal_init_parameters
from liteasr.nets.initialization import set_forget_bias_to_one
from liteasr.nets.rnn_decoder import RNNDecoder
from liteasr.nets.transformer_encoder import TransformerEncoder
from liteasr.utils.mask import padding_mask


class Hypothesis(object):
    def __init__(
        self,
        score: float,
        yseq: List[int],
        str_yseq: str,
        state_h: List[Tensor],
        state_c: List[Tensor],
    ):
        self.score = score
        self.yseq = yseq
        self.str_yseq = str_yseq
        self.state_h = state_h
        self.state_c = state_c


class EncoderArch(Enum):
    Transformer = "transformer"
    Conformer = "conformer"


class DecoderArch(Enum):
    LSTM = "lstm"


@dataclass
class TransducerConfig(LiteasrDataclass):
    name: Optional[str] = field(default="transducer")

    # transducer
    joint_dim: int = field(default=768)
    dropout_rate: float = field(default=0.0)

    # encoder
    enc_arch: EncoderArch = field(default=EncoderArch.Transformer)
    use_rel: bool = field(default=True)
    input_dim: int = field(default=MISSING)
    enc_dim: int = field(default=256)
    enc_ff_dim: int = field(default=2048)
    enc_attn_heads: int = field(default=4)
    enc_dropout_rate: float = II("model.dropout_rate")
    enc_pos_dropout_rate: float = II("model.enc_dropout_rate")
    enc_attn_dropout_rate: float = II("model.enc_dropout_rate")
    enc_ff_dropout_rate: float = II("model.enc_dropout_rate")
    enc_layers: int = field(default=4)
    activation: str = field(default="relu")

    # decoder
    dec_arch: DecoderArch = field(default=DecoderArch.LSTM)
    vocab_size: int = field(default=MISSING)
    dec_dim: int = field(default=256)
    dec_units: int = field(default=2048)
    dec_dropout_rate: float = II("model.dropout_rate")
    dec_layers: int = field(default=2)


@register_model("transducer", dataclass=TransducerConfig)
class Transducer(LiteasrModel):
    def __init__(self, cfg: TransducerConfig, task=None):
        super().__init__()

        assert cfg.enc_arch in EncoderArch
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
            arch=cfg.enc_arch.value,
        )

        assert cfg.dec_arch in DecoderArch
        self.decoder = RNNDecoder(
            i_dim=cfg.vocab_size,
            h_dim=cfg.dec_dim,
            h_units=cfg.dec_units,
            n_layer=cfg.dec_layers,
            dropout_rate=cfg.dec_dropout_rate,
        )

        self.lin_enc = nn.Linear(cfg.enc_dim, cfg.joint_dim)
        self.lin_dec = nn.Linear(cfg.dec_units, cfg.joint_dim, bias=False)
        self.lin_jnt = nn.Linear(cfg.joint_dim, cfg.vocab_size)
        self.joint_activation = nn.Tanh()
        self.ignore = -1

        self._init_module()

    def forward(self, xs, xlens, ys, ylens):
        """Forward function of Transducer.

        :param Tensor xs: Padded audio input with shape (`Batch`, `Tmax`, `*`)
        :param Tensor xlens: Time duration of x (`Batch`)
        :param Tensor ys: Padded token indices with shape (`Batch`, `Lmax`)
        :param Tensor ylens: Length of transcripted text of y (`Batch`)
        :return: Joint tensor (`Batch`, `Fmax`, `Lmax` + 1, `Vocab`)
        :rtype: Tensor

        """

        xs_in, ys_in, xs_mask, ys_mask = self._preprocess(xs, xlens, ys, ylens)
        h_enc = self.encoder(xs_in, mask=xs_mask).unsqueeze(2)
        h_dec = self.decoder(ys_in).unsqueeze(1)
        h_jnt = self.joint(h_enc, h_dec)
        return h_jnt

    def inference(self, x):
        # implement beam search
        h = self.encoder(x)  # (1, time, 83) -> (1, frame, 256)
        state_h, state_c = self.decoder.init_state(h)
        blank_tensor = torch.zeros(1, dtype=torch.long, device=h.device)
        cache = {}
        init_hyp = Hypothesis(0.0, [0], "0", state_h, state_c)
        kept_hyps: List[Hypothesis] = [init_hyp]
        hyps: List[Hypothesis] = []

        for i, hi in enumerate(h[0]):
            hyps = kept_hyps
            kept_hyps = []

            while True:
                hyp_max = max(hyps, key=lambda hyp: hyp.score)
                hyps.remove(hyp_max)

                if hyp_max.str_yseq in cache:
                    y, state_h, state_c = cache[hyp_max.str_yseq]
                else:
                    token = torch.full(
                        (1, 1), hyp_max.yseq[-1], dtype=torch.long, device=h.device
                    )
                    token_embed = self.decoder.embed(token)
                    y, state_h, state_c = self.decoder.rnn_forward(
                        token_embed[:, 0, :], hyp_max.state_h, hyp_max.state_c
                    )
                    cache[hyp_max.str_yseq] = (y, state_h, state_c)

                ytu = F.log_softmax(self.joint(hi, y[0]), dim=-1)
                top_k = ytu[1:].topk(10, dim=-1)
                ytu = (
                    torch.cat((top_k[0], ytu[0:1])),
                    torch.cat((top_k[1] + 1, blank_tensor)),
                )

                for logp, k in zip(*ytu):
                    new_hyp = Hypothesis(
                        score=hyp_max.score + float(logp),
                        yseq=hyp_max.yseq[:],
                        str_yseq=hyp_max.str_yseq,
                        state_h=hyp_max.state_h,
                        state_c=hyp_max.state_c,
                    )

                    if k == 0:
                        kept_hyps.append(new_hyp)
                    else:
                        new_hyp.state_h = state_h
                        new_hyp.state_c = state_c
                        new_hyp.yseq.append(int(k))
                        new_hyp.str_yseq = hyp_max.str_yseq + "_" + str(k)
                        hyps.append(new_hyp)

                # max_hyp_score = float(max(hyps, key=lambda x: x.score).score)
                # kept_most_prob = sorted(
                #     [hyp for hyp in kept_hyps if hyp.score > max_hyp_score],
                #     key=lambda x: x.score,
                # )
                # if len(kept_most_prob) >= 10:
                #     kept_hyps = kept_most_prob
                #     break
                if len(kept_hyps) >= 10:
                    break

        best_hyp = sorted(kept_hyps, key=lambda x: x.score / len(x.yseq), reverse=True)[
            0
        ]
        return best_hyp.yseq

    def get_pred_len(self, xlens):
        pred_len = ((xlens - 1) // 2 - 1) // 2
        return pred_len

    def get_target(self, ys, ylens):
        target = ys
        return target

    def get_target_len(self, ylens):
        target_len = ylens
        return target_len

    def joint(self, h_enc, h_dec):
        h_enc = self.lin_enc(h_enc)
        h_dec = self.lin_dec(h_dec)
        h_jnt = self.lin_jnt(self.joint_activation(h_enc + h_dec))
        return h_jnt

    def _preprocess(self, xs, xlens, ys, ylens):
        # xs_in
        xs_in = xs

        # xs_mask
        xs_mask = padding_mask(xlens)

        # ys_in
        ys_ = ys.masked_fill(ys == self.ignore, 0)
        blank = torch.zeros(ys.size(0), 1).to(dtype=ys.dtype, device=ys.device)
        ys_in = torch.cat([blank, ys_], dim=1)

        # ys_mask
        ys_mask = padding_mask(ylens + 1)

        return xs_in, ys_in, xs_mask, ys_mask

    def _init_module(self):
        # TODO: support dynamic init afterwards
        lecun_normal_init_parameters(self.decoder)
        lecun_normal_init_parameters(self.lin_enc)
        lecun_normal_init_parameters(self.lin_dec)
        lecun_normal_init_parameters(self.lin_jnt)
        self.decoder.embed.weight.data.normal_(0, 1)
        for i in range(len(self.decoder.dec_layers)):
            set_forget_bias_to_one(self.decoder.dec_layers[i].bias_ih)

    @classmethod
    def build_model(cls, cfg, task=None):
        cfg.input_dim = task.feat_dim
        cfg.vocab_size = task.vocab_size
        return cls(cfg, task)
