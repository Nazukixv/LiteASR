"""Label smoothing loss."""

from dataclasses import dataclass
from dataclasses import field

from omegaconf import MISSING
import torch
import torch.nn as nn

from liteasr.config import LiteasrDataclass
from liteasr.criterions import LiteasrLoss
from liteasr.criterions import register_criterion
from liteasr.models import LiteasrModel


@dataclass
class HybridCTCLossConfig(LiteasrDataclass):
    vocab_size: int = field(default=MISSING)
    padding_idx: int = field(default=-1)
    smoothing: float = field(default=0.0)
    normalize_length: bool = field(default=False)
    ctc_weight: float = field(default=0.0)


@register_criterion("hybrid_ctc", dataclass=HybridCTCLossConfig)
class HybridCTCLoss(LiteasrLoss):

    def __init__(self, cfg: HybridCTCLossConfig, task=None):
        super().__init__(cfg)
        self._loss_attn = nn.KLDivLoss(reduction="none")
        self._loss_ctc = nn.CTCLoss(reduction="sum")

    @classmethod
    def build_criterion(cls, cfg, task):
        cfg.vocab_size = task.vocab_size
        return cls(cfg, task)

    def __call__(self, model: LiteasrModel, xs, xlens, ys, ylens):
        # get predict hidden map
        h_attn, h_ctc = model(xs, xlens, ys, ylens)
        self._check_pred(h_attn, h_ctc)

        # get target
        tgt_attn, tgt_ctc = model.get_target(ys, ylens)
        self._check_target(tgt_attn, tgt_ctc)

        # attention loss
        tgt_attn = tgt_attn.view(-1)
        ign_attn = tgt_attn == self.cfg.padding_idx
        tgt_attn = tgt_attn.masked_fill(ign_attn, 0)

        h_attn = h_attn.view(-1, self.cfg.vocab_size)
        true_dist = torch.zeros_like(h_attn).fill_(
            self.cfg.smoothing / (self.cfg.vocab_size - 1)
        )
        true_dist.scatter_(
            dim=1,
            index=tgt_attn.unsqueeze(1),
            value=1.0 - self.cfg.smoothing,
        )
        kl = self._loss_attn(torch.log_softmax(h_attn, dim=1), true_dist)
        loss_attn = kl.masked_fill(ign_attn.unsqueeze(1), 0).sum()
        loss_attn = loss_attn / ys.size(0)

        # ctc loss
        h_ctc = h_ctc.transpose(0, 1)
        h_ctc = h_ctc.log_softmax(-1)
        loss_ctc = self._loss_ctc(
            h_ctc,
            tgt_ctc,
            model.get_pred_len(xlens),
            torch.tensor(ylens),
        )
        loss_ctc = loss_ctc / ys.size(0)

        # hybrid loss
        loss = (
            self.cfg.ctc_weight * loss_ctc +
            (1 - self.cfg.ctc_weight) * loss_attn
        )
        return loss

    def _check_pred(self, h_attn, h_ctc):
        # TODO
        pass

    def _check_target(self, tgt_attn, tgt_ctc):
        # TODO
        pass
