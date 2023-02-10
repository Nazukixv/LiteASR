"""Paraformer loss."""

from dataclasses import dataclass
from dataclasses import field
import logging
from typing import Optional

from omegaconf import MISSING
import torch.nn as nn

from liteasr.config import LiteasrDataclass
from liteasr.criterions import LiteasrLoss
from liteasr.criterions import register_criterion
from liteasr.models import LiteasrModel

logger = logging.getLogger(__name__)


@dataclass
class ParaformerLossConfig(LiteasrDataclass):
    name: Optional[str] = field(default="paraformer_loss")
    vocab_size: int = field(default=MISSING)
    gamma: float = field(default=1.0)


@register_criterion("paraformer_loss", dataclass=ParaformerLossConfig)
class ParaformerLoss(LiteasrLoss):

    def __init__(self, cfg: ParaformerLossConfig, task=None):
        super().__init__(cfg)
        self._loss_ce = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
        self._loss_mae = nn.L1Loss(reduction="mean")

    @classmethod
    def build_criterion(cls, cfg, task):
        cfg.vocab_size = task.vocab_size
        return cls(cfg, task)

    def __call__(self, model: LiteasrModel, xs, xlens, ys, ylens):
        # get predict hidden map
        hs_attn, sum_alpha = model(xs, xlens, ys, ylens)

        # get target
        tgt_attn = model.get_target(ys, ylens)

        # cross entropy loss
        tgt_attn = tgt_attn.view(-1)
        hs_attn = hs_attn.view(-1, self.cfg.vocab_size)
        loss_ce = self._loss_ce(hs_attn, tgt_attn)

        # mae loss
        loss_mae = self._loss_mae(sum_alpha, ylens)

        # hybrid loss
        loss = self.cfg.gamma * loss_ce + loss_mae

        return loss
