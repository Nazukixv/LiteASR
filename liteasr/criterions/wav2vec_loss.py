"""Wav2vec 2.0 loss."""

from dataclasses import dataclass
from dataclasses import field

import torch.nn as nn

from liteasr.config import LiteasrDataclass
from liteasr.criterions import LiteasrLoss
from liteasr.criterions import register_criterion
from liteasr.models import LiteasrModel


@dataclass
class Wav2Vec2LossConfig(LiteasrDataclass):
    infonce: bool = field(default=False)


@register_criterion("wav2vec", dataclass=Wav2Vec2LossConfig)
class Wav2Vec2Loss(LiteasrLoss):

    def __init__(self, cfg: Wav2Vec2LossConfig, task=None):
        super().__init__(cfg)
        self._loss_contrastive = nn.CrossEntropyLoss()

    @classmethod
    def build_criterion(cls, cfg, task):
        return cls(cfg, task)

    def __call__(self, model: LiteasrModel, xs, xlens, ys, ylens):
        logits = model(xs)
        target = model.get_target(logits, xlens)

        contrastive_loss = self._loss_contrastive(logits, target)

        return contrastive_loss
