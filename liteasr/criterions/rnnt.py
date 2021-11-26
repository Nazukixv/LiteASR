"""RNNT loss."""

from dataclasses import dataclass, field

import torch
from torch import cuda

from liteasr.config import LiteasrDataclass
from liteasr.criterions import LiteasrLoss
from liteasr.criterions import register_criterion


@dataclass
class RNNTLossConfig(LiteasrDataclass):
    trans_type: str = field(default="warp-transducer")
    blank_id: int = field(default=0)


@register_criterion("rnnt", dataclass=RNNTLossConfig)
class RNNTLoss(LiteasrLoss):

    def __init__(self, cfg: RNNTLossConfig, task=None):
        super().__init__(cfg)
        if cfg.trans_type == "warp-transducer":
            from warprnnt_pytorch import RNNTLoss
            self._loss = RNNTLoss(blank=cfg.blank_id)
        elif cfg.trans_type == "warp-rnnt":
            if cuda.is_available():
                from warp_rnnt import rnnt_loss
                self._loss = rnnt_loss
        else:
            raise NotImplementedError

        self.trans_type = cfg.trans_type
        self.blank_id = cfg.blank_id

    @classmethod
    def build_criterion(cls, cfg, task):
        return cls(cfg, task)

    def __call__(self, pred_pad, target, pred_len, target_len):
        if self.trans_type == "warp-rnnt":
            log_probs = torch.log_softmax(pred_pad, dim=-1)

            loss = self._loss(
                log_probs,
                target,
                pred_len,
                target_len,
                reduction="mean",
                blank=self.blank_id,
                gather=True,
            )
        else:
            loss = self._loss(pred_pad, target, pred_len, target_len)

        return loss
