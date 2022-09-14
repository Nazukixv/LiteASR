from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import torch

from liteasr.config import LiteasrDataclass
from liteasr.optims import LiteasrOptimizer
from liteasr.optims import register_optimzer


@dataclass
class AdamConfig(LiteasrDataclass):
    name: Optional[str] = field(default="adam")
    lr: float = field(default=1e-3)
    beta1: float = field(default=0.9)
    beta2: float = field(default=0.999)
    eps: float = field(default=1e-8)
    weight_decay: float = field(default=0.0)
    amsgrad: bool = field(default=False)


@register_optimzer("adam", dataclass=AdamConfig)
class Adam(LiteasrOptimizer):

    def __init__(self, params, cfg: AdamConfig, task=None):
        super().__init__(cfg)
        self._optimizer = torch.optim.Adam(
            params,
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            amsgrad=cfg.amsgrad,
        )

    @classmethod
    def build_optimizer(cls, params, cfg, task=None):
        return cls(params, cfg, task)
