from dataclasses import dataclass
from dataclasses import field
from typing import Optional

from liteasr.optims import register_optimzer
from liteasr.optims.adam import Adam
from liteasr.optims.adam import AdamConfig


@dataclass
class NoamConfig(AdamConfig):
    name: Optional[str] = field(default="noam")
    beta2: float = field(default=0.98)
    eps: float = field(default=1e-9)
    model_dim: int = field(default=256)
    factor: float = field(default=1.0)
    warmup: int = field(default=25000)


@register_optimzer("noam", dataclass=NoamConfig)
class Noam(Adam):
    def __init__(self, params, cfg: NoamConfig, task=None):
        super().__init__(params, cfg, task)
        self.model_dim = cfg.model_dim
        self.factor = cfg.factor
        self.warmup = cfg.warmup
        self._step = 0
        self._rate = 0

        for param_group in self.param_groups:
            self.add_param_group(param_group)

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self):
        return (
            self.factor
            * self.model_dim ** (-0.5)
            * min(self._step ** (-0.5), self._step * self.warmup ** (-1.5))
        )

    @classmethod
    def build_optimizer(cls, params, cfg, task=None):
        return cls(params, cfg, task)

    def add_param_group(self, param_group):
        cfg = {
            "model_dim": self.model_dim,
            "factor": self.factor,
            "warmup": self.warmup,
        }
        param_group.update(cfg)
