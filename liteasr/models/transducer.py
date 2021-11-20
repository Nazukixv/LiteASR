from dataclasses import dataclass, field

import torch.nn as nn

from liteasr.config import LiteasrDataclass
from liteasr.models import LiteasrModel
from liteasr.models import register_model

from liteasr.nets.attention import MultiHeadAttention
from liteasr.nets.feed_forward import PositionwiseFeedForward
from liteasr.nets.transformer_layer import EncoderLayer


@dataclass
class TransducerConfig(LiteasrDataclass):
    dropout_rate: float = field(default=0.1)
    elayers: int = field(default=4)
    activation: str = field(default="relu")


@register_model("transducer", dataclass=TransducerConfig)
class Transducer(LiteasrModel):

    def __init__(self, cfg: TransducerConfig, task=None):
        super().__init__()
        self.lin = nn.Linear(83, 256)
        self.encoder = nn.ModuleList([
            EncoderLayer(
                size=256,
                self_attn=MultiHeadAttention(4, 256, 0.1),
                feed_forward=PositionwiseFeedForward(256, 1024, 0.1),
                dropout_rate=0.1
            ) for n in range(cfg.elayers)
        ])

    def forward(self, x):
        x = self.lin(x)
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
        return x

    @classmethod
    def build_model(cls, cfg, task=None):
        return cls(cfg, task)
