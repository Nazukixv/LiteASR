"""Initialize sub package."""

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Optional

import torch


@dataclass
class LiteasrDataclass(object):
    name: Optional[str] = None


@dataclass
class CommonConfig(LiteasrDataclass):
    seed: int = field(default=1)


@dataclass
class DistributedConfig(LiteasrDataclass):
    world_size: int = field(default=max(1, torch.cuda.device_count()))
    rank: int = field(default=0)
    backend: str = field(default="NCCL")
    init_method: Optional[str] = field(default=None)


@dataclass
class LiteasrConfig(LiteasrDataclass):
    common: CommonConfig = CommonConfig()
    distributed: DistributedConfig = DistributedConfig()
    task: Any = None
    model: Any = None
    criterion: Any = None
    optimizer: Any = None
