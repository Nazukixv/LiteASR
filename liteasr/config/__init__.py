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
class DatasetConfig(LiteasrDataclass):
    batch_size: Optional[int] = field(default=None)
    min_batch_size: Optional[int] = field(default=None)
    max_len_in: Optional[int] = field(default=None)
    max_len_out: Optional[int] = field(default=None)


@dataclass
class DistributedConfig(LiteasrDataclass):
    world_size: int = field(default=max(1, torch.cuda.device_count()))
    rank: int = field(default=0)
    backend: str = field(default="NCCL")
    init_method: Optional[str] = field(default=None)
    device_id: int = field(default=0)


@dataclass
class OptimizationConfig(LiteasrDataclass):
    max_epoch: int = field(default=-1)
    max_iter: int = field(default=-1)
    accum_grad: int = field(default=1)
    clip_grad_norm: float = field(default=0.0)


@dataclass
class LiteasrConfig(LiteasrDataclass):
    common: CommonConfig = CommonConfig()
    dataset: DatasetConfig = DatasetConfig()
    distributed: DistributedConfig = DistributedConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    task: Any = None
    model: Any = None
    criterion: Any = None
    optimizer: Any = None
