"""Initialize sub package."""

from dataclasses import dataclass
from dataclasses import field
from typing import Any, List, Optional

from omegaconf import II
from omegaconf import MISSING
import torch


@dataclass
class LiteasrDataclass(object):
    name: Optional[str] = None


@dataclass
class _TriggerConfig(LiteasrDataclass):
    interval: int = field(default=1)
    unit: str = field(default="epoch")


@dataclass
class CommonConfig(LiteasrDataclass):
    seed: int = field(default=1)
    trigger: List[_TriggerConfig] = field(default_factory=lambda: [])


@dataclass
class DatasetConfig(LiteasrDataclass):
    batch_count: str = field(default="seq")
    batch_size: Optional[int] = field(default=None)
    min_batch_size: Optional[int] = field(default=None)
    max_len_in: Optional[int] = field(default=None)
    max_len_out: Optional[int] = field(default=None)
    max_frame_in: Optional[int] = field(default=None)
    max_frame_out: Optional[int] = field(default=None)
    max_frame_inout: Optional[int] = field(default=None)


@dataclass
class _SpecAugmentConfig(object):
    time_warp: int = field(default=80)
    freq_mask: int = field(default=27)
    freq_mask_times: int = field(default=1)
    time_mask: int = field(default=100)
    time_mask_times: int = field(default=1)
    inplace: bool = field(default=True)
    replace_with_zero: bool = field(default=False)


@dataclass
class PostProcessConfig(LiteasrDataclass):
    spec_aug: _SpecAugmentConfig = _SpecAugmentConfig()
    workflow: List[str] = field(default_factory=lambda: ["spec_aug"])


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
class InferenceConfig(LiteasrDataclass):
    ckpt_path: str = II("task.save_dir")
    ckpt_name: Optional[str] = field(default=MISSING)
    model_avg: bool = field(default=False)
    avg_num: int = field(default=1)
    thread_num: int = field(default=32)


@dataclass
class LiteasrConfig(LiteasrDataclass):
    common: CommonConfig = CommonConfig()
    dataset: DatasetConfig = DatasetConfig()
    postprocess: PostProcessConfig = PostProcessConfig()
    distributed: DistributedConfig = DistributedConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    inference: InferenceConfig = InferenceConfig()
    task: Any = None
    model: Any = None
    criterion: Any = None
    optimizer: Any = None
