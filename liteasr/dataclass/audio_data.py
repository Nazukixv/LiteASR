from dataclasses import dataclass
from typing import Optional, Tuple

import soundfile as sf
import torch

from liteasr.utils.kaldiio import load_mat


@dataclass
class Audio(object):
    __slots__ = [
        "fd",
        "start",
        "shape",
        "tokenids",
        "text",
    ]

    fd: str
    start: Optional[int]
    shape: int
    tokenids: Optional[Tuple[int]]
    text: Optional[str]

    @property
    def x(self):
        if self.start is None:  # feature map
            x = torch.from_numpy(load_mat(self.fd))
        else:  # pcm samples
            samples, _ = sf.read(self.fd)
            x = torch.from_numpy(samples).float()
            x = x[self.start : self.start + self.xlen]
        return x

    @property
    def xlen(self):
        return self.shape

    @property
    def y(self):
        y = torch.tensor(self.tokenids) if self.tokenids is not None else None
        return y

    @property
    def ylen(self):
        ylen = len(self.tokenids) if self.tokenids is not None else 0
        return ylen
