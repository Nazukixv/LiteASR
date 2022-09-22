from dataclasses import dataclass
from typing import List, Optional, Union

import soundfile as sf
import torch

from liteasr.utils.kaldiio import load_mat


@dataclass
class Audio(object):
    uttid: str
    fd: str
    start: Union[int, float]
    end: Union[int, float]
    shape: List[int]
    tokenids: Optional[List[int]] = None
    text: Optional[str] = None

    @property
    def x(self):
        if self.start == -1 or self.end == -1:  # feature map
            x = torch.from_numpy(load_mat(self.fd))
        else:  # pcm samples
            samples, rate = sf.read(self.fd)
            x = torch.from_numpy(samples).float()
            stt = round(self.start * rate)
            x = x[stt:stt + self.shape[0]]
        return x

    @property
    def xlen(self):
        return self.shape[0]

    @property
    def y(self):
        y = torch.tensor(self.tokenids) if self.tokenids is not None else None
        return y

    @property
    def ylen(self):
        ylen = len(self.tokenids) if self.tokenids is not None else 0
        return ylen
