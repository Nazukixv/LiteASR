from dataclasses import dataclass
from typing import List, Optional, Union

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

    @property
    def x(self):
        if self.start == -1 or self.end == -1:  # feature map
            x = torch.from_numpy(load_mat(self.fd))
        else:  # pcm samples
            # TODO: pcm -> tensor
            pass
        return x

    @property
    def y(self):
        y = torch.tensor(self.tokenids) if self.tokenids is not None else None
        return y
