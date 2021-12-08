"""Functions of mask generation."""

from typing import List, Optional

import torch
from torch import Tensor


def padding_mask(size: List[int]) -> Tensor:
    """Returns mask which masks the padding part of tensor.

    :param size: [description]
    :type size: List[int]
    """

    batch = len(size)
    t_max = max(size)
    base = torch.arange(0, t_max).unsqueeze(0).expand(batch, t_max)
    lens = torch.tensor(size).unsqueeze(1).expand(batch, t_max)
    mask = base >= lens
    return mask


def triangle_mask(
    width: int,
    height: Optional[int] = None,
    diagonal: int = 1,
) -> Tensor:
    """Returns triangle mask."""

    if height is None:
        base = torch.ones(width, width).bool()
    else:
        base = torch.ones(width, height).bool()
    return torch.triu(base, diagonal=diagonal)
