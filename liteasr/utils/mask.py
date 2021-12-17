"""Functions of mask generation."""

from typing import List

import torch
from torch import Tensor


def padding_mask(size: List[int]) -> Tensor:
    """Returns mask which masks the padding part of tensor.

    :param size: sequence of unmasking lengths
    :type size: List[int]
    :return: mask
    :rtype: Tensor

    :Example (ignoring zeros):

    >>> padding_mask([5, 3, 1])
    [[ ,  ,  ,  ,  ],
     [ ,  ,  , 1, 1],
     [ , 1, 1, 1, 1]]

    """

    base = torch.arange(max(size)).unsqueeze(0)
    lens = torch.tensor(size).unsqueeze(1)
    mask = base >= lens
    return mask


def triangle_mask(
    row: int,
    col: int = 0,
    stage: int = 1,
    diagonal: int = 1,
) -> Tensor:
    """Returns triangle mask.

    :param int row: number of rows of mask
    :param int col: number of columns of mask
    :param int stage: width of each `stage` of triangle mask
    :param int diagonal: diagonal position
    :return: mask
    :rtype: Tensor

    :Example (ignoring zeros):

    >>> triangle_mask(row=5)
    [[ , 1, 1, 1, 1],
     [ ,  , 1, 1, 1],
     [ ,  ,  , 1, 1],
     [ ,  ,  ,  , 1],
     [ ,  ,  ,  ,  ]]

    >>> triangle_mask(row=3, col=5)
    [[ , 1, 1, 1, 1],
     [ ,  , 1, 1, 1],
     [ ,  ,  , 1, 1]]

    >>> triangle_mask(row=3, col=5, diagonal=2)
    [[ ,  , 1, 1, 1],
     [ ,  ,  , 1, 1],
     [ ,  ,  ,  , 1]]

    >>> triangle_mask(row=8, stage=2)
    [[ ,  , 1, 1, 1, 1, 1, 1],
     [ ,  , 1, 1, 1, 1, 1, 1],
     [ ,  ,  ,  , 1, 1, 1, 1],
     [ ,  ,  ,  , 1, 1, 1, 1],
     [ ,  ,  ,  ,  ,  , 1, 1],
     [ ,  ,  ,  ,  ,  , 1, 1],
     [ ,  ,  ,  ,  ,  ,  ,  ],
     [ ,  ,  ,  ,  ,  ,  ,  ]]

    >>> triangle_mask(row=8, stage=2, diagonal=2)
    [[ ,  ,  ,  , 1, 1, 1, 1],
     [ ,  ,  ,  , 1, 1, 1, 1],
     [ ,  ,  ,  ,  ,  , 1, 1],
     [ ,  ,  ,  ,  ,  , 1, 1],
     [ ,  ,  ,  ,  ,  ,  ,  ],
     [ ,  ,  ,  ,  ,  ,  ,  ],
     [ ,  ,  ,  ,  ,  ,  ,  ],
     [ ,  ,  ,  ,  ,  ,  ,  ]]

    """

    col = row if col == 0 else col
    row_idx = torch.arange(row).unsqueeze(1)
    col_idx = torch.arange(col).unsqueeze(0)
    mask = (col_idx // stage) > ((row_idx // stage) + (diagonal - 1))
    return mask
