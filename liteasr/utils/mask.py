"""Functions of mask generation."""

from typing import List

import numpy as np
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


def span_mask(
    batch: int,
    frame: int,
    prob: float,
    length: int,
    policy: str = "static",
    no_overlap: bool = False,
    min_mask_num: int = 0,
    min_interval: int = 0,
) -> Tensor:
    """Returns random span mask.

    :param int batch: batch size of source
    :param int frame: timesteps of source
    :param float prob: Probability for each token to be chosen as start of the
        span to be masked. This will be multiplied by number of timesteps
        divided by length of mask span to mask approximately this percentage
        of all elements. However due to overlaps, the actual number will be
        smaller (unless no_overlap is True)
    :param int length: basic mask span length
    :param policy: How to compute mask lengths, defaults to "static"
        static = fixed size
        uniform = sample from uniform distribution [mask_other, mask_length*2]
        normal = sample from normal distribution with mean mask_length and
            stdev mask_other. mask is min 1 element
        poisson = sample from possion distribution with lambda = mask length
    :type policy: str, optional
    :param no_overlap: Whether the masked spans can overlap, defaults to True
    :type no_overlap: bool, optional
    :param min_mask_num: Minimum number of masked spans, defaults to 0
    :type min_mask_num: int, optional
    :param min_interval: How many elements to keep unmasked between spans
        only used if no_overlap is True, defaults to 0
    :type min_interval: int, optional
    :return: Span mask
    :rtype: Tensor

    :Example (ignoring zeros):

    >>> span_mask(batch=3, frame=20, prob=0.6, length=3)
    [[1, 1, 1, 1, 1, 1, 1, 1,  ,  ,  ,  ,  ,  ,  ,  , 1, 1, 1,  ],
     [ , 1, 1, 1,  , 1, 1, 1,  , 1, 1, 1, 1, 1,  ,  ,  ,  ,  ,  ],
     [ ,  , 1, 1, 1, 1, 1, 1, 1, 1,  ,  ,  , 1, 1, 1,  ,  ,  ,  ]]

    >>> span_mask(batch=3, frame=20, prob=0.6, length=3, no_overlap=True)
    [[1, 1, 1,  , 1, 1, 1,  ,  ,  ,  ,  , 1, 1, 1,  , 1, 1, 1,  ],
     [ , 1, 1, 1,  , 1, 1, 1,  ,  , 1, 1, 1,  ,  , 1, 1, 1,  ,  ],
     [ , 1, 1, 1, 1, 1, 1,  ,  , 1, 1, 1,  , 1, 1, 1,  ,  ,  ,  ]]

    """

    mask = torch.zeros((batch, frame), dtype=torch.bool)

    mask_num = int(prob * frame / float(length) + np.random.rand())
    mask_num = max(min_mask_num, mask_num)

    mask_idcs = []
    for i in range(batch):

        # policy selection
        if policy == "static":
            spans = np.full(mask_num, length)
        elif policy == "uniform":
            spans = np.random.randint(0.0, length * 2 + 1, size=mask_num)
        elif policy == "normal":
            spans = np.random.normal(length, 0.0, size=mask_num)
            spans = [max(1, int(round(x))) for x in spans]
        elif policy == "poisson":
            spans = np.random.poisson(length, size=mask_num)
            spans = [int(round(x)) for x in spans]
        else:
            raise Exception("unknown mask selection " + policy)

        if sum(spans) == 0:
            spans[0] = min(length, frame - 1)

        # overlap process
        if no_overlap:
            mask_idc = []

            def new_se_pairs(stt, end, length, keep_length):
                # record mask index [span_stt ~ span_stt + length)
                span_stt = np.random.randint(stt, end - length)
                mask_idc.extend(span_stt + i for i in range(length))

                # push new (stt, end) pair into stack
                new_se = []
                if stt + keep_length + min_interval <= span_stt:
                    s = stt
                    e = span_stt - min_interval + 1
                    new_se.append((s, e))
                if span_stt + length + min_interval + keep_length < end:
                    s = span_stt + length + min_interval
                    e = end
                    new_se.append((s, e))
                return new_se

            se_pairs = [(0, frame)]
            min_span_size = min(spans)
            for size in sorted(spans, reverse=True):
                se_lens = np.fromiter(
                    (
                        e - s if e - s >= size + min_interval else 0
                        for s, e in se_pairs
                    ), int
                )
                l_sum = np.sum(se_lens)
                if l_sum == 0:
                    break
                probs = se_lens / l_sum  # uniform distribution
                which_se = np.random.choice(len(se_pairs), p=probs)
                s, e = se_pairs.pop(which_se)
                se_pairs.extend(new_se_pairs(s, e, size, min_span_size))

            mask_idc = np.asarray(mask_idc)
        else:
            min_span_size = min(spans)
            if frame - min_span_size <= mask_num:
                min_span_size = frame - mask_num - 1

            mask_idc = np.random.choice(
                frame - min_span_size, mask_num, replace=False
            )

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(spans[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < frame]))

    # make total mask length of each sample equal
    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask
