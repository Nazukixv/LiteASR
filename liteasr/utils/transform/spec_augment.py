"""SpecAugment."""

import random

import numpy
from PIL import Image
from PIL.Image import BICUBIC
import torch

from liteasr.config import _SpecAugmentConfig
from liteasr.utils.transform import register_transformation


@register_transformation("spec_aug")
class SpecAugment(object):
    def __init__(self, cfg: _SpecAugmentConfig):
        self.cfg = cfg

    def time_warp(self, x):
        """time warp for spec augment

        move random center frame by the random width ~ uniform(-window, window)
        :param numpy.ndarray x: spectrogram (time, freq)
        :param bool inplace: overwrite x with the result
        :param str mode: "PIL" (default, fast, not differentiable)
            or "sparse_image_warp" (slow, differentiable)
        :returns numpy.ndarray: time warped spectrogram (time, freq)
        """

        window = self.cfg.time_warp
        t = x.shape[0]
        if t - window <= window:
            return x
        # NOTE: randrange(a, b) emits a, a + 1, ..., b - 1
        center = random.randrange(window, t - window)
        warped = random.randrange(center - window, center + window) + 1  # 1 ... t - 1

        left = Image.fromarray(x[:center]).resize(
            (x.shape[1], warped),
            BICUBIC,
        )
        right = Image.fromarray(x[center:]).resize(
            (x.shape[1], t - warped),
            BICUBIC,
        )
        if self.cfg.inplace:
            x[:warped] = left
            x[warped:] = right
            return x
        return numpy.concatenate((left, right), 0)

    def freq_mask(self, x):
        """freq mask for spec agument

        :param numpy.ndarray x: (time, freq)
        :param bool inplace: overwrite
        :param bool replace_with_zero: pad zero on mask if true else use mean
        """

        if self.cfg.inplace:
            cloned = x
        else:
            cloned = x.copy()

        num_mel_channels = cloned.shape[1]
        fs = numpy.random.randint(
            0, self.cfg.freq_mask, size=(self.cfg.freq_mask_times, 2)
        )

        for f, mask_end in fs:
            f_zero = random.randrange(0, num_mel_channels - f)
            mask_end += f_zero

            # avoids randrange error if values are equal and range is empty
            if f_zero == f_zero + f:
                continue

            if self.cfg.replace_with_zero:
                cloned[:, f_zero:mask_end] = 0
            else:
                cloned[:, f_zero:mask_end] = cloned.mean()
        return cloned

    def time_mask(self, x):
        """freq mask for spec agument

        :param numpy.ndarray x: (time, freq)
        :param bool inplace: overwrite
        :param bool replace_with_zero: pad zero on mask if true else use mean
        """
        if self.cfg.inplace:
            cloned = x
        else:
            cloned = x.copy()
        len_spectro = cloned.shape[0]
        ts = numpy.random.randint(
            0, self.cfg.time_mask, size=(self.cfg.time_mask_times, 2)
        )
        for t, mask_end in ts:
            # avoid randint range error
            if len_spectro - t <= 0:
                continue
            t_zero = random.randrange(0, len_spectro - t)

            # avoids randrange error if values are equal and range is empty
            if t_zero == t_zero + t:
                continue

            mask_end += t_zero
            if self.cfg.replace_with_zero:
                cloned[t_zero:mask_end] = 0
            else:
                cloned[t_zero:mask_end] = cloned.mean()
        return cloned

    def __call__(self, x):
        x = x.numpy()
        assert isinstance(x, numpy.ndarray)
        assert x.ndim == 2

        x = self.time_warp(x)
        x = self.freq_mask(x)
        x = self.time_mask(x)

        return torch.from_numpy(x)
