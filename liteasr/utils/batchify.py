"""Batchfied dataset."""

import logging

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from liteasr.config import DatasetConfig
from liteasr.config import PostProcessConfig
from liteasr.utils.transform.spec_augment import SpecAugment

logger = logging.getLogger(__name__)


class BatchfiedDataset(Dataset):

    def __init__(
        self,
        samples,
        split: str,
        dataset_cfg: DatasetConfig,
        postprocess_cfg: PostProcessConfig,
    ):
        self.data = []
        self.minibatch = []
        self.split = split
        self.dataset_cfg = dataset_cfg
        self.postprocess_cfg = postprocess_cfg

        self.spec_aug = SpecAugment(
            resize_mode="PIL",
            max_time_warp=postprocess_cfg.spec_aug.time_warp,
            max_freq_width=postprocess_cfg.spec_aug.freq_mask,
            n_freq_mask=postprocess_cfg.spec_aug.freq_mask_times,
            max_time_width=postprocess_cfg.spec_aug.time_mask,
            n_time_mask=postprocess_cfg.spec_aug.time_mask_times,
            inplace=True,
            replace_with_zero=False,
        )

    @property
    def empty(self) -> bool:
        """Determine whether the current minibatch is empty."""

        return len(self.minibatch) == 0

    @property
    def full(self) -> bool:
        """Determine whether the current minibatch is full."""

        raise NotImplementedError

    def push(self, sample):
        """Push one sample into minibatch."""

        raise NotImplementedError

    def pop(self):
        """Pop the minibatch into data, then empty the minibatch."""

        self.data.append(self.minibatch)
        self.minibatch = []

    def refresh(self):
        """Refresh the state of minibatch."""

        raise NotImplementedError

    def batchify(self, samples):
        self.refresh()
        for sample in samples:
            self.sample = sample
            if self.full:
                self.pop()
                self.refresh()
            self.push(sample)

        if not self.empty:
            self.pop()
            self.refresh()

    def __getitem__(self, index):
        audios = self.data[index]
        xs, xlens, ys, ylens = [], [], [], []
        for audio in audios:
            xs.append(
                torch.from_numpy(
                    self.spec_aug(audio.x.numpy(), train=self.split == "train")
                )
            )
            xlens.append(audio.xlen)
            ys.append(audio.y)
            ylens.append(audio.ylen)
        padded_xs = pad_sequence(xs, batch_first=True, padding_value=0)
        padded_ys = pad_sequence(ys, batch_first=True, padding_value=-1)
        return padded_xs, xlens, padded_ys, ylens

    def __len__(self):
        return len(self.data)


class SeqDataset(BatchfiedDataset):

    def __init__(
        self,
        samples,
        split: str,
        dataset_cfg: DatasetConfig,
        postprocess_cfg: PostProcessConfig,
    ):
        super().__init__(samples, split, dataset_cfg, postprocess_cfg)

        samples = sorted(samples, key=lambda a: a.xlen, reverse=True)
        self.batchify(samples)

    @property
    def full(self):
        return len(self.minibatch) == self.dynamic_batch_size

    def push(self, sample):
        if self.empty:
            self.minibatch.append(sample)
            self.refresh()
        else:
            self.minibatch.append(sample)

    def refresh(self):
        if self.empty:
            self.factor = 0
            self.dynamic_batch_size = self.dataset_cfg.batch_size
        else:
            ilen = self.minibatch[0].xlen
            olen = self.minibatch[0].ylen
            self.factor = max(
                int(ilen / self.dataset_cfg.max_len_in),
                int(olen / self.dataset_cfg.max_len_out),
            )
            self.dynamic_batch_size = max(
                self.dataset_cfg.min_batch_size,
                int(self.dataset_cfg.batch_size / (1 + self.factor)),
            )
