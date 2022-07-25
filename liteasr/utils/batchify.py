"""Batchfied dataset."""

import logging

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from liteasr.config import DatasetConfig
from liteasr.config import PostProcessConfig
from liteasr.utils.transform import PostProcess

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
        self.postprocess = PostProcess(postprocess_cfg)

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
        not_train = self.split != "train"
        for audio in audios:
            xs.append(audio.x if not_train else self.postprocess(audio.x))
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


class FrameDataset(BatchfiedDataset):

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
        max_ilen = max(self.max_ilen, self.sample.xlen)
        max_olen = max(self.max_olen, self.sample.ylen)
        exp_size = len(self.minibatch) + 1

        # in full
        if max_ilen * exp_size > self.dataset_cfg.max_frame_in:
            return True
        # out full
        elif max_olen * exp_size > self.dataset_cfg.max_frame_out:
            return True
        # inout full
        elif (
            max_ilen + max_olen
        ) * exp_size > self.dataset_cfg.max_frame_inout:
            return True
        else:
            return False

    def push(self, sample):
        self.minibatch.append(sample)
        self.refresh()

    def refresh(self):
        if self.empty:
            self.max_ilen = 0
            self.max_olen = 0
        else:
            self.max_ilen = max(self.max_ilen, self.sample.xlen)
            self.max_olen = max(self.max_olen, self.sample.ylen)
