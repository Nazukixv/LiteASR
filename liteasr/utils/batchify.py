"""Batchfied dataset."""

import logging

from liteasr.config import DatasetConfig
from liteasr.dataclass.audio_data import Audio

logger = logging.getLogger(__name__)


class BatchifyPolicy(object):
    sample: Audio

    def __init__(self, dataset_cfg: DatasetConfig):
        super().__init__()
        self._num = 0
        self.data = []
        self.minibatch = []
        self.dataset_cfg = dataset_cfg

    @property
    def empty(self) -> bool:
        """Determine whether the current minibatch is empty."""

        return len(self.minibatch) == 0

    @property
    def full(self) -> bool:
        """Determine whether the current minibatch is full."""

        raise NotImplementedError

    def push(self, idx):
        """Push one sample into minibatch."""

        raise NotImplementedError

    def pop(self):
        """Pop the minibatch into data, then empty the minibatch."""

        self.data.append(self.minibatch)
        self._num += len(self.minibatch)
        self.minibatch = []

        if self._num / self._all >= self._threshold / 20:
            logger.info(
                "|{:<20}| {:>4.0%} ({}/{})".format(
                    "#" * self._threshold,
                    self._threshold / 20,
                    self._num,
                    self._all,
                )
            )
            self._threshold += 1

    def refresh(self):
        """Refresh the state of minibatch."""

        raise NotImplementedError

    def batchify(self, indices, samples):
        assert len(indices) == len(samples), f"{len(samples)}"
        self.refresh()
        self._all = len(indices)
        self._threshold = 1

        logger.info("percentage of bachified data:")
        for idx in indices:
            self.sample = samples[idx]
            if self.full:
                self.pop()
                self.refresh()
            self.push(idx)

        if not self.empty:
            self.pop()
            self.refresh()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SeqBatch(BatchifyPolicy):
    factor: int
    dynamic_batch_size: int
    max_ilen: int
    max_olen: int

    def __init__(self, dataset_cfg: DatasetConfig):
        super().__init__(dataset_cfg)

    @property
    def full(self):
        return len(self.minibatch) == self.dynamic_batch_size

    def push(self, idx):
        if self.empty:
            self.minibatch.append(idx)
            self.refresh()
        else:
            self.minibatch.append(idx)

    def refresh(self):
        if self.empty:
            self.factor = 0
            self.dynamic_batch_size = self.dataset_cfg.batch_size
            self.max_ilen = 0
            self.max_olen = 0
        else:
            self.max_ilen = self.sample.xlen
            self.max_olen = self.sample.ylen
            self.factor = max(
                int(self.max_ilen / self.dataset_cfg.max_len_in),
                int(self.max_olen / self.dataset_cfg.max_len_out),
            )
            self.dynamic_batch_size = max(
                self.dataset_cfg.min_batch_size,
                int(self.dataset_cfg.batch_size / (1 + self.factor)),
            )


class FrameBatch(BatchifyPolicy):
    max_ilen: int
    max_olen: int

    def __init__(self, dataset_cfg: DatasetConfig):
        super().__init__(dataset_cfg)

    @property
    def full(self):
        max_ilen = max(self.max_ilen, self.sample.xlen)
        max_olen = max(self.max_olen, self.sample.ylen)
        exp_size = len(self.minibatch) + 1

        # in full
        if (
            self.dataset_cfg.max_frame_in
            and max_ilen * exp_size > self.dataset_cfg.max_frame_in
        ):
            return True
        # out full
        elif (
            self.dataset_cfg.max_frame_out
            and max_olen * exp_size > self.dataset_cfg.max_frame_out
        ):
            return True
        # inout full
        elif (
            self.dataset_cfg.max_frame_inout and
            (max_ilen + max_olen) * exp_size > self.dataset_cfg.max_frame_inout
        ):
            return True
        else:
            return False

    def push(self, idx):
        self.minibatch.append(idx)
        self.refresh()

    def refresh(self):
        if self.empty:
            self.max_ilen = 0
            self.max_olen = 0
        else:
            self.max_ilen = max(self.max_ilen, self.sample.xlen)
            self.max_olen = max(self.max_olen, self.sample.ylen)


class Wav2VecBatch(BatchifyPolicy):
    min_frame: int
    max_batch_frame: int = 1400000  # should be configurable

    def __init__(self, dataset_cfg: DatasetConfig):
        super().__init__(dataset_cfg)

    @property
    def full(self) -> bool:
        min_frame = min(self.min_frame, self.sample.xlen)
        return (len(self.minibatch) + 1) * min_frame > self.max_batch_frame

    def push(self, idx: int):
        self.minibatch.append(idx)
        self.refresh()

    def refresh(self):
        if self.empty:
            self.min_frame = 250000  # should be configurable
        else:
            self.min_frame = min(self.min_frame, self.sample.xlen)
