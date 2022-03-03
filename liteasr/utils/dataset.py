"""Lite-ASR dataset implementation for training."""

import logging

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from liteasr.config import DatasetConfig
from liteasr.dataclass.audio_data import Audio
from liteasr.dataclass.sheet import AudioSheet
from liteasr.dataclass.sheet import TextSheet

logger = logging.getLogger(__name__)


class SortedDataset(Dataset):

    def __init__(self, samples, cfg: DatasetConfig):
        self.data = []
        self.cfg = cfg
        self.batchify(samples)

    def batchify(self, samples):
        samples = sorted(samples, key=lambda a: a.xlen, reverse=True)
        start = 0
        while True:
            ilen = samples[start].xlen
            olen = samples[start].ylen
            factor = max(
                int(ilen / self.cfg.max_len_in),
                int(olen / self.cfg.max_len_out),
            )
            bs = max(
                self.cfg.min_batch_size,
                int(self.cfg.batch_size / (1 + factor)),
            )
            end = min(len(samples), start + bs)

            self.data.append(samples[start:end])

            if end == len(samples):
                break

            start = end

    def __getitem__(self, index):
        audios = self.data[index]
        xs, xlens, ys, ylens = [], [], [], []
        for audio in audios:
            xs.append(audio.x)
            xlens.append(audio.xlen)
            ys.append(audio.y)
            ylens.append(audio.ylen)
        padded_xs = pad_sequence(xs, batch_first=True, padding_value=0)
        padded_ys = pad_sequence(ys, batch_first=True, padding_value=-1)
        return padded_xs, xlens, padded_ys, ylens

    def __len__(self):
        return len(self.data)


class AudioFileDataset(Dataset):

    def __init__(
        self,
        scp: str,
        segments: str,
        text: str,
        vocab,
    ):
        self.data = []
        _as = AudioSheet(scp=scp, segments=segments)
        _ts = TextSheet(text=text, vocab=vocab)
        for audio_info, text_info in zip(_as, _ts):
            uttid, fd, start, end, shape = audio_info
            uttid_t, tokenids = text_info
            assert uttid_t == uttid
            self.data.append(
                Audio(uttid, fd, start, end, shape, tokenids=tokenids)
            )
            if len(self.data) % 10000 == 0:
                logger.info("number of loaded data: {}".format(len(self.data)))
        logger.info("number of loaded data: {}".format(len(self.data)))
        self.feat_dim = self.data[0].shape[-1]

    def batchify(self, cfg: DatasetConfig) -> Dataset:
        return SortedDataset(self.data, cfg)

    def __getitem__(self, index):
        """overload [] operator"""
        return self.data[index]

    def __len__(self):
        """overload len() method"""
        return len(self.data)
