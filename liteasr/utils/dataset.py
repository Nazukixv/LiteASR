"""Lite-ASR dataset implementation for training."""

import logging
from typing import List

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from liteasr.config import DatasetConfig
from liteasr.dataclass.audio_data import Audio
from liteasr.dataclass.sheet import AudioSheet
from liteasr.dataclass.sheet import TextSheet

logger = logging.getLogger(__name__)


class AudioFileDataset(Dataset):

    def __init__(
        self,
        scp: str,
        segments: str,
        text: str,
        vocab,
        cfg: DatasetConfig,
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

        self.batch_data = self.batchify(self.data, cfg)
        self.feat_dim = self.data[0].shape[-1]

    def batchify(self, samples, cfg):
        samples = sorted(samples, key=lambda a: a.shape[0], reverse=True)
        batch_data = []
        while len(batch_data) * cfg.batch_size < len(samples):
            batch_data.append(
                samples[len(batch_data) * cfg.batch_size:(len(batch_data) + 1)
                        * cfg.batch_size]
            )
        return batch_data

    def __getitem__(self, index):
        """overload [] operator"""
        audios: List[Audio] = self.batch_data[index]
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
        """overload len() method"""
        return len(self.batch_data)
