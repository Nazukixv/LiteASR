"""Lite-ASR dataset implementation for training."""

import logging

from torch.utils.data import Dataset

from liteasr.config import DatasetConfig
from liteasr.config import PostProcessConfig
from liteasr.dataclass.audio_data import Audio
from liteasr.dataclass.sheet import AudioSheet
from liteasr.dataclass.sheet import TextSheet
from liteasr.utils.batchify import SeqDataset

logger = logging.getLogger(__name__)


class AudioFileDataset(Dataset):

    def __init__(
        self,
        data_cfg: str,
        vocab,
        keep_raw=False,
    ):
        self.data = []
        _as = AudioSheet(data_cfg)
        _ts = TextSheet(data_cfg, vocab=vocab)
        for audio_info, text_info in zip(_as, _ts):
            uttid, fd, start, end, shape = audio_info
            uttid_t, tokenids, text = text_info
            assert uttid_t == uttid
            if not keep_raw:
                info = uttid, fd, start, end, shape, tokenids
            else:
                info = uttid, fd, start, end, shape, tokenids, text
            self.data.append(Audio(*info))

            if len(self.data) % 10000 == 0:
                logger.info("number of loaded data: {}".format(len(self.data)))
        logger.info("number of loaded data: {}".format(len(self.data)))
        self.feat_dim = self.data[0].shape[-1]

    def batchify(
        self,
        split: str,
        dataset_cfg: DatasetConfig,
        postprocess_cfg: PostProcessConfig,
    ) -> Dataset:
        return SeqDataset(
            samples=self.data,
            split=split,
            dataset_cfg=dataset_cfg,
            postprocess_cfg=postprocess_cfg,
        )

    def __getitem__(self, index):
        """overload [] operator"""
        return self.data[index]

    def __len__(self):
        """overload len() method"""
        return len(self.data)
