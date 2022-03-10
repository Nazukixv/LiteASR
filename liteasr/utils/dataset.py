"""Lite-ASR dataset implementation for training."""

import logging

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from liteasr.config import DatasetConfig
from liteasr.config import PostProcessConfig
from liteasr.dataclass.audio_data import Audio
from liteasr.dataclass.sheet import AudioSheet
from liteasr.dataclass.sheet import TextSheet
from liteasr.utils.transform.spec_augment import SpecAugment

logger = logging.getLogger(__name__)


class SortedDataset(Dataset):

    def __init__(
        self,
        samples,
        split: str,
        dataset_cfg: DatasetConfig,
        postprocess_cfg: PostProcessConfig,
    ):
        self.data = []
        self.split = split
        self.dataset_cfg = dataset_cfg
        self.postprocess_cfg = postprocess_cfg
        self.batchify(samples)

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

    def batchify(self, samples):
        samples = sorted(samples, key=lambda a: a.xlen, reverse=True)
        start = 0
        while True:
            ilen = samples[start].xlen
            olen = samples[start].ylen
            factor = max(
                int(ilen / self.dataset_cfg.max_len_in),
                int(olen / self.dataset_cfg.max_len_out),
            )
            bs = max(
                self.dataset_cfg.min_batch_size,
                int(self.dataset_cfg.batch_size / (1 + factor)),
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


class AudioFileDataset(Dataset):

    def __init__(
        self,
        scp: str,
        segments: str,
        text: str,
        vocab,
        keep_raw=False,
    ):
        self.data = []
        _as = AudioSheet(scp=scp, segments=segments)
        _ts = TextSheet(text=text, vocab=vocab)
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
        return SortedDataset(
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
