import json
import logging
import math
import os
from pathlib import Path
from typing import List

from torch.nn.utils.rnn import pad_sequence

from liteasr.config import DatasetConfig
from liteasr.config import PostProcessConfig
from liteasr.dataclass.audio_data import Audio
from liteasr.dataclass.sheet import AudioSheet
from liteasr.dataclass.sheet import TextSheet
from liteasr.dataset.liteasr_dataset import LiteasrDataset
from liteasr.utils.batchify import FrameBatch
from liteasr.utils.batchify import SeqBatch
from liteasr.utils.transform import PostProcess

logger = logging.getLogger(__name__)


class SplittedFile(object):

    def __init__(self, file_list, sector_size) -> None:
        self.file_list = file_list
        self.sector_size = sector_size

    def __getitem__(self, index):
        sector = index // self.sector_size
        offset = index % self.sector_size
        for i, line in enumerate(self.file_list[sector]):
            if i == offset:
                info = json.loads(line)
                self.file_list[sector].seek(0)
                return Audio(**info)


class AudioFileDataset(LiteasrDataset):

    def __init__(
        self,
        split: str,
        data_dir: str,
        dataset_cfg: DatasetConfig,
        postprocess_cfg: PostProcessConfig,
        vocab,
        keep_raw=False,
        memory_save=False,
    ):
        super().__init__()
        self.split = split
        self.data = []
        self.batchify_policy = None
        self._flag = memory_save and not os.path.isdir(f"{data_dir}/.split")

        sector_size = 100

        if self._flag:
            Path(f"{data_dir}/.split").mkdir(parents=True)
            index = 0
            ftgt = open(f"{data_dir}/.split/audios.{index}.json", "w")

        _as = AudioSheet(data_dir)
        _ts = TextSheet(data_dir, vocab=vocab)
        for audio_info, text_info in zip(_as, _ts):
            uttid, fd, start, shape = audio_info
            uttid_t, tokenids, text = text_info
            assert uttid_t == uttid
            info = fd, start, shape, tokenids, text if keep_raw else None
            self.data.append(Audio(*info))

            if len(self.data) % sector_size == 0:
                logger.info("number of loaded data: {}".format(len(self.data)))
                if self._flag:
                    index += 1
                    ftgt.close()
                    ftgt = open(f"{data_dir}/.split/audios.{index}.json", "w")

            if self._flag:
                ftgt.write(
                    json.dumps(
                        {
                            "fd": fd,
                            "start": start,
                            "shape": shape,
                            "tokenids": tokenids,
                            "text": text,
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )
        if len(self.data) % sector_size != 0:
            logger.info("number of loaded data: {}".format(len(self.data)))

        self.feat_dim = self.data[0].x.shape[-1]

        # |              | train | valid | test  |
        # | batchify     |   Y   |   Y   |   N   |
        # | post-process |   Y   |   N   |   N   |
        if self.split != "test":
            self.batchify(dataset_cfg)
        if self.split == "train":
            self.set_postprocess(postprocess_cfg)

        if memory_save:
            self.files = SplittedFile(
                [
                    open(f"{data_dir}/.split/audios.{i}.json", "r")
                    for i in range(math.ceil(len(self.data) / sector_size))
                ],
                sector_size=sector_size,
            )
            self.data = None

    def batchify(self, dataset_cfg: DatasetConfig):
        if dataset_cfg.batch_count == "seq":
            BatchifyPolicy = SeqBatch
        elif dataset_cfg.batch_count == "frame":
            BatchifyPolicy = FrameBatch
        else:
            logger.error(f"unsupport strategy {dataset_cfg.batch_count}")
            raise ValueError

        self.batchify_policy = BatchifyPolicy(dataset_cfg)
        indices, _ = zip(
            *
            sorted(enumerate(self.data), key=lambda d: d[1].xlen, reverse=True)
        )
        self.batchify_policy.batchify(indices, self.data)

    def set_postprocess(self, postprocess_cfg: PostProcessConfig):
        self.postprocess = PostProcess(postprocess_cfg)

    def collator(self, samples: List[List[Audio]]):
        xs, xlens, ys, ylens = [], [], [], []
        for sample in samples[0]:
            xs.append(self.postprocess(sample.x) if self.train else sample.x)
            xlens.append(sample.xlen)
            ys.append(sample.y)
            ylens.append(sample.ylen)
        padded_xs = pad_sequence(xs, batch_first=True, padding_value=0)
        padded_ys = pad_sequence(ys, batch_first=True, padding_value=-1)
        return padded_xs, xlens, padded_ys, ylens

    @property
    def train(self):
        return self.split == "train"

    def __getitem__(self, index):
        """overload [] operator"""
        if self.batchify_policy is None:
            return self.data[index]
        elif self.data is not None:
            return [self.data[idx] for idx in self.batchify_policy[index]]
        else:
            return [self.files[idx] for idx in self.batchify_policy[index]]

    def __len__(self):
        """overload len() method"""
        # |                 | train | valid | test  |
        # | data            |   N   |   N   |   Y   |
        # | batchify_policy |   Y   |   Y   |   N   |
        if self.batchify_policy is None:
            return len(self.data)
        else:
            return len(self.batchify_policy)
