import logging
from pathlib import Path
import pickle
from typing import List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from liteasr.config import DatasetConfig
from liteasr.config import PostProcessConfig
from liteasr.dataclass.audio_data import Audio
from liteasr.dataclass.sheet import AudioSheet
from liteasr.dataclass.sheet import TextSheet
from liteasr.dataset.liteasr_dataset import LiteasrDataset
from liteasr.utils.batchify import FrameBatch
from liteasr.utils.batchify import SeqBatch
from liteasr.utils.progress_bar import ProgressBar
from liteasr.utils.transform import PostProcess
from liteasr.utils.utils import dec2hex

logger = logging.getLogger(__name__)


class AudioFileDataset(LiteasrDataset):
    def __init__(
        self,
        split: str,
        data_dir: str,
        delimiter: Optional[str],
        dataset_cfg: Optional[DatasetConfig],
        postprocess_cfg: Optional[PostProcessConfig],
        vocab,
        keep_raw=False,
        memory_save=False,
    ):
        super().__init__()
        self.split = split
        self.data = []
        self.batchify_policy = None
        self.dump_path = Path(data_dir, ".dump")
        if postprocess_cfg is not None:
            self.set_postprocess(postprocess_cfg)

        _is_prior = memory_save and not self.dump_path.is_dir()
        _is_other = memory_save and self.dump_path.is_dir()

        _as = AudioSheet(data_dir)
        _ts = TextSheet(data_dir, vocab=vocab, delimiter=delimiter)
        assert len(_as) == len(_ts)

        pb = ProgressBar(total=len(_as), title="loaded data")
        for audio_info, text_info in zip(_as, _ts):
            uttid, fd, start, shape = audio_info
            uttid_t, tokenids, text = text_info
            assert uttid_t == uttid
            info = fd, start, shape, tokenids, text if keep_raw else None
            self.data.append(Audio(*info))

            # In `memory_save` mode, processes that are not prior
            # do not have to load whole dataset.They load first data
            # just to get `self.feat_dim`.
            if _is_other:
                break

            pb.update(len(self.data))

        self.feat_dim = self.data[0].x.shape[-1]

        # |                   | train | valid | test  |
        # | ----------------------------------------- |
        # | batchify (normal) |   Y   |   Y   |   N   |
        # | ----------------------------------------- |
        # | batchify (prior)  |   Y   |   Y   |   N   |
        # | batchify (other)  |   N   |   Y   |   N   |
        if not memory_save or _is_prior:
            if dataset_cfg is not None:
                self.batchify(dataset_cfg)

        # dump all the batches
        if _is_prior:
            self.dump_path.mkdir(parents=True)
            for i, batch_indices in enumerate(self.batchify_policy):
                prefix, infix, suffix = dec2hex(i)
                (self.dump_path / prefix / infix).mkdir(parents=True, exist_ok=True)
                pickle.dump(
                    [self.data[idx] for idx in batch_indices],
                    file=(self.dump_path / prefix / infix / f"{suffix}.batch").open(
                        "wb"
                    ),
                )

        # release memory
        if memory_save:
            self.data = []
            self.batchify_policy = None

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
            *sorted(enumerate(self.data), key=lambda d: d[1].xlen, reverse=True)
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
        tensor_xlens = torch.tensor(xlens).long()
        tensor_ylens = torch.tensor(ylens).long()
        return padded_xs, tensor_xlens, padded_ys, tensor_ylens

    @property
    def train(self):
        return self.split == "train"

    def __getitem__(self, index):
        """overload [] operator"""
        if self.batchify_policy is not None:
            return [self.data[idx] for idx in self.batchify_policy[index]]
        elif self.data != []:
            return self.data[index]
        else:
            prefix, infix, suffix = dec2hex(index)
            return pickle.load(
                file=(self.dump_path / prefix / infix / f"{suffix}.batch").open("rb")
            )

    def __len__(self):
        """overload len() method"""
        if self.batchify_policy is not None:
            return len(self.batchify_policy)
        elif self.data != []:
            return len(self.data)
        else:
            count = 0
            for suffix in self.dump_path.iterdir():
                for infix in (self.dump_path / suffix).iterdir():
                    count += len(list((self.dump_path / suffix / infix).iterdir()))
            return count
