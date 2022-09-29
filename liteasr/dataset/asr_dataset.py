import logging
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


class AudioFileDataset(LiteasrDataset):

    def __init__(
        self,
        split: str,
        data_dir: str,
        vocab,
        keep_raw=False,
    ):
        super().__init__()
        self.split = split
        self.data = []
        self.batchify_policy = None

        _as = AudioSheet(data_dir)
        _ts = TextSheet(data_dir, vocab=vocab)
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
        if len(self.data) % 10000 != 0:
            logger.info("number of loaded data: {}".format(len(self.data)))
        self.feat_dim = self.data[0].shape[-1]

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
        else:
            return [self.data[idx] for idx in self.batchify_policy[index]]

    def __len__(self):
        """overload len() method"""
        if self.batchify_policy is None:
            return len(self.data)
        else:
            return len(self.batchify_policy)
