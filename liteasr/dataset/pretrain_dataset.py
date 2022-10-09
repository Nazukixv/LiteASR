import logging
from typing import List

from torch.nn.utils.rnn import pad_sequence

from liteasr.config import DatasetConfig
from liteasr.config import PostProcessConfig
from liteasr.dataclass.audio_data import Audio
from liteasr.dataclass.sheet import AudioSheet
from liteasr.dataset.liteasr_dataset import LiteasrDataset
from liteasr.utils.batchify import Wav2VecBatch

logger = logging.getLogger(__name__)


class RawAudioFileDataset(LiteasrDataset):

    def __init__(self, data_cfg: str) -> None:
        super().__init__()
        self.data = []
        self.batchify_policy = None

        _as = AudioSheet(data_cfg)
        for audio_info in _as:
            uttid, fd, start, shape = audio_info
            info = fd, start, shape, None, None
            self.data.append(Audio(*info))

            if len(self.data) % 10000 == 0:
                logger.info("number of loaded data: {}".format(len(self.data)))
        if len(self.data) % 10000 != 0:
            logger.info("number of loaded data: {}".format(len(self.data)))

    def batchify(self, dataset_cfg: DatasetConfig):
        self.batchify_policy = Wav2VecBatch(dataset_cfg)
        indices, _ = zip(
            *
            sorted(enumerate(self.data), key=lambda d: d[1].xlen, reverse=True)
        )
        self.batchify_policy.batchify(indices, self.data)

    def set_postprocess(self, postprocess_cfg: PostProcessConfig):
        pass

    def collator(self, samples: List[List[Audio]]):
        xs = []
        min_batch_frame = min(samples[0][-1].xlen, 250000)
        for sample in samples[0]:
            xs.append(sample.x[:min_batch_frame])
        return pad_sequence(xs, batch_first=True), None, None, None

    def __getitem__(self, index: int):
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
