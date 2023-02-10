"""LiteASR dataset class."""

import logging
from typing import List

from torch.utils.data import Dataset

from liteasr.config import DatasetConfig
from liteasr.config import PostProcessConfig
from liteasr.dataclass.audio_data import Audio

logger = logging.getLogger(__name__)


class LiteasrDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def batchify(self, dataset_cfg: DatasetConfig) -> None:
        raise NotImplementedError

    def set_postprocess(self, postprocess_cfg: PostProcessConfig) -> None:
        raise NotImplementedError

    def collator(self, samples: List[List[Audio]]):
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
