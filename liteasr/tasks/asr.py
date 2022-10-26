from dataclasses import dataclass
from dataclasses import field
import logging
import os
from pathlib import Path
from typing import List, Optional, Union

from omegaconf import MISSING
from omegaconf.listconfig import ListConfig

from liteasr.config import DatasetConfig
from liteasr.config import LiteasrDataclass
from liteasr.config import PostProcessConfig
from liteasr.dataclass.vocab import Vocab
from liteasr.dataset import AudioFileDataset
from liteasr.models import LiteasrModel
from liteasr.tasks import LiteasrTask
from liteasr.tasks import register_task

logger = logging.getLogger(__name__)


@dataclass
class DataConfig(object):
    scp: str = field(default=MISSING)
    segments: Optional[str] = None
    text: str = field(default=MISSING)


@dataclass
class ASRConfig(LiteasrDataclass):
    vocab: str = field(default=MISSING)
    train: str = field(default=MISSING)
    valid: str = field(default=MISSING)
    test: List[str] = field(default_factory=list)
    save_dir: str = field(default="ckpts")


@register_task('asr', dataclass=ASRConfig)
class ASRTask(LiteasrTask):

    def __init__(self, cfg: ASRConfig):
        super().__init__(cfg)
        self.vocab = Vocab(cfg.vocab)
        self.save_dir = cfg.save_dir
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        self.vocab_size = len(self.vocab)
        self.feat_dim = 0

    def load_dataset(
        self,
        split: str,
        data_dir: Union[str, ListConfig],
        dataset_cfg: Optional[DatasetConfig] = None,
        postprocess_cfg: Optional[PostProcessConfig] = None,
        memory_save: bool = False,
    ):
        assert split in ["train", "valid", "test"]

        if isinstance(data_dir, str):
            logger.info("loading {} data from {}".format(split, data_dir))
            self.datasets[split] = AudioFileDataset(
                split=split,
                data_dir=data_dir,
                dataset_cfg=dataset_cfg,
                postprocess_cfg=postprocess_cfg,
                vocab=self.vocab,
                keep_raw=split == "test",
                memory_save=memory_save,
            )
            self.feat_dim = self.datasets[split].feat_dim
        elif isinstance(data_dir, ListConfig):
            self.datasets[split] = []
            for d_dir in data_dir:
                logger.info("loading {} data from {}".format(split, d_dir))
                self.datasets[split].append(
                    AudioFileDataset(
                        split=split,
                        data_dir=d_dir,
                        dataset_cfg=dataset_cfg,
                        postprocess_cfg=postprocess_cfg,
                        vocab=self.vocab,
                        keep_raw=split == "test",
                    )
                )
            self.feat_dim = self.datasets[split][0].feat_dim
        else:
            raise TypeError(
                "data_dir with type {} cannot be parsed".format(type(data_dir))
            )

    def inference(self, x, model: LiteasrModel):
        tokenids = model.inference(x)
        return "".join(self.vocab.lookupi(tokenids, convert=True))

    def save_model(self, model_name: str, model: LiteasrModel):
        model_path = os.sep.join((self.save_dir, model_name))
        model.save(model_path)
