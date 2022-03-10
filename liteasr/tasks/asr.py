from dataclasses import dataclass
from dataclasses import field
import logging
import os
from typing import List, Optional

from omegaconf import MISSING
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from liteasr.config import LiteasrDataclass
from liteasr.dataclass.vocab import Vocab
from liteasr.models import LiteasrModel
from liteasr.tasks import LiteasrTask
from liteasr.tasks import register_task
from liteasr.utils.dataset import AudioFileDataset

logger = logging.getLogger(__name__)


@dataclass
class DataConfig(object):
    scp: str = field(default=MISSING)
    segments: Optional[str] = None
    text: str = field(default=MISSING)


@dataclass
class ASRConfig(LiteasrDataclass):
    vocab: str = field(default=MISSING)
    train: DataConfig = DataConfig()
    valid: DataConfig = DataConfig()
    test: List[DataConfig] = field(default_factory=list)
    save_dir: str = field(default=MISSING)


@register_task('asr', dataclass=ASRConfig)
class ASRTask(LiteasrTask):

    def __init__(self, cfg: ASRConfig):
        super().__init__(cfg)
        self.vocab = Vocab(cfg.vocab)
        self.save_dir = cfg.save_dir

        self.vocab_size = len(self.vocab)
        self.feat_dim = 0

    def load_dataset(
        self,
        split: str,
        data_cfg,
    ):
        assert split in ["train", "valid", "test"]

        if isinstance(data_cfg, DictConfig):
            logger.info(
                "loading {} data from {}".format(
                    split, os.path.dirname(data_cfg.scp)
                )
            )
            self.datasets[split] = AudioFileDataset(
                scp=data_cfg.scp,
                segments=data_cfg.segments,
                text=data_cfg.text,
                vocab=self.vocab,
                keep_raw=split == "test",
            )
            self.feat_dim = self.datasets[split].feat_dim
        elif isinstance(data_cfg, ListConfig):
            self.datasets[split] = []
            for cfg in data_cfg:
                logger.info(
                    "loading {} data from {}".format(
                        split, os.path.dirname(cfg.scp)
                    )
                )
                self.datasets[split].append(
                    AudioFileDataset(
                        scp=cfg.scp,
                        segments=cfg.segments,
                        text=cfg.text,
                        vocab=self.vocab,
                        keep_raw=split == "test",
                    )
                )
            self.feat_dim = self.datasets[split][0].feat_dim
        else:
            raise TypeError(
                "data_cfg with type {} cannot be parsed".format(type(data_cfg))
            )

    def inference(self, x, model: LiteasrModel):
        tokenids = model.inference(x)
        return "".join(self.vocab.lookupi(tokenids, convert=True))

    def save_model(self, model_name: str, model: LiteasrModel):
        model_path = os.sep.join((self.save_dir, model_name))
        model.save(model_path)
