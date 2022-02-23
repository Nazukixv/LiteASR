from dataclasses import dataclass
from dataclasses import field
import logging
import os
from typing import Optional

from omegaconf import MISSING

from liteasr.config import DatasetConfig
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
        dataset_cfg: DatasetConfig,
    ):
        assert split in ["train", "valid", "test"]

        if not isinstance(data_cfg, list):
            logger.info(
                "loading {} data from {}".format(
                    split, os.path.dirname(data_cfg.scp)
                )
            )
            self.dataset[split] = AudioFileDataset(
                scp=data_cfg.scp,
                segments=data_cfg.segments,
                text=data_cfg.text,
                vocab=self.vocab,
                cfg=dataset_cfg,
            )
            self.feat_dim = self.dataset[split].feat_dim
        else:
            self.dataset[split] = []
            for cfg in data_cfg:
                logger.info(
                    "loading {} data from {}".format(
                        split, os.path.dirname(cfg.scp)
                    )
                )
                self.dataset[split].append(
                    AudioFileDataset(
                        scp=cfg.scp,
                        segments=cfg.segments,
                        text=cfg.text,
                        vocab=self.vocab,
                        cfg=dataset_cfg,
                    )
                )
            self.feat_dim = self.dataset[split][0].feat_dim

    def inference(self, x, model: LiteasrModel):
        tokenids = model.inference(x)
        tokens = self.vocab.lookup(tokenids)
        return ''.join(tokens[1:])

    def save_model(self, model_name: str, model: LiteasrModel):
        model_path = os.sep.join((self.save_dir, model_name))
        model.save(model_path)
