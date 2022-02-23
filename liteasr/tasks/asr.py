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
        super().__init__()
        self.vocab = Vocab(cfg.vocab)
        self.train = cfg.train
        self.valid = cfg.valid
        self.save_dir = cfg.save_dir

        self.vocab_size = len(self.vocab)
        self.feat_dim = 0

    def load_data(self, cfg: DatasetConfig):
        logger.info(
            "loading training data from {}".format(
                os.path.dirname(self.train.scp)
            )
        )
        self.train_set = AudioFileDataset(
            scp=self.train.scp,
            segments=self.train.segments,
            text=self.train.text,
            vocab=self.vocab,
            cfg=cfg,
        )
        self.feat_dim = self.train_set.feat_dim

        logger.info(
            "loading validation data from {}".format(
                os.path.dirname(self.valid.scp)
            )
        )
        self.valid_set = AudioFileDataset(
            scp=self.valid.scp,
            segments=self.valid.segments,
            text=self.valid.text,
            vocab=self.vocab,
            cfg=cfg,
        )

    def inference(self, x, model: LiteasrModel):
        tokenids = model.inference(x)
        tokens = self.vocab.lookup(tokenids)
        return ''.join(tokens[1:])

    def save_model(self, model_name: str, model: LiteasrModel):
        model_path = os.sep.join((self.save_dir, model_name))
        model.save(model_path)
