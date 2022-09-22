from dataclasses import dataclass
from dataclasses import field
import logging
import os
from pathlib import Path

from omegaconf import MISSING

from liteasr.config import LiteasrDataclass
from liteasr.models import LiteasrModel
from liteasr.tasks import LiteasrTask
from liteasr.tasks import register_task
from liteasr.utils.dataset import RawAudioFileDataset

logger = logging.getLogger(__name__)


@dataclass
class PreTrainConfig(LiteasrDataclass):
    train: str = field(default=MISSING)
    valid: str = field(default=MISSING)
    save_dir: str = field(default="ckpts")


@register_task("pretrain", dataclass=PreTrainConfig)
class PreTrainTask(LiteasrTask):

    def __init__(self, cfg: PreTrainConfig):
        super().__init__(cfg)
        self.save_dir = cfg.save_dir
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def load_dataset(self, split, data_cfg):
        assert split in ["train", "valid"]

        logger.info("loading {} data from {}".format(split, data_cfg))
        self.datasets[split] = RawAudioFileDataset(data_cfg)

    def save_model(self, model_name: str, model: LiteasrModel):
        model_path = os.sep.join((self.save_dir, model_name))
        model.save(model_path)
