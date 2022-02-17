from dataclasses import dataclass
from dataclasses import field
import logging
import os
from typing import Optional

from omegaconf import MISSING

from liteasr.config import DatasetConfig
from liteasr.config import LiteasrDataclass
from liteasr.dataclass.audio_data import Audio
from liteasr.dataclass.sheet import AudioSheet
from liteasr.dataclass.sheet import TextSheet
from liteasr.dataclass.vocab import Vocab
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
        train_data = self._load_dataset(self.train)
        self.feat_dim = train_data[0].shape[-1]

        logger.info(
            "loading validation data from {}".format(
                os.path.dirname(self.valid.scp)
            )
        )
        valid_data = self._load_dataset(self.valid)

        # TODO: batchify
        train_data = sorted(
            train_data, key=lambda audio: audio.shape[0], reverse=True
        )

        while len(self._train_data) * cfg.batch_size < len(train_data):
            self._train_data.append(
                train_data[len(self._train_data)
                           * cfg.batch_size:(len(self._train_data) + 1)
                           * cfg.batch_size]
            )

        valid_data = sorted(
            valid_data, key=lambda audio: audio.shape[0], reverse=True
        )

        while len(self._valid_data) * cfg.batch_size < len(valid_data):
            self._valid_data.append(
                valid_data[len(self._valid_data)
                           * cfg.batch_size:(len(self._valid_data) + 1)
                           * cfg.batch_size]
            )

        from liteasr.utils.dataset import AudioFileDataset
        self.train_set = AudioFileDataset(self._train_data)
        self.valid_set = AudioFileDataset(self._valid_data)

    def inference(self, x, model: LiteasrModel):
        tokenids = model.inference(x)
        tokens = self.vocab.lookup(tokenids)
        return ''.join(tokens[1:])

    def save_model(self, model_name: str, model: LiteasrModel):
        model_path = os.sep.join((self.save_dir, model_name))
        model.save(model_path)

    def _load_dataset(self, cfg: DataConfig):
        data = []
        audio_sheet = AudioSheet(scp=cfg.scp, segments=cfg.segments)
        text_sheet = TextSheet(text=cfg.text, vocab=self.vocab)
        for audio_info, text_info in zip(audio_sheet, text_sheet):
            uttid, fd, start, end, shape = audio_info
            uttid_t, tokenids = text_info
            assert uttid == uttid_t
            data.append(Audio(uttid, fd, start, end, shape, tokenids=tokenids))
            if len(data) % 10000 == 0:
                logger.info("number of loaded data: {}".format(len(data)))
        logger.info("number of loaded data: {}".format(len(data)))
        return data
