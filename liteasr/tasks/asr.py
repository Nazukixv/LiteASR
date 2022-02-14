from dataclasses import dataclass
from dataclasses import field
import logging
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
class ASRConfig(LiteasrDataclass):
    scp: str = field(default=MISSING)
    segments: Optional[str] = None
    vocab: str = field(default=MISSING)
    text: str = field(default=MISSING)


@register_task('asr', dataclass=ASRConfig)
class ASRTask(LiteasrTask):

    def __init__(self, cfg: ASRConfig):
        super().__init__()
        self.scp = cfg.scp
        self.segments = cfg.segments
        self.vocab = Vocab(cfg.vocab)
        self.text = cfg.text

        self.vocab_size = len(self.vocab)
        self.feat_dim = 0

    def load_data(self, cfg: DatasetConfig):
        data = []
        audio_sheet = AudioSheet(self.scp, self.segments)
        text_sheet = TextSheet(self.text, self.vocab)
        for audio_info, text_info in zip(audio_sheet, text_sheet):
            uttid, fd, start, end, shape = audio_info
            uttid_t, tokenids = text_info
            assert uttid == uttid_t
            data.append(Audio(uttid, fd, start, end, shape, tokenids))
            if len(data) % 10000 == 0:
                logger.info("number of loaded data: {}".format(len(data)))
        logger.info("number of loaded data: {}".format(len(data)))

        # TODO: batchify
        data = sorted(data, key=lambda audio: audio.shape[0], reverse=True)
        self.feat_dim = data[0].shape[-1]

        while len(self.data) * cfg.batch_size < len(data):
            self.data.append(
                data[len(self.data) * cfg.batch_size:(len(self.data) + 1)
                     * cfg.batch_size]
            )

        from liteasr.utils.dataset import AudioFileDataset
        self.dataset = AudioFileDataset(self.data)

    def inference(self, x, model: LiteasrModel):
        tokenids = model.inference(x)
        tokens = self.vocab.lookup(tokenids)
        return ''.join(tokens[1:])
