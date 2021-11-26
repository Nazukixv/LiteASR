from dataclasses import dataclass
from dataclasses import field
from typing import Optional

from omegaconf import MISSING

from liteasr.config import LiteasrDataclass
from liteasr.dataclass.audio_data import Audio
from liteasr.dataclass.sheet import AudioSheet
from liteasr.dataclass.sheet import TextSheet
from liteasr.dataclass.vocab import Vocab
from liteasr.tasks import LiteasrTask
from liteasr.tasks import register_task


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

    def load_data(self):
        data = []
        audio_sheet = AudioSheet(self.scp, self.segments)
        text_sheet = TextSheet(self.text, self.vocab)
        for audio_info, text_info in zip(audio_sheet, text_sheet):
            uttid, fd, start, end, shape = audio_info
            uttid_t, tokenids = text_info
            assert uttid == uttid_t
            data.append(Audio(uttid, fd, start, end, shape, tokenids))
        data = sorted(data, key=lambda audio: audio.shape[0], reverse=True)
        batch_num, batch_size = 0, 20
        while batch_num * batch_size < len(data):
            self.data.append(
                data[batch_num * batch_size:(batch_num + 1) * batch_size]
            )
            batch_num += 1

        from liteasr.utils.dataset import AudioFileDataset
        self.dataset = AudioFileDataset(self.data)
