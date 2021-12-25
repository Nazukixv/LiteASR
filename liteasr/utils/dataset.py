"""Lite-ASR dataset implementation for training."""

from typing import List

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from liteasr.dataclass.audio_data import Audio


class AudioFileDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        """overload [] operator"""
        audios: List[Audio] = self.data[index]
        xs, ys, xlens, ylens = [], [], [], []
        for audio in audios:
            xs.append(audio.x)
            ys.append(audio.y)
            xlens.append(audio.xlen)
            ylens.append(audio.ylen)
        padded_xs = pad_sequence(xs, batch_first=True, padding_value=0)
        padded_ys = pad_sequence(ys, batch_first=True, padding_value=-1)
        return padded_xs, xlens, padded_ys, ylens

    def __len__(self):
        """overload len() method"""
        return len(self.data)
