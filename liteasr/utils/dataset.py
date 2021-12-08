"""Lite-ASR dataset implementation for training."""

from typing import List

from torch.utils.data import Dataset

from liteasr.dataclass.audio_data import Audio
from liteasr.utils.padding import pad


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
            xlens.append(int(len(audio.x)))
            ylens.append(int(len(audio.y)))
        padded_xs = pad(xs, 0)
        padded_ys = pad(ys, 0)
        return padded_xs, padded_ys, xlens, ylens

    def __len__(self):
        """overload len() method"""
        return len(self.data)
