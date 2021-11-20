"""Lite-ASR dataset implementation for training."""

from typing import List

import torch
from torch.utils.data import Dataset

from liteasr.dataclass.audio_data import Audio
from liteasr.utils.kaldiio import load_mat


class AudioFileDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        """overload [] operator"""
        audios: List[Audio] = self.data[index]
        xs, ys = [], []
        for audio in audios:
            ys.append(torch.tensor(audio.tokenids))
            if audio.start == -1:
                feat = torch.from_numpy(load_mat(audio.fd)).float()
            else:
                # TODO: pcm -> tensor
                pass
            xs.append(feat)

        # padding xs
        batch = len(xs)
        t_max = max(x.size(0) for x in xs)
        padded_xs = xs[0].new(batch, t_max, *xs[0].size()[1:]).fill_(0)
        for i in range(batch):
            padded_xs[i, :xs[i].size(0)] = xs[i]
        return padded_xs, ys

    def __len__(self):
        """overload len() method"""
        return len(self.data)
