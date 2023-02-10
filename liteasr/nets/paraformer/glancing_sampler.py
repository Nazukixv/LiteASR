"""Glancing sampler."""

import random

import torch
import torch.nn as nn


class GlancingSampler(nn.Module):
    def __init__(self, sample_ratio: float):
        super().__init__()
        self.sample_ratio = sample_ratio

    def forward(self, hs, embed_ys, ys, ys_hat, ylens):
        # hs       (B, U, D)
        # embed_ys (B, U, D)
        # ys       (B, U)
        # ys_hat   (B, U)
        # ylens    (B)

        # hamming distance
        distance = (ys_hat != ys).sum(-1)
        sample_num = torch.ceil(self.sample_ratio * distance).long()

        replace_map = torch.zeros_like(ys, device=ys.device).bool()
        for b in range(ys.size(0)):
            sample_idx = random.sample(range(ylens[b]), sample_num[b])
            replace_map[b][sample_idx] = True

        embed_ys_mixed = torch.where(replace_map.unsqueeze(-1), embed_ys, hs)

        return embed_ys_mixed
