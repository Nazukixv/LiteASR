"""Paraformer predictor."""

from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from liteasr.utils.mask import padding_mask


class Predictor(nn.Module):

    def __init__(self, size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=size,
            out_channels=size,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.lin = nn.Linear(size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        xs,
        xlens: Optional[Tensor] = None,
        ylens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # xs (B, T, D) -> (B, T, 1) -> alpha (B, T)
        alpha = self.relu(self.conv(xs.transpose(1, 2)))
        alpha = self.sigmoid(self.lin(alpha.transpose(1, 2)))
        alpha = torch.squeeze(alpha, dim=-1)

        # alpha (B, T)
        if xlens is not None:
            # train
            alpha_mask = padding_mask(xlens)
            alpha = alpha.masked_fill(alpha_mask, 0)
        else:
            # inference
            alpha = alpha

        # beta (B)
        sum_alpha = alpha.sum(-1)
        if ylens is not None:
            # train
            ulens = ylens
        else:
            # inference
            ulens = torch.round(sum_alpha).int()
        beta = sum_alpha / ulens - 1e-4  # prevent precision error

        # Prepare for IF
        B, U, D = xs.size()
        accum_alpha = torch.zeros(B, 0).to(xs.device)
        accum_state = torch.zeros(B, 0, D).to(xs.device)
        fired_state = torch.zeros(B, 0, D).to(xs.device)

        for u in range(U):
            cur_alpha = alpha[:, u]
            cur_state = xs[:, u, :]

            if u == 0:
                prev_alpha = torch.zeros(B).to(xs.device)
                prev_state = torch.zeros(B, D).to(xs.device)
            else:
                prev_alpha = accum_alpha[:, u - 1]
                prev_state = accum_state[:, u - 1, :]

            # bool tensor with size (B, 1)
            new_alpha = prev_alpha + cur_alpha
            is_fired = (new_alpha >= beta).unsqueeze(1)
            # is_close = torch.isclose(new_alpha, beta, atol=1e-5).unsqueeze(1)
            # is_fired = is_fired | is_close

            # both size are (B, 1)
            # `left_alpha`: make sense where `is_fired` are False
            # `right_alpha`: make sense where `is_fired` are True
            left_alpha = (beta - prev_alpha).unsqueeze(1)
            right_alpha = (new_alpha - beta).unsqueeze(1)

            # update info according to `is_fired`
            next_alpha = torch.where(
                is_fired,
                right_alpha,
                new_alpha.unsqueeze(1),
            )
            next_state = torch.where(
                is_fired,
                right_alpha * cur_state,
                prev_state + left_alpha * cur_state,
            ).unsqueeze(1)
            next_fired = torch.where(
                is_fired,
                prev_state + left_alpha * cur_state,
                torch.zeros_like(prev_state),
            ).unsqueeze(1)

            accum_alpha = torch.cat([accum_alpha, next_alpha], dim=1)
            accum_state = torch.cat([accum_state, next_state], dim=1)
            fired_state = torch.cat([fired_state, next_fired], dim=1)

        fired_marks = (torch.abs(fired_state).sum(-1) != 0.0).int()
        cif_output = torch.zeros(0, U, D).to(xs.device)
        for b in range(B):
            rearanged_fired_state = torch.cat(
                [
                    fired_state[b][fired_marks[b, :] == 1],
                    fired_state[b][fired_marks[b, :] == 0],
                ],
                dim=0
            ).unsqueeze(0)
            cif_output = torch.cat([cif_output, rearanged_fired_state], dim=0)
        h_cif = cif_output[:, :max(ulens), :]

        return h_cif, sum_alpha
