"""Gumbel vector quantizer."""

from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class GumbelVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_vars: int,
        temp: Tuple[float, float, float],
        groups: int,
        vq_dim: int,
        combine_groups: bool,
    ):
        """Vector quantization using gumbel softmax.

        :param int dim: input dimension (channels)
        :param int num_vars: number of quantized vectors per group
        :param temp: temperature for training.
            this should be a tuple of 3 elements: (start, stop, decay factor)
        :type temp: Tuple[float, float, float]
        :param int groups: number of groups for vector quantization
        :param int vq_dim: dimensionality of the resulting quantized vector
        :param bool combine_groups: whether to use the vectors for all groups
        """

        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by {groups} for concatenation"

        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim))
        nn.init.uniform_(self.vars)

        self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
        nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
        nn.init.zeros_(self.weight_proj.bias)

        if isinstance(temp, str):
            import ast

            temp = ast.literal_eval(temp)
        assert len(temp) == 3, f"{temp}, {len(temp)}"

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

    def forward(self, x: Tensor):
        """Gumbel vector quantization.

        :param x: (Batch, Frame, Dimension)
        :type x: Tensor
        :return: quantized input & average probabilities
        :rtype: Tuple[Tensor, Tensor]
        """

        # projection
        b, t, d = x.shape
        x = x.reshape(-1, d)  # (b * t, d)
        x = self.weight_proj(x)  # (b * t, g * nv)
        x = x.view(b * t * self.groups, -1)  # (b * t * g, nv)

        # compute codebook index
        _, k = x.max(-1)  # (b * t * g)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(b * t, self.groups, -1)
        )  # (b * t, g, nv)

        avg_x = torch.softmax(
            x.view(b * t, self.groups, -1).float(), dim=-1
        )  # (b * t, g, nv)
        avg_probs = torch.mean(avg_x, dim=0)  # (g, nv)

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(
                x
            )  # (b * t * g, nv)
        else:
            x = hard_x

        x = x.view(b * t, -1)  # (b * t, g * nv)

        vars = self.vars  # (1, g * nv, vq_dim // g)
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)

        # concatenate
        x = x.unsqueeze(-1) * vars  # (b * t, g * nv, vq_dim // g)
        x = x.view(b * t, self.groups, self.num_vars, -1)  # (b * t, g, nv, vq_dim // g)
        x = x.sum(-2)  # (b * t, g, vq_dim // g)
        x = x.view(b, t, -1)  # (b, t, vq_dim)

        return x, avg_probs
