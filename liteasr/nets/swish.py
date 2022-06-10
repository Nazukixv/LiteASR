"""Swish activation function for Conformer."""

import torch
import torch.nn as nn


class Swish(nn.Module):
    """Construct an Swish object.

    .. math::
        x * sigma(x)
    """

    def forward(self, x):
        """Return Swich activation function."""
        return x * torch.sigmoid(x)
