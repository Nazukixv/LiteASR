from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        n_head: int,
        i_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        assert i_dim % n_head == 0
        self.d_k = i_dim // n_head
        self.scaling = self.d_k**-0.5
        self.h = n_head
        self.linear_q = nn.Linear(i_dim, i_dim)
        self.linear_k = nn.Linear(i_dim, i_dim)
        self.linear_v = nn.Linear(i_dim, i_dim)
        self.linear_o = nn.Linear(i_dim, i_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def project_qkv(
        self,
        query,
        key,
        value,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch = query.size(0)

        q = self.linear_q(query).contiguous().view(batch, -1, self.h, self.d_k)
        k = self.linear_k(key).contiguous().view(batch, -1, self.h, self.d_k)
        v = self.linear_v(value).contiguous().view(batch, -1, self.h, self.d_k)

        # (Batch, Head, Length, Dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def apply_attention(
        self,
        scores,
        value,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch = value.size(0)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e38)
        attn = self.dropout(self.softmax(scores))
        x = torch.matmul(attn, value)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_k)
        x = self.linear_o(x)
        return x

    def forward(
        self,
        query,
        key,
        value,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        q, k, v = self.project_qkv(query, key, value)
        scores = self.scaling * torch.matmul(q, k.transpose(-2, -1))
        x = self.apply_attention(scores, v, mask=mask)
        return x
