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


class RelativeMultiHeadAttention(MultiHeadAttention):
    """Multi-Head Attention layer with relative positional encoding.

    Paper: https://arxiv.org/abs/1901.02860

    """

    def __init__(
        self,
        n_head: int,
        i_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__(n_head, i_dim, dropout_rate)

        # linear transformation for positional ecoding
        self.linear_pos = nn.Linear(i_dim, i_dim, bias=False)

        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu: bool = False) -> Tensor:
        """Compute relative positinal encoding."""

        zero_pad = torch.zeros((x.size()[:3] + (1,)), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(
            x.size()[:2]
            + (
                x.size(3) + 1,
                x.size(2),
            )
        )
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(
        self,
        query,
        key,
        value,
        pos_emb,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        q, k, v = self.project_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb)
        p = p.view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) * self.scaling  # (batch, head, time1, time2)

        return self.apply_attention(scores, v, mask=mask)
