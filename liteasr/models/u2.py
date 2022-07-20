"""U2."""

from dataclasses import dataclass
from dataclasses import defaultdict
from dataclasses import field
from enum import Enum
import math
from typing import List, Optional, Tuple

from omegaconf import II
from omegaconf import MISSING
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from liteasr.config import LiteasrDataclass
from liteasr.models import LiteasrModel
from liteasr.models import register_model
from liteasr.nets.ctc import CTC
from liteasr.nets.transformer_decoder import TransformerDecoder
from liteasr.nets.transformer_encoder import TransformerEncoder
from liteasr.utils.mask import padding_mask
from liteasr.utils.mask import triangle_mask


class EncoderArch(Enum):
    Transformer = "transformer"
    Conformer = "conformer"


class DecoderArch(Enum):
    Transformer = "transformer"


@dataclass
class U2Config(LiteasrDataclass):
    name: Optional[str] = field(default="U2")

    # U2
    dropout_rate: float = field(default=0.0)

    # shared encoder
    enc_arch: EncoderArch = field(default=EncoderArch.Conformer)
    use_rel: bool = field(default=True)
    input_dim: int = field(default=MISSING)
    enc_dim: int = field(default=256)
    enc_ff_dim: int = field(default=2048)
    enc_attn_heads: int = field(default=4)
    enc_dropout_rate: float = II("model.dropout_rate")
    enc_pos_dropout_rate: float = II("model.enc_dropout_rate")
    enc_attn_dropout_rate: float = II("model.enc_dropout_rate")
    enc_ff_dropout_rate: float = II("model.enc_dropout_rate")
    enc_layers: int = field(default=12)
    activation: str = field(default="swish")

    # attention decoder
    dec_arch: DecoderArch = field(default=DecoderArch.Transformer)
    vocab_size: int = field(default=MISSING)
    dec_dim: int = field(default=256)
    dec_ff_dim: int = field(default=2048)
    dec_attn_heads: int = field(default=4)
    dec_dropout_rate: float = II("model.dropout_rate")
    dec_pos_dropout_rate: float = II("model.dec_dropout_rate")
    dec_self_attn_dropout_rate: float = II("model.dec_dropout_rate")
    dec_src_attn_dropout_rate: float = II("model.dec_dropout_rate")
    dec_ff_dropout_rate: float = II("model.dec_dropout_rate")
    dec_layers: int = field(default=6)


@register_model("U2", dataclass=U2Config)
class U2(LiteasrModel):

    def __init__(self, cfg: U2Config, task=None):
        super().__init__()

        assert cfg.enc_arch in EncoderArch
        self.encoder = TransformerEncoder(
            use_rel=cfg.use_rel,
            i_dim=cfg.input_dim,
            h_dim=cfg.enc_dim,
            ff_dim=cfg.enc_ff_dim,
            n_head=cfg.enc_attn_heads,
            n_layer=cfg.enc_layers,
            dropout_rate=cfg.enc_dropout_rate,
            pos_dropout_rate=cfg.enc_pos_dropout_rate,
            attn_dropout_rate=cfg.enc_attn_dropout_rate,
            ff_dropout_rate=cfg.enc_ff_dropout_rate,
            activation=cfg.activation,
            arch=cfg.enc_arch.value,
        )

        assert cfg.dec_arch in DecoderArch
        self.decoder = TransformerDecoder(
            i_dim=cfg.vocab_size,
            h_dim=cfg.dec_dim,
            ff_dim=cfg.dec_ff_dim,
            n_head=cfg.dec_attn_heads,
            n_layer=cfg.dec_layers,
            dropout_rate=cfg.dec_dropout_rate,
            pos_dropout_rate=cfg.dec_pos_dropout_rate,
            self_attn_dropout_rate=cfg.dec_self_attn_dropout_rate,
            src_attn_dropout_rate=cfg.dec_src_attn_dropout_rate,
            ff_dropout_rate=cfg.dec_ff_dropout_rate,
            arch=cfg.dec_arch.value,
        )

        self.ctc = CTC(
            i_dim=cfg.enc_dim,
            o_dim=cfg.vocab_size,
            dropout_rate=cfg.dropout_rate,
        )
        self.ignore = -1
        self.blank = 0
        self.sos = cfg.vocab_size - 1
        self.eos = cfg.vocab_size - 1

    def forward(
        self,
        xs,
        xlens: List[int],
        ys,
        ylens: List[int],
    ):
        xs_in, xs_mask, ys_in, ys_mask = self._preprocess(xs, xlens, ys, ylens)

        # encoder
        h_enc = self.encoder(xs_in, mask=xs_mask)

        # attention part
        """
        e.g. len(ys)=[5, 2, 1] batch=3 lmax=5

        ys_mask (B, Q)
        >>> [[ ,  ,  ,  ,  ],
             [ ,  , 1, 1, 1],
             [ , 1, 1, 1, 1]]

        dec_mask (Q, K)
        >>> [[ , 1, 1, 1, 1],
             [ ,  , 1, 1, 1],
             [ ,  ,  , 1, 1],
             [ ,  ,  ,  , 1],
             [ ,  ,  ,  ,  ]]

        ys_attn_mask (B, Q, K)
        >>> [[[ , 1, 1, 1, 1], [[ , 1, 1, 1, 1], [[ , 1, 1, 1, 1],
              [ ,  , 1, 1, 1],  [ ,  , 1, 1, 1],  [ , 1, 1, 1, 1],
              [ ,  ,  , 1, 1],  [ ,  , 1, 1, 1],  [ , 1, 1, 1, 1],
              [ ,  ,  ,  , 1],  [ ,  , 1, 1, 1],  [ , 1, 1, 1, 1],
              [ ,  ,  ,  ,  ]], [ ,  , 1, 1, 1]], [ , 1, 1, 1, 1]]]

        """
        lmax = ys_mask.shape[1]
        dec_mask = triangle_mask(lmax).to(device=ys_in.device)
        ys_attn_mask = ys_mask.unsqueeze(1) | dec_mask.unsqueeze(0)
        h_attn = self.decoder(
            ys_in,
            mask=ys_attn_mask,
            memory=h_enc,
            memory_mask=xs_mask,
        )

        # ctc part
        h_ctc = self.ctc(h_enc)

        return h_attn, h_ctc

    def inference(self, x):
        return self.attention_rescore(x)

    def attention(self, x):
        h = self.encoder(x)  # (1, time, 83) -> (1, frame, 256)
        max_len = h.size(1)
        beam_size = 10
        h = h.repeat(beam_size, 1, 1)
        hyps = torch.ones([beam_size, 1], dtype=torch.long,
                          device=h.device).fill_(self.sos)
        init_scores = [0.0] + [-float("inf")] * (beam_size - 1)
        scores = torch.tensor(init_scores, dtype=torch.float, device=h.device)
        scores = scores.unsqueeze(1)
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=h.device)
        cache = None

        for i in range(1, max_len + 1):
            if end_flag.sum() == beam_size:
                break

            hyps_mask = triangle_mask(i).unsqueeze(0).to(h.device)

            logp, cache = self.decoder.forward_one_step(
                y=hyps,
                mask=hyps_mask,
                memory=h,
                memory_mask=None,
                cache=cache,
            )

            score_topk, index_topk = torch.topk(logp, beam_size)
            score_topk[end_flag.squeeze()] = torch.tensor(
                init_scores, device=h.device
            )
            index_topk[end_flag.squeeze()] = self.eos

            scores = (scores + score_topk).view(-1, beam_size * beam_size)
            scores, index_global = torch.topk(scores, beam_size)

            scores = scores.view(-1, 1)
            index_global = index_global.view(-1, 1)

            select_hyp_index = index_global // beam_size
            select_hyp_offset = index_global % beam_size

            hyps = torch.cat(
                [
                    hyps[select_hyp_index.squeeze()],
                    index_topk[select_hyp_index, select_hyp_offset],
                ],
                dim=1,
            )

            end_flag = torch.eq(hyps[:, -1], self.eos).view(-1, 1)

        scores = scores.view(-1, beam_size)
        _, best_hyp_index = scores.max(dim=-1)
        best_hyp = hyps[best_hyp_index][0]

        return best_hyp.tolist()

    def _ctc_prefix_beam_search(self, x):
        h = self.encoder(x)  # (1, time, 83) -> (1, frame, 256)
        beam_size = 10
        ctc_probs = self.ctc.log_softmax(h).squeeze(0)  # (frame, vocab)

        cur_hyps = [(tuple(), (0.0, -float('inf')))]

        for logp in ctc_probs:
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            score_topk, index_topk = torch.topk(logp, beam_size)
            for s in index_topk:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s,)
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s,)
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            next_hyps = sorted(
                next_hyps.items(),
                key=lambda x: log_add(list(x[1])),
                reverse=True
            )
            cur_hyps = next_hyps[:beam_size]

        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]

        return hyps, h

    def ctc_prefix_beam_search(self, x):
        hyps, _ = self._ctc_prefix_beam_search(x)
        return hyps[0][0]

    def attention_rescore(self, x):
        hyps, h = self._ctc_prefix_beam_search(x)

        # repeat h to satisfy batch computation
        h_in = h.repeat(len(hyps), 1, 1)  # (beam, frame, 256)

        # padding hyps
        hyps_pad = pad_sequence(
            [
                torch.tensor(hyp[0], device=h.device, dtype=torch.long)
                for hyp in hyps
            ],
            batch_first=True,
            padding_value=self.ignore,
        )

        # replace ignore with <eos>, add <sos> at beginning
        _, _, hyps_in, hyps_mask = self._preprocess(
            xs=h,
            xlens=[h.size(1)],
            ys=hyps_pad,
            ylens=[len(hyp[0]) for hyp in hyps],
        )

        # construct hyps_in_mask
        dec_mask = triangle_mask(hyps_mask.shape[1]).to(device=hyps_in.device)
        hyps_in_mask = hyps_mask.unsqueeze(1) | dec_mask.unsqueeze(0)

        # (beam, max_ylen, vocab)
        h_attn = self.decoder(
            hyps_in,
            mask=hyps_in_mask,
            memory=h_in,
            memory_mask=None,
        )
        attn_score = torch.nn.functional.log_softmax(h_attn, dim=-1)

        # calculate hypothesis score
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += attn_score[i][j][w]
            score += attn_score[i][len(hyp[0])][self.eos]
            # add ctc score
            score += hyp[1] * 0.5  # ctc-weight
            if score > best_score:
                best_score = score
                best_index = i

        return hyps[best_index][0]

    def get_pred_len(self, xlens: List[int]) -> Tensor:
        pred_len = torch.tensor(xlens)
        pred_len = ((pred_len - 1) // 2 - 1) // 2
        return pred_len

    def get_target(self, ys, ylens: List[int]) -> Tuple[Tensor, Tensor]:
        # tgt_attn
        ignore = torch.tensor(self.ignore).repeat(ys.size(0), 1)
        tgt_attn = torch.cat([ys, ignore.to(ys.device)], dim=1)
        index = (torch.arange(len(ylens)), torch.tensor(ylens))
        tgt_attn[index] = self.eos

        # tgt_ctc
        tgt_ctc = ys

        return tgt_attn, tgt_ctc

    def get_target_len(self, ylens: List[int]) -> Tensor:
        target_len = torch.tensor(ylens)
        return target_len

    def _preprocess(
        self,
        xs: Tensor,
        xlens: List[int],
        ys: Tensor,
        ylens: List[int],
    ):
        # xs_in
        xs_in = xs

        # xs_mask
        xs_mask = padding_mask(xlens).to(device=xs.device)

        # ys_in
        ys_ = ys.masked_fill(ys == self.ignore, self.eos)
        sos = (
            torch.ones(ys.size(0),
                       1).fill_(self.sos).to(dtype=ys.dtype, device=ys.device)
        )
        ys_in = torch.cat([sos, ys_], dim=1)

        # ys_mask
        ys_mask = padding_mask([yl + 1 for yl in ylens]).to(device=ys.device)

        return xs_in, xs_mask, ys_in, ys_mask

    @classmethod
    def build_model(cls, cfg: U2Config, task=None):
        cfg.input_dim = task.feat_dim
        cfg.vocab_size = task.vocab_size
        return cls(cfg, task)


def log_add(args: List[int]) -> float:
    """
    Stable log add
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp
