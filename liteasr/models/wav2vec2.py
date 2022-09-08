"""wav2vec 2.0."""

from dataclasses import dataclass
from dataclasses import field
from typing import Tuple

import torch
import torch.nn as nn

from liteasr.config import LiteasrDataclass
from liteasr.models import LiteasrModel
from liteasr.models import register_model
from liteasr.nets.gumbel_vector_quantizer import GumbelVectorQuantizer
from liteasr.nets.layer_norm import LayerNorm
from liteasr.nets.transformer_encoder import Wav2Vec2TansformerEncoder
from liteasr.nets.wav2vec2_convolution import Convolution
from liteasr.utils.mask import span_mask


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


@dataclass
class Wav2Vec2Config(LiteasrDataclass):
    # extractor_mode: EXTRACTOR_MODE_CHOICES = field(
    #     default="default",
    #     metadata={
    #         "help":
    #             "mode for feature extractor. default has a single group norm with d "
    #             "groups in the first conv block, whereas layer_norm has layer norms in "
    #             "every block (meant to use with normalize=True)"
    #     },
    # )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    # activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
    #     default="gelu", metadata={"help": "activation function to use"}
    # )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"}
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"}
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help":
                "project final representations and targets to this many dimensions."
                "set to encoder_embed_dim is <= 0"
        },
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"}
    )
    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help":
                "string describing convolutional feature extraction layers in form of a python list that contains "
                "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    quantize_targets: bool = field(
        default=True, metadata={"help": "use quantized targets"}
    )
    quantize_input: bool = field(
        default=False, metadata={"help": "use quantized inputs"}
    )
    same_quantizer: bool = field(
        default=False,
        metadata={"help": "use same quantizer for inputs and targets"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"}
    )
    latent_vars: int = field(
        default=320,
        metadata={
            "help": "number of latent variables V in each group of the codebook"
        },
    )
    latent_groups: int = field(
        default=2,
        metadata={
            "help": "number of groups G of latent variables in the codebook"
        },
    )
    latent_dim: int = field(
        default=0,
        metadata={
            "help":
                "if > 0, uses this dimensionality for latent variables. "
                "otherwise uses final_dim / latent_groups"
        },
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"}
    )
    # mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
    #     default="static", metadata={"help": "how to choose mask length"}
    # )
    mask_other: float = field(
        default=0,
        metadata={
            "help":
                "secondary mask argument (used for more complex distributions), "
                "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"}
    )
    # mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
    #     default="static",
    #     metadata={"help": "how to choose mask length for channel masking"},
    # )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help":
                "secondary mask argument (used for more complex distributions), "
                "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"}
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # negative selection
    num_negatives: int = field(
        default=7,
        metadata={"help": "number of negative examples from the same sample"},
    )
    negatives_from_everywhere: bool = field(
        default=False,
        metadata={
            "help": "sample negatives from everywhere, not just masked states"
        },
    )
    cross_sample_negatives: int = field(
        default=0,
        metadata={"help": "number of negative examples from the any sample"}
    )
    codebook_negatives: int = field(
        default=0, metadata={"help": "number of negative examples codebook"}
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={
            "help": "number of filters for convolutional positional embeddings"
        },
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={
            "help": "number of groups for convolutional positional embedding"
        },
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={
            "help":
                "temperature for latent variable sampling. "
                "can be tuple of 3 values (start, end, decay)"
        },
    )


@register_model("wav2vec2", dataclass=Wav2Vec2Config)
class Wav2Vec2(LiteasrModel):

    def __init__(self, cfg: Wav2Vec2Config, task=None):
        super().__init__()
        self.cfg = cfg

        # convolution feature extractor + linear
        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]
        self.feature_extractor = Convolution(
            conv_layers=feature_enc_layers,
            conv_bias=cfg.conv_bias,
        )
        self.linear_input = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim else None
        )
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        # target quantizer
        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
        self.quantizer = GumbelVectorQuantizer(
            dim=self.embed,
            num_vars=cfg.latent_vars,
            temp=cfg.latent_temp,
            groups=cfg.latent_groups,
            combine_groups=False,
            vq_dim=vq_dim,
        )
        self.linear_quantizer = nn.Linear(vq_dim, final_dim)

        # shared trained feature vector used in masking
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        # contextualized Transformer encoder
        self.encoder = Wav2Vec2TansformerEncoder(
            i_dim=cfg.encoder_embed_dim,
            h_dim=cfg.encoder_embed_dim,
            ff_dim=cfg.encoder_ffn_embed_dim,
            n_head=cfg.encoder_attention_heads,
            n_layer=cfg.encoder_layers,
            dropout_rate=cfg.dropout,
            attn_dropout_rate=cfg.dropout,
            ff_dropout_rate=cfg.dropout,
        )

        self.layer_norm = LayerNorm(self.embed)
        self.linear_final = nn.Linear(cfg.encoder_embed_dim, final_dim)

    def forward(self, source):
        # 1. feature extraction
        features: torch.Tensor = self.feature_extractor(source)
        # features_pen = features.float().pow(2).mean()
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        unmasked_features = features.clone()
        unmasked_features = self.dropout_features(unmasked_features)

        features = self.linear_input(features)
        features = self.dropout_input(features)

        # 2. mask applying
        x, mask_idxs = self.apply_mask(features)

        # 3. contextualized representation
        x = self.encoder(x)
        x = x[mask_idxs].view(x.size(0), -1, x.size(-1))
        x = self.linear_final(x)

        # 4. quantization
        y = unmasked_features[mask_idxs].view(
            unmasked_features.size(0), -1, unmasked_features.size(-1)
        )
        y, _ = self.quantizer(y)
        y = self.linear_quantizer(y)

        # 5. sample negatives
        if self.cfg.negatives_from_everywhere:
            # sample from unmasked features
            q, _ = self.quantizer(unmasked_features)
            negs, _ = self.sample_negatives(
                sample_source=q,
                num_mask=y.size(1),
            )
            negs = self.linear_quantizer(negs)
        else:
            # sample from y
            negs, _ = self.sample_negatives(
                sample_source=y,
                num_mask=y.size(1),
            )

        logits = self.compute_logits(x, pos=y, negs=negs)

        return logits

    def apply_mask(self, x):
        batch, frame, _ = x.shape
        if self.cfg.mask_prob > 0:
            mask_indices = span_mask(
                batch=batch,
                frame=frame,
                prob=self.cfg.mask_prob,
                length=self.cfg.mask_length,
                policy="static",
                no_overlap=self.cfg.no_mask_overlap,
                min_mask_num=2,
                min_interval=self.cfg.mask_min_space,
            )
            mask_indices = mask_indices.to(x.device)
            x[mask_indices] = self.mask_emb
        return x, mask_indices

    def sample_negatives(self, sample_source, num_mask):
        batch, frame, dim = sample_source.shape
        sample_source = sample_source.view(-1, dim)

        with torch.no_grad():
            idx = (
                buffered_arange(num_mask).unsqueeze(-1).expand(
                    -1, self.cfg.num_negatives
                ).flatten()
            )

            negative_indices = torch.randint(
                low=0,
                high=frame - 1,
                size=(batch, num_mask * self.cfg.num_negatives),
            )

            negative_indices[negative_indices >= idx] += 1

        for i in range(1, batch):
            negative_indices[i] += i * frame

        negatives = sample_source[negative_indices.view(-1)]
        negatives = negatives.view(
            batch, num_mask, self.cfg.num_negatives, dim
        ).permute(2, 0, 1, 3)

        return negatives, negative_indices

    def compute_logits(self, x, pos, negs):
        pos = pos.unsqueeze(0)  # (1, B, M, D)
        tgt = torch.cat([pos, negs], dim=0)  # (N+1, B, M, D)

        logits = torch.cosine_similarity(
            x.float(), tgt.float(), dim=-1
        ).type_as(x)
        logits = logits / self.cfg.logit_temp  # (N+1, B, M)

        neg_is_pos = (negs == pos).all(-1)  # (N, B, M)
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        logits = logits.transpose(0, 2)  # (M, B, N+1)
        logits = logits.reshape(-1, logits.size(-1))  # (M*B, N+1)

        return logits

    def get_target(self, ys, ylens):
        # `ys` here is actually `logits` with shape (M*B, N+1)
        return ys.new_zeros(ys.size(0), dtype=torch.long)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Config, task=None):
        return cls(cfg, task)
