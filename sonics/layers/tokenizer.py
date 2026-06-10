import math
import torch
import torch.nn as nn
from sonics.layers.embedding import (
    SinusoidPositionalEncoding,
    LearnedPositionalEncoding,
)


class STTokenizer(nn.Module):
    def __init__(
        self,
        input_spec_dim,
        input_temp_dim,
        t_clip,
        f_clip,
        embed_dim,
        pre_norm=False,
        pe_learnable=False,
    ):
        super(STTokenizer, self).__init__()
        self.input_spec_dim = input_spec_dim
        self.input_temp_dim = input_temp_dim
        self.t_clip = t_clip
        self.f_clip = f_clip
        self.embed_dim = embed_dim
        self.pre_norm = pre_norm
        self.pe_learnable = pe_learnable

        self.num_temporal_tokens = math.floor(
            (input_temp_dim - t_clip) / t_clip + 1
        )
        self.num_spectral_tokens = math.floor(
            (input_spec_dim - f_clip) / f_clip + 1
        )
        self.num_tokens = (
            self.num_temporal_tokens + self.num_spectral_tokens
        )

        self.temporal_tokenizer = Tokenizer1D(
            input_spec_dim,
            embed_dim,
            clip_size=t_clip,
            num_clips=self.num_temporal_tokens,
            pre_norm=pre_norm,
            pe_learnable=pe_learnable,
        )
        self.spectral_tokenizer = Tokenizer1D(
            input_temp_dim,
            embed_dim,
            clip_size=f_clip,
            num_clips=self.num_spectral_tokens,
            pre_norm=pre_norm,
            pe_learnable=pe_learnable,
        )

    def forward(self, x):
        temporal_input = x
        temporal_tokens = self.temporal_tokenizer(
            temporal_input
        )

        spectral_input = x.permute(0, 2, 1)
        spectral_tokens = self.spectral_tokenizer(
            spectral_input
        )

        spectro_temporal_tokens = torch.cat(
            (temporal_tokens, spectral_tokens), dim=1
        )
        return spectro_temporal_tokens

class Tokenizer1D(nn.Module):
    """Teimporal/Spectral Tokenizer

    Whisper uses temporal tokenizer but time_clip_size is too small, stride=1,  thus
    complexity is very high. We use stride=clip_size - 1 to reduce complexity.
    """

    def __init__(
        self,
        input_dim,
        token_dim,
        clip_size,
        num_clips,
        pre_norm=False,
        pe_learnable=False,
    ):
        super(Tokenizer1D, self).__init__()
        self.conv1d = nn.Conv1d(
            input_dim,
            token_dim,
            clip_size,
            stride=clip_size,
            bias=not pre_norm,
        )
        self.act = nn.GELU()
        self.pos_encoder = (
            SinusoidPositionalEncoding(token_dim)
            if not pe_learnable
            else LearnedPositionalEncoding(token_dim, num_clips)
        )
        self.norm_pre = nn.LayerNorm(token_dim, eps=1e-6) if pre_norm else nn.Identity()

    def forward(self, x):
        x = x
        x = self.conv1d(x)
        x = self.act(x)
        x = x.transpose(1, 2)
        x = self.pos_encoder(x)
        x = self.norm_pre(x)
        return x
