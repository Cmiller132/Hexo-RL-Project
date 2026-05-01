"""Crop transformer trunk and RestNet family builder."""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from hexorl.models.constants import BOARD_SIZE
from hexorl.models.inputs import CropInputs
from hexorl.models.trunks.crop_cnn import CropCnnTrunk, GatedResBlock, build_crop_model

RESTNET_TRUNK = "restnet"


class SpatialTransformerBlock(nn.Module):
    """PreNorm spatial attention block over the 33x33 crop."""

    def __init__(self, channels: int, heads: int = 8, mlp_ratio: float = 2.0, dropout: float = 0.0, attention_dropout: float = 0.0, relative_bias: bool = False):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by attention heads")
        if relative_bias:
            raise ValueError("relative_bias is reserved for a later ablation")
        hidden = max(channels, int(math.ceil(channels * mlp_ratio)))
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=attention_dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden * 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden * 2, channels), nn.Dropout(dropout))
        self.coord_mlp = nn.Sequential(nn.Linear(2, channels), nn.SiLU(), nn.Linear(channels, channels))
        coords = torch.linspace(-1.0, 1.0, BOARD_SIZE)
        q, r = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("coords", torch.stack([q.reshape(-1), r.reshape(-1)], dim=-1).unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h != BOARD_SIZE or w != BOARD_SIZE:
            raise ValueError(f"SpatialTransformerBlock expects {BOARD_SIZE}x{BOARD_SIZE}, got {h}x{w}")
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        attn_in = self.norm1(tokens + self.coord_mlp(self.coords.to(device=tokens.device, dtype=tokens.dtype)))
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + self.drop1(attn_out)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w).contiguous()


class CropXformerTrunk(CropCnnTrunk):
    def __init__(self, *, channels: int, blocks: int, attention_positions: list[int], attention_heads: int, attention_mlp_ratio: float, attention_dropout: float, dropout: float, relative_bias: bool):
        super().__init__(channels=channels, blocks=0, dropout=dropout)
        attention_set = set(attention_positions)
        self.res_blocks = nn.ModuleList(
            SpatialTransformerBlock(channels, attention_heads, attention_mlp_ratio, dropout, attention_dropout, relative_bias)
            if idx in attention_set
            else GatedResBlock(channels, dropout=dropout)
            for idx in range(1, blocks + 1)
        )


def build_restnet_model(spec, cfg, *, device: torch.device | None = None, inference: bool = False) -> nn.Module:
    params = spec.params
    trunk = CropXformerTrunk(
        channels=params.channels,
        blocks=params.blocks,
        attention_positions=list(params.attention_positions),
        attention_heads=params.attention_heads,
        attention_mlp_ratio=params.attention_mlp_ratio,
        attention_dropout=params.attention_dropout,
        dropout=params.dropout,
        relative_bias=params.relative_bias,
    )
    return build_crop_model(trunk, spec, cfg, device=device, inference=inference)


__all__ = ["RESTNET_TRUNK", "CropXformerTrunk", "SpatialTransformerBlock", "build_restnet_model"]
