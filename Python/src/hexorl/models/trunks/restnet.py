"""RestNet trunk construction."""

from __future__ import annotations

from typing import Any

import math
import torch
import torch.nn as nn

from hexorl.models.constants import BOARD_SIZE
from hexorl.models.specs import ModelSpec
from hexorl.models.trunks.dense_cnn import build_crop_trunk_model

RESTNET_TRUNK = "restnet"


class SpatialTransformerBlock(nn.Module):
    """PreNorm spatial attention block over the 33x33 crop."""

    def __init__(
        self,
        channels: int,
        heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        relative_bias: bool = False,
    ):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by attention heads")
        if relative_bias:
            raise ValueError("relative_bias is reserved for a later ablation")
        hidden = max(channels, int(math.ceil(channels * mlp_ratio)))
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels,
            heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, channels),
            nn.Dropout(dropout),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        coords = torch.linspace(-1.0, 1.0, BOARD_SIZE)
        q, r = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer(
            "coords",
            torch.stack([q.reshape(-1), r.reshape(-1)], dim=-1).unsqueeze(0),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h != BOARD_SIZE or w != BOARD_SIZE:
            raise ValueError(f"SpatialTransformerBlock expects {BOARD_SIZE}x{BOARD_SIZE}, got {h}x{w}")
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        coord = self.coord_mlp(self.coords.to(device=tokens.device, dtype=tokens.dtype))
        y = tokens + coord
        attn_in = self.norm1(y)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + self.drop1(attn_out)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w).contiguous()


def build_restnet_model(
    spec: ModelSpec,
    cfg: Any,
    *,
    device: torch.device | None = None,
    inference: bool = False,
) -> nn.Module:
    return build_crop_trunk_model(spec, cfg, family_kind=RESTNET_TRUNK, device=device, inference=inference)


__all__ = ["RESTNET_TRUNK", "SpatialTransformerBlock", "build_restnet_model"]
