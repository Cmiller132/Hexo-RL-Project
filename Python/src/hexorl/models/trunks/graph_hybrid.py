"""Graph-hybrid crop trunk construction."""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from hexorl.models.constants import BOARD_AREA, BOARD_SIZE
from hexorl.models.inputs import CropInputs
from hexorl.models.trunks.crop_cnn import CropCnnTrunk, build_crop_model

GRAPH_HYBRID_TRUNK = "graph_hybrid"


class SparseHexGraphHybrid0Encoder(nn.Module):
    """Sparse token Transformer trunk for the graph_hybrid_0 scout."""

    TOKEN_SETS = {
        "graph256_cells": 256,
        "graph384_windows": 384,
        "graph512_cover": 512,
        "graph512_turn": 512,
        "graph512_turn_pair_prior": 512,
        "graph768_champion": 768,
    }

    def __init__(
        self,
        channels: int,
        *,
        token_budget: int = 512,
        token_set: str = "graph512_turn_pair_prior",
        heads: int = 8,
        layers: int = 3,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by graph attention heads")
        token_set = token_set.lower()
        default_budget = self.TOKEN_SETS.get(token_set, 512)
        self.token_set = token_set
        self.token_budget = max(16, min(int(token_budget or default_budget), 768))
        self.cell_budget = max(1, min(self.token_budget - 3, BOARD_AREA))

        hidden = max(channels, int(math.ceil(channels * mlp_ratio)))
        self.special_tokens = nn.Parameter(torch.randn(3, channels) * 0.02)
        self.type_embedding = nn.Embedding(8, channels)
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=channels,
                    nhead=heads,
                    dim_feedforward=hidden * 2,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(max(1, int(layers)))
            ]
        )
        self.state_to_map = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
        )
        self.out_norm = nn.BatchNorm2d(channels)

        coords = torch.linspace(-1.0, 1.0, BOARD_SIZE)
        q, r = torch.meshgrid(coords, coords, indexing="ij")
        coord = torch.stack([q.reshape(-1), r.reshape(-1)], dim=-1)
        self.register_buffer("coords", coord.unsqueeze(0), persistent=False)
        dist = torch.maximum(torch.maximum(q.abs(), r.abs()), (q + r).abs())
        self.register_buffer("center_bias", (1.0 - dist.reshape(-1)).unsqueeze(0), persistent=False)
        tie_break = torch.linspace(0.0, 1e-4, BOARD_AREA).unsqueeze(0)
        self.register_buffer("tie_break", tie_break, persistent=False)

    def _selection_score(self, raw: torch.Tensor) -> torch.Tensor:
        flat = raw.flatten(2)
        own = flat[:, 0].abs()
        opp = flat[:, 1].abs()
        empty = flat[:, 2].abs()
        legal = flat[:, 3].abs()
        turn = flat[:, 4].abs()
        first = flat[:, 5].abs()
        own_recent = flat[:, 7].abs()
        opp_recent = flat[:, 8].abs()
        opp_hot = flat[:, 9].abs()
        own_hot = flat[:, 10].abs()
        centroid = flat[:, 11].abs()
        opp_last = flat[:, 12].abs()

        score = (
            4.0 * (own + opp)
            + 5.0 * legal
            + 7.0 * (own_hot + opp_hot)
            + 2.0 * (own_recent + opp_recent)
            + 1.5 * (turn + first + opp_last)
            + 0.15 * empty
            + 0.05 * centroid
        )
        if "windows" in self.token_set:
            score = score + 4.0 * (own_hot + opp_hot)
        if "cover" in self.token_set:
            score = score + 2.0 * opp_hot + 1.0 * legal
        if "turn" in self.token_set or "pair" in self.token_set:
            score = score + 2.0 * legal + 1.0 * (own_hot + opp_hot)
        return score + 0.01 * self.center_bias.to(score) + self.tie_break.to(score)

    def _token_types(self, raw: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        flat = raw.flatten(2)
        gather = indices.unsqueeze(1).expand(-1, flat.shape[1], -1)
        planes = flat.gather(2, gather)
        own = planes[:, 0] > 0.5
        opp = planes[:, 1] > 0.5
        legal = planes[:, 3] > 0.5
        hot = (planes[:, 9].abs() + planes[:, 10].abs()) > 0.0
        recent = (planes[:, 7].abs() + planes[:, 8].abs() + planes[:, 12].abs()) > 0.0
        type_ids = torch.ones_like(indices, dtype=torch.long)
        type_ids = torch.where(legal, torch.full_like(type_ids, 2), type_ids)
        type_ids = torch.where(own | opp, torch.full_like(type_ids, 3), type_ids)
        type_ids = torch.where(hot, torch.full_like(type_ids, 4), type_ids)
        type_ids = torch.where(recent, torch.full_like(type_ids, 5), type_ids)
        if "cover" in self.token_set:
            type_ids = torch.where(hot & legal, torch.full_like(type_ids, 6), type_ids)
        if "pair" in self.token_set or "turn" in self.token_set:
            type_ids = torch.where(legal & ~hot, torch.full_like(type_ids, 7), type_ids)
        return type_ids

    def forward(self, features: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        b, c, h, w = features.shape
        if h != BOARD_SIZE or w != BOARD_SIZE:
            raise ValueError(f"SparseHexGraphHybrid0Encoder expects {BOARD_SIZE}x{BOARD_SIZE}, got {h}x{w}")
        k = min(self.cell_budget, h * w)
        flat = features.flatten(2).transpose(1, 2).contiguous()
        score = self._selection_score(raw)
        indices = torch.topk(score, k=k, dim=1, sorted=False).indices
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, c)
        cell_tokens = flat.gather(1, gather_idx)

        coord = self.coords.to(device=features.device, dtype=features.dtype)
        coord_tokens = self.coord_mlp(coord.expand(b, -1, -1).gather(1, indices.unsqueeze(-1).expand(-1, -1, 2)))
        type_tokens = self.type_embedding(self._token_types(raw, indices)).to(dtype=features.dtype)
        special = self.special_tokens.to(device=features.device, dtype=features.dtype).unsqueeze(0).expand(b, -1, -1)
        tokens = torch.cat([special, cell_tokens + coord_tokens + type_tokens], dim=1)

        for block in self.blocks:
            tokens = block(tokens)

        state = self.state_to_map(tokens[:, 0]).unsqueeze(1)
        updated_cells = tokens[:, 3:]
        delta = updated_cells - cell_tokens
        flat = flat.scatter_add(1, gather_idx, delta)
        flat = flat + state
        out = flat.transpose(1, 2).reshape(b, c, h, w).contiguous()
        return torch.relu(self.out_norm(out))


class CropGraphHybridTrunk(CropCnnTrunk):
    def __init__(self, *, channels: int, blocks: int, graph_token_budget: int, graph_token_set: str, attention_heads: int, graph_layers: int, attention_mlp_ratio: float, attention_dropout: float, dropout: float):
        local_blocks = max(1, min(blocks, max(2, blocks // 4)))
        super().__init__(channels=channels, blocks=local_blocks, dropout=dropout)
        self.graph_encoder = SparseHexGraphHybrid0Encoder(
            channels,
            token_budget=graph_token_budget,
            token_set=graph_token_set,
            heads=attention_heads,
            layers=graph_layers,
            mlp_ratio=attention_mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

    def forward(self, inputs: CropInputs) -> torch.Tensor:
        features = super().forward(inputs)
        return self.graph_encoder(features, inputs.tensor)


def build_graph_hybrid_model(spec, cfg, *, device: torch.device | None = None, inference: bool = False) -> nn.Module:
    params = spec.params
    trunk = CropGraphHybridTrunk(
        channels=params.channels,
        blocks=params.blocks,
        graph_token_budget=params.graph_token_budget,
        graph_token_set=params.graph_token_set,
        attention_heads=params.attention_heads,
        graph_layers=params.graph_layers,
        attention_mlp_ratio=params.attention_mlp_ratio,
        attention_dropout=params.attention_dropout,
        dropout=params.dropout,
    )
    return build_crop_model(trunk, spec, cfg, device=device, inference=inference)


__all__ = ["GRAPH_HYBRID_TRUNK", "CropGraphHybridTrunk", "SparseHexGraphHybrid0Encoder", "build_graph_hybrid_model"]
