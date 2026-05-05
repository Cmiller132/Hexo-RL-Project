"""KataGo-style CNN for Hexo with binned value heads and configurable multi-head architecture.

Input: (B, 13, 33, 33) f32 tensor
Output: dict of head_name → tensor

Supports heads: policy, value (binned), lookahead_* (binned), opp_policy,
axis (3-class), axis_delta_norm (6-plane map), regret_rank (scalar),
regret_value (binned), moves_left (scalar).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from hexorl.action_contract.candidates import CANDIDATE_FEATURES
from hexorl.model.global_graph import GlobalHexGraphNet


BOARD_SIZE = 33
BOARD_AREA = BOARD_SIZE * BOARD_SIZE
DEFAULT_CANDIDATE_FEATURES = CANDIDATE_FEATURES


class HexConv2d(nn.Conv2d):
    """3x3 convolution constrained to the valid axial hex neighborhood."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.kernel_size != (3, 3):
            raise ValueError("HexConv2d is only defined for 3x3 kernels")
        mask = torch.ones_like(self.weight)
        mask[:, :, 0, 0] = 0.0
        mask[:, :, 2, 2] = 0.0
        self.register_buffer("hex_mask", mask, persistent=False)
        self.apply_hex_mask_()

    @torch.no_grad()
    def apply_hex_mask_(self) -> None:
        self.weight.mul_(self.hex_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight * self.hex_mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class GatedResBlock(nn.Module):
    """Gated residual block with BatchNorm2d for training stability."""

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = HexConv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = HexConv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.conv_gate = HexConv2d(channels, channels, 3, padding=1, bias=False)
        self.bn_gate = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.dropout(x)
        gate = torch.sigmoid(self.bn_gate(self.conv_gate(residual)))
        x = x * gate
        return x + residual


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


class SparseHexGraphHybrid0Encoder(nn.Module):
    """Sparse token Transformer trunk for the graph_hybrid_0 scout.

    The public model contract remains `(B,13,33,33) -> dense heads`, but this
    module replaces part of the crop trunk with deterministic sparse attention
    over tactically relevant crop cells. The action identity path is the
    candidate/action-keyed sparse policy head, which keeps global `(q,r)` priors
    out of the dense crop projection. This is not the true global sparse graph
    model from the architecture spec.
    """

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


class PolicyHead(nn.Module):
    """Policy head: (B, C, 33, 33) -> (B, 1089) logits."""

    def __init__(self, channels: int, policy_filters: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(channels, policy_filters, kernel_size=1)
        self.fc = nn.Linear(policy_filters * BOARD_AREA, BOARD_AREA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class ValueBinnedHead(nn.Module):
    """Binned value head: (B, C, 33, 33) -> (B, N_BINS) logits.

    Used for: value, lookahead_*, regret_value.
    """

    def __init__(self, channels: int, n_bins: int = 65, hidden: int = 64):
        super().__init__()
        self.n_bins = n_bins
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.fc1 = nn.Linear(BOARD_AREA, hidden)
        self.fc2 = nn.Linear(hidden, n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class AuxPolicyHead(nn.Module):
    """Auxiliary policy head - same structure as PolicyHead. Used for opp_policy."""

    def __init__(self, channels: int, policy_filters: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(channels, policy_filters, kernel_size=1)
        self.fc = nn.Linear(policy_filters * BOARD_AREA, BOARD_AREA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class AxisHead(nn.Module):
    """Axis classification head: global avg pool → (B, 3) logits."""

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Linear(channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=[2, 3])
        return self.fc(x)


class AxisMapHead(nn.Module):
    """Dense axis-map regression head: (B, C, 33, 33) -> (B, 6, 33, 33)."""

    def __init__(self, channels: int, planes: int = 6):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, max(8, channels // 2), kernel_size=1)
        self.conv2 = nn.Conv2d(max(8, channels // 2), planes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        return F.softplus(self.conv2(x))


class RegretRankHead(nn.Module):
    """Regret ranking head: global avg pool → Linear → ReLU → Linear → scalar φ(s)."""

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=[2, 3])
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class MovesLeftHead(nn.Module):
    """Moves-left head: global avg pool → Linear → ReLU → Linear → softplus."""

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=[2, 3])
        x = torch.relu(self.fc1(x))
        return F.softplus(self.fc2(x))


class SparsePolicyHead(nn.Module):
    """Candidate/action-keyed policy head.

    The head consumes reusable trunk features, candidate features, optional
    in-crop dense logits, and optional in-crop trunk samples. Invalid
    candidates are masked by the loss, not by this module.
    """

    def __init__(
        self,
        channels: int,
        candidate_feature_dim: int = DEFAULT_CANDIDATE_FEATURES,
        hidden: int = 128,
    ):
        super().__init__()
        self.candidate_feature_dim = candidate_feature_dim
        self.net = nn.Sequential(
            nn.Linear(channels + candidate_feature_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        dense_policy_logits: Optional[torch.Tensor],
        candidate_features: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = features.shape
        k = candidate_features.shape[1]
        flat_features = features.permute(0, 2, 3, 1).reshape(b, h * w, c)
        idx = candidate_indices.to(device=features.device, dtype=torch.long)
        valid = (idx >= 0) & (idx < h * w)
        idx_clamped = idx.clamp(0, h * w - 1)
        gather_idx = idx_clamped.unsqueeze(-1).expand(-1, -1, c)
        sampled = flat_features.gather(1, gather_idx)
        sampled = sampled * valid.unsqueeze(-1).to(dtype=sampled.dtype)

        if dense_policy_logits is None:
            dense = torch.zeros(b, k, 1, device=features.device, dtype=features.dtype)
        else:
            dense_vals = dense_policy_logits.gather(1, idx_clamped)
            dense_vals = dense_vals * valid.to(dtype=dense_vals.dtype)
            dense = dense_vals.unsqueeze(-1)

        cand = candidate_features.to(device=features.device, dtype=features.dtype)
        x = torch.cat([sampled, cand, dense], dim=-1)
        return self.net(x).squeeze(-1)


class PairPolicyHead(nn.Module):
    """Auxiliary pair-action head over selected candidate rows."""

    def __init__(
        self,
        channels: int,
        candidate_feature_dim: int = DEFAULT_CANDIDATE_FEATURES,
        hidden: int = 128,
    ):
        super().__init__()
        self.base_dim = channels + candidate_feature_dim + 1
        self.net = nn.Sequential(
            nn.Linear(self.base_dim * 4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def _candidate_embeddings(
        self,
        features: torch.Tensor,
        dense_policy_logits: Optional[torch.Tensor],
        candidate_features: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = features.shape
        flat_features = features.permute(0, 2, 3, 1).reshape(b, h * w, c)
        idx = candidate_indices.to(device=features.device, dtype=torch.long)
        valid = (idx >= 0) & (idx < h * w)
        idx_clamped = idx.clamp(0, h * w - 1)
        gather_idx = idx_clamped.unsqueeze(-1).expand(-1, -1, c)
        sampled = flat_features.gather(1, gather_idx)
        sampled = sampled * valid.unsqueeze(-1).to(dtype=sampled.dtype)
        if dense_policy_logits is None:
            dense = torch.zeros(
                b,
                candidate_features.shape[1],
                1,
                device=features.device,
                dtype=features.dtype,
            )
        else:
            dense_vals = dense_policy_logits.gather(1, idx_clamped)
            dense_vals = dense_vals * valid.to(dtype=dense_vals.dtype)
            dense = dense_vals.unsqueeze(-1)
        cand = candidate_features.to(device=features.device, dtype=features.dtype)
        return torch.cat([sampled, cand, dense], dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        dense_policy_logits: Optional[torch.Tensor],
        candidate_features: torch.Tensor,
        candidate_indices: torch.Tensor,
        pair_candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        base = self._candidate_embeddings(
            features,
            dense_policy_logits,
            candidate_features,
            candidate_indices,
        )
        b, k, d = base.shape
        pair_idx = pair_candidate_indices.to(device=features.device, dtype=torch.long)
        valid = (pair_idx[..., 0] >= 0) & (pair_idx[..., 0] < k) & (pair_idx[..., 1] >= 0) & (pair_idx[..., 1] < k)
        clamped = pair_idx.clamp(0, max(k - 1, 0))
        first = base.gather(1, clamped[..., 0].unsqueeze(-1).expand(-1, -1, d))
        second = base.gather(1, clamped[..., 1].unsqueeze(-1).expand(-1, -1, d))
        x = torch.cat([first, second, (first - second).abs(), first * second], dim=-1)
        logits = self.net(x).squeeze(-1)
        return logits.masked_fill(~valid, -80.0)


class HexNet(nn.Module):
    """KataGo-style network for Hex with configurable multi-head architecture.

    Input:  (B, 13, 33, 33)
    Output: dict of head_name → tensor

    Heads:
        policy      — (B, 1089) policy logits
        value       — (B, N_BINS) binned value logits
        lookahead_* — (B, N_BINS) binned lookahead value logits
        opp_policy  — (B, 1089) opponent policy logits
        axis        — (B, 3) hex axis classification logits
        axis_delta_norm — (B, 6, 33, 33) delta-norm axis-map regression
        regret_rank — (B, 1) ranking score scalar
        regret_value— (B, N_BINS) binned regret value logits
        moves_left  — (B, 1) moves-left scalar (softplus)
    """

    def __init__(
        self,
        channels: int = 128,
        blocks: int = 16,
        heads: Optional[List[str]] = None,
        n_bins: int = 65,
        architecture: str = "cnn",
        attention_positions: Optional[List[int]] = None,
        attention_heads: int = 8,
        attention_mlp_ratio: float = 2.0,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        relative_bias: bool = False,
        graph_token_set: str = "graph512_turn_pair_prior",
        graph_token_budget: int = 512,
        graph_layers: int = 3,
        sparse_policy: bool = False,
        candidate_feature_dim: int = DEFAULT_CANDIDATE_FEATURES,
    ):
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.n_bins = n_bins
        self.architecture = architecture.lower()
        self.attention_positions = sorted(set(attention_positions or []))
        self.sparse_policy_enabled = bool(sparse_policy)
        self.candidate_feature_dim = candidate_feature_dim
        self.graph_token_set = graph_token_set.lower()
        self.graph_token_budget = int(graph_token_budget)
        self.graph_layers = int(graph_layers)

        if heads is None:
            heads = ["policy", "value"]
        self.head_names = list(heads)

        self.conv_in = HexConv2d(13, channels, kernel_size=3, padding=1)

        self.res_blocks = nn.ModuleList()
        self.graph_encoder: Optional[SparseHexGraphHybrid0Encoder] = None
        attention_set = set(self.attention_positions)
        if self.architecture == "graph":
            self.architecture = "graph_hybrid_0"
        if self.architecture == "graph_hybrid_0":
            local_blocks = max(1, min(blocks, max(2, blocks // 4)))
            for _ in range(local_blocks):
                self.res_blocks.append(GatedResBlock(channels, dropout=dropout))
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
        else:
            for idx in range(1, blocks + 1):
                if self.architecture == "restnet" and idx in attention_set:
                    self.res_blocks.append(
                        SpatialTransformerBlock(
                            channels,
                            heads=attention_heads,
                            mlp_ratio=attention_mlp_ratio,
                            dropout=dropout,
                            attention_dropout=attention_dropout,
                            relative_bias=relative_bias,
                        )
                    )
                else:
                    self.res_blocks.append(GatedResBlock(channels, dropout=dropout))

        head_modules: Dict[str, nn.Module] = {}
        for name in self.head_names:
            if name == "policy":
                head_modules[name] = PolicyHead(channels)
            elif name == "opp_policy":
                head_modules[name] = AuxPolicyHead(channels)
            elif name == "value" or name == "regret_value" or name.startswith("lookahead_"):
                head_modules[name] = ValueBinnedHead(channels, n_bins)
            elif name == "axis":
                head_modules[name] = AxisHead(channels)
            elif name == "axis_delta_norm":
                head_modules[name] = AxisMapHead(channels, planes=6)
            elif name == "regret_rank":
                head_modules[name] = RegretRankHead(channels)
            elif name == "moves_left":
                head_modules[name] = MovesLeftHead(channels)
            elif name == "sparse_policy":
                # Built separately below because it is not a dense trunk head.
                continue
            elif name == "pair_policy":
                # Built separately below because it depends on pair candidates.
                continue
            else:
                raise ValueError(f"Unknown head: {name}")
        self.heads = nn.ModuleDict(head_modules)
        self.sparse_policy_head = (
            SparsePolicyHead(
                channels,
                candidate_feature_dim=candidate_feature_dim,
                hidden=max(64, min(256, channels * 2)),
            )
            if self.sparse_policy_enabled or "sparse_policy" in self.head_names
            else None
        )
        self.pair_policy_head = (
            PairPolicyHead(
                channels,
                candidate_feature_dim=candidate_feature_dim,
                hidden=max(64, min(256, channels * 2)),
            )
            if "pair_policy" in self.head_names
            else None
        )

        self._init_weights()
        self.apply_hex_masks_()

    def _init_weights(self):
        """Kaiming normal initialization for Conv2d and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def apply_hex_masks_(self) -> None:
        for m in self.modules():
            if isinstance(m, HexConv2d):
                m.apply_hex_mask_()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        raw = x
        x = torch.relu(self.conv_in(x))
        for block in self.res_blocks:
            x = block(x)
        if self.graph_encoder is not None:
            x = self.graph_encoder(x, raw)
        return x

    def forward(
        self,
        x: torch.Tensor,
        candidate_features: Optional[torch.Tensor] = None,
        candidate_indices: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        pair_candidate_features: Optional[torch.Tensor] = None,
        pair_candidate_row_indices: Optional[torch.Tensor] = None,
        pair_candidate_indices: Optional[torch.Tensor] = None,
        pair_candidate_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, 13, 33, 33) float tensor.

        Returns:
            Dict mapping head name to output tensor.
        """
        x = self.forward_features(x)

        out: Dict[str, torch.Tensor] = {}
        for name in self.head_names:
            if name in {"sparse_policy", "pair_policy"}:
                continue
            out[name] = self.heads[name](x)
        if self.sparse_policy_head is not None and candidate_features is not None:
            if candidate_indices is None:
                raise ValueError("candidate_indices are required for sparse_policy")
            dense_logits = out.get("policy")
            sparse = self.sparse_policy_head(
                x,
                dense_logits,
                candidate_features,
                candidate_indices,
            )
            if candidate_mask is not None:
                sparse = sparse.masked_fill(~candidate_mask.to(device=sparse.device, dtype=torch.bool), -80.0)
            out["sparse_policy"] = sparse
        if (
            self.pair_policy_head is not None
            and candidate_features is not None
            and candidate_indices is not None
            and pair_candidate_indices is not None
        ):
            pair_features = pair_candidate_features if pair_candidate_features is not None else candidate_features
            pair_rows = pair_candidate_row_indices if pair_candidate_row_indices is not None else candidate_indices
            pair = self.pair_policy_head(
                x,
                out.get("policy"),
                pair_features,
                pair_rows,
                pair_candidate_indices,
            )
            if pair_candidate_mask is not None:
                pair = pair.masked_fill(~pair_candidate_mask.to(device=pair.device, dtype=torch.bool), -80.0)
            out["pair_policy"] = pair

        return out

    @staticmethod
    def value_to_bins(t: torch.Tensor, n_bins: int = 65) -> torch.Tensor:
        """Convert continuous value in [-1, 1] to binned soft target.

        Uses linear interpolation between the two nearest bins —
        mirrors KataGo's value head target projection exactly.

        Args:
            t: (...,) tensor of continuous values in [-1, 1].
            n_bins: Number of bins (default 65).

        Returns:
            (..., n_bins) tensor of target probabilities summing to 1.
        """
        bin_width = 2.0 / (n_bins - 1)
        idx = (t + 1.0) / bin_width

        lo = idx.floor().long()
        hi = lo + 1
        hi = hi.clamp(min=0, max=n_bins - 1)
        lo = lo.clamp(min=0, max=n_bins - 1)

        w_hi = idx - lo.float()
        w_lo = 1.0 - w_hi

        target = torch.zeros(*t.shape, n_bins, device=t.device, dtype=torch.float32)
        target.scatter_add_(-1, lo.unsqueeze(-1), w_lo.unsqueeze(-1))
        target.scatter_add_(-1, hi.unsqueeze(-1), w_hi.unsqueeze(-1))

        return target

    @staticmethod
    def bins_to_value(logits: torch.Tensor) -> torch.Tensor:
        """Convert bin logits to expected value in [-1, 1].

        Args:
            logits: (..., N_BINS) tensor of logits.

        Returns:
            (...,) tensor of expected values in [-1, 1].
        """
        n_bins = logits.shape[-1]
        probs = torch.softmax(logits, dim=-1)
        bin_centers = torch.linspace(-1.0, 1.0, n_bins, device=logits.device, dtype=logits.dtype)
        return (probs * bin_centers).sum(dim=-1)

    def half(self) -> "HexNet":
        """Convert to FP16 (chained, like nn.Module.half())."""
        super().half()
        return self

    @torch.no_grad()
    def forward_batch(
        self,
        x: torch.Tensor,
        autocast: bool = False,
        requested_heads: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Inference-only forward with optional heads filter and autocast.

        Args:
            x: (B, 13, 33, 33) input tensor.
            autocast: If True and CUDA available, use FP16 autocast.
            requested_heads: Optional list of head names to compute (filters output).

        Returns:
            Dict of requested head outputs (all heads if requested_heads is None).
        """
        if autocast and torch.cuda.is_available():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = self.forward(x)
        else:
            out = self.forward(x)

        if requested_heads is not None:
            out = {k: v for k, v in out.items() if k in requested_heads}

        return out


def from_config(cfg, device: Optional[torch.device] = None) -> nn.Module:
    """Create a HexNet from a Config object.

    Args:
        cfg: hexorl.config.Config instance.
        device: Target device (cpu, cuda, mps). Defaults to best available.

    Returns:
        HexNet instance on the requested device, optionally in FP16.
    """
    model = build_model_from_config(cfg, device=device, inference=True)
    model.eval()
    return model


def build_model_from_config(
    cfg,
    device: Optional[torch.device] = None,
    inference: bool = False,
) -> nn.Module:
    """Create a HexNet from config while preserving default CNN compatibility."""
    model_cfg = cfg.model
    inference_cfg = cfg.inference
    arch = getattr(model_cfg, "architecture", "cnn").lower()
    if GlobalHexGraphNet.is_global_graph_architecture(arch):
        graph_heads = set(getattr(model_cfg, "heads", []))
        graph_heads.update(f"lookahead_{h}" for h in getattr(getattr(cfg, "buffer", object()), "lookahead_horizons", []))
        model = GlobalHexGraphNet(
            channels=model_cfg.channels,
            layers=getattr(model_cfg, "graph_layers", 3),
            heads=getattr(model_cfg, "attention_heads", 8),
            architecture=arch,
            dropout=getattr(model_cfg, "dropout", 0.0),
            output_heads=sorted(graph_heads),
        )
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        model = model.to(device)
        if inference and inference_cfg.fp16 and device.type == "cuda":
            model = model.half()
        if inference:
            model.eval()
        return model

    heads = list(model_cfg.heads)
    if getattr(model_cfg, "sparse_policy", False) and "sparse_policy" not in heads:
        heads.append("sparse_policy")

    model = HexNet(
        channels=model_cfg.channels,
        blocks=model_cfg.blocks,
        heads=heads,
        architecture=getattr(model_cfg, "architecture", "cnn"),
        attention_positions=list(getattr(model_cfg, "attention_positions", [])),
        attention_heads=getattr(model_cfg, "attention_heads", 8),
        attention_mlp_ratio=getattr(model_cfg, "attention_mlp_ratio", 2.0),
        attention_dropout=getattr(model_cfg, "attention_dropout", 0.0),
        dropout=getattr(model_cfg, "dropout", 0.0),
        relative_bias=getattr(model_cfg, "relative_bias", False),
        graph_token_set=getattr(model_cfg, "graph_token_set", "graph512_turn_pair_prior"),
        graph_token_budget=getattr(model_cfg, "graph_token_budget", 512),
        graph_layers=getattr(model_cfg, "graph_layers", 3),
        sparse_policy=getattr(model_cfg, "sparse_policy", False),
    )

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = model.to(device)

    if inference and inference_cfg.fp16 and device.type == "cuda":
        model = model.half()

    if inference:
        model.eval()
    return model


def strip_compiled_prefix(state_dict: dict) -> dict:
    """Remove torch.compile's _orig_mod prefix when present."""
    if state_dict and all(str(k).startswith("_orig_mod.") for k in state_dict):
        return {str(k).removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict


def load_model_state(model: nn.Module, state_dict: dict, *, allow_partial: bool = False):
    target = getattr(model, "_orig_mod", None)
    if target is not None:
        state_dict = strip_compiled_prefix(state_dict)
        result = target.load_state_dict(state_dict, strict=not allow_partial)
        if hasattr(target, "apply_hex_masks_"):
            target.apply_hex_masks_()
        return result
    state_dict = strip_compiled_prefix(state_dict)
    result = model.load_state_dict(state_dict, strict=not allow_partial)
    if hasattr(model, "apply_hex_masks_"):
        model.apply_hex_masks_()
    return result
