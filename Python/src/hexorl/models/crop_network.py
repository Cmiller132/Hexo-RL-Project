"""Crop-input model wrapper that composes trunk blocks with named heads."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from hexorl.models.constants import DEFAULT_CANDIDATE_FEATURES
from hexorl.models.heads import (
    AuxPolicyHead,
    AxisHead,
    AxisMapHead,
    MovesLeftHead,
    PairPolicyHead,
    PolicyHead,
    RegretRankHead,
    SparsePolicyHead,
    ValueBinnedHead,
    bins_to_value,
    value_to_bins,
)
from hexorl.models.trunks.dense_cnn import GatedResBlock, HexConv2d
from hexorl.models.trunks.graph_hybrid import SparseHexGraphHybrid0Encoder
from hexorl.models.trunks.restnet import SpatialTransformerBlock


class HexNet(nn.Module):
    """Crop-input Hex network with configurable trunk family and output heads."""

    def __init__(
        self,
        channels: int = 128,
        blocks: int = 16,
        heads: Optional[List[str]] = None,
        n_bins: int = 65,
        family_kind: str = "dense_cnn",
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
        self.family_kind = family_kind.lower()
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
        if self.family_kind == "graph_hybrid":
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
                if self.family_kind == "restnet" and idx in attention_set:
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
            elif name in {"sparse_policy", "pair_policy"}:
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

    value_to_bins = staticmethod(value_to_bins)
    bins_to_value = staticmethod(bins_to_value)

    def half(self) -> "HexNet":
        super().half()
        return self

    @torch.no_grad()
    def forward_batch(
        self,
        x: torch.Tensor,
        autocast: bool = False,
        requested_heads: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        if autocast and torch.cuda.is_available():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = self.forward(x)
        else:
            out = self.forward(x)

        if requested_heads is not None:
            out = {k: v for k, v in out.items() if k in requested_heads}

        return out


__all__ = ["HexNet"]
