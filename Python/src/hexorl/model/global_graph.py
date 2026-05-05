"""Global graph model family for all-legal Hexo action rows."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hexorl.graph.batch import (
    GRAPH_FEATURE_DIM,
    GRAPH_FEATURE_PLACEMENTS_REMAINING,
    GraphTokenType,
)


class RelationBiasedSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("global graph dim must be divisible by attention heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.relation_embedding = nn.Embedding(32, heads)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        relation_type: Optional[torch.Tensor] = None,
        relation_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, t, d = x.shape
        if token_mask.shape != (b, t):
            raise ValueError(f"token_mask must have shape {(b, t)}, got {tuple(token_mask.shape)}")
        qkv = self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if relation_type is not None:
            if relation_type.shape != (b, t, t):
                raise ValueError(
                    f"relation_type must have shape {(b, t, t)}, got {tuple(relation_type.shape)}"
                )
            rel = self.relation_embedding(
                relation_type.clamp_min(0).clamp_max(31).to(device=x.device)
            )
            score = score + rel.permute(0, 3, 1, 2).to(dtype=score.dtype)
        if relation_bias is not None:
            if relation_bias.ndim != 4 or relation_bias.shape[0] != b or relation_bias.shape[2:] != (t, t):
                raise ValueError(
                    "relation_bias must have shape (B, 1 or heads, T, T); "
                    f"got {tuple(relation_bias.shape)}"
                )
            if relation_bias.shape[1] not in {1, self.heads}:
                raise ValueError(
                    "relation_bias head dimension must be 1 or match attention heads; "
                    f"got {relation_bias.shape[1]} for {self.heads} heads"
                )
            rb = relation_bias.to(device=x.device, dtype=score.dtype)
            if rb.shape[1] == 1:
                score = score + rb
            else:
                score = score + rb
        mask = token_mask.to(device=x.device, dtype=torch.bool)
        score = score.masked_fill(~mask[:, None, None, :], -80.0)
        attn = torch.softmax(score, dim=-1)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, d)
        return self.proj(y)


class GraphBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = max(dim, int(dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelationBiasedSelfAttention(dim, heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, token_mask, relation_type=None, relation_bias=None):
        x = x + self.attn(self.norm1(x), token_mask, relation_type, relation_bias)
        x = x + self.mlp(self.norm2(x))
        return x


class LegalContextCrossAttention(nn.Module):
    """Legal-action queries attending to non-legal context tokens."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("global graph dim must be divisible by attention heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, legal_vec: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        b, a, d = legal_vec.shape
        t = context.shape[1]
        q = self.q(legal_vec).reshape(b, a, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).reshape(b, t, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = context_mask.to(device=score.device, dtype=torch.bool)
        score = score.masked_fill(~mask[:, None, None, :], -80.0)
        attn = self.dropout(torch.softmax(score, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, a, d)
        return self.proj(out)


class GlobalHexGraphNet(nn.Module):
    """Spec-matching graph network over sparse global token/action rows.

    The primary policy output is `policy_place` over `LEGAL` rows.  Dense 1089
    crop policy is intentionally absent from this model.
    """

    ARCHITECTURES = {
        "global_graph_option1",
        "global_xattn_0",
        "global_line_window_0",
        "global_pair_twostage_0",
        "global_graph_full_0",
        "global_hybrid_action_0",
        "global_graph768_champion",
    }
    RELATION_REQUIRED_ARCHITECTURES = {
        "global_graph_option1",
        "global_line_window_0",
        "global_graph_full_0",
        "global_graph768_champion",
    }

    @classmethod
    def is_global_graph_architecture(cls, architecture: object) -> bool:
        """Return whether a config architecture names a registered global graph family."""
        return str(architecture).lower() in cls.ARCHITECTURES

    def __init__(
        self,
        channels: int = 128,
        layers: int = 4,
        heads: int = 8,
        n_bins: int = 65,
        architecture: str = "global_graph_option1",
        dropout: float = 0.0,
        output_heads: Optional[list[str]] = None,
    ):
        super().__init__()
        architecture = architecture.lower()
        if architecture not in self.ARCHITECTURES:
            raise ValueError(f"unknown global graph architecture {architecture}")
        self.architecture = architecture
        self.channels = channels
        self.n_bins = n_bins
        self._all_heads = output_heads is None
        self.head_names = set(output_heads or [])
        self.lookahead_heads = sorted(name for name in self.head_names if name.startswith("lookahead_"))
        self.input = nn.Linear(GRAPH_FEATURE_DIM, channels)
        self.type_embedding = nn.Embedding(max(int(t) for t in GraphTokenType) + 1, channels)
        self.coord = nn.Sequential(nn.Linear(3, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.architecture_family = {
            "global_graph_option1": "relation_graph",
            "global_xattn_0": "context_cross_attention",
            "global_line_window_0": "line_window_cover",
            "global_pair_twostage_0": "pair_two_stage",
            "global_graph_full_0": "full_relation_graph",
            "global_hybrid_action_0": "crop_diagnostic_global_action",
            "global_graph768_champion": "scaled_relation_graph",
        }[architecture]
        block_count = max(1, int(layers))
        if architecture == "global_xattn_0":
            block_count = max(1, min(block_count, 2))
        elif architecture == "global_graph768_champion":
            block_count = max(block_count, 6)
        self.blocks = nn.ModuleList([GraphBlock(channels, heads, dropout=dropout) for _ in range(block_count)])
        self.legal_cross_attention = LegalContextCrossAttention(channels, heads, dropout=dropout)
        self.norm = nn.LayerNorm(channels)
        self.context_to_action = nn.Sequential(nn.Linear(channels * 2, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.line_window_gate = nn.Sequential(nn.Linear(channels * 2, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.hybrid_action_gate = nn.Sequential(nn.Linear(channels + GRAPH_FEATURE_DIM, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.crop_context = nn.Sequential(nn.Linear(13, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.policy_place = nn.Linear(channels, 1)
        self.policy_pair_first = nn.Linear(channels, 1)
        self.policy_opp = nn.Linear(channels, 1)
        self.pair_first_refine: Optional[nn.Module] = None
        self.pair_second_refine: Optional[nn.Module] = None
        if architecture == "global_pair_twostage_0":
            self.pair_first_refine = nn.Sequential(
                nn.Linear(channels * 2, channels),
                nn.SiLU(),
                nn.Linear(channels, channels),
            )
            self.pair_second_refine = nn.Sequential(
                nn.Linear(channels * 3, channels),
                nn.SiLU(),
                nn.Linear(channels, channels),
            )
        self.pair_joint = nn.Sequential(
            nn.Linear(channels * 4, channels),
            nn.SiLU(),
            nn.Linear(channels, 1),
        )
        self.pair_second = nn.Sequential(
            nn.Linear(channels * 4, channels),
            nn.SiLU(),
            nn.Linear(channels, 1),
        )
        self.value = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, n_bins))
        self.regret_rank = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, 1))
        self.regret_value = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, n_bins))
        self.lookahead = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, n_bins))
            for name in self.lookahead_heads
        })
        self.moves_left = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, 1), nn.Softplus())
        self.tactical = nn.Linear(channels, 4)
        self.axis = nn.Linear(channels, 3)
        self.axis_delta_norm = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, 6 * 33 * 33),
        )
        self.legal_token_quality = nn.Linear(channels, 1)

    def _wants(self, *names: str) -> bool:
        return self._all_heads or any(name in self.head_names for name in names)

    def forward(
        self,
        token_features: torch.Tensor,
        token_type: torch.Tensor,
        token_qr: torch.Tensor,
        token_mask: torch.Tensor,
        legal_token_indices: torch.Tensor,
        legal_mask: torch.Tensor,
        opp_legal_qr: Optional[torch.Tensor] = None,
        opp_legal_mask: Optional[torch.Tensor] = None,
        pair_first_indices: Optional[torch.Tensor] = None,
        pair_second_indices: Optional[torch.Tensor] = None,
        pair_token_indices: Optional[torch.Tensor] = None,
        relation_type: Optional[torch.Tensor] = None,
        relation_bias: Optional[torch.Tensor] = None,
        crop_tensor: Optional[torch.Tensor] = None,
        **_unused,
    ) -> Dict[str, torch.Tensor]:
        if self.architecture in self.RELATION_REQUIRED_ARCHITECTURES and (
            relation_type is None or relation_bias is None
        ):
            raise ValueError(f"{self.architecture} requires relation_type and relation_bias tensors")
        qr = token_qr.to(device=token_features.device, dtype=token_features.dtype)
        coord = torch.stack([qr[..., 0], qr[..., 1], qr[..., 0] + qr[..., 1]], dim=-1) / 64.0
        x = self.input(token_features)
        x = x + self.type_embedding(token_type.to(device=x.device).clamp_min(0)).to(dtype=x.dtype)
        x = x + self.coord(coord)
        mask = token_mask.to(device=x.device, dtype=torch.bool)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        if legal_token_indices.shape != legal_mask.shape:
            raise ValueError("legal_token_indices and legal_mask must have matching shape")
        legal_mask_bool = legal_mask.to(device=x.device, dtype=torch.bool)
        legal_idx_raw = legal_token_indices.to(device=x.device, dtype=torch.long)
        legal_idx_valid = (legal_idx_raw >= 0) & (legal_idx_raw < x.shape[1])
        legal_mask_bool = legal_mask_bool & legal_idx_valid
        for block in self.blocks:
            x = block(x, mask, relation_type, relation_bias)
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        x = self.norm(x)
        state = x[:, 0]

        legal_idx = legal_idx_raw.clamp(0, max(x.shape[1] - 1, 0))
        legal_vec = x.gather(1, legal_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        if self.architecture == "global_xattn_0":
            context_type = token_type.to(device=x.device)
            context_mask = mask & (context_type != int(GraphTokenType.LEGAL))
            legal_vec = legal_vec + self.legal_cross_attention(legal_vec, x, context_mask)
        elif self.architecture == "global_line_window_0":
            tactical_types = torch.tensor(
                [int(GraphTokenType.WINDOW6), int(GraphTokenType.LINE), int(GraphTokenType.COVER_SET)],
                device=x.device,
            )
            tactical_mask = torch.isin(token_type.to(device=x.device), tactical_types) & mask
            denom = tactical_mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=x.dtype)
            tactical_context = (x * tactical_mask.unsqueeze(-1).to(dtype=x.dtype)).sum(dim=1) / denom
            legal_vec = legal_vec + self.line_window_gate(
                torch.cat([legal_vec, tactical_context.unsqueeze(1).expand_as(legal_vec)], dim=-1)
            )
        elif self.architecture == "global_hybrid_action_0":
            legal_features = token_features.gather(
                1,
                legal_idx.unsqueeze(-1).expand(-1, -1, token_features.shape[-1]),
            )
            legal_vec = legal_vec + self.hybrid_action_gate(torch.cat([legal_vec, legal_features], dim=-1))
            if crop_tensor is not None:
                crop_context = crop_tensor.to(device=x.device, dtype=x.dtype).mean(dim=(-1, -2))
                legal_vec = legal_vec + self.crop_context(crop_context).unsqueeze(1)
        out: Dict[str, torch.Tensor] = {}
        if self._wants("policy_place", "policy"):
            out["policy_place"] = self.policy_place(legal_vec).squeeze(-1).masked_fill(
                ~legal_mask_bool,
                -80.0,
            )
        if self._wants("policy_pair_first", "pair_policy"):
            pair_first_vec = legal_vec
            if self.pair_first_refine is not None:
                state_action = state.unsqueeze(1).expand_as(legal_vec)
                pair_first_vec = pair_first_vec + self.pair_first_refine(
                    torch.cat([pair_first_vec, state_action], dim=-1)
                )
            first_pair_phase = token_features[:, 0, GRAPH_FEATURE_PLACEMENTS_REMAINING].to(
                device=x.device,
                dtype=x.dtype,
            ) > 0.75
            pair_first_mask = legal_mask_bool & first_pair_phase.unsqueeze(-1)
            out["policy_pair_first"] = self.policy_pair_first(pair_first_vec).squeeze(-1).masked_fill(
                ~pair_first_mask,
                -80.0,
            )
        if self._wants("value"):
            out["value"] = self.value(state)
        if self._wants("regret_rank"):
            out["regret_rank"] = self.regret_rank(state)
        if self._wants("regret_value"):
            out["regret_value"] = self.regret_value(state)
        if self._wants("moves_left"):
            out["moves_left"] = self.moves_left(state)
        if self._wants("tactical"):
            out["tactical"] = self.tactical(state)
        if self._wants("axis"):
            out["axis"] = self.axis(state)
        if self._wants("axis_delta_norm"):
            out["axis_delta_norm"] = self.axis_delta_norm(state).reshape(state.shape[0], 6, 33, 33)
        if self._wants("legal_token_quality"):
            out["legal_token_quality"] = self.legal_token_quality(legal_vec).squeeze(-1).masked_fill(
                ~legal_mask_bool,
                -80.0,
            )
        for name, head in self.lookahead.items():
            out[name] = head(state)
        if self._wants("opp_policy") and opp_legal_qr is not None and opp_legal_mask is not None:
            if opp_legal_qr.ndim != 3 or opp_legal_qr.shape[-1] != 2:
                raise ValueError("opp_legal_qr must have shape (B, A_opp, 2)")
            if opp_legal_mask.shape != opp_legal_qr.shape[:2]:
                raise ValueError("opp_legal_mask must match opp_legal_qr leading dimensions")
            oqr = opp_legal_qr.to(device=x.device, dtype=token_features.dtype)
            ocoord = torch.stack([oqr[..., 0], oqr[..., 1], oqr[..., 0] + oqr[..., 1]], dim=-1) / 64.0
            opp_vec = (
                state.unsqueeze(1)
                + self.type_embedding(
                    torch.full(
                        opp_legal_mask.shape,
                        int(GraphTokenType.LEGAL),
                        device=x.device,
                        dtype=torch.long,
                    )
                ).to(dtype=x.dtype)
                + self.coord(ocoord)
            )
            opp_mask = opp_legal_mask.to(device=x.device, dtype=torch.bool)
            out["opp_policy"] = self.policy_opp(opp_vec).squeeze(-1).masked_fill(~opp_mask, -80.0)
        wants_joint = self._wants("policy_pair_joint", "pair_policy")
        wants_second = self._wants("policy_pair_second", "pair_policy")
        if (wants_joint or wants_second) and pair_first_indices is not None and pair_second_indices is not None:
            if pair_first_indices.shape != pair_second_indices.shape:
                raise ValueError("pair_first_indices and pair_second_indices must have matching shape")
            first_raw = pair_first_indices.to(device=x.device, dtype=torch.long)
            second_raw = pair_second_indices.to(device=x.device, dtype=torch.long)
            pair_mask = (
                (first_raw >= 0)
                & (first_raw < x.shape[1])
                & (second_raw >= 0)
                & (second_raw < x.shape[1])
                & (first_raw != second_raw)
            )
            first = first_raw.clamp(0, max(x.shape[1] - 1, 0))
            second = second_raw.clamp(0, max(x.shape[1] - 1, 0))
            token_type_device = token_type.to(device=x.device)
            first_type = token_type_device.gather(1, first)
            second_type = token_type_device.gather(1, second)
            first_active = mask.gather(1, first)
            second_active = mask.gather(1, second)
            legal_type = int(GraphTokenType.LEGAL)
            stone_type = int(GraphTokenType.STONE)
            pair_action_type = int(GraphTokenType.PAIR_ACTION)
            pair_mask = pair_mask & first_active & second_active
            pair_mask = pair_mask & (second_type == legal_type)
            pair_mask = pair_mask & ((first_type == legal_type) | (first_type == stone_type))
            if pair_token_indices is not None:
                pair_token_raw = pair_token_indices.to(device=x.device, dtype=torch.long)
                if pair_token_raw.shape != pair_first_indices.shape:
                    raise ValueError("pair_token_indices must match pair_first_indices shape")
                materialized_pair_token = pair_token_raw >= 0
                pair_token = pair_token_raw.clamp(0, max(x.shape[1] - 1, 0))
                pair_token_type = token_type_device.gather(1, pair_token)
                pair_token_active = mask.gather(1, pair_token)
                pair_mask = pair_mask & (
                    (~materialized_pair_token)
                    | (
                        (pair_token_raw < x.shape[1])
                        & pair_token_active
                        & (pair_token_type == pair_action_type)
                    )
                )
            first_vec = x.gather(1, first.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
            second_vec = x.gather(1, second.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
            state_pair = state.unsqueeze(1).expand_as(first_vec)
            if self.pair_first_refine is not None:
                first_joint_vec = first_vec + self.pair_first_refine(torch.cat([first_vec, state_pair], dim=-1))
                second_joint_vec = second_vec + self.pair_first_refine(torch.cat([second_vec, state_pair], dim=-1))
            else:
                first_joint_vec = first_vec
                second_joint_vec = second_vec
            first_cond_vec = first_joint_vec
            second_cond_vec = second_vec
            if self.pair_second_refine is not None:
                second_cond_vec = second_cond_vec + self.pair_second_refine(
                    torch.cat([second_cond_vec, first_cond_vec, state_pair], dim=-1)
                )
            if wants_joint:
                joint_features = torch.cat(
                    [
                        state_pair,
                        first_joint_vec + second_joint_vec,
                        (first_joint_vec - second_joint_vec).abs(),
                        first_joint_vec * second_joint_vec,
                    ],
                    dim=-1,
                )
                joint = self.pair_joint(joint_features).squeeze(-1)
                out["policy_pair_joint"] = joint.masked_fill(~pair_mask, -80.0)
            if wants_second:
                second_features = torch.cat(
                    [
                        state_pair,
                        first_cond_vec,
                        second_cond_vec,
                        (first_cond_vec - second_cond_vec).abs(),
                    ],
                    dim=-1,
                )
                second_logits = self.pair_second(second_features).squeeze(-1)
                out["policy_pair_second"] = second_logits.masked_fill(~pair_mask, -80.0)
        return out

    @staticmethod
    def graph_policy_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=logits.device, dtype=torch.bool)
        target = target.to(device=logits.device, dtype=logits.dtype)
        mass = (target * mask.to(dtype=target.dtype)).sum(dim=-1)
        valid = mask.any(dim=-1) & (mass > 0)
        if not torch.any(valid):
            return logits.sum() * 0.0
        log_probs = F.log_softmax(logits.masked_fill(~mask, -80.0), dim=-1)
        norm = torch.zeros_like(target)
        norm[valid] = target[valid] / mass[valid].unsqueeze(-1).clamp(min=1e-6)
        return -(norm[valid] * log_probs[valid]).sum(dim=-1).mean()
