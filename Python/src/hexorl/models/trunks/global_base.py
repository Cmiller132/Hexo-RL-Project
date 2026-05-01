"""Shared global graph trunk modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from hexorl.graph.semantic_builder import GRAPH_FEATURE_DIM, GraphTokenType
from hexorl.models.inference_contracts import graph_input_tensors
from hexorl.models.inputs import GlobalTrunkOutputs, GraphInputs


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

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor, relation_type: torch.Tensor | None = None, relation_bias: torch.Tensor | None = None) -> torch.Tensor:
        b, t, d = x.shape
        if token_mask.shape != (b, t):
            raise ValueError(f"token_mask must have shape {(b, t)}, got {tuple(token_mask.shape)}")
        qkv = self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if relation_type is not None:
            if relation_type.shape != (b, t, t):
                raise ValueError(f"relation_type must have shape {(b, t, t)}, got {tuple(relation_type.shape)}")
            rel = self.relation_embedding(relation_type.clamp_min(0).clamp_max(31).to(device=x.device))
            score = score + rel.permute(0, 3, 1, 2).to(dtype=score.dtype)
        if relation_bias is not None:
            if relation_bias.ndim != 4 or relation_bias.shape[0] != b or relation_bias.shape[2:] != (t, t):
                raise ValueError(f"relation_bias must have shape (B, 1 or heads, T, T); got {tuple(relation_bias.shape)}")
            if relation_bias.shape[1] not in {1, self.heads}:
                raise ValueError(f"relation_bias head dimension must be 1 or match attention heads; got {relation_bias.shape[1]} for {self.heads} heads")
            score = score + relation_bias.to(device=x.device, dtype=score.dtype)
        score = score.masked_fill(~token_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :], -80.0)
        attn = self.dropout(torch.softmax(score, dim=-1))
        y = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, d)
        return self.proj(y)


class GraphBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = max(dim, int(dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelationBiasedSelfAttention(dim, heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor, relation_type: torch.Tensor | None = None, relation_bias: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), token_mask, relation_type, relation_bias)
        return x + self.mlp(self.norm2(x))


class LegalContextCrossAttention(nn.Module):
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
        score = torch.matmul(q, kv[0].transpose(-2, -1)) * self.scale
        score = score.masked_fill(~context_mask.to(device=score.device, dtype=torch.bool)[:, None, None, :], -80.0)
        out = torch.matmul(self.dropout(torch.softmax(score, dim=-1)), kv[1]).transpose(1, 2).reshape(b, a, d)
        return self.proj(out)


class BaseGlobalGraphTrunk(nn.Module):
    input_tensors = graph_input_tensors()

    def __init__(self, *, channels: int, layers: int, heads: int, dropout: float, relation_required: bool, variant_name: str):
        super().__init__()
        self.feature_dim = int(channels)
        self.variant_name = variant_name
        self.relation_required = bool(relation_required)
        self.input = nn.Linear(GRAPH_FEATURE_DIM, channels)
        self.type_embedding = nn.Embedding(max(int(t) for t in GraphTokenType) + 1, channels)
        self.coord = nn.Sequential(nn.Linear(3, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.blocks = nn.ModuleList(GraphBlock(channels, heads, dropout=dropout) for _ in range(max(1, int(layers))))
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: GraphInputs) -> GlobalTrunkOutputs:
        if self.relation_required and (inputs.relation_type is None or inputs.relation_bias is None):
            raise ValueError("relation graph trunk requires relation_type and relation_bias tensors")
        x = self._embed(inputs)
        mask = inputs.token_mask.to(device=x.device, dtype=torch.bool)
        for block in self.blocks:
            x = block(x, mask, inputs.relation_type, inputs.relation_bias)
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        x = self.norm(x)
        legal, legal_mask = self._legal_states(x, inputs)
        legal = self.transform_legal_states(x, legal, mask, inputs)
        pair_states, pair_mask = self._pair_states(x, inputs)
        opp_states, opp_mask = self._opp_legal_states(x[:, 0], inputs)
        return GlobalTrunkOutputs(x, legal, x[:, 0], legal_mask, pair_states, pair_mask, opp_states, opp_mask)

    def transform_legal_states(self, token_states: torch.Tensor, legal_states: torch.Tensor, token_mask: torch.Tensor, inputs: GraphInputs) -> torch.Tensor:
        del token_states, token_mask, inputs
        return legal_states

    def _embed(self, inputs: GraphInputs) -> torch.Tensor:
        qr = inputs.token_qr.to(device=inputs.token_features.device, dtype=inputs.token_features.dtype)
        coord = torch.stack([qr[..., 0], qr[..., 1], qr[..., 0] + qr[..., 1]], dim=-1) / 64.0
        x = self.input(inputs.token_features)
        x = x + self.type_embedding(inputs.token_type.to(device=x.device).clamp_min(0)).to(dtype=x.dtype)
        return (x + self.coord(coord)) * inputs.token_mask.to(device=x.device, dtype=torch.bool).unsqueeze(-1).to(dtype=x.dtype)

    def _legal_states(self, x: torch.Tensor, inputs: GraphInputs) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.legal_token_indices.shape != inputs.legal_mask.shape:
            raise ValueError("legal_token_indices and legal_mask must have matching shape")
        idx_raw = inputs.legal_token_indices.to(device=x.device, dtype=torch.long)
        valid = (idx_raw >= 0) & (idx_raw < x.shape[1])
        idx = idx_raw.clamp(0, max(x.shape[1] - 1, 0))
        return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])), inputs.legal_mask.to(device=x.device, dtype=torch.bool) & valid

    def _pair_states(self, x: torch.Tensor, inputs: GraphInputs) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if inputs.pair_first_indices is None or inputs.pair_second_indices is None:
            return None, None
        if inputs.pair_first_indices.shape != inputs.pair_second_indices.shape:
            raise ValueError("pair_first_indices and pair_second_indices must have matching shape")
        first_raw = inputs.pair_first_indices.to(device=x.device, dtype=torch.long)
        second_raw = inputs.pair_second_indices.to(device=x.device, dtype=torch.long)
        pair_mask = (first_raw >= 0) & (first_raw < x.shape[1]) & (second_raw >= 0) & (second_raw < x.shape[1]) & (first_raw != second_raw)
        if inputs.pair_token_indices is not None:
            pair_mask = pair_mask & (inputs.pair_token_indices.to(device=x.device, dtype=torch.long) >= 0)
        first = x.gather(1, first_raw.clamp(0, max(x.shape[1] - 1, 0)).unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        second = x.gather(1, second_raw.clamp(0, max(x.shape[1] - 1, 0)).unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        state = x[:, 0].unsqueeze(1).expand_as(first)
        return torch.cat([state, first, second, (first - second).abs()], dim=-1), pair_mask

    def _opp_legal_states(self, state: torch.Tensor, inputs: GraphInputs) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if inputs.opp_legal_qr is None or inputs.opp_legal_mask is None:
            return None, None
        if inputs.opp_legal_qr.ndim != 3 or inputs.opp_legal_qr.shape[-1] != 2:
            raise ValueError("opp_legal_qr must have shape (B, A_opp, 2)")
        if inputs.opp_legal_mask.shape != inputs.opp_legal_qr.shape[:2]:
            raise ValueError("opp_legal_mask must match opp_legal_qr leading dimensions")
        oqr = inputs.opp_legal_qr.to(device=state.device, dtype=state.dtype)
        ocoord = torch.stack([oqr[..., 0], oqr[..., 1], oqr[..., 0] + oqr[..., 1]], dim=-1) / 64.0
        type_ids = torch.full(inputs.opp_legal_mask.shape, int(GraphTokenType.LEGAL), device=state.device, dtype=torch.long)
        opp = state.unsqueeze(1) + self.type_embedding(type_ids).to(dtype=state.dtype) + self.coord(ocoord)
        return opp, inputs.opp_legal_mask.to(device=state.device, dtype=torch.bool)


__all__ = ["BaseGlobalGraphTrunk", "GraphBlock", "LegalContextCrossAttention", "RelationBiasedSelfAttention"]
