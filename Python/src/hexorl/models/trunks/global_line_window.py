"""Global line/window graph trunk."""

from __future__ import annotations

import torch
import torch.nn as nn

from hexorl.graph.semantic_builder import GraphTokenType
from hexorl.models.inputs import GraphInputs
from hexorl.models.trunks.global_base import BaseGlobalGraphTrunk
from hexorl.models.trunks.global_xattn import build_global_model

GLOBAL_LINE_WINDOW_TRUNK = "global_line_window"


class GlobalLineWindowTrunk(BaseGlobalGraphTrunk):
    def __init__(self, *, channels: int, layers: int, heads: int, dropout: float):
        super().__init__(channels=channels, layers=layers, heads=heads, dropout=dropout, relation_required=True, variant_name="line_window_cover")
        self.line_window_gate = nn.Sequential(nn.Linear(channels * 2, channels), nn.SiLU(), nn.Linear(channels, channels))

    def transform_legal_states(self, token_states: torch.Tensor, legal_states: torch.Tensor, token_mask: torch.Tensor, inputs: GraphInputs) -> torch.Tensor:
        tactical_types = torch.tensor([int(GraphTokenType.WINDOW6), int(GraphTokenType.LINE), int(GraphTokenType.COVER_SET)], device=token_states.device)
        tactical_mask = torch.isin(inputs.token_type.to(device=token_states.device), tactical_types) & token_mask
        denom = tactical_mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=token_states.dtype)
        context = (token_states * tactical_mask.unsqueeze(-1).to(dtype=token_states.dtype)).sum(dim=1) / denom
        return legal_states + self.line_window_gate(torch.cat([legal_states, context.unsqueeze(1).expand_as(legal_states)], dim=-1))


def build_global_line_window_model(spec, cfg, *, device: torch.device | None = None, inference: bool = False) -> nn.Module:
    params = spec.params
    trunk = GlobalLineWindowTrunk(channels=params.channels, layers=params.graph_layers, heads=params.attention_heads, dropout=params.dropout)
    return build_global_model(trunk, spec, cfg, device=device, inference=inference)


__all__ = ["GLOBAL_LINE_WINDOW_TRUNK", "GlobalLineWindowTrunk", "build_global_line_window_model"]
