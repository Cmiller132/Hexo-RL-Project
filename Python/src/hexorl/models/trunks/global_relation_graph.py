"""Global relation graph trunk."""

from __future__ import annotations

import torch
import torch.nn as nn

from hexorl.models.trunks.global_base import BaseGlobalGraphTrunk
from hexorl.models.trunks.global_xattn import build_global_model

GLOBAL_RELATION_GRAPH_TRUNK = "global_relation_graph"


class GlobalRelationGraphTrunk(BaseGlobalGraphTrunk):
    def __init__(self, *, channels: int, layers: int, heads: int, dropout: float):
        super().__init__(channels=channels, layers=layers, heads=heads, dropout=dropout, relation_required=True, variant_name="relation_graph")


def build_global_relation_graph_model(spec, cfg, *, device: torch.device | None = None, inference: bool = False) -> nn.Module:
    params = spec.params
    trunk = GlobalRelationGraphTrunk(channels=params.channels, layers=params.graph_layers, heads=params.attention_heads, dropout=params.dropout)
    return build_global_model(trunk, spec, cfg, device=device, inference=inference)


__all__ = ["GLOBAL_RELATION_GRAPH_TRUNK", "GlobalRelationGraphTrunk", "build_global_relation_graph_model"]
