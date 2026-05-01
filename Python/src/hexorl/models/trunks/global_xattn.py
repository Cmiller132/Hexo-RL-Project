"""Global cross-attention graph trunk."""

from __future__ import annotations

import torch
import torch.nn as nn

from hexorl.graph.semantic_builder import GraphTokenType
from hexorl.models.heads import build_heads_for_family
from hexorl.models.inputs import GraphInputs
from hexorl.models.trunks.crop_cnn import resolve_device
from hexorl.models.trunks.global_base import BaseGlobalGraphTrunk, LegalContextCrossAttention

GLOBAL_XATTN_TRUNK = "global_xattn"


class GlobalXAttnTrunk(BaseGlobalGraphTrunk):
    def __init__(self, *, channels: int, layers: int, heads: int, dropout: float):
        super().__init__(channels=channels, layers=max(1, min(int(layers), 2)), heads=heads, dropout=dropout, relation_required=False, variant_name="context_cross_attention")
        self.legal_cross_attention = LegalContextCrossAttention(channels, heads, dropout=dropout)

    def transform_legal_states(self, token_states: torch.Tensor, legal_states: torch.Tensor, token_mask: torch.Tensor, inputs: GraphInputs) -> torch.Tensor:
        context_mask = token_mask & (inputs.token_type.to(device=token_states.device) != int(GraphTokenType.LEGAL))
        return legal_states + self.legal_cross_attention(legal_states, token_states, context_mask)


def build_global_model(trunk: nn.Module, spec, cfg, *, device: torch.device | None, inference: bool) -> nn.Module:
    from hexorl.models.composers import GlobalModel

    model = GlobalModel(trunk, build_heads_for_family(spec, cfg, trunk))
    model.n_bins = int(getattr(spec.params, "n_bins", 65))
    resolved = resolve_device(device)
    model = model.to(resolved)
    if inference and cfg.inference.fp16 and resolved.type == "cuda":
        model = model.half()
    if inference:
        model.eval()
    return model


def build_global_xattn_model(spec, cfg, *, device: torch.device | None = None, inference: bool = False) -> nn.Module:
    params = spec.params
    trunk = GlobalXAttnTrunk(channels=params.channels, layers=params.graph_layers, heads=params.attention_heads, dropout=params.dropout)
    return build_global_model(trunk, spec, cfg, device=device, inference=inference)


__all__ = ["GLOBAL_XATTN_TRUNK", "GlobalXAttnTrunk", "build_global_model", "build_global_xattn_model"]
