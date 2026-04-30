"""Graph-hybrid crop trunk construction."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from hexorl.models.network import SparseHexGraphHybrid0Encoder
from hexorl.models.specs import ModelSpec
from hexorl.models.trunks.dense_cnn import build_crop_trunk_model

GRAPH_HYBRID_TRUNK = "graph_hybrid"


def build_graph_hybrid_model(
    spec: ModelSpec,
    cfg: Any,
    *,
    device: torch.device | None = None,
    inference: bool = False,
) -> nn.Module:
    return build_crop_trunk_model(spec, cfg, family_kind=GRAPH_HYBRID_TRUNK, device=device, inference=inference)


__all__ = ["GRAPH_HYBRID_TRUNK", "SparseHexGraphHybrid0Encoder", "build_graph_hybrid_model"]
