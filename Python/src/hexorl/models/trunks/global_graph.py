"""Global graph trunk construction."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from hexorl.models.global_graph import GlobalHexGraphNet
from hexorl.models.specs import ModelSpec
from hexorl.models.trunks.dense_cnn import resolve_device

GLOBAL_GRAPH_TRUNK = "global_graph"
GLOBAL_XATTN_VARIANT = "context_cross_attention"
GLOBAL_LINE_WINDOW_VARIANT = "line_window_cover"
GLOBAL_RELATION_GRAPH_VARIANT = "relation_graph"


def build_global_graph_model(
    spec: ModelSpec,
    cfg: Any,
    *,
    family_kind: str,
    device: torch.device | None,
    inference: bool,
) -> nn.Module:
    model_cfg = cfg.model
    inference_cfg = cfg.inference
    graph_heads = set(getattr(model_cfg, "heads", []))
    graph_heads.update(f"lookahead_{h}" for h in getattr(getattr(cfg, "buffer", object()), "lookahead_horizons", []))
    model = GlobalHexGraphNet(
        channels=model_cfg.channels,
        layers=getattr(model_cfg, "graph_layers", 3),
        heads=getattr(model_cfg, "attention_heads", 8),
        family_kind=family_kind,
        dropout=getattr(model_cfg, "dropout", 0.0),
        output_heads=sorted(graph_heads),
    )
    resolved = resolve_device(device)
    model = model.to(resolved)
    if inference and inference_cfg.fp16 and resolved.type == "cuda":
        model = model.half()
    if inference:
        model.eval()
    return model


def build_global_xattn_model(
    spec: ModelSpec,
    cfg: Any,
    *,
    device: torch.device | None = None,
    inference: bool = False,
) -> nn.Module:
    return build_global_graph_model(
        spec,
        cfg,
        family_kind=GLOBAL_XATTN_VARIANT,
        device=device,
        inference=inference,
    )


def build_global_line_window_model(
    spec: ModelSpec,
    cfg: Any,
    *,
    device: torch.device | None = None,
    inference: bool = False,
) -> nn.Module:
    return build_global_graph_model(
        spec,
        cfg,
        family_kind=GLOBAL_LINE_WINDOW_VARIANT,
        device=device,
        inference=inference,
    )


def build_global_relation_graph_model(
    spec: ModelSpec,
    cfg: Any,
    *,
    device: torch.device | None = None,
    inference: bool = False,
) -> nn.Module:
    return build_global_graph_model(
        spec,
        cfg,
        family_kind=GLOBAL_RELATION_GRAPH_VARIANT,
        device=device,
        inference=inference,
    )


__all__ = [
    "GLOBAL_GRAPH_TRUNK",
    "GLOBAL_LINE_WINDOW_VARIANT",
    "GLOBAL_RELATION_GRAPH_VARIANT",
    "GLOBAL_XATTN_VARIANT",
    "GlobalHexGraphNet",
    "build_global_graph_model",
    "build_global_line_window_model",
    "build_global_relation_graph_model",
    "build_global_xattn_model",
]
