"""Dense CNN trunk construction."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from hexorl.models.network import GatedResBlock, HexConv2d, HexNet
from hexorl.models.specs import ModelSpec

DENSE_CNN_TRUNK = "dense_cnn"


def resolve_device(device: torch.device | None) -> torch.device:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_crop_trunk_model(
    spec: ModelSpec,
    cfg: Any,
    *,
    family_kind: str,
    device: torch.device | None,
    inference: bool,
) -> nn.Module:
    model_cfg = cfg.model
    inference_cfg = cfg.inference
    heads = list(spec.params["heads"])
    if spec.params["sparse_policy"] and "sparse_policy" not in heads:
        heads.append("sparse_policy")
    model = HexNet(
        channels=model_cfg.channels,
        blocks=model_cfg.blocks,
        heads=heads,
        family_kind=family_kind,
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
    resolved = resolve_device(device)
    model = model.to(resolved)
    if inference and inference_cfg.fp16 and resolved.type == "cuda":
        model = model.half()
    if inference:
        model.eval()
    return model


def build_dense_cnn_model(
    spec: ModelSpec,
    cfg: Any,
    *,
    device: torch.device | None = None,
    inference: bool = False,
) -> nn.Module:
    return build_crop_trunk_model(spec, cfg, family_kind=DENSE_CNN_TRUNK, device=device, inference=inference)


__all__ = [
    "DENSE_CNN_TRUNK",
    "GatedResBlock",
    "HexConv2d",
    "build_crop_trunk_model",
    "build_dense_cnn_model",
    "resolve_device",
]
