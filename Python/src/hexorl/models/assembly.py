"""Model assembly entry points backed by registered architecture specs."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hexorl.models.recipes import build_model_family
from hexorl.models.recipes import bins_to_value as _bins_to_value
from hexorl.models.recipes import load_model_state as _load_model_state
from hexorl.models.registry import is_global_graph_architecture, resolve_model_spec


def build_model_from_config(
    cfg,
    device: Optional[torch.device] = None,
    inference: bool = False,
) -> nn.Module:
    resolved = resolve_model_spec(cfg)
    model = build_model_family(cfg, resolved, device=device, inference=inference)
    return model


def from_config(cfg, device: Optional[torch.device] = None) -> nn.Module:
    model = build_model_from_config(cfg, device=device, inference=True)
    model.eval()
    return model


def bins_to_value(logits: torch.Tensor) -> torch.Tensor:
    return _bins_to_value(logits)


def load_model_state(model: nn.Module, state_dict: dict, *, allow_partial: bool = False):
    return _load_model_state(model, state_dict, allow_partial=allow_partial)


def is_global_graph_model(model: nn.Module) -> bool:
    arch = getattr(model, "hexorl_architecture", getattr(model, "architecture", ""))
    return is_global_graph_architecture(arch)
