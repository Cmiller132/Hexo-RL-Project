"""Model loading/provider boundary for runtime consumers."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hexorl.models.assembly import build_model_from_config
from hexorl.models.recipes import bins_to_value as decode_binned_value_logits
from hexorl.models.recipes import load_model_state as _restore_recipe_state


def build_runtime_model(
    cfg,
    *,
    device: Optional[torch.device] = None,
    inference: bool = False,
) -> nn.Module:
    """Build a model through registered architecture recipes for runtime use."""
    model = build_model_from_config(cfg, device=device, inference=inference)
    if inference:
        model.eval()
    return model


def restore_model_weights(
    model: nn.Module,
    state_dict: dict,
    *,
    allow_partial: bool = False,
) -> None:
    """Restore model weights through the registered recipe implementation."""
    _restore_recipe_state(model, state_dict, allow_partial=allow_partial)
