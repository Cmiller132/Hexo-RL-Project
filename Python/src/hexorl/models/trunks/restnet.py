"""RestNet trunk construction."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from hexorl.models.network import SpatialTransformerBlock
from hexorl.models.specs import ModelSpec
from hexorl.models.trunks.dense_cnn import build_crop_trunk_model

RESTNET_TRUNK = "restnet"


def build_restnet_model(
    spec: ModelSpec,
    cfg: Any,
    *,
    device: torch.device | None = None,
    inference: bool = False,
) -> nn.Module:
    return build_crop_trunk_model(spec, cfg, family_kind=RESTNET_TRUNK, device=device, inference=inference)


__all__ = ["RESTNET_TRUNK", "SpatialTransformerBlock", "build_restnet_model"]
