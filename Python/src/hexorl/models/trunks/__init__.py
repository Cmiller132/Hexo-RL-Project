"""Trunk registry and trunk interfaces."""

from __future__ import annotations

from typing import Callable, Protocol

import torch
import torch.nn as nn

from hexorl.models.inference_contracts import TensorSpec
from hexorl.models.inputs import CropInputs, GlobalTrunkOutputs, GraphInputs


class CropTrunk(Protocol):
    feature_channels: int
    input_tensors: tuple[TensorSpec, ...]

    def __call__(self, inputs: CropInputs) -> torch.Tensor: ...


class GlobalTrunk(Protocol):
    feature_dim: int
    variant_name: str
    input_tensors: tuple[TensorSpec, ...]

    def __call__(self, inputs: GraphInputs) -> GlobalTrunkOutputs: ...


TrunkBuilder = Callable[..., nn.Module]
TRUNK_REGISTRY: dict[str, TrunkBuilder] = {}


def register_trunk(name: str, builder: TrunkBuilder) -> None:
    if name in TRUNK_REGISTRY:
        raise ValueError(f"trunk already registered: {name}")
    TRUNK_REGISTRY[name] = builder


from hexorl.models.trunks.crop_cnn import CropCnnTrunk, GatedResBlock, HexConv2d, build_dense_cnn_model
from hexorl.models.trunks.crop_graph_hybrid import (
    CropGraphHybridTrunk,
    SparseHexGraphHybrid0Encoder,
    build_graph_hybrid_model,
)
from hexorl.models.trunks.crop_xformer import CropXformerTrunk, SpatialTransformerBlock, build_restnet_model
from hexorl.models.trunks.global_line_window import GlobalLineWindowTrunk, build_global_line_window_model
from hexorl.models.trunks.global_relation_graph import GlobalRelationGraphTrunk, build_global_relation_graph_model
from hexorl.models.trunks.global_xattn import GlobalXAttnTrunk, build_global_xattn_model

register_trunk("dense_cnn", build_dense_cnn_model)
register_trunk("restnet", build_restnet_model)
register_trunk("graph_hybrid", build_graph_hybrid_model)
register_trunk("global_xattn", build_global_xattn_model)
register_trunk("global_line_window", build_global_line_window_model)
register_trunk("global_relation_graph", build_global_relation_graph_model)

__all__ = [
    "CropCnnTrunk",
    "CropGraphHybridTrunk",
    "CropTrunk",
    "CropXformerTrunk",
    "GatedResBlock",
    "GlobalLineWindowTrunk",
    "GlobalRelationGraphTrunk",
    "GlobalTrunk",
    "GlobalTrunkOutputs",
    "GlobalXAttnTrunk",
    "HexConv2d",
    "SparseHexGraphHybrid0Encoder",
    "SpatialTransformerBlock",
    "TRUNK_REGISTRY",
    "build_dense_cnn_model",
    "build_global_line_window_model",
    "build_global_relation_graph_model",
    "build_global_xattn_model",
    "build_graph_hybrid_model",
    "build_restnet_model",
    "register_trunk",
]
