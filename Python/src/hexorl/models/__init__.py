"""Model registry, specs, capabilities, and family implementations."""

from hexorl.models.capabilities import (
    CROP_INPUT,
    DENSE_PLACE_POLICY,
    GLOBAL_GRAPH_INPUT,
    GLOBAL_PLACE_POLICY,
    JOINT_PAIR_POLICY,
    PAIR_FIRST_POLICY,
    PAIR_SECOND_POLICY,
    REGRET_HEAD,
    SPARSE_PLACE_POLICY,
    CapabilitySet,
)
from hexorl.models.factory import (
    REGISTRY,
    build_inference_model,
    build_model,
    get_model_registry,
    inference_manifest,
    model_capabilities,
    model_spec_from_config,
    model_uses_crop,
    model_uses_global_graph,
)
from hexorl.models.specs import ModelSpec

__all__ = [
    "CROP_INPUT",
    "DENSE_PLACE_POLICY",
    "GLOBAL_GRAPH_INPUT",
    "GLOBAL_PLACE_POLICY",
    "JOINT_PAIR_POLICY",
    "PAIR_FIRST_POLICY",
    "PAIR_SECOND_POLICY",
    "REGRET_HEAD",
    "SPARSE_PLACE_POLICY",
    "CapabilitySet",
    "ModelSpec",
    "REGISTRY",
    "build_inference_model",
    "build_model",
    "get_model_registry",
    "inference_manifest",
    "model_capabilities",
    "model_spec_from_config",
    "model_uses_crop",
    "model_uses_global_graph",
]
