"""Registry-backed model construction and public model facets."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from hexorl.models.capabilities import CapabilitySet
from hexorl.models.facets import InferenceAdapter, LossPlan, PolicyProvider
from hexorl.models.inference_contracts import ModelInferenceContract
from hexorl.models.families import builtin_descriptors
from hexorl.models.registry import ModelFamilyRegistry
from hexorl.models.specs import REQUIRED_MODEL_KINDS, model_spec_from_config


def default_registry() -> ModelFamilyRegistry:
    registry = ModelFamilyRegistry()
    for descriptor in builtin_descriptors():
        registry.register(descriptor)
    return registry


def get_model_registry() -> ModelFamilyRegistry:
    return REGISTRY


def build_model(cfg: Any, *, device: torch.device | None = None, inference: bool = False) -> nn.Module:
    spec = model_spec_from_config(cfg)
    descriptor = REGISTRY.resolve(spec)
    return descriptor.model_builder(spec, cfg, device=device, inference=inference)


def build_inference_model(cfg: Any, *, device: torch.device | None = None) -> nn.Module:
    model = build_model(cfg, device=device, inference=True)
    model.eval()
    return model


def train_adapter_for(model: nn.Module, cfg: Any, *, device: torch.device):
    spec = model_spec_from_config(cfg)
    descriptor = REGISTRY.resolve(spec)
    return descriptor.train_adapter_factory(spec, cfg, model, device=device)


def inference_manifest(cfg: Any) -> ModelInferenceContract:
    spec = model_spec_from_config(cfg)
    descriptor = REGISTRY.resolve(spec)
    return descriptor.inference_adapter_factory(spec, cfg, torch.nn.Identity()).contract


def inference_contract(cfg: Any) -> ModelInferenceContract:
    spec = model_spec_from_config(cfg)
    descriptor = REGISTRY.resolve(spec)
    return descriptor.inference_contract_factory(spec, cfg)


def model_capabilities(cfg: Any) -> CapabilitySet:
    return REGISTRY.resolve(model_spec_from_config(cfg)).capabilities


def model_uses_global_graph(cfg: Any) -> bool:
    return model_spec_from_config(cfg).is_global_graph


def model_uses_crop(cfg: Any) -> bool:
    return model_spec_from_config(cfg).is_crop


__all__ = [
    "InferenceAdapter",
    "LossPlan",
    "PolicyProvider",
    "REQUIRED_MODEL_KINDS",
    "REGISTRY",
    "build_inference_model",
    "build_model",
    "default_registry",
    "get_model_registry",
    "inference_contract",
    "inference_manifest",
    "model_capabilities",
    "model_spec_from_config",
    "model_uses_crop",
    "model_uses_global_graph",
]


REGISTRY = default_registry()
