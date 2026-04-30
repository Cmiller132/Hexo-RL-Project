"""Registry-backed model construction and public model facets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

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
from hexorl.models.global_graph import GlobalHexGraphNet
from hexorl.models.network import HexNet
from hexorl.models.registry import FamilyComponents, ModelFamilyDescriptor, ModelFamilyRegistry
from hexorl.models.specs import (
    MODEL_SPEC_VERSION,
    REQUIRED_MODEL_KINDS,
    ModelSpec,
    model_spec_from_config,
)


@dataclass(frozen=True)
class InferenceManifest:
    protocol_version: int
    request_kind: str
    model_family: str
    model_spec_version: int
    input_contract: str
    output_contract: str
    action_contract: str
    graph_schema_version: int | None
    relation_schema_version: int | None
    max_tokens: int | None
    max_legal_rows: int
    max_pair_rows: int
    required_heads: tuple[str, ...]
    optional_heads: tuple[str, ...]
    capabilities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "request_kind": self.request_kind,
            "model_family": self.model_family,
            "model_spec_version": self.model_spec_version,
            "input_contract": self.input_contract,
            "output_contract": self.output_contract,
            "action_contract": self.action_contract,
            "graph_schema_version": self.graph_schema_version,
            "relation_schema_version": self.relation_schema_version,
            "max_tokens": self.max_tokens,
            "max_legal_rows": self.max_legal_rows,
            "max_pair_rows": self.max_pair_rows,
            "required_heads": list(self.required_heads),
            "optional_heads": list(self.optional_heads),
            "capabilities": list(self.capabilities),
        }


@dataclass(frozen=True)
class InferenceAdapter:
    spec: ModelSpec
    manifest: InferenceManifest


@dataclass(frozen=True)
class PolicyProvider:
    model: nn.Module
    capabilities: CapabilitySet


@dataclass(frozen=True)
class LossPlan:
    weights: dict[str, float]
    masked_heads: tuple[str, ...]
    finite_required: bool = True


def default_registry() -> ModelFamilyRegistry:
    registry = ModelFamilyRegistry()
    for descriptor in _builtin_descriptors():
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
    return REGISTRY.resolve(spec).train_adapter_factory(spec, cfg, model, device=device)


def inference_manifest(cfg: Any) -> InferenceManifest:
    spec = model_spec_from_config(cfg)
    descriptor = REGISTRY.resolve(spec)
    return descriptor.inference_adapter_factory(spec, cfg, torch.nn.Identity()).manifest


def model_capabilities(cfg: Any) -> CapabilitySet:
    return REGISTRY.resolve(model_spec_from_config(cfg)).capabilities


def model_uses_global_graph(cfg: Any) -> bool:
    return model_spec_from_config(cfg).is_global_graph


def model_uses_crop(cfg: Any) -> bool:
    return model_spec_from_config(cfg).is_crop


def _resolve_device(device: torch.device | None) -> torch.device:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_crop_model(
    spec: ModelSpec,
    cfg: Any,
    *,
    device: torch.device | None,
    inference: bool,
) -> nn.Module:
    model_cfg = cfg.model
    inference_cfg = cfg.inference
    heads = list(spec.params["heads"])
    if spec.params["sparse_policy"] and "sparse_policy" not in heads:
        heads.append("sparse_policy")
    family_kind = {
        "dense_cnn": "dense_cnn",
        "restnet": "restnet",
        "graph_hybrid": "graph_hybrid",
    }[spec.kind]
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
    resolved = _resolve_device(device)
    model = model.to(resolved)
    if inference and inference_cfg.fp16 and resolved.type == "cuda":
        model = model.half()
    if inference:
        model.eval()
    return model


def _build_global_model(
    spec: ModelSpec,
    cfg: Any,
    *,
    device: torch.device | None,
    inference: bool,
) -> nn.Module:
    model_cfg = cfg.model
    inference_cfg = cfg.inference
    graph_heads = set(getattr(model_cfg, "heads", []))
    graph_heads.update(f"lookahead_{h}" for h in getattr(getattr(cfg, "buffer", object()), "lookahead_horizons", []))
    variant = {
        "global_xattn": "context_cross_attention",
        "global_line_window": "line_window_cover",
        "global_relation_graph": "relation_graph",
    }[spec.kind]
    model = GlobalHexGraphNet(
        channels=model_cfg.channels,
        layers=getattr(model_cfg, "graph_layers", 3),
        heads=getattr(model_cfg, "attention_heads", 8),
        family_kind=variant,
        dropout=getattr(model_cfg, "dropout", 0.0),
        output_heads=sorted(graph_heads),
    )
    resolved = _resolve_device(device)
    model = model.to(resolved)
    if inference and inference_cfg.fp16 and resolved.type == "cuda":
        model = model.half()
    if inference:
        model.eval()
    return model


def _loss_plan(spec: ModelSpec, cfg: Any) -> LossPlan:
    masked = (
        ("policy_place", "policy_pair_first", "policy_pair_second", "policy_pair_joint")
        if spec.is_global_graph
        else ("sparse_policy", "pair_policy")
    )
    return LossPlan(weights=dict(getattr(cfg.train, "loss_weights", {})), masked_heads=masked)


def _recipe(spec_kind: str):
    def _provider(host_profile: Any) -> dict[str, Any]:
        return {
            "model_kind": spec_kind,
            "host_profile": str(host_profile),
            "channels": 16,
            "blocks": 1,
            "graph_layers": 1,
            "valid": True,
        }
    return _provider


def _tune_space(spec_kind: str):
    def _provider(host_profile: Any) -> dict[str, Any]:
        return {
            "model_kind": spec_kind,
            "mutations": {
                "channels": [8, 16, 32],
                "blocks": [1, 2],
                "graph_layers": [1, 2] if spec_kind.startswith("global") else [1],
            },
        }
    return _provider


def _manifest_provider(spec: ModelSpec, cfg: Any) -> dict[str, Any]:
    descriptor = REGISTRY.resolve(spec)
    return {
        "model_family": descriptor.name,
        "model_spec_version": MODEL_SPEC_VERSION,
        "model_spec": spec.manifest(),
        "capabilities": descriptor.capabilities.to_manifest(),
        "inference_protocol": descriptor.inference_adapter_factory(spec, cfg, torch.nn.Identity()).manifest.to_dict(),
    }


def _inference_adapter(spec: ModelSpec, cfg: Any, model: nn.Module) -> InferenceAdapter:
    descriptor = REGISTRY.resolve(spec)
    is_global = spec.is_global_graph
    required_heads = ("value", "policy_place") if is_global else ("value", "policy")
    manifest = InferenceManifest(
        protocol_version=1,
        request_kind="global_graph" if is_global else "crop",
        model_family=descriptor.name,
        model_spec_version=MODEL_SPEC_VERSION,
        input_contract="global_graph_v1" if is_global else "crop_tensor_v1",
        output_contract="global_place_value_v1" if is_global else "dense_place_value_v1",
        action_contract="legal_action_table_v1",
        graph_schema_version=1 if is_global else None,
        relation_schema_version=1 if is_global else None,
        max_tokens=1024 if is_global else None,
        max_legal_rows=1024 if is_global else 1089,
        max_pair_rows=4096 if is_global else int(getattr(cfg.model, "pair_strategy_max_pairs", 0)),
        required_heads=required_heads,
        optional_heads=tuple(sorted(set(getattr(cfg.model, "heads", [])) - set(required_heads))),
        capabilities=tuple(descriptor.capabilities.to_manifest()),
    )
    return InferenceAdapter(spec=spec, manifest=manifest)


def _policy_provider(spec: ModelSpec, cfg: Any, model: nn.Module) -> PolicyProvider:
    return PolicyProvider(model=model, capabilities=REGISTRY.resolve(spec).capabilities)


def _train_adapter(spec: ModelSpec, cfg: Any, model: nn.Module, *, device: torch.device):
    from hexorl.train.adapters import TrainAdapter

    return TrainAdapter(spec=spec, cfg=cfg, model=model, device=device, loss_plan=_loss_plan(spec, cfg))


def _descriptor(
    name: str,
    aliases: tuple[str, ...],
    capabilities: tuple[str, ...],
    builder,
    trunk: str,
    heads: tuple[str, ...],
) -> ModelFamilyDescriptor:
    return ModelFamilyDescriptor(
        name=name,
        aliases=frozenset(aliases),
        capabilities=CapabilitySet.of(capabilities),
        spec_schema=ModelSpec,
        components=FamilyComponents(trunk=trunk, heads=heads),
        model_builder=builder,
        train_adapter_factory=_train_adapter,
        inference_adapter_factory=_inference_adapter,
        policy_provider_factory=_policy_provider,
        loss_plan_provider=_loss_plan,
        recipe_provider=_recipe(name),
        tune_space_provider=_tune_space(name),
        checkpoint_manifest_provider=_manifest_provider,
    )


def _builtin_descriptors() -> tuple[ModelFamilyDescriptor, ...]:
    return (
        _descriptor("dense_cnn", ("cnn",), (CROP_INPUT, DENSE_PLACE_POLICY, REGRET_HEAD), _build_crop_model, "dense_cnn", ("policy", "value")),
        _descriptor("restnet", (), (CROP_INPUT, DENSE_PLACE_POLICY, REGRET_HEAD), _build_crop_model, "restnet", ("policy", "value")),
        _descriptor("graph_hybrid", ("graph", "graph_hybrid_0"), (CROP_INPUT, DENSE_PLACE_POLICY, SPARSE_PLACE_POLICY, JOINT_PAIR_POLICY, REGRET_HEAD), _build_crop_model, "graph_hybrid", ("policy", "sparse_policy", "pair_policy", "value")),
        _descriptor("global_xattn", ("global_xattn_0",), (GLOBAL_GRAPH_INPUT, GLOBAL_PLACE_POLICY, PAIR_FIRST_POLICY, PAIR_SECOND_POLICY, JOINT_PAIR_POLICY, REGRET_HEAD), _build_global_model, "global_graph", ("policy_place", "policy_pair_first", "policy_pair_second", "policy_pair_joint", "value")),
        _descriptor("global_line_window", ("global_line_window_0",), (GLOBAL_GRAPH_INPUT, GLOBAL_PLACE_POLICY, PAIR_FIRST_POLICY, PAIR_SECOND_POLICY, JOINT_PAIR_POLICY, REGRET_HEAD), _build_global_model, "global_graph", ("policy_place", "policy_pair_first", "policy_pair_second", "policy_pair_joint", "value")),
        _descriptor("global_relation_graph", ("global_graph_option1", "global_pair_twostage_0", "global_graph_full_0", "global_hybrid_action_0", "global_graph768_champion"), (GLOBAL_GRAPH_INPUT, GLOBAL_PLACE_POLICY, PAIR_FIRST_POLICY, PAIR_SECOND_POLICY, JOINT_PAIR_POLICY, REGRET_HEAD), _build_global_model, "global_graph", ("policy_place", "policy_pair_first", "policy_pair_second", "policy_pair_joint", "value")),
    )


__all__ = [
    "InferenceAdapter",
    "InferenceManifest",
    "LossPlan",
    "PolicyProvider",
    "REQUIRED_MODEL_KINDS",
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


REGISTRY = default_registry()
