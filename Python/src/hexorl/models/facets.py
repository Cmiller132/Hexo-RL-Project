"""Shared model family facet implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

from hexorl.models.capabilities import CapabilitySet
from hexorl.models.registry import FamilyComponents, ModelFamilyDescriptor
from hexorl.models.specs import MODEL_SPEC_VERSION, ModelSpec


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


def crop_loss_plan(spec: ModelSpec, cfg: Any) -> LossPlan:
    return LossPlan(
        weights=dict(getattr(cfg.train, "loss_weights", {})),
        masked_heads=("sparse_policy", "pair_policy"),
    )


def global_loss_plan(spec: ModelSpec, cfg: Any) -> LossPlan:
    return LossPlan(
        weights=dict(getattr(cfg.train, "loss_weights", {})),
        masked_heads=("policy_place", "policy_pair_first", "policy_pair_second", "policy_pair_joint"),
    )


def train_adapter(spec: ModelSpec, cfg: Any, model: nn.Module, *, device: torch.device):
    from hexorl.train.adapters import TrainAdapter

    loss_plan = global_loss_plan(spec, cfg) if spec.is_global_graph else crop_loss_plan(spec, cfg)
    return TrainAdapter(spec=spec, cfg=cfg, model=model, device=device, loss_plan=loss_plan)


def recipe_provider(spec_kind: str) -> Callable[[Any], dict[str, Any]]:
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


def tune_space_provider(spec_kind: str) -> Callable[[Any], dict[str, Any]]:
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


def make_inference_adapter(
    *,
    family_name: str,
    capabilities: CapabilitySet,
    required_heads: tuple[str, ...],
    graph: bool,
) -> Callable[[ModelSpec, Any, nn.Module], InferenceAdapter]:
    def _adapter(spec: ModelSpec, cfg: Any, model: nn.Module) -> InferenceAdapter:
        manifest = InferenceManifest(
            protocol_version=1,
            request_kind="global_graph" if graph else "crop",
            model_family=family_name,
            model_spec_version=MODEL_SPEC_VERSION,
            input_contract="global_graph_v1" if graph else "crop_tensor_v1",
            output_contract="global_place_value_v1" if graph else "dense_place_value_v1",
            action_contract="legal_action_table_v1",
            graph_schema_version=1 if graph else None,
            relation_schema_version=1 if graph else None,
            max_tokens=1024 if graph else None,
            max_legal_rows=1024 if graph else 1089,
            max_pair_rows=4096 if graph else int(getattr(cfg.model, "pair_strategy_max_pairs", 0)),
            required_heads=required_heads,
            optional_heads=tuple(sorted(set(getattr(cfg.model, "heads", [])) - set(required_heads))),
            capabilities=tuple(capabilities.to_manifest()),
        )
        return InferenceAdapter(spec=spec, manifest=manifest)

    return _adapter


def make_policy_provider(capabilities: CapabilitySet) -> Callable[[ModelSpec, Any, nn.Module], PolicyProvider]:
    def _provider(spec: ModelSpec, cfg: Any, model: nn.Module) -> PolicyProvider:
        return PolicyProvider(model=model, capabilities=capabilities)

    return _provider


def make_checkpoint_manifest_provider(
    *,
    family_name: str,
    capabilities: CapabilitySet,
    inference_adapter_factory: Callable[[ModelSpec, Any, nn.Module], InferenceAdapter],
) -> Callable[[ModelSpec, Any], dict[str, Any]]:
    def _provider(spec: ModelSpec, cfg: Any) -> dict[str, Any]:
        return {
            "model_family": family_name,
            "model_spec_version": MODEL_SPEC_VERSION,
            "model_spec": spec.manifest(),
            "capabilities": capabilities.to_manifest(),
            "inference_protocol": inference_adapter_factory(spec, cfg, torch.nn.Identity()).manifest.to_dict(),
        }

    return _provider


def make_descriptor(
    *,
    name: str,
    aliases: tuple[str, ...],
    capabilities: CapabilitySet,
    builder: Callable[[ModelSpec, Any], nn.Module],
    components: FamilyComponents,
    required_heads: tuple[str, ...],
    graph: bool,
) -> ModelFamilyDescriptor:
    inference_adapter_factory = make_inference_adapter(
        family_name=name,
        capabilities=capabilities,
        required_heads=required_heads,
        graph=graph,
    )
    return ModelFamilyDescriptor(
        name=name,
        aliases=frozenset(aliases),
        capabilities=capabilities,
        spec_schema=ModelSpec,
        components=components,
        model_builder=builder,
        train_adapter_factory=train_adapter,
        inference_adapter_factory=inference_adapter_factory,
        policy_provider_factory=make_policy_provider(capabilities),
        loss_plan_provider=global_loss_plan if graph else crop_loss_plan,
        recipe_provider=recipe_provider(name),
        tune_space_provider=tune_space_provider(name),
        checkpoint_manifest_provider=make_checkpoint_manifest_provider(
            family_name=name,
            capabilities=capabilities,
            inference_adapter_factory=inference_adapter_factory,
        ),
    )
