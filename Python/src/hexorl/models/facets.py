"""Shared model family facet implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

from hexorl.models.capabilities import CapabilitySet
from hexorl.models.inference_contracts import (
    ModelInferenceContract,
    make_crop_contract,
    make_graph_contract,
)
from hexorl.models.registry import FamilyComponents, ModelFamilyDescriptor
from hexorl.models.specs import MODEL_SPEC_VERSION, ModelParams, ModelSpec


@dataclass(frozen=True)
class InferenceAdapter:
    spec: ModelSpec
    contract: ModelInferenceContract


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
    output_heads: tuple[str, ...],
    graph: bool,
) -> Callable[[ModelSpec, Any, nn.Module], InferenceAdapter]:
    def _adapter(spec: ModelSpec, cfg: Any, model: nn.Module) -> InferenceAdapter:
        contract = (
            make_graph_contract(family_name=family_name, cfg=cfg, required_heads=required_heads, output_heads=output_heads)
            if graph
            else make_crop_contract(
                family_name=family_name,
                capabilities=tuple(capabilities.to_manifest()),
                cfg=cfg,
                required_heads=required_heads,
                output_heads=output_heads,
                graph_hybrid=capabilities.has("SPARSE_PLACE_POLICY") or capabilities.has("JOINT_PAIR_POLICY"),
            )
        )
        return InferenceAdapter(spec=spec, contract=contract)

    return _adapter


def make_inference_contract_provider(
    *,
    family_name: str,
    capabilities: CapabilitySet,
    required_heads: tuple[str, ...],
    output_heads: tuple[str, ...],
    graph: bool,
) -> Callable[[ModelSpec, Any], ModelInferenceContract]:
    def _provider(spec: ModelSpec, cfg: Any) -> ModelInferenceContract:
        if graph:
            return make_graph_contract(family_name=family_name, cfg=cfg, required_heads=required_heads, output_heads=output_heads)
        return make_crop_contract(
            family_name=family_name,
            capabilities=tuple(capabilities.to_manifest()),
            cfg=cfg,
            required_heads=required_heads,
            output_heads=output_heads,
            graph_hybrid=capabilities.has("SPARSE_PLACE_POLICY") or capabilities.has("JOINT_PAIR_POLICY"),
        )

    return _provider


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
            "model_inference_contract": inference_adapter_factory(spec, cfg, torch.nn.Identity()).contract.canonical_dict(),
        }

    return _provider


def make_descriptor(
    *,
    name: str,
    aliases: tuple[str, ...],
    capabilities: CapabilitySet,
    builder: Callable[[ModelSpec, Any], nn.Module],
    components: FamilyComponents,
    params_schema: type[ModelParams],
    required_heads: tuple[str, ...],
    graph: bool,
) -> ModelFamilyDescriptor:
    inference_adapter_factory = make_inference_adapter(
        family_name=name,
        capabilities=capabilities,
        required_heads=required_heads,
        output_heads=components.heads,
        graph=graph,
    )
    inference_contract_factory = make_inference_contract_provider(
        family_name=name,
        capabilities=capabilities,
        required_heads=required_heads,
        output_heads=components.heads,
        graph=graph,
    )
    return ModelFamilyDescriptor(
        name=name,
        aliases=frozenset(aliases),
        capabilities=capabilities,
        spec_schema=ModelSpec,
        params_schema=params_schema,
        components=components,
        model_builder=builder,
        train_adapter_factory=train_adapter,
        inference_adapter_factory=inference_adapter_factory,
        inference_contract_factory=inference_contract_factory,
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
