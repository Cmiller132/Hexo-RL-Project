"""Model family registry and descriptor/facet contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

import torch.nn as nn

from hexorl.models.capabilities import CapabilitySet
from hexorl.models.specs import ModelSpec


class ModelBuilder(Protocol):
    def __call__(self, spec: ModelSpec, cfg: Any, *, device: Any | None = None, inference: bool = False) -> nn.Module: ...


class TrainAdapterFactory(Protocol):
    def __call__(self, spec: ModelSpec, cfg: Any, model: nn.Module, *, device: Any) -> Any: ...


class InferenceAdapterFactory(Protocol):
    def __call__(self, spec: ModelSpec, cfg: Any, model: nn.Module) -> Any: ...


@dataclass(frozen=True)
class FamilyComponents:
    trunk: str
    heads: tuple[str, ...]


@dataclass(frozen=True)
class ModelFamilyDescriptor:
    name: str
    aliases: frozenset[str]
    capabilities: CapabilitySet
    spec_schema: type[ModelSpec]
    components: FamilyComponents
    model_builder: ModelBuilder
    train_adapter_factory: TrainAdapterFactory
    inference_adapter_factory: InferenceAdapterFactory
    policy_provider_factory: Callable[[ModelSpec, Any, nn.Module], Any]
    loss_plan_provider: Callable[[ModelSpec, Any], Any]
    recipe_provider: Callable[[Any], dict[str, Any]]
    tune_space_provider: Callable[[Any], dict[str, Any]]
    checkpoint_manifest_provider: Callable[[ModelSpec, Any], dict[str, Any]]

    def validate_complete(self) -> None:
        required = (
            self.model_builder,
            self.train_adapter_factory,
            self.inference_adapter_factory,
            self.policy_provider_factory,
            self.loss_plan_provider,
            self.recipe_provider,
            self.tune_space_provider,
            self.checkpoint_manifest_provider,
        )
        if any(value is None for value in required):
            raise ValueError(f"model family {self.name} has an incomplete descriptor")


@dataclass
class ModelFamilyRegistry:
    _families: dict[str, ModelFamilyDescriptor] = field(default_factory=dict)
    _aliases: dict[str, str] = field(default_factory=dict)

    def register(self, descriptor: ModelFamilyDescriptor) -> None:
        descriptor.validate_complete()
        if descriptor.name in self._families:
            raise ValueError(f"model family already registered: {descriptor.name}")
        self._families[descriptor.name] = descriptor
        for alias in descriptor.aliases:
            if alias in self._aliases:
                raise ValueError(f"model family alias already registered: {alias}")
            self._aliases[alias] = descriptor.name

    def resolve(self, spec_or_name: ModelSpec | str) -> ModelFamilyDescriptor:
        name = spec_or_name.kind if isinstance(spec_or_name, ModelSpec) else str(spec_or_name)
        canonical = self._aliases.get(name, name)
        try:
            return self._families[canonical]
        except KeyError as exc:
            raise ValueError(f"unregistered model family: {name}") from exc

    def descriptors(self) -> tuple[ModelFamilyDescriptor, ...]:
        return tuple(self._families[name] for name in sorted(self._families))

    def names(self) -> tuple[str, ...]:
        return tuple(descriptor.name for descriptor in self.descriptors())

    def capability_matrix(self) -> dict[str, list[str]]:
        return {
            descriptor.name: descriptor.capabilities.to_manifest()
            for descriptor in self.descriptors()
        }

    def clone(self) -> "ModelFamilyRegistry":
        other = ModelFamilyRegistry()
        for descriptor in self.descriptors():
            other.register(descriptor)
        return other


def assert_required_families(registry: ModelFamilyRegistry, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(registry.names()))
    if missing:
        raise ValueError(f"missing required model families: {missing}")
