"""Registry-owned pair strategy descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, TypeAlias

from hexorl.contracts.pairs import PairGenerationMode
from hexorl.contracts.validation import ContractValidationError

PairStrategyName: TypeAlias = str
PairStrategyCapField = Literal["max_root_pair_rows", "max_full_pair_rows"]


@dataclass(frozen=True)
class PairStrategySpec:
    name: PairStrategyName = "none"
    enabled_sources: tuple[str, ...] = ()
    root_enabled: bool = False
    leaf_enabled: bool = False
    max_root_pair_rows: int = 0
    max_leaf_pair_rows: int = 0
    max_full_pair_rows: int = 0
    chunk_size: int = 0
    phase_eligibility: tuple[str, ...] = ("root",)
    known_first_required: bool = False
    diagnostic: bool = False
    telemetry_level: str = "summary"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).lower())
        for attr in ("max_root_pair_rows", "max_leaf_pair_rows", "max_full_pair_rows", "chunk_size"):
            object.__setattr__(self, attr, int(getattr(self, attr)))
        object.__setattr__(self, "root_enabled", bool(self.root_enabled))
        object.__setattr__(self, "leaf_enabled", bool(self.leaf_enabled))
        object.__setattr__(self, "diagnostic", bool(self.diagnostic))
        object.__setattr__(self, "enabled_sources", tuple(str(item) for item in self.enabled_sources))
        object.__setattr__(self, "phase_eligibility", tuple(str(item) for item in self.phase_eligibility))
        object.__setattr__(self, "known_first_required", bool(self.known_first_required))
        object.__setattr__(self, "telemetry_level", str(self.telemetry_level))

    @property
    def root_pair_row_cap(self) -> int:
        return max(int(self.max_root_pair_rows), int(self.max_full_pair_rows))


@dataclass(frozen=True)
class PairStrategyDescriptor:
    name: str
    aliases: frozenset[str] = frozenset()
    generation_mode: PairGenerationMode = "none"
    root_enabled: bool = False
    leaf_enabled: bool = False
    diagnostic: bool = False
    max_pair_rows_field: PairStrategyCapField = "max_root_pair_rows"
    chunk_cap: int = 0
    allow_full: bool = False
    enabled_sources: tuple[str, ...] = ()
    phase_eligibility: tuple[str, ...] = ("root",)
    known_first_required: bool = False
    telemetry_level: str = "summary"
    requires_pair_head: bool = False
    recipe_family_tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        name = str(self.name).lower()
        aliases = frozenset(str(alias).lower() for alias in self.aliases)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "root_enabled", bool(self.root_enabled))
        object.__setattr__(self, "leaf_enabled", bool(self.leaf_enabled))
        object.__setattr__(self, "diagnostic", bool(self.diagnostic))
        object.__setattr__(self, "chunk_cap", int(self.chunk_cap))
        object.__setattr__(self, "allow_full", bool(self.allow_full))
        object.__setattr__(self, "enabled_sources", tuple(str(item) for item in self.enabled_sources))
        object.__setattr__(self, "phase_eligibility", tuple(str(item) for item in self.phase_eligibility))
        object.__setattr__(self, "known_first_required", bool(self.known_first_required))
        object.__setattr__(self, "telemetry_level", str(self.telemetry_level))
        object.__setattr__(self, "requires_pair_head", bool(self.requires_pair_head))
        object.__setattr__(self, "recipe_family_tags", frozenset(str(tag) for tag in self.recipe_family_tags))

    @property
    def scoring_enabled(self) -> bool:
        return self.root_enabled or self.leaf_enabled

    def matches_recipe_family(self, family_tags: set[str]) -> bool:
        return bool(self.recipe_family_tags & family_tags)

    def build_spec(self, *, max_pairs: int) -> PairStrategySpec:
        cap = int(max_pairs)
        values = {
            "name": self.name,
            "enabled_sources": self.enabled_sources,
            "root_enabled": self.root_enabled,
            "leaf_enabled": self.leaf_enabled,
            "max_root_pair_rows": 0,
            "max_leaf_pair_rows": 0,
            "max_full_pair_rows": 0,
            "chunk_size": 0 if cap <= 0 else min(cap, self.chunk_cap),
            "phase_eligibility": self.phase_eligibility,
            "known_first_required": self.known_first_required,
            "diagnostic": self.diagnostic,
            "telemetry_level": self.telemetry_level,
        }
        if self.scoring_enabled:
            values[self.max_pair_rows_field] = cap
        spec = PairStrategySpec(**values)
        self.validate_spec(spec)
        return spec

    def build_table_strategy(self, *, max_pairs: int):
        from hexorl.contracts.pairs import PairStrategy

        return PairStrategy(
            generation_mode=self.generation_mode,
            max_pairs=0 if not self.scoring_enabled else int(max_pairs),
            allow_full=self.allow_full,
        )

    def validate_config(self, *, max_pairs: int, pair_prior_mix: float) -> None:
        cap = int(max_pairs)
        if not self.scoring_enabled:
            if cap != 0:
                raise ValueError("model.pair_strategy_max_pairs must be 0 when model.pair_strategy='none'")
            return
        if float(pair_prior_mix) <= 0.0:
            raise ValueError("non-none model.pair_strategy requires model.pair_prior_mix > 0")
        if cap <= 0:
            raise ValueError("non-none model.pair_strategy requires model.pair_strategy_max_pairs > 0")

    def validate_spec(self, spec: PairStrategySpec) -> None:
        if spec.name != self.name:
            raise ContractValidationError(
                f"pair strategy spec name {spec.name!r} does not match descriptor {self.name!r}",
                owner="PairStrategyDescriptor",
            )
        expected = {
            "root_enabled": self.root_enabled,
            "leaf_enabled": self.leaf_enabled,
            "diagnostic": self.diagnostic,
        }
        for attr, value in expected.items():
            if bool(getattr(spec, attr)) != value:
                raise ContractValidationError(
                    f"{self.name} pair strategy has invalid {attr}",
                    owner="PairStrategyDescriptor",
                )
        caps = {
            "max_root_pair_rows": int(spec.max_root_pair_rows),
            "max_leaf_pair_rows": int(spec.max_leaf_pair_rows),
            "max_full_pair_rows": int(spec.max_full_pair_rows),
        }
        if not self.scoring_enabled:
            if any(caps.values()):
                raise ContractValidationError("none pair strategy must not enable or cap scoring", owner="PairStrategyDescriptor")
            return
        required_cap = caps[self.max_pair_rows_field]
        if required_cap <= 0:
            raise ContractValidationError(
                f"{self.name} requires positive {self.max_pair_rows_field}",
                owner="PairStrategyDescriptor",
            )
        for attr, value in caps.items():
            if attr != self.max_pair_rows_field and value != 0:
                raise ContractValidationError(
                    f"{self.name} pair strategy must not set {attr}",
                    owner="PairStrategyDescriptor",
                )
        if self.diagnostic and (not spec.root_enabled or spec.leaf_enabled or spec.max_full_pair_rows <= 0):
            raise ContractValidationError(
                f"{self.name} must be diagnostic root-only with max_full_pair_rows",
                owner="PairStrategyDescriptor",
            )


@dataclass(frozen=True)
class PairStrategyRegistry:
    descriptors: Mapping[str, PairStrategyDescriptor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        by_name: dict[str, PairStrategyDescriptor] = {}
        for descriptor in self.descriptors.values():
            for key in (descriptor.name, *descriptor.aliases):
                if key in by_name:
                    raise ContractValidationError(f"duplicate pair strategy registration {key!r}", owner="PairStrategyRegistry")
                by_name[key] = descriptor
        object.__setattr__(self, "_by_name", by_name)
        object.__setattr__(self, "descriptors", {descriptor.name: descriptor for descriptor in by_name.values()})

    def resolve(self, name: str | PairStrategyDescriptor | PairStrategySpec) -> PairStrategyDescriptor:
        if isinstance(name, PairStrategyDescriptor):
            return name
        raw = name.name if isinstance(name, PairStrategySpec) else name
        key = str(raw).lower()
        try:
            return self._by_name[key]
        except KeyError as exc:
            raise ContractValidationError(f"unknown pair strategy {key!r}", owner="PairStrategyRegistry") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.descriptors))

    def aliases(self) -> dict[str, str]:
        return {
            alias: descriptor.name
            for descriptor in self.descriptors.values()
            for alias in sorted(descriptor.aliases)
        }

    def build_spec(self, name: str, *, max_pairs: int) -> PairStrategySpec:
        return self.resolve(name).build_spec(max_pairs=max_pairs)

    def validate_spec(self, spec: PairStrategySpec) -> PairStrategySpec:
        descriptor = self.resolve(spec)
        descriptor.validate_spec(spec)
        return spec

    def normalize_name(self, name: str) -> str:
        return self.resolve(name).name

    def recipe_names_for_family(self, *, family: str, is_global: bool, is_graph_hybrid: bool) -> tuple[str, ...]:
        tags = {str(family)}
        if is_global:
            tags.add("global")
        if is_graph_hybrid:
            tags.add("graph_hybrid")
        return tuple(
            descriptor.name
            for descriptor in self.descriptors.values()
            if descriptor.matches_recipe_family(tags)
        )
