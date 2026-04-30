"""Typed model recipes for Phase 08 autotune."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from hexorl.contracts.identity import stable_digest
from hexorl.models.factory import get_model_registry
from hexorl.models.specs import MODEL_SPEC_VERSION


@dataclass(frozen=True)
class ModelRecipe:
    recipe_id: str
    model_family: str
    channels: int = 16
    blocks: int = 1
    heads: tuple[str, ...] = ("policy", "value")
    graph_layers: int = 1
    graph_token_budget: int = 256
    candidate_budget: int = 128
    pair_strategy: str = "none"
    pair_strategy_max_pairs: int = 0
    inference_protocol_version: int = 1
    model_spec_version: int = MODEL_SPEC_VERSION
    seed: int = 0

    def __post_init__(self) -> None:
        get_model_registry().resolve(self.model_family)
        if self.channels <= 0 or self.blocks <= 0:
            raise ValueError("recipe channels and blocks must be positive")
        if self.graph_layers <= 0:
            raise ValueError("recipe graph_layers must be positive")
        if not 16 <= self.graph_token_budget <= 768:
            raise ValueError("recipe graph_token_budget must be in [16, 768]")
        if not 1 <= self.candidate_budget <= 512:
            raise ValueError("recipe candidate_budget must be in [1, 512]")
        if self.pair_strategy not in {"none", "diagnostic_full_pair"}:
            raise ValueError("recipe pair_strategy must be typed and registered")
        if self.pair_strategy == "none" and self.pair_strategy_max_pairs != 0:
            raise ValueError("pair_strategy_max_pairs must be zero when pair_strategy is none")
        if self.inference_protocol_version != 1:
            raise ValueError("unsupported inference protocol version")

    @property
    def config_hash(self) -> str:
        return stable_digest(("ModelRecipe", self._hash_payload()))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "model_family": self.model_family,
            "channels": self.channels,
            "blocks": self.blocks,
            "heads": list(self.heads),
            "graph_layers": self.graph_layers,
            "graph_token_budget": self.graph_token_budget,
            "candidate_budget": self.candidate_budget,
            "pair_strategy": self.pair_strategy,
            "pair_strategy_max_pairs": self.pair_strategy_max_pairs,
            "inference_protocol_version": self.inference_protocol_version,
            "model_spec_version": self.model_spec_version,
            "seed": self.seed,
            "config_hash": self.config_hash,
        }

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "model_family": self.model_family,
            "channels": self.channels,
            "blocks": self.blocks,
            "heads": list(self.heads),
            "graph_layers": self.graph_layers,
            "graph_token_budget": self.graph_token_budget,
            "candidate_budget": self.candidate_budget,
            "pair_strategy": self.pair_strategy,
            "pair_strategy_max_pairs": self.pair_strategy_max_pairs,
            "inference_protocol_version": self.inference_protocol_version,
            "model_spec_version": self.model_spec_version,
            "seed": self.seed,
        }

    def transform(self, transform: "RecipeTransform") -> "ModelRecipe":
        updates = transform.updates
        forbidden = sorted(set(updates) - set(self.__dataclass_fields__))
        if forbidden:
            raise ValueError(f"recipe transform contains unknown typed fields: {forbidden}")
        if "model_family" in updates:
            raise ValueError("recipe transforms cannot mutate model_family")
        return replace(self, **updates, recipe_id=f"{self.recipe_id}:{transform.name}")


@dataclass(frozen=True)
class RecipeTransform:
    name: str
    updates: dict[str, Any]

    @classmethod
    def from_raw_config(cls, raw: dict[str, Any]) -> "RecipeTransform":
        raise TypeError("autotune must mutate ModelRecipe through typed RecipeTransform values, not untyped dictionaries")
