"""Typed model recipes for Phase 08 autotune."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from hexorl.contracts.pair_strategy import PAIR_STRATEGY_REGISTRY
from hexorl.contracts.identity import stable_digest
from hexorl.models.factory import get_model_registry
from hexorl.models.specs import MODEL_SPEC_VERSION, model_spec_from_config


@dataclass(frozen=True)
class ModelRecipe:
    recipe_id: str
    model_family: str
    channels: int = 16
    blocks: int = 1
    heads: tuple[str, ...] = ("policy", "value")
    attention_positions: tuple[int, ...] = ()
    attention_heads: int = 8
    attention_mlp_ratio: float = 2.0
    graph_layers: int = 1
    graph_token_budget: int = 256
    candidate_budget: int = 128
    sparse_policy: bool = False
    sparse_prior_stage: int = 0
    sparse_prior_mix: float = 0.25
    pair_strategy: str = "none"
    pair_strategy_max_pairs: int = 0
    inference_protocol_version: int = 1
    model_spec_version: int = MODEL_SPEC_VERSION
    seed: int = 0

    def __post_init__(self) -> None:
        get_model_registry().resolve(self.model_family)
        if self.channels <= 0 or self.blocks <= 0:
            raise ValueError("recipe channels and blocks must be positive")
        if self.attention_heads <= 0:
            raise ValueError("recipe attention_heads must be positive")
        if self.channels % self.attention_heads != 0:
            raise ValueError("recipe channels must be divisible by attention_heads")
        invalid_positions = [pos for pos in self.attention_positions if pos < 1 or pos > self.blocks]
        if invalid_positions:
            raise ValueError(f"recipe attention_positions outside block range: {invalid_positions}")
        if self.attention_mlp_ratio <= 0.0:
            raise ValueError("recipe attention_mlp_ratio must be positive")
        if self.graph_layers <= 0:
            raise ValueError("recipe graph_layers must be positive")
        if not 16 <= self.graph_token_budget <= 768:
            raise ValueError("recipe graph_token_budget must be in [16, 768]")
        if not 1 <= self.candidate_budget <= 512:
            raise ValueError("recipe candidate_budget must be in [1, 512]")
        pair_descriptor = PAIR_STRATEGY_REGISTRY.resolve(self.pair_strategy)
        object.__setattr__(self, "pair_strategy", pair_descriptor.name)
        if self.sparse_prior_stage not in {0, 1, 2}:
            raise ValueError("recipe sparse_prior_stage must be 0, 1, or 2")
        if not 0.0 <= self.sparse_prior_mix <= 1.0:
            raise ValueError("recipe sparse_prior_mix must be in [0, 1]")
        if ("sparse_policy" in self.heads or "pair_policy" in self.heads) and not self.sparse_policy:
            raise ValueError("recipe sparse heads require sparse_policy=True")
        pair_descriptor.validate_config(max_pairs=self.pair_strategy_max_pairs, pair_prior_mix=1.0)
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
            "attention_positions": list(self.attention_positions),
            "attention_heads": self.attention_heads,
            "attention_mlp_ratio": self.attention_mlp_ratio,
            "graph_layers": self.graph_layers,
            "graph_token_budget": self.graph_token_budget,
            "candidate_budget": self.candidate_budget,
            "sparse_policy": self.sparse_policy,
            "sparse_prior_stage": self.sparse_prior_stage,
            "sparse_prior_mix": self.sparse_prior_mix,
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
            "attention_positions": list(self.attention_positions),
            "attention_heads": self.attention_heads,
            "attention_mlp_ratio": self.attention_mlp_ratio,
            "graph_layers": self.graph_layers,
            "graph_token_budget": self.graph_token_budget,
            "candidate_budget": self.candidate_budget,
            "sparse_policy": self.sparse_policy,
            "sparse_prior_stage": self.sparse_prior_stage,
            "sparse_prior_mix": self.sparse_prior_mix,
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


@dataclass(frozen=True)
class ConfigSectionTransform:
    name: str
    run: Mapping[str, Any] = field(default_factory=dict)
    selfplay: Mapping[str, Any] = field(default_factory=dict)
    inference: Mapping[str, Any] = field(default_factory=dict)
    buffer: Mapping[str, Any] = field(default_factory=dict)
    train: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)


CONFIG_ARCHITECTURE_BY_FAMILY = {
    "dense_cnn": "cnn",
    "restnet": "restnet",
    "graph_hybrid": "graph_hybrid_0",
    "global_xattn": "global_xattn_0",
    "global_line_window": "global_line_window_0",
    "global_relation_graph": "global_graph_option1",
}


def config_from_recipe(
    base_cfg: Any,
    recipe: ModelRecipe,
    *,
    section_transform: ConfigSectionTransform | None = None,
) -> Any:
    """Return a validated config produced from a typed model recipe."""

    payload = base_cfg.model_dump(mode="python")
    model_payload = payload["model"]
    model_payload.update(
        {
            "architecture": CONFIG_ARCHITECTURE_BY_FAMILY[recipe.model_family],
            "channels": recipe.channels,
            "blocks": recipe.blocks,
            "heads": list(recipe.heads),
            "attention_positions": list(recipe.attention_positions),
            "attention_heads": recipe.attention_heads,
            "attention_mlp_ratio": recipe.attention_mlp_ratio,
            "graph_layers": recipe.graph_layers,
            "graph_token_budget": recipe.graph_token_budget,
            "candidate_budget": recipe.candidate_budget,
            "sparse_policy": recipe.sparse_policy,
            "sparse_prior_stage": recipe.sparse_prior_stage,
            "sparse_prior_mix": recipe.sparse_prior_mix,
            "pair_strategy": recipe.pair_strategy,
            "pair_strategy_max_pairs": recipe.pair_strategy_max_pairs,
        }
    )
    if section_transform is not None:
        for section_name in ("run", "selfplay", "inference", "buffer", "train", "runtime"):
            updates = dict(getattr(section_transform, section_name))
            _merge_section(payload, section_name, updates)
    cfg_type = type(base_cfg)
    return cfg_type.model_validate(payload)


def recipe_from_config(base_cfg: Any, *, recipe_id: str) -> ModelRecipe:
    spec = model_spec_from_config(base_cfg)
    model_cfg = base_cfg.model
    return ModelRecipe(
        recipe_id=recipe_id,
        model_family=spec.kind,
        channels=int(model_cfg.channels),
        blocks=int(model_cfg.blocks),
        heads=tuple(model_cfg.heads),
        attention_positions=tuple(int(pos) for pos in model_cfg.attention_positions),
        attention_heads=int(model_cfg.attention_heads),
        attention_mlp_ratio=float(model_cfg.attention_mlp_ratio),
        graph_layers=int(model_cfg.graph_layers),
        graph_token_budget=int(model_cfg.graph_token_budget),
        candidate_budget=int(model_cfg.candidate_budget),
        sparse_policy=bool(model_cfg.sparse_policy),
        sparse_prior_stage=int(model_cfg.sparse_prior_stage),
        sparse_prior_mix=float(model_cfg.sparse_prior_mix),
        pair_strategy=str(model_cfg.pair_strategy),
        pair_strategy_max_pairs=int(model_cfg.pair_strategy_max_pairs),
        seed=int(base_cfg.run.seed),
    )


def _merge_section(payload: dict[str, Any], section_name: str, updates: Mapping[str, Any]) -> None:
    section = payload[section_name]
    unknown = sorted(set(updates) - set(section))
    if unknown:
        raise ValueError(f"{section_name} transform contains unknown fields: {unknown}")
    section.update(dict(updates))
