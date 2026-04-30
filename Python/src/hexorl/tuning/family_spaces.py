"""Family-specific typed recipe spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hexorl.models.factory import get_model_registry
from hexorl.models.specs import GLOBAL_MODEL_KINDS
from hexorl.tuning.recipes import ModelRecipe, RecipeTransform


@dataclass(frozen=True)
class FamilySpace:
    family: str
    channel_choices: tuple[int, ...]
    block_choices: tuple[int, ...]
    graph_layer_choices: tuple[int, ...]
    graph_token_budget_choices: tuple[int, ...]
    candidate_budget_choices: tuple[int, ...]
    pair_strategy_choices: tuple[str, ...]

    def default_recipe(self, *, seed: int = 0) -> ModelRecipe:
        heads = ("policy_place", "value") if self.family in GLOBAL_MODEL_KINDS else ("policy", "value")
        if self.family == "graph_hybrid":
            heads = ("policy", "sparse_policy", "pair_policy", "value")
        return ModelRecipe(
            recipe_id=f"{self.family}:default",
            model_family=self.family,
            channels=self.channel_choices[0],
            blocks=self.block_choices[0],
            heads=heads,
            graph_layers=self.graph_layer_choices[0],
            graph_token_budget=self.graph_token_budget_choices[0],
            candidate_budget=self.candidate_budget_choices[0],
            pair_strategy="none",
            pair_strategy_max_pairs=0,
            seed=seed,
        )

    def transforms(self) -> tuple[RecipeTransform, ...]:
        return (
            RecipeTransform("wider", {"channels": self.channel_choices[-1]}),
            RecipeTransform("deeper", {"blocks": self.block_choices[-1]}),
            RecipeTransform("larger_runtime_rows", {"candidate_budget": self.candidate_budget_choices[-1]}),
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "channel_choices": list(self.channel_choices),
            "block_choices": list(self.block_choices),
            "graph_layer_choices": list(self.graph_layer_choices),
            "graph_token_budget_choices": list(self.graph_token_budget_choices),
            "candidate_budget_choices": list(self.candidate_budget_choices),
            "pair_strategy_choices": list(self.pair_strategy_choices),
        }


def family_space(family: str) -> FamilySpace:
    descriptor = get_model_registry().resolve(family)
    is_global = descriptor.name in GLOBAL_MODEL_KINDS
    is_graph_hybrid = descriptor.name == "graph_hybrid"
    return FamilySpace(
        family=descriptor.name,
        channel_choices=(16, 32),
        block_choices=(1, 2),
        graph_layer_choices=(1, 2) if is_global else (1,),
        graph_token_budget_choices=(256, 512) if is_global else (256,),
        candidate_budget_choices=(128, 256, 512) if is_graph_hybrid else (128, 256),
        pair_strategy_choices=("none", "diagnostic_full_pair") if is_global or is_graph_hybrid else ("none",),
    )


def all_family_spaces() -> dict[str, FamilySpace]:
    return {name: family_space(name) for name in get_model_registry().names()}


def valid_recipe_examples() -> dict[str, ModelRecipe]:
    examples: dict[str, ModelRecipe] = {}
    for family, space in all_family_spaces().items():
        examples[family] = space.default_recipe(seed=17)
    return examples
