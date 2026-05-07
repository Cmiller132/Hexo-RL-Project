"""Typed scout recipes and validated recipe-to-Config materialization."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hexorl.config import Config
from hexorl.config.schema import AUTOTUNE_PAIR_STRATEGY_MODES
from hexorl.models.registry import global_graph_architecture_ids, normalize_architecture_id


RECIPE_SCHEMA_VERSION = 1
PairMode = Literal["none", "root_pair_mcts", "full_pair_mcts"]
_PAIR_HEADS = ("policy_pair_first", "policy_pair_joint", "policy_pair_second")


class ModelRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    architecture_id: str
    architecture_contract_version: str = "global_graph:v1"
    channels: int = 128
    blocks: int = 16
    attention_heads: int = 8
    graph_token_set: str = "graph256_cells"
    graph_token_budget: int = 256
    graph_layers: int = 1
    candidate_budget: int = 256
    head_bundle: str = "policy_value"
    output_contract: str = "global_legal:v1"
    output_heads: list[str] = Field(default_factory=lambda: ["policy_place", "value"])

    @model_validator(mode="after")
    def validate_model_recipe(self) -> "ModelRecipe":
        self.architecture_id = normalize_architecture_id(self.architecture_id)
        if self.architecture_id not in set(global_graph_architecture_ids()):
            raise ValueError("ModelRecipe.architecture_id must be a global graph architecture")
        if self.channels <= 0:
            raise ValueError("ModelRecipe.channels must be positive")
        if self.blocks <= 0:
            raise ValueError("ModelRecipe.blocks must be positive")
        if self.attention_heads <= 0:
            raise ValueError("ModelRecipe.attention_heads must be positive")
        if self.channels % self.attention_heads != 0:
            raise ValueError("ModelRecipe.channels must be divisible by attention_heads")
        if not 16 <= self.graph_token_budget <= 768:
            raise ValueError("ModelRecipe.graph_token_budget must be in [16, 768]")
        if self.graph_layers <= 0:
            raise ValueError("ModelRecipe.graph_layers must be positive")
        if self.candidate_budget <= 0:
            raise ValueError("ModelRecipe.candidate_budget must be positive")
        if not self.output_heads:
            raise ValueError("ModelRecipe.output_heads must not be empty")
        return self


class PairStrategySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    mode: PairMode = "none"
    pair_row_budget: int = 0
    pair_prior_mix: float = 0.35
    max_pair_batch_rows: int = 256
    chunk_rows: int = 128
    root_only: bool = True

    @model_validator(mode="after")
    def validate_pair_strategy(self) -> "PairStrategySpec":
        if self.mode not in AUTOTUNE_PAIR_STRATEGY_MODES:
            raise ValueError(f"PairStrategySpec.mode must be one of {list(AUTOTUNE_PAIR_STRATEGY_MODES)}")
        if self.mode == "none":
            if self.pair_row_budget != 0:
                raise ValueError("PairStrategySpec.pair_row_budget must be 0 when mode='none'")
            self.root_only = False
            return self
        if self.pair_row_budget <= 0:
            raise ValueError("PairStrategySpec.pair_row_budget must be positive for pair MCTS modes")
        if not 0.0 < self.pair_prior_mix <= 1.0:
            raise ValueError("PairStrategySpec.pair_prior_mix must be in (0, 1]")
        if self.max_pair_batch_rows <= 0 or self.chunk_rows <= 0:
            raise ValueError("PairStrategySpec batching rows must be positive")
        self.root_only = self.mode == "root_pair_mcts"
        return self


class SearchRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    full_mcts_simulations: int = 512
    pcr_low_sims: int = 128
    pcr_low_sim_prob: float = 0.75
    policy_target_top_k: int = 96
    temperature_family: str = "slow_cool"
    c_puct: float = 1.5
    c_puct_init: float = 19652.0
    dirichlet_fraction: float = 0.25
    dirichlet_alpha_scale: float = 0.3


class ScheduleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    peak_lr: float = 3e-4
    lr_multiplier: float = 1.0
    weight_decay: float = 1e-4
    recency_decay: float = 0.99
    value_loss_weight: float = 1.0
    auxiliary_loss_weight: float = 0.05
    graph_loss_weight: float = 1.0
    pair_loss_weight: float = 0.05


class RuntimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    selfplay_workers: int = 0
    batch_size_per_worker: int = 0
    inference_max_batch_size: int = 0
    inference_wait_us: int = 0
    memory_safety_envelope: str = "runtime_probe_required"


class CandidateRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: int = RECIPE_SCHEMA_VERSION
    model: ModelRecipe
    pair_strategy: PairStrategySpec = Field(default_factory=PairStrategySpec)
    search: SearchRecipe = Field(default_factory=SearchRecipe)
    schedule: ScheduleSpec = Field(default_factory=ScheduleSpec)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.model.architecture_id}__{self.pair_strategy.mode}"
            f"__v{self.schema_version}"
        )

    def materialize_config(self, base_config: Config) -> Config:
        """Deep-copy ``base_config``, apply recipe fields, and revalidate."""

        data = base_config.model_copy(deep=True).model_dump(mode="json")
        model = data.setdefault("model", {})
        pair_enabled = self.pair_strategy.mode != "none"
        model.update(
            {
                "architecture": self.model.architecture_id,
                "channels": self.model.channels,
                "blocks": self.model.blocks,
                "attention_heads": self.model.attention_heads,
                "graph_token_set": self.model.graph_token_set,
                "graph_token_budget": self.model.graph_token_budget,
                "graph_layers": self.model.graph_layers,
                "candidate_budget": self.model.candidate_budget,
                "sparse_policy": False,
                "sparse_prior_stage": 0,
                "sparse_prior_mix": 0.0,
                "heads": list(self.model.output_heads),
                "pair_strategy": self.pair_strategy.mode,
                "pair_strategy_max_pairs": int(self.pair_strategy.pair_row_budget) if pair_enabled else 0,
                "pair_prior_mix": float(self.pair_strategy.pair_prior_mix) if pair_enabled else 0.0,
            }
        )

        selfplay = data.setdefault("selfplay", {})
        selfplay_update = {
            "mcts_simulations": self.search.full_mcts_simulations,
            "pcr_low_sims": self.search.pcr_low_sims,
            "pcr_low_sim_prob": self.search.pcr_low_sim_prob,
            "policy_target_top_k": self.search.policy_target_top_k,
            "temperature_schedule": _temperature_schedule(self.search.temperature_family),
            "c_puct": self.search.c_puct,
            "c_puct_init": self.search.c_puct_init,
            "dirichlet_fraction": self.search.dirichlet_fraction,
            "dirichlet_alpha": self.search.dirichlet_alpha_scale,
        }
        if self.runtime.selfplay_workers > 0:
            selfplay_update["num_workers"] = self.runtime.selfplay_workers
        if self.runtime.batch_size_per_worker > 0:
            selfplay_update["batch_size_per_worker"] = self.runtime.batch_size_per_worker
        selfplay.update(selfplay_update)

        inference = data.setdefault("inference", {})
        inference_update = {}
        if self.runtime.inference_max_batch_size > 0:
            inference_update["max_batch_size"] = self.runtime.inference_max_batch_size
        if self.runtime.inference_wait_us > 0:
            inference_update["max_wait_us"] = self.runtime.inference_wait_us
        inference.update(inference_update)

        buffer = data.setdefault("buffer", {})
        buffer["recency_decay"] = self.schedule.recency_decay

        train = data.setdefault("train", {})
        train["peak_lr"] = self.schedule.peak_lr * self.schedule.lr_multiplier
        train["weight_decay"] = self.schedule.weight_decay
        loss_weights = dict(train.get("loss_weights", {}))
        loss_weights.update(
            {
                "policy_place": self.schedule.graph_loss_weight,
                "value": self.schedule.value_loss_weight,
                "tactical": self.schedule.auxiliary_loss_weight,
                "legal_token_quality": self.schedule.auxiliary_loss_weight,
            }
        )
        for head in _PAIR_HEADS:
            if head in self.model.output_heads:
                loss_weights[head] = self.schedule.pair_loss_weight
        train["loss_weights"] = loss_weights

        return Config.model_validate(data)


def candidate_recipes_from_config(config: Config) -> tuple[CandidateRecipe, ...]:
    """Build the validated initial scout recipe plan from ``config.autotune``."""

    recipes = []
    for entry in config.autotune.scout.candidate_plan:
        architecture_id, pair_mode = entry.split(":", 1)
        recipes.append(
            CandidateRecipe(
                model=_model_recipe_for_architecture(architecture_id, pair_mode),
                pair_strategy=_pair_strategy_for_mode(pair_mode),
                metadata={"source": "autotune.scout.candidate_plan", "plan_entry": entry},
            )
        )
    ids = [recipe.candidate_id for recipe in recipes]
    if len(set(ids)) != len(ids):
        raise ValueError("autotune scout candidate ids must be unique")
    return tuple(recipes)


def _model_recipe_for_architecture(architecture_id: str, pair_mode: str) -> ModelRecipe:
    architecture_id = normalize_architecture_id(architecture_id)
    output_heads = ["policy_place", "value"]
    head_bundle = "policy_value"
    if pair_mode != "none":
        output_heads.extend(_PAIR_HEADS)
        head_bundle = "pair_mcts"
    kwargs: dict[str, Any] = {}
    if architecture_id == "global_graph768_champion":
        kwargs.update({"graph_token_set": "graph768_champion", "graph_token_budget": 768, "graph_layers": 6})
    return ModelRecipe(
        architecture_id=architecture_id,
        head_bundle=head_bundle,
        output_heads=output_heads,
        **kwargs,
    )


def _pair_strategy_for_mode(pair_mode: str) -> PairStrategySpec:
    if pair_mode == "none":
        return PairStrategySpec(mode="none")
    return PairStrategySpec(mode=pair_mode, pair_row_budget=256, pair_prior_mix=0.35)


def _temperature_schedule(family: str) -> list[list[float]]:
    if family == "slow_cool":
        return [[0, 1.0], [30, 0.0]]
    if family == "fixed_cold":
        return [[0, 0.25], [30, 0.0]]
    if family == "fixed_zero":
        return [[0, 0.0]]
    raise ValueError(f"unknown temperature family {family!r}")
