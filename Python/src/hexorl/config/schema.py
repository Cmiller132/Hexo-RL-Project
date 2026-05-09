"""Pydantic configuration schema."""

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import List

from hexorl.models.registry import (
    architecture_ids,
    architecture_spec,
    deprecated_aliases,
    global_graph_architecture_ids,
    normalize_architecture_id,
    resolve_model_spec,
)
from hexorl.models.specs import merge_resolved_loss_weights
from hexorl.search.pair_strategy import build_pair_strategy


AUTOTUNE_PAIR_STRATEGY_MODES = ("none", "root_pair_mcts", "full_pair_mcts")
DEFAULT_AUTOTUNE_CANDIDATE_PLAN = (
    "global_xattn_0:none",
    "global_line_window_0:none",
    "global_pair_twostage_0:none",
    "global_graph_full_0:none",
    "global_graph768_champion:none",
    "global_pair_twostage_0:root_pair_mcts",
    "global_pair_twostage_0:full_pair_mcts",
    "global_graph_full_0:root_pair_mcts",
)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    seed: int = 42
    output_dir: str = "./runs/{name}"
    log_level: str = "INFO"
    deterministic: bool = False


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    channels: int = 128
    blocks: int = 16
    heads: List[str] = Field(default_factory=lambda: ["policy", "value"])
    architecture: str = "cnn"
    attention_positions: List[int] = Field(default_factory=list)
    attention_heads: int = 8
    attention_mlp_ratio: float = 2.0
    attention_dropout: float = 0.0
    dropout: float = 0.0
    relative_bias: bool = False
    graph_token_set: str = "graph512_turn_pair_prior"
    graph_token_budget: int = 512
    graph_layers: int = 3
    sparse_policy: bool = False
    candidate_budget: int = 256
    sparse_prior_stage: int = 0
    sparse_prior_mix: float = 0.25
    pair_prior_mix: float = 0.35
    pair_strategy: str = "none"
    pair_strategy_max_pairs: int = 0
    global_graph_leaf_eval: bool = False

    @model_validator(mode="after")
    def validate_model_config(self) -> "ModelConfig":
        arch = self.architecture.lower()
        try:
            arch = normalize_architecture_id(arch)
        except ValueError as exc:
            if arch in deprecated_aliases():
                raise ValueError(str(exc)) from exc
            raise ValueError(
                "model.architecture must be one of "
                f"{sorted(architecture_ids())}"
            ) from exc
        spec = architecture_spec(arch)
        self.architecture = arch
        fields_set = set(self.model_fields_set)
        if arch == "restnet":
            if "blocks" not in fields_set:
                self.blocks = 10
            if "attention_heads" not in fields_set:
                self.attention_heads = 4
            if "attention_positions" not in fields_set:
                self.attention_positions = [4, 7, 10]
            if "relative_bias" not in fields_set:
                self.relative_bias = True
            if "heads" not in fields_set:
                self.heads = ["policy", "value", "opp_policy"]
            if self.blocks != 10:
                raise ValueError("canonical restnet requires model.blocks = 10")
            if self.attention_heads != 4:
                raise ValueError("canonical restnet requires model.attention_heads = 4")
            if list(self.attention_positions) != [4, 7, 10]:
                raise ValueError(
                    "canonical restnet requires model.attention_positions = [4, 7, 10]"
                )
            if self.relative_bias is not True:
                raise ValueError("canonical restnet requires model.relative_bias = true")
            if set(self.heads) != {"policy", "value", "opp_policy"} or len(self.heads) != 3:
                raise ValueError("canonical restnet supports only policy, value, and opp_policy heads")
            if self.sparse_policy or self.sparse_prior_stage != 0:
                raise ValueError("canonical restnet does not support sparse policy heads")
            if self.pair_strategy != "none" or self.pair_strategy_max_pairs != 0:
                raise ValueError("canonical restnet requires model.pair_strategy='none'")
        if self.blocks <= 0:
            raise ValueError("model.blocks must be positive")
        if self.channels <= 0:
            raise ValueError("model.channels must be positive")
        if self.attention_heads <= 0:
            raise ValueError("model.attention_heads must be positive")
        if (
            spec.requires_attention_head_divisibility
            or self.attention_positions
        ) and self.channels % self.attention_heads != 0:
            raise ValueError("model.channels must be divisible by model.attention_heads")
        if self.attention_mlp_ratio <= 0.0:
            raise ValueError("model.attention_mlp_ratio must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("model.attention_dropout must be in [0, 1)")
        if self.relative_bias and arch != "restnet":
            raise ValueError("model.relative_bias is reserved and must remain false")
        valid_graph_sets = {
            "graph256_cells",
            "graph384_windows",
            "graph512_cover",
            "graph512_turn",
            "graph512_turn_pair_prior",
            "graph768_champion",
            "graph768_devwin",
        }
        self.graph_token_set = self.graph_token_set.lower()
        if self.graph_token_set not in valid_graph_sets:
            raise ValueError(f"model.graph_token_set must be one of {sorted(valid_graph_sets)}")
        if not 16 <= self.graph_token_budget <= 768:
            raise ValueError("model.graph_token_budget must be in [16, 768]")
        if self.graph_layers <= 0:
            raise ValueError("model.graph_layers must be positive")
        if self.candidate_budget <= 0:
            raise ValueError("model.candidate_budget must be positive")
        if self.candidate_budget > 512:
            raise ValueError("model.candidate_budget must be <= 512 for the shared-memory protocol")
        if self.sparse_prior_stage not in {0, 1, 2}:
            raise ValueError("model.sparse_prior_stage must be 0, 1, or 2")
        if not 0.0 <= self.sparse_prior_mix <= 1.0:
            raise ValueError("model.sparse_prior_mix must be in [0, 1]")
        if not 0.0 <= self.pair_prior_mix <= 1.0:
            raise ValueError("model.pair_prior_mix must be in [0, 1]")
        build_pair_strategy(
            self.pair_strategy,
            max_pairs=self.pair_strategy_max_pairs,
            prior_mix=self.pair_prior_mix,
        )
        invalid_positions = [
            pos for pos in self.attention_positions if pos < 1 or pos > self.blocks
        ]
        if invalid_positions:
            raise ValueError(
                "model.attention_positions must be 1-based block positions within "
                f"1..{self.blocks}; got {invalid_positions}"
            )
        if self.attention_positions and not spec.supports_attention_positions:
            raise ValueError("model.attention_positions are only used by ResTNet dense architectures")
        return self


class SelfPlayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    num_workers: int = 24
    games_per_epoch: int = 4096
    states_per_epoch: int = 400_000
    max_game_moves: int = 256
    batch_size_per_worker: int = 8
    mcts_simulations: int = 800
    c_puct: float = 1.5
    c_puct_init: float = 19652.0
    temperature_schedule: List[List[float]] = Field(default_factory=lambda: [[0, 1.0], [30, 0.0]])
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    pcr_low_sim_prob: float = 0.75
    pcr_low_sims: int = 192
    policy_target_top_k: int = 64
    train_policy_on_full_search_only: bool = True
    near_radius: int = 8
    constrain_threats: bool = True
    subtree_reuse: bool = False
    train_on_truncated_games: bool = False
    rgsc_beta: float = 0.0
    rgsc_prb_capacity: int = 100
    rgsc_prb_temperature: float = 0.1
    rgsc_prb_ema_alpha: float = 0.5


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_batch_size: int = 128
    max_wait_us: int = 200
    fp16: bool = True
    ema_update_every: int = 100


class BufferConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    capacity: int = 2_000_000
    recency_decay: float = 0.99
    pcr_weight: float = 0.25
    regret_fraction: float = 0.08
    regret_replay_only: bool = True
    lookahead_horizons: List[int] = Field(default_factory=lambda: [4, 12, 36])
    lookahead_lambdas: List[float] = Field(default_factory=lambda: [0.75, 0.90, 0.97])


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    batch_size: int = 256
    graph_microbatch_size: int = 0
    graph_microbatch_autotune: bool = True
    graph_microbatch_autotune_max_size: int = 32
    graph_microbatch_memory_headroom: float = 0.75
    batches_per_epoch: int = 2000
    prefetch_batches: int = 2
    optimizer: str = "adamw"
    lr_schedule: str = "cosine"
    peak_lr: float = 3e-3
    weight_decay: float = 1e-4
    loss_weights: dict[str, float] = Field(default_factory=lambda: {
        "policy": 1.0,
        "value": 1.5,
        "lookahead_4": 0.15,
        "lookahead_12": 0.15,
        "lookahead_36": 0.1,
        "regret_rank": 0.1,
        "regret_value": 0.1,
        "opp_policy": 0.15,
        "axis": 0.05,
        "moves_left": 0.05,
        "entropy": 0.01,
    })


class AutotuneScoutConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = True
    max_candidates: int = 8
    min_epochs: int = 12
    estimated_epoch_seconds: int = 600
    estimated_candidate_hours: int = 2
    min_generated_selfplay_positions_per_epoch: int = 3000
    target_phase_hours: int = 48
    schedule_quantum_epochs: int = 2
    include_dense_control: bool = False
    candidate_plan: List[str] = Field(
        default_factory=lambda: list(DEFAULT_AUTOTUNE_CANDIDATE_PLAN)
    )

    @model_validator(mode="after")
    def validate_scout_plan(self) -> "AutotuneScoutConfig":
        if not 1 <= self.max_candidates <= 8:
            raise ValueError("autotune.scout.max_candidates must be in [1, 8]")
        positive_fields = {
            "min_epochs": self.min_epochs,
            "estimated_epoch_seconds": self.estimated_epoch_seconds,
            "estimated_candidate_hours": self.estimated_candidate_hours,
            "min_generated_selfplay_positions_per_epoch": self.min_generated_selfplay_positions_per_epoch,
            "target_phase_hours": self.target_phase_hours,
            "schedule_quantum_epochs": self.schedule_quantum_epochs,
        }
        invalid_positive = [name for name, value in positive_fields.items() if int(value) <= 0]
        if invalid_positive:
            raise ValueError(f"autotune.scout fields must be positive: {invalid_positive}")
        if len(self.candidate_plan) > self.max_candidates:
            raise ValueError(
                "autotune.scout.candidate_plan must not exceed autotune.scout.max_candidates"
            )
        global_architectures = set(global_graph_architecture_ids())
        normalized_plan: list[str] = []
        for entry in self.candidate_plan:
            parts = entry.split(":")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(
                    "autotune.scout.candidate_plan entries must use "
                    "'<architecture_id>:<pair_strategy_mode>'"
                )
            architecture_id, pair_mode = parts[0].lower(), parts[1].lower()
            normalized_plan.append(f"{architecture_id}:{pair_mode}")
            if pair_mode not in AUTOTUNE_PAIR_STRATEGY_MODES:
                raise ValueError(
                    "autotune.scout.candidate_plan pair modes must be one of "
                    f"{list(AUTOTUNE_PAIR_STRATEGY_MODES)}"
                )
            if not self.include_dense_control and architecture_id not in global_architectures:
                raise ValueError(
                    "autotune.scout.candidate_plan must be global-graph-only when "
                    "autotune.scout.include_dense_control is false"
                )
        if len(set(normalized_plan)) != len(normalized_plan):
            raise ValueError("autotune.scout.candidate_plan contains duplicate candidates")
        if "global_graph768_champion:none" not in normalized_plan:
            raise ValueError(
                "autotune.scout.candidate_plan must include global_graph768_champion:none"
            )
        self.candidate_plan = normalized_plan
        return self


class AutotuneOptunaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    storage: str = "sqlite:///runs/<run_id>/optuna.sqlite3"
    phase1_sampler: str = "queued_tpe_shell"
    phase1_pruner: str = "nop"
    phase1_enqueue_candidate_plan: bool = True
    phase3_sampler: str = "tpe"
    tpe_multivariate: bool = True
    tpe_group: bool = True
    tpe_startup_trials: int = 8
    phase3_pruner: str = "successive_halving_after_floor"
    pruner_min_resource_epochs: int = 12
    pruner_reduction_factor: int = 2

    @model_validator(mode="after")
    def validate_optuna_surface(self) -> "AutotuneOptunaConfig":
        expected = {
            "phase1_sampler": (self.phase1_sampler, "queued_tpe_shell"),
            "phase1_pruner": (self.phase1_pruner, "nop"),
            "phase3_sampler": (self.phase3_sampler, "tpe"),
            "phase3_pruner": (self.phase3_pruner, "successive_halving_after_floor"),
        }
        invalid = [name for name, (actual, wanted) in expected.items() if actual != wanted]
        if invalid:
            raise ValueError(f"unsupported autotune.optuna values for Scout Foundation: {invalid}")
        if self.tpe_startup_trials <= 0:
            raise ValueError("autotune.optuna.tpe_startup_trials must be positive")
        if self.pruner_min_resource_epochs <= 0:
            raise ValueError("autotune.optuna.pruner_min_resource_epochs must be positive")
        if self.pruner_reduction_factor <= 1:
            raise ValueError("autotune.optuna.pruner_reduction_factor must be > 1")
        return self


class AutotuneRuntimeProbeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = True
    speed_quarantine_positions_per_sec: float = 2.0
    measure: str = "generated_selfplay_positions_per_second"
    mode: str = "calibrate_and_select"
    behavior_invariant: bool = True

    @model_validator(mode="after")
    def validate_runtime_probe_surface(self) -> "AutotuneRuntimeProbeConfig":
        if self.speed_quarantine_positions_per_sec <= 0.0:
            raise ValueError("autotune.runtime_probe.speed_quarantine_positions_per_sec must be positive")
        if self.measure != "generated_selfplay_positions_per_second":
            raise ValueError("autotune.runtime_probe.measure must be generated_selfplay_positions_per_second")
        if self.mode != "calibrate_and_select":
            raise ValueError("autotune.runtime_probe.mode must be calibrate_and_select")
        return self


class AutotuneQuarantineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    continue_after_quarantine: bool = True
    allow_retest: bool = True
    ready_for_retest: List[str] = Field(default_factory=list)


class AutotunePairStrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    modes: List[str] = Field(default_factory=lambda: list(AUTOTUNE_PAIR_STRATEGY_MODES))
    full_pair_mcts_enabled: bool = True

    @model_validator(mode="after")
    def validate_pair_modes(self) -> "AutotunePairStrategyConfig":
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("autotune.pair_strategy.modes contains duplicate modes")
        invalid = sorted(set(self.modes) - set(AUTOTUNE_PAIR_STRATEGY_MODES))
        if invalid:
            raise ValueError(
                "autotune.pair_strategy.modes must be a subset of "
                f"{list(AUTOTUNE_PAIR_STRATEGY_MODES)}; invalid={invalid}"
            )
        if "none" not in self.modes:
            raise ValueError("autotune.pair_strategy.modes must include 'none'")
        return self


class AutotuneScoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    target_scalar: str = "classical_survival_lcb"
    phase1_uses_model_tournament: bool = False
    phase2_uses_model_tournament: bool = True

    @model_validator(mode="after")
    def validate_scoring_surface(self) -> "AutotuneScoringConfig":
        if self.target_scalar != "classical_survival_lcb":
            raise ValueError("autotune.scoring.target_scalar must be classical_survival_lcb")
        if self.phase1_uses_model_tournament:
            raise ValueError("autotune.scoring.phase1_uses_model_tournament must be false")
        return self


class AutotuneFinalEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    classical_arena_games: int = 400
    classical_opponent: str = "fixed_strong"

    @model_validator(mode="after")
    def validate_final_eval_surface(self) -> "AutotuneFinalEvalConfig":
        if self.classical_arena_games <= 0:
            raise ValueError("autotune.final_eval.classical_arena_games must be positive")
        if self.classical_opponent != "fixed_strong":
            raise ValueError("autotune.final_eval.classical_opponent must be fixed_strong")
        return self


class AutotuneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scout: AutotuneScoutConfig = Field(default_factory=AutotuneScoutConfig)
    optuna: AutotuneOptunaConfig = Field(default_factory=AutotuneOptunaConfig)
    runtime_probe: AutotuneRuntimeProbeConfig = Field(default_factory=AutotuneRuntimeProbeConfig)
    quarantine: AutotuneQuarantineConfig = Field(default_factory=AutotuneQuarantineConfig)
    pair_strategy: AutotunePairStrategyConfig = Field(default_factory=AutotunePairStrategyConfig)
    scoring: AutotuneScoringConfig = Field(default_factory=AutotuneScoringConfig)
    final_eval: AutotuneFinalEvalConfig = Field(default_factory=AutotuneFinalEvalConfig)

    @model_validator(mode="after")
    def validate_candidate_plan_modes(self) -> "AutotuneConfig":
        configured_modes = set(self.pair_strategy.modes)
        for entry in self.scout.candidate_plan:
            pair_mode = entry.split(":", 1)[1].lower()
            if pair_mode not in configured_modes:
                raise ValueError(
                    "autotune.scout.candidate_plan references a pair mode not listed in "
                    f"autotune.pair_strategy.modes: {pair_mode!r}"
                )
            if pair_mode == "full_pair_mcts" and not self.pair_strategy.full_pair_mcts_enabled:
                raise ValueError(
                    "autotune.scout.candidate_plan includes full_pair_mcts while "
                    "autotune.pair_strategy.full_pair_mcts_enabled is false"
                )
        return self


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    run: RunConfig = Field(default_factory=RunConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    selfplay: SelfPlayConfig = Field(default_factory=SelfPlayConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    buffer: BufferConfig = Field(default_factory=BufferConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    runtime: "RuntimeConfig" = Field(default_factory=lambda: RuntimeConfig())
    autotune: AutotuneConfig = Field(default_factory=AutotuneConfig)

    @model_validator(mode="after")
    def validate_cross_section_consistency(self) -> "Config":
        if len(self.buffer.lookahead_horizons) != len(self.buffer.lookahead_lambdas):
            raise ValueError(
                "buffer.lookahead_horizons and buffer.lookahead_lambdas must have the same length"
            )

        configured_horizons = {f"lookahead_{h}" for h in self.buffer.lookahead_horizons}
        model_lookahead_heads = {
            head for head in self.model.heads if head.startswith("lookahead_") and head != "lookahead_*"
        }
        missing_horizons = sorted(model_lookahead_heads - configured_horizons)
        if missing_horizons:
            raise ValueError(
                "model lookahead heads must match buffer.lookahead_horizons; "
                f"missing horizons for heads: {missing_horizons}"
            )
        if ("sparse_policy" in self.model.heads or "pair_policy" in self.model.heads) and not self.model.sparse_policy:
            raise ValueError(
                "model heads sparse_policy/pair_policy require explicit model.sparse_policy = true; "
                "the config is not auto-mutated"
            )
        resolved = resolve_model_spec(self)
        if self.model.pair_strategy != "none":
            if not resolved.pair_capabilities:
                raise ValueError(
                    "non-none model.pair_strategy requires an architecture with pair capability"
                )
            if not (
                {"pair_policy", "policy_pair_first", "policy_pair_second", "policy_pair_joint"}
                & set(resolved.outputs)
            ):
                raise ValueError(
                    "non-none model.pair_strategy requires an explicit pair policy head"
                )
        if self.model.sparse_policy and max(
            self.model.candidate_budget,
            self.selfplay.policy_target_top_k,
        ) > 512:
            raise ValueError(
                "sparse policy effective candidate width must be <= 512 "
                "(max(model.candidate_budget, selfplay.policy_target_top_k))"
            )
        trainable_heads = {
            name
            for name, contract in resolved.output_contracts.items()
            if contract.trainable and not contract.diagnostic_only
        }
        effective_loss_weights = merge_resolved_loss_weights(resolved, self.train.loss_weights)
        graph_auto_heads = set(resolved.outputs)
        missing_or_inactive = sorted(
            head
            for head in graph_auto_heads
            if (head in trainable_heads or head.startswith("lookahead_"))
            and float(effective_loss_weights.get(head, 0.0)) <= 0.0
        )
        if missing_or_inactive:
            raise ValueError(
                "enabled model heads require active train.loss_weights entries; "
                f"missing or inactive: {missing_or_inactive}"
            )
        regret_heads_active = all(
            head in resolved.outputs and float(effective_loss_weights.get(head, 0.0)) > 0.0
            for head in ("regret_rank", "regret_value")
        )
        if (
            self.buffer.regret_fraction > 0.0
            and not self.buffer.regret_replay_only
            and not regret_heads_active
        ):
            raise ValueError(
                "buffer.regret_fraction > 0 requires enabled and weighted regret heads "
                "or buffer.regret_replay_only = true"
            )
        if self.model.sparse_prior_stage > 0 and not self.model.sparse_policy:
            raise ValueError("model.sparse_prior_stage > 0 requires model.sparse_policy = true")
        if not 0.0 <= self.selfplay.rgsc_beta <= 1.0:
            raise ValueError("selfplay.rgsc_beta must be in [0, 1]")
        if self.selfplay.rgsc_prb_capacity < 0:
            raise ValueError("selfplay.rgsc_prb_capacity must be non-negative")
        if self.selfplay.rgsc_prb_temperature <= 0.0:
            raise ValueError("selfplay.rgsc_prb_temperature must be positive")
        if not 0.0 <= self.selfplay.rgsc_prb_ema_alpha <= 1.0:
            raise ValueError("selfplay.rgsc_prb_ema_alpha must be in [0, 1]")

        return self


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    """Host-aware performance knobs.

    Values left as ``None`` are filled in by the runtime autotuner. This keeps
    production configs portable across the 4070 Ti workstation and smaller CI
    machines while still allowing an experiment config to pin exact values.
    """

    autotune: bool = True
    cpu_threads: int | None = None
    interop_threads: int | None = None
    dataloader_workers: int | None = None
    graph_dataloader_workers: int | None = None
    dataloader_prefetch_factor: int = 2
    dataloader_pin_memory: bool | None = None
    graph_worker_torch_threads: int = 1
    graph_relation_rebuild_threads: int = 0
    graph_cache_size: int = 256
    selfplay_workers: int | None = None
    selfplay_cpu_reserve: int = 4
    channels_last: bool = True
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    compile_model: bool | None = None
    compile_inference: bool | None = None
    compile_mode: str = "reduce-overhead"
    train_memory_fraction: float = 0.62
    inference_start_timeout_s: float = 30.0


Config.model_rebuild()
