"""Pydantic configuration schema."""

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import List
import warnings

from hexorl.models.registry import (
    architecture_ids,
    architecture_spec,
    normalize_architecture_id,
    resolve_model_spec,
)
from hexorl.search.pair_strategy import build_pair_strategy


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

    @model_validator(mode="after")
    def validate_model_config(self) -> "ModelConfig":
        arch = self.architecture.lower()
        if arch == "graph":
            warnings.warn(
                "model.architecture='graph' is a deprecated crop-compatible alias; "
                "normalizing to 'graph_hybrid_0'. Use a global_graph_* architecture "
                "for the first-class global graph contract.",
                stacklevel=2,
            )
        try:
            arch = normalize_architecture_id(arch)
        except ValueError as exc:
            raise ValueError(
                "model.architecture must be one of "
                f"{sorted(architecture_ids())}"
            ) from exc
        spec = architecture_spec(arch)
        self.architecture = arch
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
        if self.relative_bias:
            raise ValueError("model.relative_bias is reserved and must remain false")
        valid_graph_sets = {
            "graph256_cells",
            "graph384_windows",
            "graph512_cover",
            "graph512_turn",
            "graph512_turn_pair_prior",
            "graph768_champion",
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
        self.pair_strategy = self.pair_strategy.lower()
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
            raise ValueError("model.attention_positions are only used by architecture='restnet'")
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


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    run: RunConfig = Field(default_factory=RunConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    selfplay: SelfPlayConfig = Field(default_factory=SelfPlayConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    buffer: BufferConfig = Field(default_factory=BufferConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    runtime: "RuntimeConfig" = Field(default_factory=lambda: RuntimeConfig())

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
        self.model.heads = list(resolved.outputs)
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
            "policy",
            "sparse_policy",
            "pair_policy",
            "opp_policy",
            "value",
            "regret_rank",
            "regret_value",
            "axis",
            "axis_delta_norm",
            "moves_left",
            "policy_place",
            "policy_pair_first",
            "policy_pair_second",
            "policy_pair_joint",
            "legal_token_quality",
            "tactical",
        }
        for name, weight in resolved.default_loss_weights.items():
            if name.startswith("lookahead_"):
                if not resolved.global_graph:
                    continue
                weight = self.train.loss_weights.get("value", weight)
                weight = float(weight) * 0.1
            self.train.loss_weights.setdefault(name, weight)
        graph_auto_heads = set(resolved.outputs)
        missing_or_inactive = sorted(
            head
            for head in graph_auto_heads
            if (head in trainable_heads or head.startswith("lookahead_"))
            and float(self.train.loss_weights.get(head, 0.0)) <= 0.0
        )
        if missing_or_inactive:
            raise ValueError(
                "enabled model heads require active train.loss_weights entries; "
                f"missing or inactive: {missing_or_inactive}"
            )
        regret_heads_active = all(
            head in self.model.heads and float(self.train.loss_weights.get(head, 0.0)) > 0.0
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
