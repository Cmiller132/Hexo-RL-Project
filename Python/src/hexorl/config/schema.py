"""Pydantic configuration schema."""

from pydantic import BaseModel, Field, model_validator
from typing import List


class RunConfig(BaseModel):
    seed: int = 42
    output_dir: str = "./runs/{name}"
    log_level: str = "INFO"
    deterministic: bool = False


class ModelConfig(BaseModel):
    channels: int = 128
    blocks: int = 16
    heads: List[str] = Field(default_factory=lambda: ["policy", "value"])


class SelfPlayConfig(BaseModel):
    num_workers: int = 24
    games_per_epoch: int = 4096
    states_per_epoch: int = 400_000
    batch_size_per_worker: int = 8
    mcts_simulations: int = 800
    c_puct: float = 1.5
    c_puct_init: float = 19652.0
    temperature_schedule: List[List[float]] = Field(default_factory=lambda: [[0, 1.0], [30, 0.0]])
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    pcr_low_sim_prob: float = 0.75
    pcr_low_sims: int = 192
    resign_threshold: float = -0.95
    resign_disable_prob: float = 0.1
    near_radius: int = 8
    constrain_threats: bool = True


class InferenceConfig(BaseModel):
    max_batch_size: int = 128
    max_wait_us: int = 200
    fp16: bool = True
    ema_update_every: int = 100


class BufferConfig(BaseModel):
    capacity: int = 2_000_000
    recency_decay: float = 0.99
    pcr_weight: float = 0.25
    regret_fraction: float = 0.08
    lookahead_horizons: List[int] = Field(default_factory=lambda: [4, 12, 36])
    lookahead_lambdas: List[float] = Field(default_factory=lambda: [0.75, 0.90, 0.97])


class TrainConfig(BaseModel):
    batch_size: int = 256
    batches_per_epoch: int = 2000
    optimizer: str = "adamw"
    lr_schedule: str = "cosine"
    peak_lr: float = 3e-3
    weight_decay: float = 1e-4
    loss_weights: dict = Field(default_factory=lambda: {
        "policy": 1.0,
        "value": 1.5,
        "lookahead_6": 0.15,
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
    run: RunConfig = Field(default_factory=RunConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    selfplay: SelfPlayConfig = Field(default_factory=SelfPlayConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    buffer: BufferConfig = Field(default_factory=BufferConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    @model_validator(mode="after")
    def validate_cross_section_consistency(self) -> "Config":
        if len(self.buffer.lookahead_horizons) != len(self.buffer.lookahead_lambdas):
            raise ValueError(
                "buffer.lookahead_horizons and buffer.lookahead_lambdas must have the same length"
            )

        configured_horizons = {f"lookahead_{h}" for h in self.buffer.lookahead_horizons}
        model_lookahead_heads = {
            head for head in self.model.heads if head.startswith("lookahead_")
        }
        missing_horizons = sorted(model_lookahead_heads - configured_horizons)
        if missing_horizons:
            raise ValueError(
                "model lookahead heads must match buffer.lookahead_horizons; "
                f"missing horizons for heads: {missing_horizons}"
            )

        return self
