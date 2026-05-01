from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AxisEvaluateRequest(BaseModel):
    history_b64: str | None = None
    game_id: int | None = None
    turn_index: int | None = None
    session_id: str | None = None
    prototype_id: str | None = None
    parameters: dict[str, float] = Field(default_factory=dict)
    parameter_overrides: dict[str, dict[str, float]] = Field(default_factory=dict)
    position: dict[str, Any] | None = None


class AxisPresetRequest(BaseModel):
    name: str
    prototype_id: str
    parameters: dict[str, float] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)


class AxisFixtureGenerateRequest(BaseModel):
    count: int = 8
    examples_per_move_count: int = 3
    move_counts: list[int] = Field(default_factory=lambda: [8, 16, 24, 32, 40])
    time_ms: int = 2
    max_depth: int = 1
    near_radius: int = 6
    noise_level: float = 0.08
    random_move_prob: float = 0.04
    opening_random_moves: int = 2
    seed: int = 0
    workers: int = 4

