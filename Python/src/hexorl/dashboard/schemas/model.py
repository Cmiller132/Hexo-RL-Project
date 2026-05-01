from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelLoadRequest(BaseModel):
    path: str


class InferRequest(BaseModel):
    history_b64: str
    model_ids: list[str] = Field(default_factory=list)
    policy_target_v2: list[Any] = Field(default_factory=list)
    pair_policy_target_v2: list[Any] = Field(default_factory=list)

