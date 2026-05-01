from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ArenaStartRequest(BaseModel):
    run_id: str | None = None
    side_a: str = "model"
    side_b: str = "classical"
    payload: dict[str, Any] = Field(default_factory=dict)

