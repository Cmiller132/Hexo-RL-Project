from __future__ import annotations

from pydantic import BaseModel


class ImportCheckpointsRequest(BaseModel):
    path: str
    run_id: str | None = None

