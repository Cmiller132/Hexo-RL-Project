from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from hexorl.dashboard.services.suite import suite_runs

router = APIRouter()


@router.get("/api/runs")
def runs(request: Request) -> list[dict[str, Any]]:
    store = request.app.state.store
    suite_root = request.app.state.suite_root
    if suite_root is not None:
        rows = suite_runs(suite_root)
        if rows:
            return rows
    return store.rows("SELECT * FROM runs ORDER BY updated_at DESC")

