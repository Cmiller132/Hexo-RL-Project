from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from hexorl.dashboard.services.suite import suite_store_for_run

router = APIRouter()


@router.get("/api/metrics/{run_id}")
def metrics(request: Request, run_id: str, limit: int = 500) -> list[dict[str, Any]]:
    source = suite_store_for_run(request.app.state.suite_root, run_id) or request.app.state.store
    return source.rows(
        """
        SELECT * FROM (
            SELECT * FROM metrics WHERE run_id=? ORDER BY created_at DESC LIMIT ?
        ) ORDER BY created_at ASC
        """,
        (run_id, max(1, min(limit, 5000))),
    )


@router.get("/api/events/{run_id}")
def events(request: Request, run_id: str, limit: int = 500) -> list[dict[str, Any]]:
    source = suite_store_for_run(request.app.state.suite_root, run_id) or request.app.state.store
    return source.rows(
        "SELECT * FROM events WHERE run_id=? ORDER BY created_at DESC LIMIT ?",
        (run_id, max(1, min(limit, 5000))),
    )

