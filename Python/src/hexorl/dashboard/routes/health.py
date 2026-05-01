from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    store = request.app.state.store
    suite_root = request.app.state.suite_root
    return {
        "ok": True,
        "schema_version": 1,
        "db_path": str(store.path),
        "suite_enabled": suite_root is not None,
        "suite_run_root": str(suite_root) if suite_root else None,
    }

