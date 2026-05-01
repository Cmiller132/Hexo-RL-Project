from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from hexorl.dashboard.checkpoints import scan_checkpoints
from hexorl.dashboard.schemas.checkpoints import ImportCheckpointsRequest
from hexorl.dashboard.services.suite import suite_checkpoints

router = APIRouter()


@router.get("/api/checkpoints")
def checkpoints(request: Request, run_id: str | None = None) -> list[dict[str, Any]]:
    store = request.app.state.store
    suite_root = request.app.state.suite_root
    if suite_root is not None:
        return suite_checkpoints(suite_root, run_id=run_id)
    if run_id:
        return store.rows("SELECT * FROM checkpoints WHERE run_id=? ORDER BY indexed_at DESC", (run_id,))
    return store.rows("SELECT * FROM checkpoints ORDER BY indexed_at DESC")


@router.post("/api/import/checkpoints")
def import_checkpoints(request: Request, req: ImportCheckpointsRequest) -> dict[str, Any]:
    results = scan_checkpoints(req.path, request.app.state.store, run_id=req.run_id)
    return {
        "indexed": len(results),
        "checkpoints": [
            {
                "checkpoint_id": r.checkpoint_id,
                "path": str(r.path),
                "run_id": r.run_id,
                "epoch": r.epoch,
                "global_step": r.global_step,
                "is_loadable": r.is_loadable,
                "model_heads": r.model_heads,
                "error": r.error,
            }
            for r in results
        ],
    }

