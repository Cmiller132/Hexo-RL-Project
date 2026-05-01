from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from hexorl.dashboard.db import decode_bytes
from hexorl.dashboard.schemas.model import InferRequest, ModelLoadRequest

router = APIRouter()


@router.post("/api/model/load")
def model_load(request: Request, req: ModelLoadRequest) -> dict[str, Any]:
    cached = request.app.state.model_cache.load(req.path)
    return {"model_id": cached.model_id, "path": str(cached.path), "device": str(cached.device)}


@router.get("/api/model/loaded")
def model_loaded(request: Request) -> list[dict[str, Any]]:
    return request.app.state.model_cache.list()


@router.delete("/api/model/{model_id}")
def model_unload(request: Request, model_id: str) -> dict[str, Any]:
    request.app.state.model_cache.unload(model_id)
    return {"ok": True}


@router.post("/api/model/{model_id}/infer")
def model_infer(request: Request, model_id: str, req: InferRequest) -> dict[str, Any]:
    try:
        return request.app.state.model_cache.infer_history(model_id, decode_bytes(req.history_b64))
    except KeyError as exc:
        raise HTTPException(404, f"Model not loaded: {model_id}") from exc
