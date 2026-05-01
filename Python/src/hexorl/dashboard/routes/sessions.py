from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from hexorl.dashboard.play import apply_move, create_session, reset_session, session_payload, undo_move
from hexorl.dashboard.schemas.sessions import CreateSessionRequest, MoveRequest

router = APIRouter()


@router.post("/api/session/create")
def session_create(request: Request, req: CreateSessionRequest) -> dict[str, Any]:
    session = create_session(request.app.state.store, run_id=req.run_id, payload=req.payload)
    return session_payload(request.app.state.store, session.session_id)


@router.get("/api/session/{session_id}")
def session_get(request: Request, session_id: str) -> dict[str, Any]:
    try:
        return session_payload(request.app.state.store, session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/api/session/{session_id}/move")
def session_move(request: Request, session_id: str, req: MoveRequest) -> dict[str, Any]:
    try:
        apply_move(request.app.state.store, session_id, req.q, req.r)
        return session_payload(request.app.state.store, session_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/api/session/{session_id}/undo")
def session_undo(request: Request, session_id: str) -> dict[str, Any]:
    try:
        undo_move(request.app.state.store, session_id)
        return session_payload(request.app.state.store, session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/api/session/{session_id}/reset")
def session_reset(request: Request, session_id: str) -> dict[str, Any]:
    try:
        reset_session(request.app.state.store, session_id)
        return session_payload(request.app.state.store, session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

