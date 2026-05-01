from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from hexorl.dashboard.schemas.arena import ArenaStartRequest

router = APIRouter()


@router.post("/api/arena/start")
def arena_start(request: Request, req: ArenaStartRequest) -> dict[str, Any]:
    match_id = request.app.state.arena.start(
        run_id=req.run_id,
        side_a=req.side_a,
        side_b=req.side_b,
        payload=req.payload,
    )
    return {"match_id": match_id, "status": "running"}


@router.get("/api/arena/history")
def arena_history(request: Request) -> list[dict[str, Any]]:
    return request.app.state.store.rows("SELECT * FROM arena_matches ORDER BY updated_at DESC LIMIT 100")


@router.get("/api/arena/history/stream")
async def arena_history_stream(request: Request) -> StreamingResponse:
    async def events():
        last_payload = ""
        while True:
            rows = arena_history(request)
            payload = json.dumps(rows)
            if payload != last_payload:
                last_payload = payload
                yield f"data: {payload}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/api/arena/{match_id}")
def arena_match(request: Request, match_id: str) -> dict[str, Any]:
    rows = request.app.state.store.rows("SELECT * FROM arena_matches WHERE match_id=?", (match_id,))
    if not rows:
        raise HTTPException(404, f"Arena match not found: {match_id}")
    return rows[0]


@router.websocket("/ws/arena/{match_id}")
async def arena_ws(ws: WebSocket, match_id: str) -> None:
    await ws.accept()
    sent = 0
    try:
        while True:
            events = ws.app.state.arena.events.get(match_id, [])
            for event in events[sent:]:
                await ws.send_json(event)
            sent = len(events)
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return
