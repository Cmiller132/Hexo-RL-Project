from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from hexorl.dashboard.services.suite import jsonl_tail, tail_jsonl

router = APIRouter()


@router.get("/api/suite/events")
def suite_events(request: Request, limit: int = 200) -> list[dict[str, Any]]:
    suite_root = request.app.state.suite_root
    if suite_root is None:
        return []
    return jsonl_tail(suite_root / "events.jsonl", limit=max(1, min(limit, 1000)))


@router.get("/api/suite/events/stream")
async def suite_events_stream(request: Request) -> StreamingResponse:
    async def events():
        suite_root = request.app.state.suite_root
        if suite_root is None:
            return
        async for row in tail_jsonl(suite_root / "events.jsonl"):
            yield f"data: {json.dumps(row)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
