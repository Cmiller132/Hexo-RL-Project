from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from hexorl.dashboard.services.suite import suite_status

router = APIRouter()


@router.get("/api/suite/status")
def status(request: Request) -> dict[str, Any]:
    suite_root = request.app.state.suite_root
    return {"enabled": False} if suite_root is None else suite_status(suite_root)


@router.get("/api/suite/status/stream")
async def status_stream(request: Request) -> StreamingResponse:
    async def events():
        last_payload = ""
        while True:
            payload = json.dumps(status(request))
            if payload != last_payload:
                last_payload = payload
                yield f"data: {payload}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(events(), media_type="text/event-stream")
