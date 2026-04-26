"""FastAPI dashboard application factory."""

from __future__ import annotations

import base64
import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hexorl.axis_policy.core import AxisPolicyInput
from hexorl.axis_policy.registry import describe_prototypes, evaluate_all, get_prototype
from hexorl.dashboard.arena_service import ArenaManager
from hexorl.dashboard.checkpoints import scan_checkpoints
from hexorl.dashboard.db import DashboardStore, decode_bytes
from hexorl.dashboard.model_cache import ModelCache
from hexorl.dashboard.play import apply_move, create_session, reset_session, session_payload, undo_move
from hexorl.dashboard.replay import get_replay_position, position_payload, replay_game


class ImportCheckpointsRequest(BaseModel):
    path: str
    run_id: str | None = None


class CreateSessionRequest(BaseModel):
    run_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class MoveRequest(BaseModel):
    q: int
    r: int


class AxisEvaluateRequest(BaseModel):
    history_b64: str | None = None
    game_id: int | None = None
    turn_index: int | None = None
    session_id: str | None = None
    prototype_id: str | None = None
    parameters: dict[str, float] = Field(default_factory=dict)
    parameter_overrides: dict[str, dict[str, float]] = Field(default_factory=dict)
    position: dict[str, Any] | None = None


class AxisPresetRequest(BaseModel):
    name: str
    prototype_id: str
    parameters: dict[str, float] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)


class ModelLoadRequest(BaseModel):
    path: str


class InferRequest(BaseModel):
    history_b64: str


class ArenaStartRequest(BaseModel):
    run_id: str | None = None
    side_a: str = "model"
    side_b: str = "classical"
    payload: dict[str, Any] = Field(default_factory=dict)


def create_app(
    db_path: Path | str = "runs/dashboard.sqlite3",
    *,
    frontend_dist: Path | str | None = None,
) -> FastAPI:
    store = DashboardStore(db_path)
    model_cache = ModelCache()
    arena_manager = ArenaManager(store)
    app = FastAPI(title="Hexo-RL Dashboard", version="0.1.0")
    app.state.store = store
    app.state.model_cache = model_cache
    app.state.arena = arena_manager

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "schema_version": 1, "db_path": str(store.path)}

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        return store.rows("SELECT * FROM runs ORDER BY updated_at DESC")

    @app.get("/api/metrics/{run_id}")
    def metrics(run_id: str, limit: int = 500) -> list[dict[str, Any]]:
        rows = store.rows(
            """
            SELECT * FROM (
                SELECT * FROM metrics WHERE run_id=? ORDER BY created_at DESC LIMIT ?
            ) ORDER BY created_at ASC
            """,
            (run_id, max(1, min(limit, 5000))),
        )
        return rows

    @app.get("/api/events/{run_id}")
    def events(run_id: str, limit: int = 500) -> list[dict[str, Any]]:
        return store.rows(
            "SELECT * FROM events WHERE run_id=? ORDER BY created_at DESC LIMIT ?",
            (run_id, max(1, min(limit, 5000))),
        )

    @app.get("/api/checkpoints")
    def checkpoints(run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id:
            return store.rows(
                "SELECT * FROM checkpoints WHERE run_id=? ORDER BY indexed_at DESC",
                (run_id,),
            )
        return store.rows("SELECT * FROM checkpoints ORDER BY indexed_at DESC")

    @app.post("/api/import/checkpoints")
    def import_checkpoints(req: ImportCheckpointsRequest) -> dict[str, Any]:
        results = scan_checkpoints(req.path, store, run_id=req.run_id)
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

    @app.get("/api/games")
    def games(run_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        if run_id:
            return store.rows(
                "SELECT * FROM games WHERE run_id=? ORDER BY created_at DESC LIMIT ?",
                (run_id, max(1, min(limit, 2000))),
            )
        return store.rows(
            "SELECT * FROM games ORDER BY created_at DESC LIMIT ?",
            (max(1, min(limit, 2000)),),
        )

    @app.get("/api/games/{game_id}/replay")
    def game_replay(game_id: int) -> dict[str, Any]:
        try:
            return replay_game(store, game_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/games/{game_id}/position/{turn_index}")
    def game_position(game_id: int, turn_index: int) -> dict[str, Any]:
        rows = store.rows("SELECT final_history_b64 FROM games WHERE game_id=?", (game_id,))
        if not rows:
            raise HTTPException(404, f"Game not found: {game_id}")
        pos = get_replay_position(rows[0]["final_history_b64"], turn_index=turn_index)
        return position_payload(pos)

    @app.post("/api/session/create")
    def session_create(req: CreateSessionRequest) -> dict[str, Any]:
        session = create_session(store, run_id=req.run_id, payload=req.payload)
        return session_payload(store, session.session_id)

    @app.get("/api/session/{session_id}")
    def session_get(session_id: str) -> dict[str, Any]:
        try:
            return session_payload(store, session_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/session/{session_id}/move")
    def session_move(session_id: str, req: MoveRequest) -> dict[str, Any]:
        try:
            apply_move(store, session_id, req.q, req.r)
            return session_payload(store, session_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/session/{session_id}/undo")
    def session_undo(session_id: str) -> dict[str, Any]:
        try:
            undo_move(store, session_id)
            return session_payload(store, session_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/session/{session_id}/reset")
    def session_reset(session_id: str) -> dict[str, Any]:
        try:
            reset_session(store, session_id)
            return session_payload(store, session_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/axis/prototypes")
    def axis_prototypes() -> list[dict[str, Any]]:
        return describe_prototypes()

    @app.post("/api/axis/evaluate")
    def axis_evaluate(req: AxisEvaluateRequest) -> dict[str, Any]:
        position = _axis_input_from_request(store, req)
        if req.prototype_id:
            proto = get_prototype(req.prototype_id)
            return proto.compute(position, req.parameters).to_json()
        return {"results": evaluate_all(position, req.parameter_overrides)}

    @app.post("/api/axis/presets")
    def axis_preset(req: AxisPresetRequest) -> dict[str, Any]:
        preset_id = store.save_axis_preset(
            name=req.name,
            prototype_id=req.prototype_id,
            parameters=req.parameters,
            payload=req.payload,
        )
        return {"preset_id": preset_id}

    @app.get("/api/axis/presets")
    def axis_presets() -> list[dict[str, Any]]:
        return store.rows("SELECT * FROM axis_presets ORDER BY created_at DESC")

    @app.post("/api/model/load")
    def model_load(req: ModelLoadRequest) -> dict[str, Any]:
        cached = model_cache.load(req.path)
        return {"model_id": cached.model_id, "path": str(cached.path), "device": str(cached.device)}

    @app.get("/api/model/loaded")
    def model_loaded() -> list[dict[str, Any]]:
        return model_cache.list()

    @app.delete("/api/model/{model_id}")
    def model_unload(model_id: str) -> dict[str, Any]:
        model_cache.unload(model_id)
        return {"ok": True}

    @app.post("/api/model/{model_id}/infer")
    def model_infer(model_id: str, req: InferRequest) -> dict[str, Any]:
        try:
            return model_cache.infer_history(model_id, decode_bytes(req.history_b64))
        except KeyError as exc:
            raise HTTPException(404, f"Model not loaded: {model_id}") from exc

    @app.post("/api/arena/start")
    def arena_start(req: ArenaStartRequest) -> dict[str, Any]:
        match_id = arena_manager.start(
            run_id=req.run_id,
            side_a=req.side_a,
            side_b=req.side_b,
            payload=req.payload,
        )
        return {"match_id": match_id, "status": "running"}

    @app.get("/api/arena/history")
    def arena_history() -> list[dict[str, Any]]:
        return store.rows("SELECT * FROM arena_matches ORDER BY updated_at DESC LIMIT 100")

    @app.get("/api/arena/{match_id}")
    def arena_match(match_id: str) -> dict[str, Any]:
        rows = store.rows("SELECT * FROM arena_matches WHERE match_id=?", (match_id,))
        if not rows:
            raise HTTPException(404, f"Arena match not found: {match_id}")
        return rows[0]

    @app.websocket("/ws/arena/{match_id}")
    async def arena_ws(ws: WebSocket, match_id: str) -> None:
        await ws.accept()
        sent = 0
        try:
            while True:
                events = arena_manager.events.get(match_id, [])
                for event in events[sent:]:
                    await ws.send_json(event)
                sent = len(events)
                await asyncio.sleep(0.25)
        except WebSocketDisconnect:
            return

    dist = Path(frontend_dist) if frontend_dist else _default_frontend_dist()
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(dist / "index.html")

        @app.get("/{path:path}")
        def spa(path: str) -> FileResponse:
            target = dist / path
            if target.exists() and target.is_file():
                return FileResponse(target)
            return FileResponse(dist / "index.html")
    else:

        @app.get("/")
        def no_frontend() -> HTMLResponse:
            return HTMLResponse(
                "<html><body><h1>Hexo-RL Dashboard API</h1>"
                "<p>Build Python/dashboard_frontend to serve the React UI.</p></body></html>"
            )

    return app


def _axis_input_from_request(store: DashboardStore, req: AxisEvaluateRequest) -> AxisPolicyInput:
    if req.position:
        return AxisPolicyInput(
            stones=list(req.position.get("stones", [])),
            legal_moves=list(req.position.get("legal_moves", [])),
        current_player=int(req.position.get("current_player", 0)),
        offset_q=int(req.position.get("offset_q", -16)),
        offset_r=int(req.position.get("offset_r", -16)),
        metadata={
            "placements_remaining": int(req.position.get("placements_remaining", 2)),
            **dict(req.position.get("metadata", {})),
        },
        )
    if req.session_id:
        payload = session_payload(store, req.session_id)
        pos = payload["position"]
    else:
        history = b""
        if req.game_id is not None:
            rows = store.rows("SELECT final_history_b64 FROM games WHERE game_id=?", (req.game_id,))
            if not rows:
                raise HTTPException(404, f"Game not found: {req.game_id}")
            history = rows[0]["final_history_b64"]
        elif req.history_b64:
            history = base64.b64decode(req.history_b64)
        position = get_replay_position(history, turn_index=req.turn_index)
        pos = position_payload(position)
    return AxisPolicyInput(
        stones=pos["stones"],
        legal_moves=pos["legal_moves"],
        current_player=int(pos["current_player"]),
        offset_q=int(pos["encoding"].get("offset_q", -16)),
        offset_r=int(pos["encoding"].get("offset_r", -16)),
        metadata={
            "source": "dashboard",
            "turn_index": int(pos.get("turn_index", 0)),
            "placements_remaining": int(pos.get("placements_remaining", 2)),
        },
    )


def _default_frontend_dist() -> Path:
    return Path(__file__).resolve().parents[3] / "dashboard_frontend" / "dist"


def default_app() -> FastAPI:
    """Factory target for ASGI runners that prefer a zero-argument callable."""
    return create_app()
