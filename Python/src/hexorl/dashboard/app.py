"""FastAPI dashboard application factory."""

from __future__ import annotations

import base64
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hexorl.axis_policy.core import AxisPolicyInput
from hexorl.contracts.candidates import (
    CANDIDATE_FEATURE_NAMES,
    CANDIDATE_FEATURE_VERSION,
    CandidateContractBuilder,
)
from hexorl.contracts.pairs import PairActionTableBuilder, PairStrategy
from hexorl.contracts.history import MoveHistory
from hexorl.contracts.symmetry import (
    transform_history,
    transform_pair_policy_target,
    transform_policy_target,
)
from hexorl.axis_policy.registry import describe_prototypes, evaluate_all, get_prototype
from hexorl.dashboard.arena_service import ArenaManager
from hexorl.dashboard.checkpoints import scan_checkpoints
from hexorl.dashboard.db import DashboardStore, decode_bytes
from hexorl.dashboard.fixtures import (
    ClassicalFixtureConfig,
    generate_classical_fixtures,
    list_axis_fixtures,
)
from hexorl.dashboard.model_cache import ModelCache
from hexorl.dashboard.play import apply_move, create_session, reset_session, session_payload, undo_move
from hexorl.dashboard.render import MatchSnapshotOptions, render_match_snapshot_png, snapshot_filename
from hexorl.dashboard.replay import get_replay_position, position_payload, replay_game
from hexorl.graph.semantic_builder import (
    GRAPH_FEATURE_DIM,
    GRAPH_SCHEMA_VERSION,
    GRAPH_CAPACITY_STRATEGY,
    RELATION_SCHEMA_VERSION,
    GraphTokenType,
    RelationType,
)
from hexorl.graph.tensorize import (
    build_graph_batch_from_history,
    graph_capacity_report,
)
from hexorl.selfplay.records import BOARD_SIZE


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


class AxisFixtureGenerateRequest(BaseModel):
    count: int = 8
    examples_per_move_count: int = 3
    move_counts: list[int] = Field(default_factory=lambda: [8, 16, 24, 32, 40])
    time_ms: int = 2
    max_depth: int = 1
    near_radius: int = 6
    noise_level: float = 0.08
    random_move_prob: float = 0.04
    opening_random_moves: int = 2
    seed: int = 0
    workers: int = 4


class ModelLoadRequest(BaseModel):
    path: str


class InferRequest(BaseModel):
    history_b64: str
    model_ids: list[str] = Field(default_factory=list)
    policy_target_v2: list[Any] = Field(default_factory=list)
    pair_policy_target_v2: list[Any] = Field(default_factory=list)


class ArenaStartRequest(BaseModel):
    run_id: str | None = None
    side_a: str = "model"
    side_b: str = "classical"
    payload: dict[str, Any] = Field(default_factory=dict)


def _game_summary(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload_json", {}) or {}
    return {
        "game_id": row["game_id"],
        "run_id": row["run_id"],
        "trial_id": row.get("trial_id") or row["run_id"],
        "external_game_id": row["external_game_id"],
        "source": row["source"],
        "epoch": row["epoch"],
        "outcome": row["outcome"],
        "move_count": row["move_count"],
        "created_at": row["created_at"],
        "terminal_reason": payload.get("terminal_reason", ""),
        "truncated": bool(payload.get("truncated", False)),
        "positions": payload.get("positions"),
        "payload": payload,
    }


def create_app(
    db_path: Path | str = "runs/dashboard.sqlite3",
    *,
    frontend_dist: Path | str | None = None,
    run_root: Path | str | None = None,
) -> FastAPI:
    store = DashboardStore(db_path)
    suite_root = Path(run_root).expanduser().resolve() if run_root else None
    model_cache = ModelCache()
    arena_manager = ArenaManager(store)
    app = FastAPI(title="Hexo-RL Dashboard", version="0.1.0")
    app.state.store = store
    app.state.suite_root = suite_root
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
        return {
            "ok": True,
            "schema_version": 1,
            "db_path": str(store.path),
            "suite_enabled": suite_root is not None,
            "suite_run_root": str(suite_root) if suite_root else None,
        }

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        if suite_root is not None:
            rows = _suite_runs(suite_root)
            if rows:
                return rows
        return store.rows("SELECT * FROM runs ORDER BY updated_at DESC")

    @app.get("/api/metrics/{run_id}")
    def metrics(run_id: str, limit: int = 500) -> list[dict[str, Any]]:
        trial_store = _suite_store_for_run(suite_root, run_id)
        source = trial_store or store
        rows = source.rows(
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
        trial_store = _suite_store_for_run(suite_root, run_id)
        source = trial_store or store
        return source.rows(
            "SELECT * FROM events WHERE run_id=? ORDER BY created_at DESC LIMIT ?",
            (run_id, max(1, min(limit, 5000))),
        )

    @app.get("/api/checkpoints")
    def checkpoints(run_id: str | None = None) -> list[dict[str, Any]]:
        if suite_root is not None:
            return _suite_checkpoints(suite_root, run_id=run_id)
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
        if suite_root is not None:
            rows = _suite_games(suite_root, run_id=run_id, limit=max(1, min(limit, 2000)))
            return [_game_summary(row) for row in rows]
        if run_id:
            rows = store.rows(
                "SELECT * FROM games WHERE run_id=? ORDER BY created_at DESC LIMIT ?",
                (run_id, max(1, min(limit, 2000))),
            )
        else:
            rows = store.rows(
                "SELECT * FROM games ORDER BY created_at DESC LIMIT ?",
                (max(1, min(limit, 2000)),),
            )
        return [_game_summary(row) for row in rows]

    @app.get("/api/games/{game_id}/replay")
    def game_replay(game_id: int, run_id: str | None = None) -> dict[str, Any]:
        source = _suite_store_for_run(suite_root, run_id) if run_id else store
        try:
            return replay_game(source or store, game_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/games/{game_id}/position/{turn_index}")
    def game_position(game_id: int, turn_index: int, run_id: str | None = None) -> dict[str, Any]:
        source = _suite_store_for_run(suite_root, run_id) if run_id else store
        rows = (source or store).rows("SELECT final_history_b64 FROM games WHERE game_id=?", (game_id,))
        if not rows:
            raise HTTPException(404, f"Game not found: {game_id}")
        pos = get_replay_position(rows[0]["final_history_b64"], turn_index=turn_index)
        return position_payload(pos)

    @app.get("/api/games/{game_id}/snapshot.png")
    def game_snapshot(
        game_id: int,
        run_id: str | None = None,
        turn_index: int = -1,
        width: int = 1280,
        height: int = 960,
        context_rings: int = 2,
        show_numbers: bool = True,
        show_legal: bool = False,
        fit: str = "played",
        near_radius: int = 8,
    ) -> Response:
        row = _game_row_for_request(store, suite_root, game_id, run_id)
        if row is None:
            raise HTTPException(404, f"Game not found: {game_id}")
        history = row["final_history_b64"]
        turn = None if turn_index < 0 else turn_index
        legal_moves = None
        if show_legal:
            try:
                legal_moves = get_replay_position(
                    history,
                    turn_index=turn,
                    near_radius=max(1, min(near_radius, 64)),
                    constrain_threats=False,
                ).legal_moves
            except Exception:
                legal_moves = None
        title = f"{row.get('source') or 'match'} epoch {row.get('epoch') if row.get('epoch') is not None else '-'}"
        png = render_match_snapshot_png(
            history,
            options=MatchSnapshotOptions(
                width=width,
                height=height,
                turn_index=turn,
                context_rings=context_rings,
                show_numbers=show_numbers,
                show_legal=show_legal,
                fit=fit,
                title=title,
            ),
            legal_moves=legal_moves,
            metadata=row,
        )
        filename = snapshot_filename(row, turn_index=turn)
        return Response(
            content=png,
            media_type="image/png",
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )

    @app.get("/api/suite/status")
    def suite_status() -> dict[str, Any]:
        if suite_root is None:
            return {"enabled": False}
        return _suite_status(suite_root)

    @app.get("/api/suite/trials")
    def suite_trials() -> list[dict[str, Any]]:
        if suite_root is None:
            return []
        return _suite_trials(suite_root)

    @app.get("/api/suite/trials/{trial_id}")
    def suite_trial_detail(trial_id: str) -> dict[str, Any]:
        if suite_root is None:
            raise HTTPException(404, "Suite run root is not configured")
        detail = _suite_trial_detail(suite_root, trial_id)
        if not detail:
            raise HTTPException(404, f"Trial not found: {trial_id}")
        return detail

    @app.get("/api/suite/best-checkpoints")
    def suite_best_checkpoints(limit: int = 50) -> list[dict[str, Any]]:
        if suite_root is None:
            return []
        return _suite_best_checkpoints(suite_root, limit=max(1, min(limit, 500)))

    @app.get("/api/suite/events")
    def suite_events(limit: int = 200) -> list[dict[str, Any]]:
        if suite_root is None:
            return []
        return _jsonl_tail(suite_root / "events.jsonl", limit=max(1, min(limit, 1000)))

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

    @app.get("/api/axis/fixtures")
    def axis_fixtures(limit: int = 200) -> list[dict[str, Any]]:
        return list_axis_fixtures(store, limit=limit)

    @app.post("/api/axis/fixtures/generate")
    def axis_fixtures_generate(req: AxisFixtureGenerateRequest) -> dict[str, Any]:
        try:
            fixtures = generate_classical_fixtures(
                store,
                ClassicalFixtureConfig(
                    count=max(1, min(req.count, 64)),
                    examples_per_move_count=max(0, min(req.examples_per_move_count, 16)),
                    move_counts=tuple(req.move_counts),
                    time_ms=max(1, min(req.time_ms, 500)),
                    max_depth=max(1, min(req.max_depth, 8)),
                    near_radius=max(1, min(req.near_radius, 32)),
                    noise_level=max(0.0, min(req.noise_level, 2.0)),
                    random_move_prob=max(0.0, min(req.random_move_prob, 1.0)),
                    opening_random_moves=max(0, min(req.opening_random_moves, 12)),
                    seed=req.seed,
                    workers=max(1, min(req.workers, 8)),
                ),
            )
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"fixtures": fixtures}

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

    @app.get("/api/debug/contracts")
    def debug_contracts() -> dict[str, Any]:
        return {
            "candidate": {
                "feature_version": CANDIDATE_FEATURE_VERSION,
                "feature_names": list(CANDIDATE_FEATURE_NAMES),
                "feature_width": len(CANDIDATE_FEATURE_NAMES),
            },
            "graph": {
                "schema_version": GRAPH_SCHEMA_VERSION,
                "relation_schema_version": RELATION_SCHEMA_VERSION,
                "feature_dim": GRAPH_FEATURE_DIM,
                "capacity_strategy": GRAPH_CAPACITY_STRATEGY,
                "token_types": {token.name: int(token) for token in GraphTokenType},
                "relation_types": {relation.name: int(relation) for relation in RelationType},
            },
        }

    @app.post("/api/debug/graph")
    def debug_graph(req: InferRequest) -> dict[str, Any]:
        history = decode_bytes(req.history_b64)
        policy_target = _parse_policy_target_v2(req.policy_target_v2)
        pair_target = _parse_pair_policy_target_v2(req.pair_policy_target_v2)
        return _graph_debug_payload(history, policy_target=policy_target, pair_policy_target=pair_target)

    @app.post("/api/debug/d6")
    def debug_d6(req: InferRequest) -> dict[str, Any]:
        history = decode_bytes(req.history_b64)
        base_policy_target = _parse_policy_target_v2(req.policy_target_v2)
        base_pair_target = _parse_pair_policy_target_v2(req.pair_policy_target_v2)
        transforms = []
        for sym_idx in range(12):
            transformed = transform_history(history, sym_idx)
            policy_target = transform_policy_target(base_policy_target, sym_idx)
            pair_target = transform_pair_policy_target(base_pair_target, sym_idx)
            graph = _graph_debug_payload(
                transformed,
                policy_target=policy_target,
                pair_policy_target=pair_target,
            )
            position = position_payload(get_replay_position(transformed, constrain_threats=False))
            model_logits = {}
            for model_id in req.model_ids:
                try:
                    model_logits[model_id] = model_cache.infer_history(model_id, transformed)
                except KeyError as exc:
                    raise HTTPException(404, f"Model not loaded: {model_id}") from exc
            transforms.append(
                {
                    "symmetry_index": sym_idx,
                    "history_b64": base64.b64encode(transformed).decode("ascii"),
                    "current_player": position["current_player"],
                    "placements_remaining": position["placements_remaining"],
                    "legal_count": len(position["legal_moves"]),
                    "graph": graph,
                    "contracts": _d6_contract_payload(
                        transformed,
                        position,
                        graph,
                        policy_target=policy_target,
                        pair_policy_target=pair_target,
                    ),
                    "model_logits": model_logits,
                }
            )
        return {
            "symmetry_count": 12,
            "source_history_b64": base64.b64encode(history).decode("ascii"),
            "target_checks": _d6_target_checks(transforms),
            "transforms": transforms,
        }

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


def _suite_trial_dirs(run_root: Path) -> list[Path]:
    trials = run_root / "trials"
    if not trials.exists():
        return []
    return sorted([path for path in trials.iterdir() if (path / "dashboard.sqlite3").exists()])


def _game_row_for_request(
    store: DashboardStore,
    run_root: Path | None,
    game_id: int,
    run_id: str | None,
) -> dict[str, Any] | None:
    if run_id:
        trial_store = _suite_store_for_run(run_root, run_id)
        if trial_store is not None:
            rows = trial_store.rows("SELECT * FROM games WHERE game_id=?", (game_id,))
            if rows:
                rows[0]["source_db"] = str(trial_store.path)
                return rows[0]
        rows = store.rows("SELECT * FROM games WHERE game_id=? AND run_id=?", (game_id, run_id))
        return rows[0] if rows else None
    rows = store.rows("SELECT * FROM games WHERE game_id=?", (game_id,))
    if rows:
        return rows[0]
    if run_root is None:
        return None
    for trial_dir in _suite_trial_dirs(run_root):
        db = trial_dir / "dashboard.sqlite3"
        try:
            trial_store = DashboardStore(db)
            rows = trial_store.rows("SELECT * FROM games WHERE game_id=?", (game_id,))
        except Exception:
            continue
        if rows:
            rows[0]["trial_id"] = trial_dir.name
            rows[0]["source_db"] = str(db)
            return rows[0]
    return None


def _suite_store_for_run(run_root: Path | None, run_id: str | None) -> DashboardStore | None:
    if run_root is None or not run_id:
        return None
    direct = run_root / "trials" / run_id / "dashboard.sqlite3"
    if direct.exists():
        return DashboardStore(direct)
    for trial_dir in _suite_trial_dirs(run_root):
        db = trial_dir / "dashboard.sqlite3"
        try:
            rows = DashboardStore(db).rows("SELECT run_id FROM runs WHERE run_id=? LIMIT 1", (run_id,))
        except Exception:
            continue
        if rows:
            return DashboardStore(db)
    return None


def _suite_runs(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trial_state = {trial["trial_id"]: trial for trial in _suite_state(run_root).get("trials", [])}
    for trial_dir in _suite_trial_dirs(run_root):
        db = trial_dir / "dashboard.sqlite3"
        try:
            run_rows = DashboardStore(db).rows("SELECT * FROM runs ORDER BY updated_at DESC")
        except Exception:
            continue
        for row in run_rows:
            trial_id = str(row.get("run_id") or trial_dir.name)
            row["trial_id"] = trial_id
            row["source_db"] = str(db)
            row["name"] = trial_id
            state = trial_state.get(trial_id, {})
            row["payload_json"] = {
                **dict(row.get("payload_json") or {}),
                "family": (state.get("family") or {}).get("name"),
                "pruned": state.get("pruned"),
                "last_score": state.get("last_score"),
            }
            rows.append(row)
    rows.sort(key=lambda row: float(row.get("updated_at") or 0.0), reverse=True)
    return rows


def _suite_games(run_root: Path, *, run_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    trial_dirs = [run_root / "trials" / run_id] if run_id else _suite_trial_dirs(run_root)
    rows: list[dict[str, Any]] = []
    for trial_dir in trial_dirs:
        db = trial_dir / "dashboard.sqlite3"
        if not db.exists():
            continue
        try:
            store = DashboardStore(db)
            if run_id:
                next_rows = store.rows(
                    "SELECT * FROM games WHERE run_id=? ORDER BY created_at DESC LIMIT ?",
                    (run_id, limit),
                )
            else:
                next_rows = store.rows("SELECT * FROM games ORDER BY created_at DESC LIMIT ?", (limit,))
        except Exception:
            continue
        for row in next_rows:
            row["trial_id"] = trial_dir.name
            row["source_db"] = str(db)
            rows.append(row)
    rows.sort(key=lambda row: float(row.get("created_at") or 0.0), reverse=True)
    return rows[:limit]


def _suite_checkpoints(run_root: Path, *, run_id: str | None = None) -> list[dict[str, Any]]:
    trial_dirs = [run_root / "trials" / run_id] if run_id else _suite_trial_dirs(run_root)
    rows: list[dict[str, Any]] = []
    score_by_trial = _suite_score_by_trial(run_root)
    for trial_dir in trial_dirs:
        db = trial_dir / "dashboard.sqlite3"
        if not db.exists():
            continue
        try:
            store = DashboardStore(db)
            if run_id:
                next_rows = store.rows(
                    "SELECT * FROM checkpoints WHERE run_id=? ORDER BY indexed_at DESC",
                    (run_id,),
                )
            else:
                next_rows = store.rows("SELECT * FROM checkpoints ORDER BY indexed_at DESC")
        except Exception:
            continue
        for row in next_rows:
            trial_id = str(row.get("run_id") or trial_dir.name)
            row["trial_id"] = trial_id
            row["source_db"] = str(db)
            row["score"] = score_by_trial.get(trial_id)
            row["scheduler_score"] = row["score"]
            rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row.get("score") if row.get("score") is not None else float("-inf")),
            int(row.get("epoch") or -1),
            float(row.get("indexed_at") or 0.0),
        ),
        reverse=True,
    )
    return rows


def _suite_best_checkpoints(run_root: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    rows = _suite_checkpoints(run_root)
    best_by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = str(row.get("path") or "")
        if not path:
            continue
        current = best_by_path.get(path)
        score = float(row.get("score") if row.get("score") is not None else float("-inf"))
        current_score = float(current.get("score") if current and current.get("score") is not None else float("-inf"))
        if current is None or (score, int(row.get("epoch") or -1)) > (current_score, int(current.get("epoch") or -1)):
            best_by_path[path] = row
    ranked = list(best_by_path.values())
    ranked.sort(
        key=lambda row: (
            float(row.get("score") if row.get("score") is not None else float("-inf")),
            int(row.get("epoch") or -1),
            float(row.get("indexed_at") or 0.0),
        ),
        reverse=True,
    )
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked[:limit]


def _suite_trials(run_root: Path) -> list[dict[str, Any]]:
    state_trials = {trial["trial_id"]: trial for trial in _suite_state(run_root).get("trials", [])}
    score_by_trial = _suite_score_by_trial(run_root)
    rows: list[dict[str, Any]] = []
    for trial_dir in _suite_trial_dirs(run_root):
        trial_id = trial_dir.name
        state = dict(state_trials.get(trial_id) or {})
        trial_json = _read_json(trial_dir / "trial.json")
        latest = _read_json(trial_dir / "LATEST.json")
        scores = _jsonl_tail(trial_dir / "scores.jsonl", limit=1)
        family = state.get("family") or trial_json.get("family") or {}
        counts = _suite_trial_counts(trial_dir)
        latest_selfplay = latest.get("selfplay") or {}
        latest_train = latest.get("train") or {}
        score = score_by_trial.get(trial_id, state.get("last_score"))
        if (score is None or score == "-inf") and scores:
            score = scores[-1].get("scheduler_score")
        row = {
            "trial_id": trial_id,
            "family": family.get("name") or latest.get("family") or "",
            "architecture": family.get("architecture") or "",
            "model_summary": _model_summary_from_trial(family, state.get("static") or trial_json.get("static") or latest.get("static") or {}),
            "stage": latest.get("stage") or state.get("stage") or trial_json.get("stage") or "",
            "epoch": state.get("epoch") or latest.get("epoch") or trial_json.get("epoch") or 0,
            "score": _finite_or_none(score),
            "pruned": bool(state.get("pruned") or trial_json.get("pruned") or False),
            "prune_reason": state.get("prune_reason") or trial_json.get("prune_reason") or "",
            "checkpoint_path": state.get("checkpoint_path") or latest.get("checkpoint_path") or "",
            "games": counts["games"],
            "positions": counts["positions"],
            "checkpoints": counts["checkpoints"],
            "metrics": counts["metrics"],
            "selfplay_positions_per_min": latest_selfplay.get("positions_per_min"),
            "positions_per_sec": _per_second(latest_selfplay.get("positions_per_min")),
            "workers": _worker_summary(latest_selfplay),
            "epoch_elapsed_s": latest.get("epoch_elapsed_s"),
            "loss_total": latest_train.get("loss_total"),
            "policy_top1_acc": latest_train.get("policy_top1_acc"),
            "sparse_policy_top1_acc": latest_train.get("sparse_policy_top1_acc"),
            "pair_policy_top1_acc": latest_train.get("pair_policy_top1_acc"),
            "runtime_sweep": state.get("runtime_sweep") or trial_json.get("runtime_sweep") or {},
            "updated_at": max(_mtime(trial_dir / "LATEST.json"), _mtime(trial_dir / "dashboard.sqlite3")),
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            not bool(row.get("pruned")),
            float(row.get("score") if row.get("score") is not None else float("-inf")),
            float(row.get("updated_at") or 0.0),
        ),
        reverse=True,
    )
    return rows


def _suite_status(run_root: Path) -> dict[str, Any]:
    manifest = _read_json(run_root / "manifest.json")
    state = _suite_state(run_root)
    events = _jsonl_tail(run_root / "events.jsonl", limit=100)
    trials = _suite_trials(run_root)
    latest_stage = _latest_suite_stage(state, events, trials)
    total_games = sum(int(trial.get("games") or 0) for trial in trials)
    total_positions = sum(int(trial.get("positions") or 0) for trial in trials)
    best = next((trial for trial in trials if not trial.get("pruned") and trial.get("score") is not None), None)
    activity = _supervisor_activity(run_root, trials, events, manifest)
    return {
        "enabled": True,
        "run_root": str(run_root),
        "latest_stage": latest_stage,
        "trial_count": len(trials),
        "live_trial_count": sum(1 for trial in trials if not trial.get("pruned")),
        "total_games": total_games,
        "total_positions": total_positions,
        "best_trial_id": best.get("trial_id") if best else None,
        "best_score": best.get("score") if best else None,
        "manifest": manifest,
        "state_elapsed_s": state.get("elapsed_s"),
        "host": manifest.get("host", {}),
        "args": manifest.get("args", {}),
        "last_event": events[-1] if events else None,
        "current_activity": activity,
        "current_trial_id": activity.get("trial_id"),
        "current_model": activity.get("model"),
        "current_positions_per_sec": activity.get("positions_per_sec"),
        "current_stage": activity.get("stage") or latest_stage,
        "last_event_name": (events[-1] if events else {}).get("event"),
        "last_event_time": (events[-1] if events else {}).get("time"),
    }


def _suite_trial_detail(run_root: Path, trial_id: str) -> dict[str, Any]:
    trial_dir = run_root / "trials" / trial_id
    if not trial_dir.exists():
        return {}
    state = next((trial for trial in _suite_state(run_root).get("trials", []) if trial.get("trial_id") == trial_id), {})
    trial_json = _read_json(trial_dir / "trial.json")
    latest = _read_json(trial_dir / "LATEST.json")
    scores = _jsonl_tail(trial_dir / "scores.jsonl", limit=12)
    summaries = _jsonl_tail(trial_dir / "summary.jsonl", limit=12)
    events = _jsonl_tail(trial_dir / "events.jsonl", limit=80)
    checkpoints = _suite_checkpoints(run_root, run_id=trial_id)
    checkpoint_path = (
        state.get("checkpoint_path")
        or latest.get("checkpoint_path")
        or (checkpoints[0].get("path") if checkpoints else "")
    )
    checkpoint = _checkpoint_metadata(Path(checkpoint_path)) if checkpoint_path else {}
    cfg = checkpoint.get("cfg") or {}
    model_metadata = checkpoint.get("model_metadata") or cfg.get("model") or {}
    family = state.get("family") or trial_json.get("family") or {}
    static = state.get("static") or trial_json.get("static") or latest.get("static") or {}
    architecture = model_metadata or {
        "architecture": family.get("architecture"),
        "channels": family.get("channels"),
        "blocks": family.get("blocks"),
        "heads": trial_json.get("heads"),
        "graph_token_set": static.get("graph_token_set"),
        "graph_token_budget": static.get("graph_token_budget"),
        "graph_layers": static.get("graph_layers"),
        "candidate_budget": static.get("candidate_budget"),
        "sparse_prior_stage": static.get("sparse_prior_stage"),
    }
    return {
        "trial_id": trial_id,
        "trial_dir": str(trial_dir),
        "trial": trial_json,
        "state": state,
        "latest": latest,
        "scores": scores,
        "summary": summaries,
        "events": events,
        "checkpoints": checkpoints,
        "checkpoint_metadata": checkpoint,
        "config": cfg,
        "model_metadata": model_metadata,
        "architecture": architecture,
        "architecture_summary": _architecture_summary(architecture, family),
        "current_activity": _trial_activity(events, latest),
    }


def _suite_state(run_root: Path) -> dict[str, Any]:
    return _read_json(run_root / "state.json")


def _suite_trial_counts(trial_dir: Path) -> dict[str, int]:
    db = trial_dir / "dashboard.sqlite3"
    counts = {"games": 0, "positions": 0, "checkpoints": 0, "metrics": 0}
    if not db.exists():
        return counts
    try:
        store = DashboardStore(db)
        for key, table in [
            ("games", "games"),
            ("positions", "positions"),
            ("checkpoints", "checkpoints"),
            ("metrics", "metrics"),
        ]:
            rows = store.rows(f"SELECT COUNT(*) AS n FROM {table}")
            counts[key] = int(rows[0]["n"] if rows else 0)
    except Exception:
        pass
    return counts


def _suite_score_by_trial(run_root: Path) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for trial in _suite_state(run_root).get("trials", []):
        trial_id = trial.get("trial_id")
        if not trial_id:
            continue
        scores[str(trial_id)] = _finite_or_none(trial.get("last_score"))
    for trial_dir in _suite_trial_dirs(run_root):
        trial_id = trial_dir.name
        if scores.get(trial_id) is not None:
            continue
        for row in reversed(_jsonl_tail(trial_dir / "scores.jsonl", limit=32)):
            score = _finite_or_none(row.get("scheduler_score"))
            if score is None:
                score = _finite_or_none(row.get("score"))
            if score is not None:
                scores[trial_id] = score
                break
    return scores


def _latest_suite_stage(
    state: dict[str, Any],
    events: list[dict[str, Any]],
    trials: list[dict[str, Any]],
) -> str:
    for key in ("stage", "current_stage", "latest_stage"):
        if state.get(key):
            return str(state[key])
    for event in reversed(events):
        if event.get("stage"):
            return str(event["stage"])
    for trial in sorted(trials, key=lambda row: float(row.get("updated_at") or 0.0), reverse=True):
        if trial.get("stage"):
            return str(trial["stage"])
    return ""


def _jsonl_tail(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _supervisor_activity(
    run_root: Path,
    trials: list[dict[str, Any]],
    events: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    trial_by_id = {str(trial.get("trial_id")): trial for trial in trials if trial.get("trial_id")}
    latest_trial_id = ""
    for event in reversed(events):
        if event.get("trial_id"):
            latest_trial_id = str(event.get("trial_id"))
            break
    trial = trial_by_id.get(latest_trial_id, {})
    log_lines = _tail_lines(run_root / "supervisor.log", limit=120)
    progress = _last_progress(log_lines)
    latest_event = events[-1] if events else {}
    max_game_moves = int((manifest.get("args") or {}).get("max_game_moves") or 0)
    positions_per_sec = None
    if progress and max_game_moves > 0 and float(progress["games_per_min"]) > 0.0:
        positions_per_sec = float(progress["games_per_min"]) * max_game_moves / 60.0
    if positions_per_sec is None:
        positions_per_sec = _event_positions_per_second(latest_event)
    if positions_per_sec is None and trial.get("positions_per_sec") is not None:
        positions_per_sec = trial.get("positions_per_sec")
    action = "Waiting for supervisor events"
    if progress:
        action = f"Self-play running, {progress['progress_pct']:.1f}% of current epoch"
    elif latest_event.get("event"):
        action = _event_blurb(str(latest_event.get("event")))
    stage = latest_event.get("stage") or trial.get("stage") or ""
    return {
        "trial_id": latest_trial_id or None,
        "model": trial.get("family") or latest_event.get("family") or None,
        "architecture": trial.get("architecture") or None,
        "stage": stage,
        "action": action,
        "positions_per_sec": positions_per_sec,
        "progress": progress,
        "last_log_line": log_lines[-1] if log_lines else "",
        "log_tail": log_lines[-20:],
        "last_event": latest_event,
    }


def _trial_activity(events: list[dict[str, Any]], latest: dict[str, Any]) -> dict[str, Any]:
    event = events[-1] if events else {}
    latest_selfplay = latest.get("selfplay") or {}
    latest_train = latest.get("train") or {}
    return {
        "event": event.get("event_type") or event.get("event") or "",
        "phase": event.get("phase") or latest.get("stage") or "",
        "epoch": latest.get("epoch") or latest_train.get("epoch"),
        "positions_per_sec": _per_second(latest_selfplay.get("positions_per_min")),
        "loss_total": latest_train.get("loss_total"),
        "updated_at": latest.get("epoch_elapsed_s"),
    }


def _last_progress(lines: list[str]) -> dict[str, Any] | None:
    pattern = re.compile(
        r"Progress:\s+([0-9.]+)%\s+\|\s+Games:\s+(\d+)\s+\(([0-9.]+)/min\)\s+\|\s+Buffer:\s+(\d+)\s+\|\s+Workers:\s+(\d+)/(\d+)"
    )
    for line in reversed(lines):
        match = pattern.search(line)
        if not match:
            continue
        return {
            "progress_pct": float(match.group(1)),
            "games_done": int(match.group(2)),
            "games_per_min": float(match.group(3)),
            "buffer_positions": int(match.group(4)),
            "workers_alive": int(match.group(5)),
            "workers_total": int(match.group(6)),
        }
    return None


def _tail_lines(path: Path, *, limit: int) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except OSError:
        return []


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "error": "checkpoint_not_found"}
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    cfg = checkpoint.get("cfg_json") or {}
    model_metadata = checkpoint.get("model_metadata") or cfg.get("model") or {}
    state = checkpoint.get("model_state_dict") or {}
    return {
        "path": str(path),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "cfg": cfg,
        "model_metadata": model_metadata,
        "action_contract_metadata": checkpoint.get("action_contract_metadata", {}),
        "model_parameter_tensors": len(state) if hasattr(state, "__len__") else None,
    }


def _architecture_summary(model: dict[str, Any], family: dict[str, Any] | None = None) -> str:
    family = family or {}
    arch = str(model.get("architecture") or family.get("architecture") or "").lower()
    channels = model.get("channels") or family.get("channels")
    blocks = model.get("blocks") or family.get("blocks")
    heads = model.get("heads") or []
    if arch in {"graph", "graph_hybrid_0"}:
        return (
            f"Graph hybrid 0, {channels} channels, {blocks} residual blocks, "
            f"{model.get('graph_token_budget', '?')} {model.get('graph_token_set', 'tokens')}, "
            f"{model.get('graph_layers', '?')} graph layers, heads: {len(heads)}."
        )
    if arch == "restnet":
        return (
            f"ResTNet hybrid trunk, {channels} channels, {blocks} blocks, "
            f"attention at {model.get('attention_positions') or []}, heads: {len(heads)}."
        )
    return f"CNN residual trunk, {channels} channels, {blocks} blocks, heads: {len(heads)}."


def _model_summary_from_trial(family: dict[str, Any], static: dict[str, Any]) -> str:
    arch = str(family.get("architecture") or "")
    if arch in {"graph", "graph_hybrid_0"}:
        return f"graph_hybrid_0 {static.get('graph_token_budget', '?')} tokens x {static.get('graph_layers', '?')} layers"
    return f"{arch or 'model'} {family.get('channels', '?')}x{family.get('blocks', '?')}"


def _worker_summary(selfplay: dict[str, Any]) -> str:
    alive = selfplay.get("workers_alive")
    total = selfplay.get("workers_total")
    if alive is None and total is None:
        return ""
    return f"{alive or 0}/{total or 0}"


def _per_second(positions_per_min: Any) -> float | None:
    try:
        return float(positions_per_min) / 60.0
    except (TypeError, ValueError):
        return None


def _event_positions_per_second(event: dict[str, Any]) -> float | None:
    for key in ("positions_per_min", "selected_positions_per_min"):
        value = _per_second(event.get(key))
        if value is not None:
            return value
    selfplay = event.get("selfplay") or {}
    value = _per_second(selfplay.get("positions_per_min"))
    if value is not None:
        return value
    throughput = event.get("throughput") or {}
    return _per_second(throughput.get("selfplay_positions_per_min"))


def _event_blurb(event: str) -> str:
    return {
        "runtime_sweep_start": "Runtime sweep is testing worker/batch settings",
        "runtime_sweep_result": "Runtime sweep recorded a probe result",
        "runtime_sweep_selected": "Runtime sweep selected the fastest stable setting",
        "trial_epoch_complete": "Epoch finished; metrics and checkpoint were written",
        "trial_evaluated": "Evaluation finished; scheduler score updated",
        "trial_pruned": "Trial was pruned by a hard gate or scheduler decision",
        "pbt_generation_start": "PBT generation started",
        "stage_start": "Autotune stage started",
    }.get(event, event.replace("_", " "))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number == float("inf") or number == float("-inf") or number != number:
        return None
    return number


def _parse_policy_target_v2(rows: list[Any]) -> list[tuple[int, int, float]]:
    parsed: list[tuple[int, int, float]] = []
    for row in rows or []:
        if isinstance(row, dict):
            q = row.get("q")
            r = row.get("r")
            prob = row.get("prob", row.get("p", row.get("weight", 0.0)))
        else:
            if len(row) != 3:
                raise HTTPException(400, "policy_target_v2 rows must be [q, r, probability]")
            q, r, prob = row
        prob_f = float(prob)
        if prob_f > 0.0:
            parsed.append((int(q), int(r), prob_f))
    return parsed


def _parse_pair_policy_target_v2(rows: list[Any]) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    parsed: list[tuple[tuple[int, int], tuple[int, int], float]] = []
    for row in rows or []:
        if isinstance(row, dict):
            first_raw = row.get("first")
            second_raw = row.get("second")
            prob = row.get("prob", row.get("p", row.get("weight", 0.0)))
        else:
            if len(row) != 3:
                raise HTTPException(400, "pair_policy_target_v2 rows must be [[q1, r1], [q2, r2], probability]")
            first_raw, second_raw, prob = row
        first = (
            (int(first_raw["q"]), int(first_raw["r"]))
            if isinstance(first_raw, dict)
            else (int(first_raw[0]), int(first_raw[1]))
        )
        second = (
            (int(second_raw["q"]), int(second_raw["r"]))
            if isinstance(second_raw, dict)
            else (int(second_raw[0]), int(second_raw[1]))
        )
        prob_f = float(prob)
        if prob_f > 0.0:
            parsed.append((first, second, prob_f))
    return parsed


def _d6_target_checks(transforms: list[dict[str, Any]]) -> dict[str, Any]:
    policy_masses = [
        float(item["contracts"]["sparse_candidates"].get("target_mass", 0.0))
        for item in transforms
    ]
    pair_masses = [
        float(item["contracts"]["pair_rows"].get("target_mass", 0.0))
        for item in transforms
    ]
    graph_policy_masses = [
        float(item["contracts"]["graph_targets"].get("target_masses", {}).get("policy", 0.0))
        for item in transforms
    ]
    graph_pair_masses = [
        float(item["contracts"]["graph_targets"].get("target_masses", {}).get("pair", 0.0))
        for item in transforms
    ]

    def stable(values: list[float]) -> bool:
        if not values:
            return True
        first = values[0]
        return all(abs(value - first) <= 1e-5 for value in values)

    return {
        "policy_target_mass_preserved": stable(policy_masses) and stable(graph_policy_masses),
        "pair_target_mass_preserved": stable(pair_masses) and stable(graph_pair_masses),
        "policy_target_masses": policy_masses,
        "pair_target_masses": pair_masses,
        "graph_policy_target_masses": graph_policy_masses,
        "graph_pair_target_masses": graph_pair_masses,
    }


def _last_history_qr(history: bytes) -> tuple[int, int] | None:
    if len(history) < 12:
        return None
    moves = MoveHistory.decode(history, source="rust").rows
    if not moves:
        return None
    _player, q, r = moves[-1]
    return q, r


def _graph_debug_payload(
    history: bytes,
    *,
    policy_target: list[tuple[int, int, float]] | None = None,
    pair_policy_target: list[tuple[tuple[int, int], tuple[int, int], float]] | None = None,
) -> dict[str, Any]:
    graph = build_graph_batch_from_history(
        history,
        policy_target=policy_target or [],
        pair_policy_target=pair_policy_target or [],
    )
    capacity = graph_capacity_report(graph)
    token_counts = {
        token.name: int((graph.token_type == int(token)).sum())
        for token in GraphTokenType
    }
    relation_counts = {
        relation.name: int((graph.relation_type == int(relation)).sum())
        for relation in RelationType
        if int((graph.relation_type == int(relation)).sum()) > 0
    }
    pair_examples = []
    token_qr = graph.token_qr
    for row in range(min(16, int(graph.pair_token_indices.shape[0]))):
        first = int(graph.pair_first_indices[row])
        second = int(graph.pair_second_indices[row])
        pair_examples.append(
            {
                "first": {"q": int(token_qr[first, 0]), "r": int(token_qr[first, 1])},
                "second": {"q": int(token_qr[second, 0]), "r": int(token_qr[second, 1])},
            }
        )
    return {
        "schema_version": graph.schema_version,
        "relation_schema_version": graph.relation_schema_version,
        "feature_dim": int(graph.token_features.shape[-1]),
        "capacity": {
            "fits_ipc": capacity.fits_ipc,
            "strategy": capacity.strategy,
            "failures": list(capacity.failures()),
            "max_tokens": capacity.max_tokens,
            "max_actions": capacity.max_actions,
            "max_pairs": capacity.max_pairs,
        },
        "token_count": int(graph.token_features.shape[0]),
        "token_counts": token_counts,
        "legal_count": int(graph.legal_qr.shape[0]),
        "legal_qr": [
            {"q": int(q), "r": int(r)}
            for q, r in graph.legal_qr[: min(64, int(graph.legal_qr.shape[0]))]
        ],
        "opp_legal_count": int(graph.opp_legal_qr.shape[0]),
        "pair_count": int(graph.pair_token_indices.shape[0]),
        "pair_examples": pair_examples,
        "relation_counts": relation_counts,
        "relation_bias_shape": [int(dim) for dim in graph.relation_bias.shape],
        "placements_remaining": int(graph.placements_remaining),
        "current_player": int(graph.current_player),
        "target_masses": {
            "policy": float(graph.policy_target.sum()),
            "pair": float(graph.pair_policy_target.sum()),
            "pair_first": float(graph.pair_first_policy_target.sum()),
            "opp_policy": float(graph.opp_policy_target.sum()),
        },
    }


def _d6_contract_payload(
    history: bytes,
    position: dict[str, Any],
    graph: dict[str, Any],
    *,
    policy_target: list[tuple[int, int, float]] | None = None,
    pair_policy_target: list[tuple[tuple[int, int], tuple[int, int], float]] | None = None,
) -> dict[str, Any]:
    legal_moves = [
        (int(row["q"]), int(row["r"]))
        for row in position.get("legal_moves", [])
    ]
    offset_q = int(position.get("encoding", {}).get("offset_q", -16))
    offset_r = int(position.get("encoding", {}).get("offset_r", -16))
    candidate_budget = min(max(len(legal_moves), 1), 512)
    candidates = CandidateContractBuilder().build(
        legal_moves,
        policy_target or [],
        offset_q=offset_q,
        offset_r=offset_r,
        budget=candidate_budget,
        storage_width=candidate_budget,
    )
    candidate_rows = [
        {
            "q": int(candidates.qr[row, 0]),
            "r": int(candidates.qr[row, 1]),
            "dense_index": int(candidates.indices[row]),
        }
        for row, active in enumerate(candidates.mask)
        if bool(active)
    ][:64]
    pair_rows: list[dict[str, Any]] = []
    pair_target_mass = 0.0
    pair_missing_mass = 0.0
    if len(legal_moves) >= 2 and int(position.get("placements_remaining", 1)) >= 2:
        pair_budget = min(512, len(legal_moves) * max(len(legal_moves) - 1, 0) // 2)
        pair = PairActionTableBuilder().build(
            candidates,
            pair_policy_target or [],
            strategy=PairStrategy(mode="capped_fill", max_pairs=max(1, pair_budget)),
            legal_moves=legal_moves,
        )
        pair_target_mass = float(pair.target.sum())
        pair_missing_mass = float(pair.missing_mass)
        for row, active in enumerate(pair.mask):
            if not bool(active) or len(pair_rows) >= 64:
                continue
            first_idx, second_idx = pair.pair_indices[row]
            pair_rows.append(
                {
                    "first": {
                        "q": int(candidates.qr[int(first_idx), 0]),
                        "r": int(candidates.qr[int(first_idx), 1]),
                    },
                    "second": {
                        "q": int(candidates.qr[int(second_idx), 0]),
                        "r": int(candidates.qr[int(second_idx), 1]),
                    },
                }
            )
    elif int(position.get("placements_remaining", 1)) == 1 and pair_policy_target:
        known_first = _last_history_qr(history)
        if known_first is not None:
            storage_width = min(max(len(legal_moves) + 1, 1), 512)
            pair_candidates = CandidateContractBuilder().build(
                [known_first] + legal_moves,
                [],
                offset_q=offset_q,
                offset_r=offset_r,
                budget=storage_width,
                storage_width=storage_width,
                critical_actions=[known_first] + legal_moves,
                source="rust:synthetic",
            )
            pair = PairActionTableBuilder().build(
                pair_candidates,
                pair_policy_target,
                strategy=PairStrategy(mode="capped_fill", max_pairs=min(max(len(legal_moves), 1), 512)),
                legal_moves=legal_moves,
                known_first=known_first,
                source="rust:synthetic",
            )
            pair_target_mass = float(pair.target.sum())
            pair_missing_mass = float(pair.missing_mass)
            for row, active in enumerate(pair.mask):
                if not bool(active) or len(pair_rows) >= 64:
                    continue
                first_idx, second_idx = pair.pair_indices[row]
                pair_rows.append(
                    {
                        "first": {
                            "q": int(pair_candidates.qr[int(first_idx), 0]),
                            "r": int(pair_candidates.qr[int(first_idx), 1]),
                        },
                        "second": {
                            "q": int(pair_candidates.qr[int(second_idx), 0]),
                            "r": int(pair_candidates.qr[int(second_idx), 1]),
                        },
                    }
                )
    axis_input = AxisPolicyInput(
        stones=list(position.get("stones", [])),
        legal_moves=list(position.get("legal_moves", [])),
        current_player=int(position.get("current_player", 0)),
        offset_q=offset_q,
        offset_r=offset_r,
        metadata={
            "source": "dashboard_d6_debug",
            "placements_remaining": int(position.get("placements_remaining", 1)),
            "history_b64": base64.b64encode(history).decode("ascii"),
        },
    )
    axis_results = evaluate_all(axis_input, {})
    return {
        "dense_legal_mask": {
            "offset_q": offset_q,
            "offset_r": offset_r,
            "legal_indices": list(position.get("encoding", {}).get("legal_mask", []))[:128],
            "legal_count": len(legal_moves),
        },
        "sparse_candidates": {
            "feature_version": CANDIDATE_FEATURE_VERSION,
            "feature_names": list(CANDIDATE_FEATURE_NAMES),
            "candidate_count": int(candidates.mask.sum()),
            "target_mass": float(candidates.target.sum()),
            "missing_mass": float(candidates.missing_mass),
            "rows": candidate_rows,
        },
        "pair_rows": {
            "available": bool(pair_rows),
            "target_mass": pair_target_mass,
            "missing_mass": pair_missing_mass,
            "rows": pair_rows,
        },
        "axis": {
            "prototype_count": len(axis_results),
            "results": axis_results[:8],
        },
        "graph_targets": {
            "legal_count": int(graph["legal_count"]),
            "pair_count": int(graph["pair_count"]),
            "opp_legal_count": int(graph["opp_legal_count"]),
            "token_counts": graph["token_counts"],
            "target_masses": graph.get("target_masses", {}),
        },
    }


def _axis_input_from_request(store: DashboardStore, req: AxisEvaluateRequest) -> AxisPolicyInput:
    if req.position:
        offset_q, offset_r = _fit_axis_offsets(
            list(req.position.get("stones", [])),
            list(req.position.get("legal_moves", [])),
            int(req.position.get("offset_q", -16)),
            int(req.position.get("offset_r", -16)),
        )
        return AxisPolicyInput(
            stones=list(req.position.get("stones", [])),
            legal_moves=list(req.position.get("legal_moves", [])),
            current_player=int(req.position.get("current_player", 0)),
            offset_q=offset_q,
            offset_r=offset_r,
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
    offset_q, offset_r = _fit_axis_offsets(
        pos["stones"],
        pos["legal_moves"],
        int(pos["encoding"].get("offset_q", -16)),
        int(pos["encoding"].get("offset_r", -16)),
    )
    return AxisPolicyInput(
        stones=pos["stones"],
        legal_moves=pos["legal_moves"],
        current_player=int(pos["current_player"]),
        offset_q=offset_q,
        offset_r=offset_r,
        metadata={
            "source": "dashboard",
            "turn_index": int(pos.get("turn_index", 0)),
            "placements_remaining": int(pos.get("placements_remaining", 2)),
        },
    )


def _fit_axis_offsets(
    stones: list[dict[str, Any]],
    legal_moves: list[dict[str, Any]],
    offset_q: int,
    offset_r: int,
) -> tuple[int, int]:
    """Choose a dashboard analysis window that keeps sparse legal cells visible."""
    primary = [(int(m["q"]), int(m["r"])) for m in legal_moves if "q" in m and "r" in m]
    secondary = [(int(s["q"]), int(s["r"])) for s in stones if "q" in s and "r" in s]
    if _all_inside(primary or secondary, offset_q, offset_r):
        return offset_q, offset_r
    coords = primary or secondary
    if not coords:
        return offset_q, offset_r
    return _best_offset_pair(coords, offset_q, offset_r)


def _all_inside(coords: list[tuple[int, int]], offset_q: int, offset_r: int) -> bool:
    return all(
        offset_q <= q < offset_q + BOARD_SIZE and offset_r <= r < offset_r + BOARD_SIZE
        for q, r in coords
    )


def _best_axis_start(values: list[int], current: int) -> int:
    if not values:
        return current
    if max(values) - min(values) < BOARD_SIZE:
        return int(round((min(values) + max(values) - BOARD_SIZE + 1) / 2))
    starts = sorted(set(values + [value - BOARD_SIZE + 1 for value in values]))
    median = sorted(values)[len(values) // 2]
    return max(
        starts,
        key=lambda start: (
            sum(1 for value in values if start <= value < start + BOARD_SIZE),
            -abs((start + (BOARD_SIZE - 1) / 2) - median),
        ),
    )


def _best_offset_pair(
    coords: list[tuple[int, int]],
    current_q: int,
    current_r: int,
) -> tuple[int, int]:
    q_values = [q for q, _r in coords]
    r_values = [r for _q, r in coords]
    if max(q_values) - min(q_values) < BOARD_SIZE and max(r_values) - min(r_values) < BOARD_SIZE:
        return _best_axis_start(q_values, current_q), _best_axis_start(r_values, current_r)
    q_starts = sorted(set(q_values + [q - BOARD_SIZE + 1 for q in q_values]))
    r_starts = sorted(set(r_values + [r - BOARD_SIZE + 1 for r in r_values]))
    q_median = sorted(q_values)[len(q_values) // 2]
    r_median = sorted(r_values)[len(r_values) // 2]
    return max(
        ((q_start, r_start) for q_start in q_starts for r_start in r_starts),
        key=lambda start: (
            sum(
                1
                for q, r in coords
                if start[0] <= q < start[0] + BOARD_SIZE
                and start[1] <= r < start[1] + BOARD_SIZE
            ),
            -abs((start[0] + (BOARD_SIZE - 1) / 2) - q_median)
            - abs((start[1] + (BOARD_SIZE - 1) / 2) - r_median),
        ),
    )


def _default_frontend_dist() -> Path:
    return Path(__file__).resolve().parents[3] / "dashboard_frontend" / "dist"


def default_app() -> FastAPI:
    """Factory target for ASGI runners that prefer a zero-argument callable."""
    return create_app()
