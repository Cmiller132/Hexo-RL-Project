"""FastAPI dashboard application factory."""

from __future__ import annotations

import base64
import asyncio
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hexorl.axis_policy.core import AxisPolicyInput
from hexorl.action_contract.candidates import (
    CANDIDATE_FEATURE_NAMES,
    CANDIDATE_FEATURE_VERSION,
    build_candidate_batch,
    build_pair_candidate_batch,
)
from hexorl.buffer.sampler import _transform_history_bytes
from hexorl.axis_policy.registry import describe_prototypes, evaluate_all, get_prototype
from hexorl.dashboard.arena_service import ArenaManager
from hexorl.dashboard.checkpoints import scan_checkpoints
from hexorl.dashboard.db import SCHEMA_VERSION as DASHBOARD_SCHEMA_VERSION
from hexorl.dashboard.db import DashboardSchemaError, DashboardStore, decode_bytes
from hexorl.dashboard.fixtures import (
    ClassicalFixtureConfig,
    generate_classical_fixtures,
    list_axis_fixtures,
)
from hexorl.dashboard.model_cache import ModelCache
from hexorl.dashboard.play import apply_move, create_session, reset_session, session_payload, undo_move
from hexorl.dashboard.render import MatchSnapshotOptions, render_match_snapshot_png, snapshot_filename
from hexorl.dashboard.replay import decode_move_history, get_replay_position, position_payload, replay_game
from hexorl.models.registry import architecture_display_summary, trial_model_summary
from hexorl.graph.batch import (
    GRAPH_FEATURE_DIM,
    GRAPH_SCHEMA_VERSION,
    GRAPH_CAPACITY_STRATEGY,
    RELATION_SCHEMA_VERSION,
    GraphTokenType,
    RelationType,
    build_graph_batch_from_history,
    graph_capacity_report,
    transform_pair_policy_target,
    transform_policy_target,
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
    replay_error = _history_replay_error(row.get("final_history_b64"))
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
        "replay_available": bool(row.get("final_history_b64")) and replay_error is None,
        "replay_error": replay_error or "",
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
            "schema_version": DASHBOARD_SCHEMA_VERSION,
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
        if trial_store is not None and not rows:
            rows = source.rows(
                """
                SELECT * FROM (
                    SELECT * FROM metrics ORDER BY created_at DESC LIMIT ?
                ) ORDER BY created_at ASC
                """,
                (max(1, min(limit, 5000)),),
            )
        return rows

    @app.get("/api/events/{run_id}")
    def events(run_id: str, limit: int = 500) -> list[dict[str, Any]]:
        trial_store = _suite_store_for_run(suite_root, run_id)
        source = trial_store or store
        rows = source.rows(
            "SELECT * FROM events WHERE run_id=? ORDER BY created_at DESC LIMIT ?",
            (run_id, max(1, min(limit, 5000))),
        )
        if trial_store is not None and not rows:
            rows = source.rows(
                "SELECT * FROM events ORDER BY created_at DESC LIMIT ?",
                (max(1, min(limit, 5000)),),
            )
        return rows

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
    def game_replay(
        game_id: int,
        run_id: str | None = None,
        include_positions: bool = False,
    ) -> dict[str, Any]:
        source = _suite_store_for_run(suite_root, run_id) if run_id else store
        try:
            rows = (source or store).rows("SELECT final_history_b64 FROM games WHERE game_id=?", (game_id,))
            if rows:
                replay_error = _history_replay_error(rows[0].get("final_history_b64"))
                if replay_error:
                    raise HTTPException(422, f"Invalid replay history: {replay_error}")
            return replay_game(source or store, game_id, include_positions=include_positions)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/games/{game_id}/position/{turn_index}")
    def game_position(
        game_id: int,
        turn_index: int,
        run_id: str | None = None,
        compact: bool = False,
    ) -> dict[str, Any]:
        source = _suite_store_for_run(suite_root, run_id) if run_id else store
        rows = (source or store).rows("SELECT final_history_b64 FROM games WHERE game_id=?", (game_id,))
        if not rows:
            raise HTTPException(404, f"Game not found: {game_id}")
        replay_error = _history_replay_error(rows[0]["final_history_b64"])
        if replay_error:
            raise HTTPException(422, f"Invalid replay history: {replay_error}")
        pos = get_replay_position(rows[0]["final_history_b64"], turn_index=turn_index)
        return position_payload(pos, include_moves=not compact)

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

    @app.get("/api/suite/phase2")
    def suite_phase2() -> dict[str, Any]:
        if suite_root is None:
            return {"enabled": False, "rows": []}
        return _suite_phase2(suite_root)

    @app.get("/api/suite/phase3")
    def suite_phase3() -> dict[str, Any]:
        if suite_root is None:
            return {"enabled": False, "rows": []}
        return _suite_phase3(suite_root)

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
        return _suite_events(suite_root, limit=max(1, min(limit, 1000)))

    @app.get("/api/suite/game-examples")
    def suite_game_examples(limit: int = 80) -> list[dict[str, Any]]:
        if suite_root is None:
            return []
        return _suite_game_examples(suite_root, limit=max(1, min(limit, 500)))

    @app.get("/api/suite/game-examples/{example_id}/replay")
    def suite_game_example_replay(example_id: str) -> dict[str, Any]:
        if suite_root is None:
            raise HTTPException(404, "Suite run root is not configured")
        row = _suite_example_by_id(suite_root, example_id)
        if not row or not row.get("final_history_b64"):
            raise HTTPException(404, f"Replay not found: {example_id}")
        replay_error = _history_replay_error(row["final_history_b64"])
        if replay_error:
            raise HTTPException(422, f"Invalid replay history: {replay_error}")
        history = _decode_history_value(row["final_history_b64"])
        public_row = dict(row)
        public_row.pop("final_history_b64", None)
        return {
            "game": public_row,
            "moves": [
                {"player": int(player), "q": int(q), "r": int(r)}
                for player, q, r in decode_move_history(history)
            ],
            "positions": [],
        }

    @app.get("/api/suite/game-examples/{example_id}/position/{turn_index}")
    def suite_game_example_position(
        example_id: str,
        turn_index: int,
        compact: bool = False,
    ) -> dict[str, Any]:
        if suite_root is None:
            raise HTTPException(404, "Suite run root is not configured")
        row = _suite_example_by_id(suite_root, example_id)
        if not row or not row.get("final_history_b64"):
            raise HTTPException(404, f"Position not found: {example_id}")
        replay_error = _history_replay_error(row["final_history_b64"])
        if replay_error:
            raise HTTPException(422, f"Invalid replay history: {replay_error}")
        return position_payload(
            get_replay_position(
                _decode_history_value(row["final_history_b64"]),
                turn_index=turn_index,
            ),
            include_moves=not compact,
        )

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
            transformed = _transform_history_bytes(history, sym_idx)
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
    return sorted(
        [
            path
            for path in trials.iterdir()
            if path.is_dir()
            and (
                (path / "dashboard.sqlite3").exists()
                or (path / "optuna_trial.json").exists()
                or (path / "full_config.json").exists()
                or (path / "events.jsonl").exists()
                or (path / "scorecards.jsonl").exists()
                or (path / "checkpoints").exists()
            )
        ]
    )


def _suite_dashboard_trial_dirs(run_root: Path) -> list[Path]:
    return [path for path in _suite_trial_dirs(run_root) if (path / "dashboard.sqlite3").exists()]


def _suite_game_trial_dirs(run_root: Path) -> list[Path]:
    """Return dashboard-backed dirs that can contribute replayable games."""
    candidates: list[Path] = []
    candidates.extend(_suite_dashboard_trial_dirs(run_root))
    phase3_root = run_root.parent / "phase3_trials"
    if phase3_root.exists():
        candidates.extend(path for path in phase3_root.iterdir() if path.is_dir() and (path / "dashboard.sqlite3").exists())
    deduped: dict[str, Path] = {}
    for path in candidates:
        # Prefer direct phase3_trials artifacts over stale mirrored suite copies.
        if path.parent.name == "phase3_trials" or path.name not in deduped:
            deduped[path.name] = path
    return sorted(deduped.values(), key=lambda path: path.name)


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
    for trial_dir in _suite_game_trial_dirs(run_root):
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
    for trial_dir in _suite_game_trial_dirs(run_root):
        db = trial_dir / "dashboard.sqlite3"
        try:
            rows = DashboardStore(db).rows("SELECT run_id FROM runs WHERE run_id=? LIMIT 1", (run_id,))
        except Exception:
            continue
        if rows:
            return DashboardStore(db)
    return None


def _suite_trial_id_for_run(run_root: Path | None, run_id: str | None) -> str | None:
    if run_root is None or not run_id:
        return None
    direct = run_root / "trials" / run_id / "dashboard.sqlite3"
    if direct.exists():
        return run_id
    for trial_dir in _suite_game_trial_dirs(run_root):
        db = trial_dir / "dashboard.sqlite3"
        try:
            rows = DashboardStore(db).rows("SELECT run_id FROM runs WHERE run_id=? LIMIT 1", (run_id,))
        except Exception:
            continue
        if rows:
            return trial_dir.name
    return None


def _suite_primary_run_id(trial_dir: Path) -> str:
    db = trial_dir / "dashboard.sqlite3"
    if db.exists():
        try:
            rows = DashboardStore(db).rows("SELECT run_id FROM runs ORDER BY updated_at DESC LIMIT 1")
            if rows and rows[0].get("run_id"):
                return str(rows[0]["run_id"])
        except Exception:
            pass
    return trial_dir.name


def _trial_display_name(trial_id: str) -> str:
    label = str(trial_id or "")
    for suffix in ("__none__v1", "__root_pair_mcts__v1", "__full_pair_mcts__v1"):
        label = label.replace(suffix, "")
    return label.replace("global_", "").replace("_", " ")


def _suite_runs(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trial_state = {trial["trial_id"]: trial for trial in _suite_state(run_root).get("trials", [])}
    for trial_dir in _suite_trial_dirs(run_root):
        db = trial_dir / "dashboard.sqlite3"
        if not db.exists():
            rows.append(
                {
                    "run_id": _suite_primary_run_id(trial_dir),
                    "trial_id": trial_dir.name,
                    "updated_at": 0,
                    "metadata_only": True,
                    **trial_state.get(trial_dir.name, {}),
                }
            )
            continue
        try:
            run_rows = DashboardStore(db).rows("SELECT * FROM runs ORDER BY updated_at DESC")
        except DashboardSchemaError as exc:
            rows.append(
                {
                    "run_id": trial_dir.name,
                    "trial_id": trial_dir.name,
                    "suite_trial_id": trial_dir.name,
                    "source_db": str(db),
                    "name": _trial_display_name(trial_dir.name),
                    "output_dir": str(trial_dir),
                    "config_json": {},
                    "payload_json": {
                        "dashboard_schema_error": str(exc),
                        "requires_rebuild": True,
                        "schema_version": DASHBOARD_SCHEMA_VERSION,
                    },
                    "created_at": 0.0,
                    "updated_at": 0.0,
                }
            )
            continue
        except Exception:
            continue
        for row in run_rows:
            trial_id = str(row.get("run_id") or trial_dir.name)
            row["trial_id"] = trial_id
            row["suite_trial_id"] = trial_dir.name
            row["source_db"] = str(db)
            row["name"] = _trial_display_name(trial_dir.name)
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
    if run_id:
        trial_store = _suite_store_for_run(run_root, run_id)
        if trial_store is None:
            return []
        try:
            rows = trial_store.rows(
                "SELECT * FROM games WHERE run_id=? ORDER BY created_at DESC LIMIT ?",
                (run_id, limit),
            )
            if not rows:
                rows = trial_store.rows(
                    "SELECT * FROM games ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            for row in rows:
                row.setdefault("trial_id", run_id)
            return rows
        except Exception:
            return []
    trial_dirs = _suite_game_trial_dirs(run_root)
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


def _suite_game_examples(run_root: Path, *, limit: int = 80) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    # Validate a wider raw window before applying the final limit. A burst of
    # invalid or non-replayable rows from one trial should not crowd out
    # viewable examples from other models.
    selfplay_limit = 500
    for row in _suite_games(run_root, limit=selfplay_limit):
        trial_id = row.get("trial_id") or row.get("run_id")
        replay_error = _history_replay_error(row.get("final_history_b64"))
        examples.append(
            {
                "example_id": f"selfplay|{row.get('run_id')}|{row.get('game_id')}",
                "kind": "selfplay",
                "opponent_type": "selfplay",
                "opponent_label": "Self-play",
                "trial_id": trial_id,
                "model_label": _trial_display_name(str(trial_id or "")),
                "run_id": row.get("run_id"),
                "game_id": row.get("game_id"),
                "source": row.get("source", "selfplay"),
                "epoch": row.get("epoch"),
                "outcome": row.get("outcome"),
                "move_count": row.get("move_count"),
                "terminal_reason": (row.get("payload_json") or {}).get("terminal_reason", ""),
                "created_at": row.get("created_at"),
                "replay_available": bool(row.get("final_history_b64")) and replay_error is None,
                "replay_error": replay_error or "",
            }
        )

    seen_classical: set[tuple[str, int | None, int]] = set()
    for trial_dir in _fixed_classical_trial_dirs(run_root):
        trial_id = trial_dir.name
        for evidence_path in sorted(trial_dir.glob("fixed_classical_epoch_*_games.jsonl")):
            epoch = _epoch_from_fixed_classical_path(evidence_path)
            for row in _jsonl_tail(evidence_path, limit=24):
                try:
                    game_index = int(row.get("game_index", -1))
                except (TypeError, ValueError):
                    continue
                key = (trial_id, epoch, game_index)
                if key in seen_classical:
                    continue
                seen_classical.add(key)
                opponent_id = str(row.get("opponent_id") or "fixed_classical")
                example_id = f"classical|{trial_id}|{epoch if epoch is not None else 0}|{game_index}"
                replay_error = _history_replay_error(row.get("final_history_b64"))
                examples.append(
                    {
                        "example_id": example_id,
                        "kind": "fixed_classical",
                        "opponent_type": opponent_id,
                        "opponent_label": opponent_id.replace("_", " "),
                        "trial_id": trial_id,
                        "model_label": _trial_display_name(trial_id),
                        "run_id": trial_id,
                        "game_id": game_index,
                        "source": "fixed_classical",
                        "epoch": epoch,
                        "outcome": row.get("outcome"),
                        "move_count": row.get("moves"),
                        "terminal_reason": row.get("reason", ""),
                        "created_at": _mtime(evidence_path),
                        "replay_available": bool(row.get("final_history_b64")) and replay_error is None,
                        "replay_error": replay_error or "",
                        "opponent_id": opponent_id,
                        "opening_is_black": row.get("opening_is_black"),
                        "checkpoint_id": row.get("checkpoint_id", ""),
                    }
                )

    examples.sort(
        key=lambda row: (
            bool(row.get("replay_available")),
            float(row.get("created_at") or 0.0),
        ),
        reverse=True,
    )
    return examples[:limit]


def _suite_example_by_id(run_root: Path, example_id: str) -> dict[str, Any] | None:
    parts = example_id.split("|")
    if len(parts) < 3:
        return None
    kind = parts[0]
    if kind == "selfplay":
        run_id = parts[1]
        try:
            game_id = int(parts[2])
        except ValueError:
            return None
        trial_store = _suite_store_for_run(run_root, run_id)
        if trial_store is None:
            return None
        rows = trial_store.rows("SELECT * FROM games WHERE game_id=?", (game_id,))
        if not rows:
            return None
        row = rows[0]
        history = row.get("final_history_b64", "")
        row = _game_summary(row)
        row["example_id"] = example_id
        row["kind"] = "selfplay"
        row["final_history_b64"] = history
        return row
    if kind != "classical":
        return None
    trial_id = parts[1]
    try:
        if len(parts) >= 4:
            target_epoch = int(parts[2])
            target_index = int(parts[3])
        else:
            target_epoch = None
            target_index = int(parts[2])
    except ValueError:
        return None
    trial_dirs = [path for path in _fixed_classical_trial_dirs(run_root) if path.name == trial_id]
    for evidence_path in sorted(path for trial_dir in trial_dirs for path in trial_dir.glob("fixed_classical_epoch_*_games.jsonl")):
        epoch = _epoch_from_fixed_classical_path(evidence_path)
        if target_epoch is not None and target_epoch != 0 and epoch != target_epoch:
            continue
        for row in _read_jsonl_lenient(evidence_path):
            if int(row.get("game_index", -1)) != target_index:
                continue
            opponent_id = str(row.get("opponent_id") or "fixed_classical")
            replay_error = _history_replay_error(row.get("final_history_b64"))
            row = dict(row)
            row.update(
                {
                    "example_id": example_id,
                    "kind": "fixed_classical",
                    "opponent_type": opponent_id,
                    "opponent_label": opponent_id.replace("_", " "),
                    "trial_id": trial_id,
                    "model_label": _trial_display_name(trial_id),
                    "run_id": trial_id,
                    "game_id": target_index,
                    "source": "fixed_classical",
                    "epoch": epoch,
                    "move_count": row.get("moves"),
                    "terminal_reason": row.get("reason", ""),
                    "created_at": _mtime(evidence_path),
                    "replay_available": bool(row.get("final_history_b64")) and replay_error is None,
                    "replay_error": replay_error or "",
                    "opponent_id": opponent_id,
                }
            )
            return row
    return None


def _fixed_classical_trial_dirs(run_root: Path) -> list[Path]:
    """Return trial/candidate dirs with persisted fixed-classical game evidence."""
    candidates: list[Path] = []
    candidates.extend(path for path in _suite_trial_dirs(run_root) if list(path.glob("fixed_classical_epoch_*_games.jsonl")))
    for root in (run_root.parent / "phase3_trials", run_root.parent / "candidates"):
        if root.exists():
            candidates.extend(path for path in root.iterdir() if path.is_dir() and list(path.glob("fixed_classical_epoch_*_games.jsonl")))
    deduped: dict[str, Path] = {}
    for path in candidates:
        deduped.setdefault(path.name, path)
    return sorted(deduped.values(), key=lambda path: path.name)


def _read_jsonl_lenient(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _decode_history_value(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    return decode_bytes(str(value or ""))


def _history_replay_error(value: Any) -> str | None:
    """Return a concise reason a persisted history cannot be replayed safely."""
    if not value:
        return "missing history"
    try:
        history = _decode_history_value(value)
    except Exception as exc:
        return f"history decode failed: {exc}"
    if len(history) % 12 != 0:
        return f"history byte length {len(history)} is not divisible by 12"
    seen: set[tuple[int, int]] = set()
    for index, (_player, q, r) in enumerate(decode_move_history(history)):
        key = (int(q), int(r))
        if key in seen:
            return f"duplicate placement at move {index}: ({int(q)}, {int(r)})"
        seen.add(key)
    return None


def _epoch_from_fixed_classical_path(path: Path) -> int | None:
    match = re.search(r"epoch_(\d+)_games", path.name)
    return int(match.group(1)) if match else None


def _suite_checkpoints(run_root: Path, *, run_id: str | None = None) -> list[dict[str, Any]]:
    if run_id:
        trial_store = _suite_store_for_run(run_root, run_id)
        if trial_store is None:
            return []
        try:
            rows = trial_store.rows(
                "SELECT * FROM checkpoints WHERE run_id=? ORDER BY indexed_at DESC",
                (run_id,),
            )
            if not rows:
                rows = trial_store.rows("SELECT * FROM checkpoints ORDER BY indexed_at DESC")
            score_by_trial = _suite_score_by_trial(run_root)
            for row in rows:
                row["run_id"] = row.get("run_id") or run_id
                row["trial_id"] = _suite_trial_id_for_run(run_root, run_id) or run_id
                row["suite_trial_id"] = row["trial_id"]
                row["trial_label"] = _trial_display_name(str(row["trial_id"]))
                row["score"] = score_by_trial.get(str(row["trial_id"]))
                row["scheduler_score"] = row["score"]
            return rows
        except Exception:
            return []
    trial_dirs = _suite_trial_dirs(run_root)
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
            row["run_id"] = row.get("run_id") or _suite_primary_run_id(trial_dir)
            row["trial_id"] = trial_dir.name
            row["suite_trial_id"] = trial_dir.name
            row["trial_label"] = _trial_display_name(trial_dir.name)
            row["source_db"] = str(db)
            row["score"] = score_by_trial.get(trial_dir.name)
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
        row["rank_basis"] = "score" if row.get("score") is not None else "epoch/recency"
    return ranked[:limit]


def _suite_latest_metrics(trial_dir: Path) -> dict[str, Any]:
    db = trial_dir / "dashboard.sqlite3"
    payload: dict[str, Any] = {"latest": {}, "train": {}, "selfplay": {}, "buffer": {}}
    if not db.exists():
        return payload
    try:
        store = DashboardStore(db)
        rows = store.rows("SELECT * FROM metrics ORDER BY created_at ASC")
    except Exception:
        return payload
    for row in rows:
        metrics = dict(row.get("metrics_json") or {})
        phase = str(row.get("phase") or "")
        created_at = row.get("created_at")
        epoch = row.get("epoch")
        global_step = row.get("global_step")
        if phase == "train":
            train = dict(metrics.get("train") or metrics)
            buffer = dict(metrics.get("buffer") or {})
            train.setdefault("epoch", epoch)
            train.setdefault("global_step", global_step)
            train["created_at"] = created_at
            payload["train"] = train
            if buffer:
                payload["buffer"] = buffer
            payload["latest"] = {
                **payload["latest"],
                "epoch": epoch,
                "global_step": global_step,
                "checkpoint_path": metrics.get("checkpoint_path") or payload["latest"].get("checkpoint_path"),
                "epoch_elapsed_s": train.get("elapsed_s"),
                "updated_at": created_at,
            }
        elif phase == "selfplay":
            selfplay = dict(metrics)
            selfplay["created_at"] = created_at
            payload["selfplay"] = selfplay
            payload["latest"] = {
                **payload["latest"],
                "selfplay_updated_at": created_at,
            }
    return payload


def _suite_trial_config(trial_dir: Path, trial_json: dict[str, Any]) -> dict[str, Any]:
    cfg = _read_json(trial_dir / "full_config.json")
    if cfg:
        return cfg
    user_attrs = trial_json.get("user_attrs") or {}
    full_config = user_attrs.get("full_config") or {}
    return full_config if isinstance(full_config, dict) else {}


def _suite_trials(run_root: Path) -> list[dict[str, Any]]:
    state_trials = {trial["trial_id"]: trial for trial in _suite_state(run_root).get("trials", [])}
    score_by_trial = _suite_score_by_trial(run_root)
    phase2_by_trial = {row["trial_id"]: row for row in _suite_phase2_rows(run_root)}
    rows: list[dict[str, Any]] = []
    for trial_dir in _suite_trial_dirs(run_root):
        trial_id = trial_dir.name
        state = dict(state_trials.get(trial_id) or {})
        trial_json = _read_json(trial_dir / "trial.json")
        if not trial_json:
            trial_json = _read_json(trial_dir / "optuna_trial.json")
        latest = _read_json(trial_dir / "LATEST.json")
        latest_metrics = _suite_latest_metrics(trial_dir)
        cfg = _suite_trial_config(trial_dir, trial_json)
        if not latest:
            latest = dict(latest_metrics.get("latest") or {})
        scores = _jsonl_tail(trial_dir / "scores.jsonl", limit=1)
        family = state.get("family") or trial_json.get("family") or {}
        model_cfg = cfg.get("model") or {}
        selfplay_cfg = cfg.get("selfplay") or {}
        counts = _suite_trial_counts(trial_dir)
        latest_event = _suite_latest_event(trial_dir)
        latest_selfplay = latest.get("selfplay") or latest_metrics.get("selfplay") or {}
        latest_train = latest.get("train") or latest_metrics.get("train") or {}
        latest_buffer = latest.get("buffer") or latest_metrics.get("buffer") or {}
        checkpoint_path = state.get("checkpoint_path") or latest.get("checkpoint_path") or ""
        if not checkpoint_path:
            checkpoints = _suite_checkpoints(run_root, run_id=trial_id)
            checkpoint_path = checkpoints[0].get("path") if checkpoints else ""
        score = score_by_trial.get(trial_id, state.get("last_score"))
        if (score is None or score == "-inf") and scores:
            score = scores[-1].get("scheduler_score")
        phase2 = dict(phase2_by_trial.get(trial_id) or {})
        row = {
            "trial_id": trial_id,
            "run_id": _suite_primary_run_id(trial_dir),
            "trial_label": _trial_display_name(trial_id),
            "family": family.get("name") or model_cfg.get("architecture") or latest.get("family") or "",
            "architecture": family.get("architecture") or model_cfg.get("architecture") or "",
            "model_summary": _model_summary_from_trial(
                family or {"architecture": model_cfg.get("architecture"), "channels": model_cfg.get("channels"), "blocks": model_cfg.get("blocks")},
                state.get("static") or trial_json.get("static") or latest.get("static") or model_cfg,
            ),
            "stage": latest.get("stage") or state.get("stage") or trial_json.get("stage") or "phase1 scout",
            "epoch": state.get("epoch") or latest.get("epoch") or latest_train.get("epoch") or trial_json.get("epoch") or 0,
            "score": _finite_or_none(score),
            "pruned": bool(state.get("pruned") or trial_json.get("pruned") or False),
            "prune_reason": state.get("prune_reason") or trial_json.get("prune_reason") or "",
            "checkpoint_path": checkpoint_path,
            "games": counts["games"],
            "positions": counts["positions"],
            "checkpoints": counts["checkpoints"],
            "metrics": counts["metrics"],
            "selfplay_positions_per_min": latest_selfplay.get("positions_per_min"),
            "positions_per_sec": _per_second(latest_selfplay.get("positions_per_min")),
            "workers": _worker_summary(latest_selfplay),
            "epoch_elapsed_s": latest.get("epoch_elapsed_s") or latest_train.get("elapsed_s"),
            "loss_total": latest_train.get("loss_total"),
            "loss_value": latest_train.get("loss_value"),
            "loss_policy_place": latest_train.get("loss_policy_place"),
            "value_weight_mean": latest_train.get("value_weight_mean"),
            "value_weight_zero_frac": latest_train.get("value_weight_zero_frac"),
            "policy_top1_acc": latest_train.get("policy_top1_acc"),
            "sparse_policy_top1_acc": latest_train.get("sparse_policy_top1_acc"),
            "pair_policy_top1_acc": latest_train.get("pair_policy_top1_acc"),
            "truncation_rate": latest_selfplay.get("truncation_rate"),
            "terminal_reason_win": latest_selfplay.get("terminal_reason_win"),
            "max_game_moves": selfplay_cfg.get("max_game_moves"),
            "mcts_simulations": selfplay_cfg.get("mcts_simulations"),
            "states_per_epoch": selfplay_cfg.get("states_per_epoch"),
            "recorder_failures": latest_selfplay.get("recorder_failures"),
            "missing_target_policy_mass": latest_buffer.get("avg_missing_target_policy_mass"),
            "candidate_recall_top1": latest_buffer.get("avg_candidate_recall_mcts_top1"),
            "graph_microbatch_size": latest_train.get("graph_microbatch_size"),
            "graph_microbatch_count": latest_train.get("graph_microbatch_count"),
            "graph_collate_s": latest_train.get("graph_collate_s"),
            "graph_to_device_s": latest_train.get("graph_to_device_s"),
            "graph_forward_s": latest_train.get("graph_forward_s"),
            "graph_loss_s": latest_train.get("graph_loss_s"),
            "graph_backward_s": latest_train.get("graph_backward_s"),
            "graph_optimizer_s": latest_train.get("graph_optimizer_s"),
            "graph_peak_cuda_allocated_mb": latest_train.get("graph_peak_cuda_allocated_mb"),
            "runtime_sweep": state.get("runtime_sweep") or trial_json.get("runtime_sweep") or {},
            "last_event": latest_event.get("event") or latest_event.get("event_type") or "",
            "last_event_phase": latest_event.get("phase") or "",
            "last_event_time": latest_event.get("time") or latest_event.get("created_at"),
            "last_event_message": _suite_event_message(latest_event),
            "phase2_status": phase2.get("phase2_status"),
            "phase2_rank": phase2.get("phase2_rank"),
            "phase2_promoted": phase2.get("phase2_promoted"),
            "phase2_excluded": phase2.get("phase2_excluded"),
            "phase2_exclusion_reason": phase2.get("phase2_exclusion_reason"),
            "classical_survival_lcb": phase2.get("classical_survival_lcb"),
            "classical_survival_mean": phase2.get("classical_survival_mean"),
            "classical_survival_games": phase2.get("classical_survival_games"),
            "classical_win_rate": phase2.get("classical_win_rate"),
            "classical_draw_rate": phase2.get("classical_draw_rate"),
            "classical_avg_moves": phase2.get("classical_avg_moves"),
            "fixed_classical_evidence_path": phase2.get("fixed_classical_evidence_path"),
            "hard_pass": phase2.get("hard_pass"),
            "updated_at": max(
                float(latest.get("updated_at") or 0.0),
                float(latest_selfplay.get("created_at") or 0.0),
                float(latest_train.get("created_at") or 0.0),
                float(latest_event.get("time") or latest_event.get("created_at") or 0.0),
                _mtime(trial_dir / "LATEST.json"),
                _mtime(trial_dir / "dashboard.sqlite3"),
            ),
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
    events = _suite_events(run_root, limit=200)
    trials = _suite_trials(run_root)
    phase2 = _suite_phase2(run_root)
    latest_stage = _latest_suite_stage(state, events, trials)
    total_games = sum(int(trial.get("games") or 0) for trial in trials)
    total_positions = sum(int(trial.get("positions") or 0) for trial in trials)
    total_checkpoints = sum(int(trial.get("checkpoints") or 0) for trial in trials)
    warning_count = sum(1 for event in events if event.get("severity") == "warning")
    best = next((trial for trial in trials if not trial.get("pruned") and trial.get("score") is not None), None)
    leading = max(
        (trial for trial in trials if not trial.get("pruned")),
        key=lambda row: (int(row.get("epoch") or 0), float(row.get("updated_at") or 0.0)),
        default=None,
    )
    activity = _supervisor_activity(run_root, trials, events, manifest)
    return {
        "enabled": True,
        "run_root": str(run_root),
        "latest_stage": latest_stage,
        "trial_count": len(trials),
        "live_trial_count": sum(1 for trial in trials if not trial.get("pruned")),
        "total_games": total_games,
        "total_positions": total_positions,
        "total_checkpoints": total_checkpoints,
        "event_count": len(events),
        "warning_count": warning_count,
        "best_trial_id": best.get("trial_id") if best else None,
        "best_score": best.get("score") if best else None,
        "leading_trial_id": leading.get("trial_id") if leading else None,
        "leading_trial_label": leading.get("trial_label") if leading else None,
        "leading_epoch": leading.get("epoch") if leading else None,
        "leading_loss_total": leading.get("loss_total") if leading else None,
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
        "phase2": phase2,
    }


def _suite_trial_detail(run_root: Path, trial_id: str) -> dict[str, Any]:
    trial_dir = run_root / "trials" / trial_id
    if not trial_dir.exists():
        return {}
    state = next((trial for trial in _suite_state(run_root).get("trials", []) if trial.get("trial_id") == trial_id), {})
    trial_json = _read_json(trial_dir / "trial.json")
    if not trial_json:
        trial_json = _read_json(trial_dir / "optuna_trial.json")
    latest = _read_json(trial_dir / "LATEST.json")
    latest_metrics = _suite_latest_metrics(trial_dir)
    if not latest:
        latest = dict(latest_metrics.get("latest") or {})
        latest["train"] = latest_metrics.get("train") or {}
        latest["selfplay"] = latest_metrics.get("selfplay") or {}
        latest["buffer"] = latest_metrics.get("buffer") or {}
    scores = _jsonl_tail(trial_dir / "scores.jsonl", limit=12)
    summaries = _jsonl_tail(trial_dir / "summary.jsonl", limit=12)
    events = _suite_events_for_trial(trial_dir, limit=80)
    checkpoints = _suite_checkpoints(run_root, run_id=trial_id)
    checkpoint_path = (
        state.get("checkpoint_path")
        or latest.get("checkpoint_path")
        or (checkpoints[0].get("path") if checkpoints else "")
    )
    checkpoint = _checkpoint_metadata(Path(checkpoint_path)) if checkpoint_path else {}
    cfg = checkpoint.get("cfg") or _suite_trial_config(trial_dir, trial_json)
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
        "run_id": _suite_primary_run_id(trial_dir),
        "trial_label": _trial_display_name(trial_id),
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
        "phase2": next((row for row in _suite_phase2_rows(run_root) if row.get("trial_id") == trial_id), {}),
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


def _suite_latest_event(trial_dir: Path) -> dict[str, Any]:
    for row in reversed(_jsonl_tail(trial_dir / "events.jsonl", limit=80)):
        if _suite_event_is_useful(row):
            normalized = _normalize_suite_event(row, trial_id=trial_dir.name)
            return normalized
    return {}


def _suite_events(run_root: Path, *, limit: int = 200) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _jsonl_tail(run_root / "events.jsonl", limit=limit):
        rows.append(_normalize_suite_event(row, trial_id=str(row.get("trial_id") or row.get("candidate_id") or "")))
    per_trial_limit = max(80, min(300, limit))
    for trial_dir in _suite_trial_dirs(run_root):
        rows.extend(_suite_events_for_trial(trial_dir, limit=per_trial_limit))
    unique: dict[tuple[str, str, str, float, int], dict[str, Any]] = {}
    for row in rows:
        if not row or not _suite_event_is_useful(row):
            continue
        key = (
            str(row.get("trial_id") or ""),
            str(row.get("event") or row.get("event_type") or ""),
            str(row.get("phase") or ""),
            float(row.get("time") or row.get("created_at") or 0.0),
            int(row.get("epoch") or -1),
        )
        unique[key] = row
    ordered = sorted(
        unique.values(),
        key=lambda row: (
            float(row.get("time") or row.get("created_at") or 0.0),
            str(row.get("trial_id") or ""),
            str(row.get("event") or ""),
        ),
    )
    return ordered[-limit:]


def _suite_events_for_trial(trial_dir: Path, *, limit: int = 200) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _jsonl_tail(trial_dir / "events.jsonl", limit=limit):
        if not _suite_event_is_useful(row):
            continue
        rows.append(_normalize_suite_event(row, trial_id=trial_dir.name))
    return rows


def _suite_event_is_useful(row: dict[str, Any]) -> bool:
    event = str(row.get("event") or row.get("event_type") or "")
    if not event:
        return False
    return event != "game_recorded"


def _normalize_suite_event(row: dict[str, Any], *, trial_id: str) -> dict[str, Any]:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    train = payload.get("train") or payload.get("train_stats") or row.get("train") or {}
    train = train if isinstance(train, dict) else {}
    buffer_stats = payload.get("buffer") or payload.get("buffer_stats") or row.get("buffer") or {}
    buffer_stats = buffer_stats if isinstance(buffer_stats, dict) else {}
    event = str(row.get("event") or row.get("event_type") or "")
    resolved_trial_id = trial_id or str(row.get("trial_id") or row.get("candidate_id") or "")
    normalized = {
        "event": event,
        "event_type": row.get("event_type") or row.get("event") or event,
        "severity": _suite_event_severity(row),
        "stage": row.get("stage") or payload.get("stage") or "",
        "trial_id": resolved_trial_id,
        "trial_label": _trial_display_name(resolved_trial_id),
        "phase": row.get("phase") or payload.get("phase") or "",
        "epoch": row.get("epoch") if row.get("epoch") is not None else payload.get("epoch") or train.get("epoch"),
        "global_step": row.get("global_step") if row.get("global_step") is not None else train.get("global_step"),
        "reason": row.get("reason") or row.get("prune_reason") or payload.get("reason") or "",
        "action": row.get("action") or payload.get("action") or "",
        "start_epoch": row.get("start_epoch") or payload.get("start_epoch"),
        "end_epoch": row.get("end_epoch") or payload.get("end_epoch"),
        "completed_epochs": row.get("completed_epochs") or payload.get("completed_epochs"),
        "expected_epoch": row.get("expected_epoch") or payload.get("expected_epoch"),
        "metric": row.get("metric") or payload.get("metric"),
        "value": row.get("value") if row.get("value") is not None else payload.get("value"),
        "score": _finite_or_none(row.get("score") or row.get("scheduler_score") or payload.get("score")),
        "loss_total": _finite_or_none(train.get("loss_total")),
        "loss_value": _finite_or_none(train.get("loss_value")),
        "value_weight_mean": _finite_or_none(train.get("value_weight_mean")),
        "value_weight_zero_frac": _finite_or_none(train.get("value_weight_zero_frac")),
        "positions_per_min": _finite_or_none(payload.get("positions_per_min") or row.get("positions_per_min")),
        "positions_done": payload.get("positions_done") or buffer_stats.get("positions_done"),
        "games_done": payload.get("games_done") or buffer_stats.get("games_done"),
        "truncation_rate": _finite_or_none(payload.get("truncation_rate") or buffer_stats.get("truncation_rate")),
        "truncated_games": payload.get("truncated_games") or buffer_stats.get("truncated_games"),
        "terminal_reason_win": payload.get("terminal_reason_win") or buffer_stats.get("terminal_reason_win"),
        "terminal_reason_max_game_moves": payload.get("terminal_reason_max_game_moves")
        or buffer_stats.get("terminal_reason_max_game_moves"),
        "recorder_failures": payload.get("recorder_failures") or buffer_stats.get("recorder_failures"),
        "checkpoint_path": row.get("checkpoint_path") or payload.get("checkpoint_path") or payload.get("path"),
        "elapsed_s": row.get("elapsed_s") or payload.get("elapsed_s"),
        "hexo_status": row.get("hexo_status") or payload.get("hexo_status") or "",
        "time": row.get("time") or row.get("created_at") or payload.get("time"),
        "source": "trial" if resolved_trial_id else "suite",
    }
    normalized["positions_per_sec"] = _event_positions_per_second(normalized)
    normalized["message"] = _suite_event_message(normalized)
    return normalized


def _suite_event_severity(row: dict[str, Any]) -> str:
    event = str(row.get("event") or row.get("event_type") or "")
    if event in {"training_signal_warning", "runtime_warning", "warning"}:
        return "warning"
    if event in {"trial_pruned", "candidate_quarantined", "runtime_quarantine", "hard_failure"}:
        return "error"
    return "info"


def _suite_event_message(event: dict[str, Any]) -> str:
    event_name = str(event.get("event") or event.get("event_type") or "")
    label = event.get("trial_label") or _trial_display_name(str(event.get("trial_id") or event.get("candidate_id") or ""))
    epoch = event.get("epoch")
    phase = str(event.get("phase") or "")
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    if event_name == "metric" and phase == "selfplay":
        positions = event.get("positions_done") or payload.get("positions_done")
        trunc = event.get("truncation_rate") if event.get("truncation_rate") is not None else payload.get("truncation_rate")
        return f"{label} self-play: {format_count_py(positions)} positions, trunc {format_float_py(trunc)}"
    if event_name == "metric" and phase == "train":
        loss = event.get("loss_total")
        if loss is None:
            train = payload.get("train") or {}
            loss = train.get("loss_total")
        return f"{label} train epoch {epoch or '-'}: loss {format_float_py(loss)}"
    if event_name == "checkpoint":
        return f"{label} checkpoint saved for epoch {epoch or '-'}"
    if event_name == "epoch_complete":
        return f"{label} completed epoch {epoch or '-'}"
    if event_name == "epoch_start":
        return f"{label} started an epoch"
    if event_name == "quantum_started":
        start = event.get("start_epoch") or payload.get("start_epoch")
        end = event.get("end_epoch") or payload.get("end_epoch")
        return f"{label} quantum started for epochs {start or '?'}-{end or '?'}"
    if event_name == "quantum_completed":
        completed = event.get("completed_epochs") or payload.get("completed_epochs")
        return f"{label} quantum completed through epoch {completed or epoch or '-'}"
    if event_name == "epoch_runner_completed":
        return f"{label} runner wrote epoch {epoch or event.get('expected_epoch') or '-'}"
    if event_name == "training_signal_warning":
        metric = event.get("metric") or payload.get("metric") or "signal"
        value = event.get("value") if event.get("value") is not None else payload.get("value")
        return f"{label} warning: {metric} {format_float_py(value)}"
    if event_name:
        return f"{label} {_event_blurb(event_name)}"
    return ""


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


def _suite_phase2(run_root: Path) -> dict[str, Any]:
    rows = _suite_phase2_rows(run_root)
    ranked = [row for row in rows if row.get("phase2_status") == "ready"]
    pending = [row for row in rows if row.get("phase2_status") == "pending"]
    excluded = [row for row in rows if row.get("phase2_excluded")]
    best = ranked[0] if ranked else None
    latest_summary = _latest_fixed_classical_summary(run_root)
    return {
        "enabled": True,
        "stage": "phase2 fixed-classical review" if ranked else "phase2 pending evidence",
        "rows": rows,
        "ranked_count": len(ranked),
        "pending_count": len(pending),
        "excluded_count": len(excluded),
        "best_trial_id": best.get("trial_id") if best else None,
        "best_label": best.get("trial_label") if best else None,
        "best_lcb": best.get("classical_survival_lcb") if best else None,
        "best_mean": best.get("classical_survival_mean") if best else None,
        "total_classical_games": sum(int(row.get("classical_survival_games") or 0) for row in ranked),
        "latest_summary_path": str(latest_summary) if latest_summary else "",
        "updated_at": max([float(row.get("phase2_updated_at") or 0.0) for row in rows] + [_mtime(latest_summary) if latest_summary else 0.0]),
    }


def _suite_phase2_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial_dir in _suite_trial_dirs(run_root):
        trial_id = trial_dir.name
        trial_json = _read_json(trial_dir / "trial.json") or _read_json(trial_dir / "optuna_trial.json")
        user_attrs = trial_json.get("user_attrs") if isinstance(trial_json.get("user_attrs"), dict) else {}
        latest = _read_json(trial_dir / "LATEST.json")
        latest_metrics = _suite_latest_metrics(trial_dir)
        cfg = _suite_trial_config(trial_dir, trial_json)
        model_cfg = cfg.get("model") or {}
        latest_train = latest.get("train") or latest_metrics.get("train") or {}
        latest_selfplay = latest.get("selfplay") or latest_metrics.get("selfplay") or {}
        completed_epochs = user_attrs.get("completed_epochs") or latest.get("epoch") or latest_train.get("epoch") or 0
        quarantine_reason = user_attrs.get("quarantine_reason")
        hexo_status = user_attrs.get("hexo_status") or ""
        scorecard = _latest_classical_scorecard(trial_dir)
        metrics = scorecard.get("component_metrics") if isinstance(scorecard.get("component_metrics"), dict) else {}
        metadata = scorecard.get("metadata") if isinstance(scorecard.get("metadata"), dict) else {}
        fixed_eval = metadata.get("fixed_classical_eval") if isinstance(metadata.get("fixed_classical_eval"), dict) else {}
        hard_gates = scorecard.get("hard_gates") if isinstance(scorecard.get("hard_gates"), dict) else {}
        checkpoint_lineage = scorecard.get("checkpoint_lineage") if isinstance(scorecard.get("checkpoint_lineage"), dict) else {}
        games = _finite_or_none(metrics.get("classical_survival_games"))
        lcb = _finite_or_none(metrics.get("classical_survival_lcb") or scorecard.get("classical_survival_lcb"))
        excluded = bool(quarantine_reason or hexo_status in {"quarantined", "failed"})
        if excluded:
            phase2_status = "excluded"
            exclusion_reason = str(quarantine_reason or hexo_status)
        elif lcb is not None and games and games > 0:
            phase2_status = "ready"
            exclusion_reason = ""
        elif int(completed_epochs or 0) >= 12:
            phase2_status = "pending"
            exclusion_reason = "awaiting_fixed_classical_evidence"
        else:
            phase2_status = "training"
            exclusion_reason = "below_epoch_floor"
        evidence_path = fixed_eval.get("evidence_path") or _latest_fixed_classical_evidence_path(trial_dir)
        rows.append(
            {
                "trial_id": trial_id,
                "run_id": _suite_primary_run_id(trial_dir),
                "trial_label": _trial_display_name(trial_id),
                "architecture": model_cfg.get("architecture") or user_attrs.get("architecture_id") or "",
                "phase2_status": phase2_status,
                "phase2_excluded": excluded,
                "phase2_exclusion_reason": exclusion_reason,
                "phase2_promoted": phase2_status == "ready",
                "phase2_rank": None,
                "epoch": scorecard.get("epoch") or completed_epochs,
                "classical_survival_lcb": lcb,
                "classical_survival_mean": _finite_or_none(metrics.get("classical_survival_mean")),
                "classical_survival_games": games,
                "classical_win_rate": _finite_or_none(metrics.get("classical_win_rate")),
                "classical_draw_rate": _finite_or_none(metrics.get("classical_draw_rate")),
                "classical_avg_moves": _finite_or_none(metrics.get("classical_avg_moves")),
                "illegal_or_crash_rate": _finite_or_none(metrics.get("illegal_or_crash_rate")),
                "hard_pass": hard_gates.get("hard_pass") if hard_gates else None,
                "fixed_classical_evidence_path": str(evidence_path) if evidence_path else "",
                "scorecard_path": str(trial_dir / "scorecards.jsonl"),
                "checkpoint_path": scorecard.get("checkpoint_path") or checkpoint_lineage.get("checkpoint_path") or user_attrs.get("latest_checkpoint_path") or "",
                "phase2_updated_at": _scorecard_time(scorecard) or _mtime(trial_dir / "scorecards.jsonl"),
                "phase1_truncation_rate": latest_selfplay.get("truncation_rate"),
                "phase1_value_weight_mean": latest_train.get("value_weight_mean"),
                "phase1_value_effective_samples": latest_train.get("value_effective_samples"),
            }
        )
    rows.sort(
        key=lambda row: (
            row.get("phase2_status") == "ready",
            float(row.get("classical_survival_lcb") if row.get("classical_survival_lcb") is not None else float("-inf")),
            not bool(row.get("phase2_excluded")),
            float(row.get("phase2_updated_at") or 0.0),
        ),
        reverse=True,
    )
    best_lcb = next((row.get("classical_survival_lcb") for row in rows if row.get("phase2_status") == "ready"), None)
    rank = 1
    for row in rows:
        if row.get("phase2_status") == "ready":
            row["phase2_rank"] = rank
            rank += 1
            if best_lcb is not None and row.get("classical_survival_lcb") is not None:
                row["phase2_gap_to_best"] = float(best_lcb) - float(row["classical_survival_lcb"])
    return rows


def _suite_phase3(run_root: Path) -> dict[str, Any]:
    run_dir = run_root.parent
    specs = _read_json_list(run_dir / "phase2_review" / "phase3_study_specs.json")
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        rows.extend(_suite_phase3_rows_for_spec(run_dir, spec))
    rows.sort(
        key=lambda row: (
            row.get("state") == "COMPLETE",
            float(row.get("value") if row.get("value") is not None else float("-inf")),
            float(row.get("updated_at") or 0.0),
        ),
        reverse=True,
    )
    for rank, row in enumerate([row for row in rows if row.get("state") == "COMPLETE"], start=1):
        row["phase3_rank"] = rank
    specs_by_candidate = {
        str(spec.get("metadata", {}).get("promoted_candidate_id") or ""): spec
        for spec in specs
        if isinstance(spec, dict)
    }
    running = [row for row in rows if row.get("state") == "RUNNING"]
    complete = [row for row in rows if row.get("state") == "COMPLETE"]
    failed = [row for row in rows if row.get("state") in {"FAIL", "FAILED"}]
    pruned = [row for row in rows if row.get("state") == "PRUNED"]
    best = complete[0] if complete else None
    summary = _latest_phase3_summary(run_dir)
    return {
        "enabled": True,
        "stage": "phase3 tuning running" if running else ("phase3 evidence complete" if complete else "phase3 studies ready"),
        "spec_count": len(specs),
        "study_count": len(specs_by_candidate),
        "trial_count": len(rows),
        "running_count": len(running),
        "complete_count": len(complete),
        "pruned_count": len(pruned),
        "failed_count": len(failed),
        "best_trial_id": best.get("phase3_candidate_id") if best else None,
        "best_promoted_candidate_id": best.get("promoted_candidate_id") if best else None,
        "best_value": best.get("value") if best else None,
        "latest_summary_path": str(summary) if summary else "",
        "updated_at": max([float(row.get("updated_at") or 0.0) for row in rows] + [_mtime(summary) if summary else 0.0]),
        "rows": rows,
        "specs": [
            {
                "promoted_candidate_id": str(spec.get("metadata", {}).get("promoted_candidate_id") or ""),
                "study_name": spec.get("study_name"),
                "storage": spec.get("storage"),
                "search_knobs": (spec.get("search_scope") or {}).get("knobs", []),
                "sampler": (spec.get("sampler") or {}).get("type", ""),
                "pruner": (spec.get("pruner") or {}).get("type", ""),
                "phase2_lcb": (spec.get("metadata") or {}).get("phase2_classical_survival_lcb"),
            }
            for spec in specs
            if isinstance(spec, dict)
        ],
    }


def _suite_phase3_rows_for_spec(run_dir: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    storage = str(spec.get("storage") or "")
    study_name = str(spec.get("study_name") or "")
    metadata = spec.get("metadata") if isinstance(spec.get("metadata"), dict) else {}
    promoted_candidate_id = str(metadata.get("promoted_candidate_id") or "")
    db_path = _sqlite_path_from_storage(storage)
    if db_path is None:
        return []
    rows: list[dict[str, Any]] = []
    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
    except sqlite3.Error:
        return rows
    try:
        trials = con.execute(
            "SELECT trial_id, number, state, datetime_start, datetime_complete FROM trials ORDER BY number"
        ).fetchall()
        for trial in trials:
            trial_id = int(trial["trial_id"])
            params = _optuna_trial_params(con, trial_id)
            user_attrs = _optuna_trial_attrs(con, trial_id, "trial_user_attributes")
            phase3_candidate_id = str(user_attrs.get("hexo_phase3_candidate_id") or f"{promoted_candidate_id}__phase3_t{int(trial['number']):04d}")
            trial_dir = Path(str(user_attrs.get("hexo_trial_dir") or run_dir / "phase3_trials" / phase3_candidate_id))
            if not trial_dir.is_absolute():
                trial_dir = Path.cwd() / trial_dir if str(trial_dir).startswith("runs") else run_dir / trial_dir
            value = _optuna_trial_value(con, trial_id)
            scorecard = _latest_classical_scorecard(trial_dir)
            metrics = scorecard.get("component_metrics") if isinstance(scorecard.get("component_metrics"), dict) else {}
            latest_train = _latest_phase3_train_scorecard(trial_dir)
            train_metrics = latest_train.get("component_metrics") if isinstance(latest_train.get("component_metrics"), dict) else {}
            manifest = _read_json(trial_dir / "trial_manifest.json")
            latest_checkpoint = user_attrs.get("hexo_latest_checkpoint_path") or scorecard.get("checkpoint_path") or latest_train.get("checkpoint_path")
            events = _jsonl_tail(trial_dir / "events.jsonl", limit=16)
            rows.append(
                {
                    "phase3_candidate_id": phase3_candidate_id,
                    "promoted_candidate_id": promoted_candidate_id,
                    "promoted_label": _trial_display_name(promoted_candidate_id),
                    "study_name": study_name,
                    "storage": storage,
                    "trial_number": int(trial["number"]),
                    "state": trial["state"],
                    "value": value,
                    "phase3_rank": None,
                    "target_epoch": user_attrs.get("hexo_target_epoch") or manifest.get("target_epoch"),
                    "completed_epochs": user_attrs.get("hexo_completed_epochs") or latest_train.get("epoch"),
                    "classical_survival_lcb": _finite_or_none(metrics.get("classical_survival_lcb") or scorecard.get("classical_survival_lcb") or value),
                    "classical_survival_mean": _finite_or_none(metrics.get("classical_survival_mean")),
                    "classical_survival_games": _finite_or_none(metrics.get("classical_survival_games")),
                    "classical_win_rate": _finite_or_none(metrics.get("classical_win_rate")),
                    "classical_draw_rate": _finite_or_none(metrics.get("classical_draw_rate")),
                    "classical_avg_moves": _finite_or_none(metrics.get("classical_avg_moves")),
                    "loss_total": _finite_or_none(train_metrics.get("loss_total")),
                    "loss_value": _finite_or_none(train_metrics.get("loss_value")),
                    "value_weight_mean": _finite_or_none(train_metrics.get("value_weight_mean")),
                    "mcts_simulations": params.get("mcts_simulations"),
                    "pcr_low_sims_ratio": params.get("pcr_low_sims_ratio"),
                    "c_puct": params.get("c_puct"),
                    "lr_multiplier": params.get("lr_multiplier"),
                    "value_loss_weight": params.get("value_loss_weight"),
                    "params": params,
                    "trial_dir": str(trial_dir),
                    "checkpoint_path": str(latest_checkpoint or ""),
                    "scorecard_path": str(trial_dir / "scorecards.jsonl") if (trial_dir / "scorecards.jsonl").exists() else "",
                    "events_path": str(trial_dir / "events.jsonl") if (trial_dir / "events.jsonl").exists() else "",
                    "last_event": (events[-1].get("event") or events[-1].get("event_type") or "") if events else "",
                    "last_event_message": _suite_event_message(_normalize_suite_event(events[-1], trial_id=phase3_candidate_id)) if events else "",
                    "started_at": trial["datetime_start"],
                    "completed_at": trial["datetime_complete"],
                    "updated_at": max(_mtime(trial_dir), _mtime(trial_dir / "events.jsonl"), _mtime(trial_dir / "scorecards.jsonl"), _timestamp_from_sqlite(trial["datetime_complete"]), _timestamp_from_sqlite(trial["datetime_start"])),
                }
            )
    finally:
        con.close()
    return rows


def _sqlite_path_from_storage(storage: str) -> Path | None:
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        return None
    return Path(storage[len(prefix):])


def _optuna_trial_value(con: sqlite3.Connection, trial_id: int) -> float | None:
    try:
        row = con.execute("SELECT value FROM trial_values WHERE trial_id=? ORDER BY objective LIMIT 1", (trial_id,)).fetchone()
    except sqlite3.Error:
        return None
    return _finite_or_none(row["value"]) if row else None


def _optuna_trial_params(con: sqlite3.Connection, trial_id: int) -> dict[str, Any]:
    params: dict[str, Any] = {}
    try:
        rows = con.execute("SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=?", (trial_id,)).fetchall()
    except sqlite3.Error:
        return params
    for row in rows:
        params[str(row["param_name"])] = _decode_optuna_param(row["param_value"], row["distribution_json"])
    return params


def _decode_optuna_param(value: Any, distribution_json: str | None) -> Any:
    try:
        dist = json.loads(distribution_json or "{}")
    except json.JSONDecodeError:
        return _finite_or_none(value)
    attrs = dist.get("attributes") if isinstance(dist.get("attributes"), dict) else {}
    choices = attrs.get("choices")
    if isinstance(choices, list):
        try:
            idx = int(float(value))
            return choices[idx]
        except (TypeError, ValueError, IndexError):
            return value
    return _finite_or_none(value)


def _optuna_trial_attrs(con: sqlite3.Connection, trial_id: int, table: str) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    try:
        rows = con.execute(f"SELECT key, value_json FROM {table} WHERE trial_id=?", (trial_id,)).fetchall()
    except sqlite3.Error:
        return attrs
    for row in rows:
        try:
            attrs[str(row["key"])] = json.loads(row["value_json"])
        except json.JSONDecodeError:
            attrs[str(row["key"])] = row["value_json"]
    return attrs


def _latest_phase3_train_scorecard(trial_dir: Path) -> dict[str, Any]:
    for row in reversed(_jsonl_tail(trial_dir / "scorecards.jsonl", limit=64)):
        metrics = row.get("component_metrics") if isinstance(row.get("component_metrics"), dict) else {}
        if metrics.get("loss_total") is not None:
            return row
    return {}


def _latest_phase3_summary(run_dir: Path) -> Path | None:
    paths = sorted(run_dir.glob("phase3_runner_summary*.json"), key=_mtime)
    return paths[-1] if paths else None


def _timestamp_from_sqlite(value: Any) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value).replace(" ", "T").replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _latest_classical_scorecard(trial_dir: Path) -> dict[str, Any]:
    for row in reversed(_jsonl_tail(trial_dir / "scorecards.jsonl", limit=64)):
        metrics = row.get("component_metrics") if isinstance(row.get("component_metrics"), dict) else {}
        games = _finite_or_none(metrics.get("classical_survival_games"))
        if games is not None and games > 0:
            return row
    return {}


def _latest_fixed_classical_evidence_path(trial_dir: Path) -> Path | None:
    paths = sorted(trial_dir.glob("fixed_classical_epoch_*_games.jsonl"), key=_mtime)
    return paths[-1] if paths else None


def _latest_fixed_classical_summary(run_root: Path) -> Path | None:
    suite_manifest = _read_json(run_root / "manifest.json")
    source = suite_manifest.get("source_run_dir")
    candidates = []
    if source:
        candidates.append(Path(str(source)))
    candidates.append(run_root.parent)
    for root in candidates:
        if not root.exists():
            continue
        summaries = sorted(root.glob("*fixed_classical*_summary*.json"), key=_mtime)
        summaries.extend(sorted(root.glob("*fixed_classical*_games.json"), key=_mtime))
        if summaries:
            return summaries[-1]
    return None


def _scorecard_time(scorecard: dict[str, Any]) -> float:
    created_at = scorecard.get("created_at")
    if isinstance(created_at, str) and created_at:
        try:
            return datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


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
    if not trial and trials:
        trial = max(trials, key=lambda row: float(row.get("updated_at") or 0.0))
        latest_trial_id = str(trial.get("trial_id") or "")
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
        action = latest_event.get("message") or _event_blurb(str(latest_event.get("event")))
    elif trial:
        action = f"{trial.get('trial_label') or latest_trial_id} last wrote epoch {trial.get('epoch') or '-'}"
    stage = latest_event.get("stage") or trial.get("stage") or ""
    return {
        "trial_id": latest_trial_id or None,
        "model": trial.get("trial_label") or trial.get("family") or latest_event.get("family") or None,
        "architecture": trial.get("architecture") or None,
        "stage": stage,
        "epoch": trial.get("epoch"),
        "loss_total": trial.get("loss_total"),
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
    return architecture_display_summary(model, family)


def _model_summary_from_trial(family: dict[str, Any], static: dict[str, Any]) -> str:
    return trial_model_summary(family, static)


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
        "metric": "metric was recorded",
        "checkpoint": "checkpoint was indexed",
        "epoch_complete": "epoch completed",
        "epoch_start": "epoch started",
        "quantum_started": "round-robin quantum started",
        "quantum_completed": "round-robin quantum completed",
        "epoch_runner_completed": "epoch runner completed",
        "training_signal_warning": "training signal warning",
        "runtime_sweep_start": "Runtime sweep is testing worker/batch settings",
        "runtime_sweep_result": "Runtime sweep recorded a probe result",
        "runtime_sweep_selected": "Runtime sweep selected the fastest stable setting",
        "trial_epoch_complete": "Epoch finished; metrics and checkpoint were written",
        "trial_evaluated": "Evaluation finished; scheduler score updated",
        "trial_pruned": "Trial was pruned by a hard gate or scheduler decision",
        "pbt_generation_start": "PBT generation started",
        "stage_start": "Autotune stage started",
    }.get(event, event.replace("_", " "))


def format_count_py(value: Any) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "-"


def format_float_py(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_json_list(path: Path) -> list[Any]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


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
    q = int.from_bytes(history[-8:-4], "little", signed=True)
    r = int.from_bytes(history[-4:], "little", signed=True)
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
            "max_relation_edges": capacity.max_relation_edges,
            "relation_edges": capacity.relation_edge_count,
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
    candidates = build_candidate_batch(
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
        pair = build_pair_candidate_batch(
            candidates.qr,
            pair_policy_target or [],
            budget=min(512, len(legal_moves) * max(len(legal_moves) - 1, 0) // 2),
            candidate_mask=candidates.mask,
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
            pair_candidates = build_candidate_batch(
                [known_first] + legal_moves,
                [],
                offset_q=offset_q,
                offset_r=offset_r,
                budget=storage_width,
                storage_width=storage_width,
                critical_actions=[known_first] + legal_moves,
            )
            pair = build_pair_candidate_batch(
                pair_candidates.qr,
                pair_policy_target,
                budget=min(max(len(legal_moves), 1), 512),
                candidate_mask=pair_candidates.mask,
                legal_moves=legal_moves,
                known_first=known_first,
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
