from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from hexorl.dashboard.render import MatchSnapshotOptions, render_match_snapshot_png, snapshot_filename
from hexorl.dashboard.replay import get_replay_position, position_payload, replay_game
from hexorl.dashboard.services.common import game_row_for_request, game_summary
from hexorl.dashboard.services.suite import suite_games, suite_store_for_run

router = APIRouter()


@router.get("/api/games")
def games(request: Request, run_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    store = request.app.state.store
    suite_root = request.app.state.suite_root
    if suite_root is not None:
        rows = suite_games(suite_root, run_id=run_id, limit=max(1, min(limit, 2000)))
        return [game_summary(row) for row in rows]
    if run_id:
        rows = store.rows("SELECT * FROM games WHERE run_id=? ORDER BY created_at DESC LIMIT ?", (run_id, max(1, min(limit, 2000))))
    else:
        rows = store.rows("SELECT * FROM games ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 2000)),))
    return [game_summary(row) for row in rows]


@router.get("/api/games/{game_id}/replay")
def game_replay(request: Request, game_id: int, run_id: str | None = None) -> dict[str, Any]:
    source = suite_store_for_run(request.app.state.suite_root, run_id) if run_id else request.app.state.store
    try:
        return replay_game(source or request.app.state.store, game_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/api/games/{game_id}/position/{turn_index}")
def game_position(request: Request, game_id: int, turn_index: int, run_id: str | None = None) -> dict[str, Any]:
    source = suite_store_for_run(request.app.state.suite_root, run_id) if run_id else request.app.state.store
    rows = (source or request.app.state.store).rows("SELECT final_history_b64 FROM games WHERE game_id=?", (game_id,))
    if not rows:
        raise HTTPException(404, f"Game not found: {game_id}")
    return position_payload(get_replay_position(rows[0]["final_history_b64"], turn_index=turn_index))


@router.get("/api/games/{game_id}/snapshot.png")
def game_snapshot(
    request: Request,
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
    row = game_row_for_request(request.app.state.store, request.app.state.suite_root, game_id, run_id)
    if row is None:
        raise HTTPException(404, f"Game not found: {game_id}")
    turn = None if turn_index < 0 else turn_index
    legal_moves = None
    if show_legal:
        try:
            legal_moves = get_replay_position(row["final_history_b64"], turn_index=turn, near_radius=max(1, min(near_radius, 64)), constrain_threats=False).legal_moves
        except Exception:
            legal_moves = None
    png = render_match_snapshot_png(
        row["final_history_b64"],
        options=MatchSnapshotOptions(width=width, height=height, turn_index=turn, context_rings=context_rings, show_numbers=show_numbers, show_legal=show_legal, fit=fit, title=f"{row.get('source') or 'match'} epoch {row.get('epoch') if row.get('epoch') is not None else '-'}"),
        legal_moves=legal_moves,
        metadata=row,
    )
    return Response(content=png, media_type="image/png", headers={"Content-Disposition": f'inline; filename="{snapshot_filename(row, turn_index=turn)}"'})

