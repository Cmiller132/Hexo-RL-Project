"""Persistent interactive play sessions for the dashboard."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from hexorl.dashboard.db import DashboardStore, encode_bytes
from hexorl.dashboard.replay import (
    Move,
    decode_move_history,
    encode_move_history,
    get_replay_position,
    position_payload,
)


@dataclass(frozen=True)
class PlaySession:
    session_id: str
    status: str
    run_id: str | None
    move_history: bytes
    payload: dict[str, Any]

    @property
    def moves(self) -> list[Move]:
        return decode_move_history(self.move_history)


def create_session(
    store: DashboardStore,
    *,
    run_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> PlaySession:
    session_id = uuid.uuid4().hex
    now = time.time()
    with store.connect() as conn:
        conn.execute(
            """
            INSERT INTO play_sessions(
                session_id, run_id, status, current_player, move_history_b64,
                payload_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                run_id,
                "active",
                0,
                "",
                _json(payload or {}),
                now,
                now,
            ),
        )
    return get_session(store, session_id)


def get_session(store: DashboardStore, session_id: str) -> PlaySession:
    rows = store.rows("SELECT * FROM play_sessions WHERE session_id=?", (session_id,))
    if not rows:
        raise KeyError(f"Play session not found: {session_id}")
    row = rows[0]
    return PlaySession(
        session_id=row["session_id"],
        status=row["status"],
        run_id=row["run_id"],
        move_history=row["move_history_b64"],
        payload=row.get("payload_json", {}),
    )


def session_payload(
    store: DashboardStore,
    session_id: str,
    *,
    near_radius: int = 8,
    constrain_threats: bool = False,
) -> dict[str, Any]:
    session = get_session(store, session_id)
    pos = get_replay_position(
        session.move_history,
        near_radius=near_radius,
        constrain_threats=constrain_threats,
    )
    return {
        "session_id": session.session_id,
        "status": session.status,
        "run_id": session.run_id,
        "payload": session.payload,
        "position": position_payload(pos),
    }


def apply_move(store: DashboardStore, session_id: str, q: int, r: int) -> PlaySession:
    session = get_session(store, session_id)
    current = get_replay_position(session.move_history, constrain_threats=False)
    if current.is_over:
        raise ValueError("Cannot play a move into a completed game")
    legal = {(m["q"], m["r"]) for m in current.legal_moves}
    if legal and (int(q), int(r)) not in legal:
        raise ValueError(f"Illegal move: ({q}, {r})")
    moves = session.moves + [(current.current_player, int(q), int(r))]
    history = encode_move_history(moves)
    next_pos = get_replay_position(history)
    status = "complete" if next_pos.is_over else "active"
    _update_session(store, session_id, history, status=status)
    return get_session(store, session_id)


def undo_move(store: DashboardStore, session_id: str) -> PlaySession:
    session = get_session(store, session_id)
    moves = session.moves[:-1]
    history = encode_move_history(moves)
    _update_session(store, session_id, history, status="active")
    return get_session(store, session_id)


def reset_session(store: DashboardStore, session_id: str) -> PlaySession:
    _update_session(store, session_id, b"", status="active")
    return get_session(store, session_id)


def _update_session(
    store: DashboardStore,
    session_id: str,
    history: bytes,
    *,
    status: str,
) -> None:
    current_player = len(decode_move_history(history)) % 2
    with store.connect() as conn:
        conn.execute(
            """
            UPDATE play_sessions
            SET status=?, current_player=?, move_history_b64=?, updated_at=?
            WHERE session_id=?
            """,
            (status, current_player, encode_bytes(history), time.time(), session_id),
        )


def _json(value: Mapping[str, Any]) -> str:
    import json

    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
