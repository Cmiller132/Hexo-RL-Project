"""Background arena match runner for dashboard spectator streams."""

from __future__ import annotations

import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from hexorl.dashboard.db import DashboardStore
from hexorl.eval.classical import classical_opponent_fn
from hexorl.engine.rust import engine_available, hex_game_class

HAS_ENGINE = engine_available()


EventSink = Callable[[str, dict[str, Any]], None]


@dataclass
class ArenaManager:
    store: DashboardStore
    events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _threads: dict[str, threading.Thread] = field(default_factory=dict)

    def start(
        self,
        *,
        run_id: str | None,
        side_a: str,
        side_b: str,
        payload: dict[str, Any] | None = None,
    ) -> str:
        payload = dict(payload or {})
        match_id = uuid.uuid4().hex
        now = time.time()
        with self.store.connect() as conn:
            conn.execute(
                """
                INSERT INTO arena_matches(
                    match_id, run_id, status, side_a, side_b, payload_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_id,
                    run_id,
                    "running",
                    side_a,
                    side_b,
                    _json(payload),
                    now,
                    now,
                ),
            )
        self.events[match_id] = []
        self._emit(match_id, {"type": "match_start", "match_id": match_id, "side_a": side_a, "side_b": side_b})
        thread = threading.Thread(
            target=self._run_match,
            args=(match_id, side_a, side_b, payload),
            daemon=True,
            name=f"arena-{match_id[:8]}",
        )
        self._threads[match_id] = thread
        thread.start()
        return match_id

    def _run_match(self, match_id: str, side_a: str, side_b: str, payload: dict[str, Any]) -> None:
        try:
            if not HAS_ENGINE:
                self._emit(match_id, {"type": "match_result", "winner": -1, "reason": "engine_unavailable"})
                self._finish(match_id, "complete", {"winner": -1, "reason": "engine_unavailable"})
                return
            game = _new_game()
            move_history: list[tuple[int, int, int]] = []
            max_moves = int(payload.get("max_moves", 200))
            players = {
                0: _player_for(side_a, seed=1),
                1: _player_for(side_b, seed=2),
            }
            while not game.is_over and len(move_history) < max_moves:
                player = int(game.current_player)
                fn = players[player]
                q, r = fn(list(move_history), 100, player)
                if q is None or r is None:
                    winner = 1 - player
                    self._emit(match_id, {"type": "match_result", "winner": winner, "reason": "no_move"})
                    self._finish(match_id, "complete", {"winner": winner, "reason": "no_move"})
                    return
                try:
                    game.place(int(q), int(r))
                except Exception as exc:
                    winner = 1 - player
                    self._emit(
                        match_id,
                        {"type": "match_result", "winner": winner, "reason": f"illegal:{exc}"},
                    )
                    self._finish(match_id, "complete", {"winner": winner, "reason": "illegal"})
                    return
                move = {"type": "move", "index": len(move_history), "player": player, "q": int(q), "r": int(r)}
                move_history.append((player, int(q), int(r)))
                self._emit(match_id, move)
                time.sleep(float(payload.get("move_delay_s", 0.02)))

            winner = -1 if game.winner is None else int(game.winner)
            result = {
                "type": "match_result",
                "winner": winner,
                "moves": len(move_history),
                "reason": "terminal" if game.is_over else "max_moves",
            }
            self._emit(match_id, result)
            self._finish(match_id, "complete", result)
        except Exception as exc:  # pragma: no cover - defensive background path
            self._emit(match_id, {"type": "match_error", "error": str(exc)})
            self._finish(match_id, "failed", {"error": str(exc)})

    def _emit(self, match_id: str, event: dict[str, Any]) -> None:
        event = {"time": time.time(), **event}
        self.events.setdefault(match_id, []).append(event)

    def _finish(self, match_id: str, status: str, payload: dict[str, Any]) -> None:
        rows = self.store.rows("SELECT payload_json FROM arena_matches WHERE match_id=?", (match_id,))
        merged = rows[0].get("payload_json", {}) if rows else {}
        merged.update({"result": payload})
        with self.store.connect() as conn:
            conn.execute(
                "UPDATE arena_matches SET status=?, payload_json=?, updated_at=? WHERE match_id=?",
                (status, _json(merged), time.time(), match_id),
            )


def _player_for(side: str, *, seed: int):
    if side == "classical":
        return classical_opponent_fn(time_ms=50, max_depth=2, noise_level=0.03)
    return _random_player(seed)


def _random_player(seed: int):
    rng = random.Random(seed)

    def _fn(move_history, time_ms_override, player):
        game = _new_game()
        for _p, q, r in move_history:
            game.place(int(q), int(r))
        legal = game.legal_moves()
        if not legal:
            return None, None
        q, r = rng.choice(list(legal))
        return int(q), int(r)

    return _fn


def _new_game():
    cls = hex_game_class(required=True)
    return cls()


def _json(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"))
