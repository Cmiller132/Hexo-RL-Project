"""SQLite persistence for dashboard, eval, checkpoints, and replay metadata.

The dashboard store intentionally uses sqlite3 directly.  The schema keeps a
small set of indexed columns for common queries and a JSON payload column for
features that are still evolving.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

SCHEMA_VERSION = 1


def _json_dumps(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"))


def _json_loads(value: str | bytes | None) -> Any:
    if value in (None, ""):
        return {}
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def encode_bytes(data: bytes | None) -> str:
    """Encode binary payloads for JSON columns."""
    if not data:
        return ""
    return base64.b64encode(data).decode("ascii")


def decode_bytes(data: str | None) -> bytes:
    """Decode a JSON-safe binary payload."""
    if not data:
        return b""
    return base64.b64decode(data.encode("ascii"))


@dataclass(frozen=True)
class DashboardStore:
    """Tiny SQLite store with explicit migration and typed helper methods."""

    path: Path | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def migrate(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)"
            )
            current = {
                int(row[0])
                for row in conn.execute("SELECT version FROM schema_migrations")
            }
            if 1 not in current:
                _apply_v1(conn)
                conn.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (1, time.time()),
                )
            conn.commit()

    def upsert_run(
        self,
        run_id: str,
        *,
        name: str | None = None,
        output_dir: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        now = time.time()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT created_at FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            created_at = float(row["created_at"]) if row else now
            conn.execute(
                """
                INSERT INTO runs(
                    run_id, name, output_dir, config_json, payload_json,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    name=excluded.name,
                    output_dir=excluded.output_dir,
                    config_json=excluded.config_json,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    run_id,
                    name or run_id,
                    str(output_dir or ""),
                    _json_dumps(config),
                    _json_dumps(payload),
                    created_at,
                    now,
                ),
            )

    def record_metric(
        self,
        run_id: str,
        *,
        epoch: int | None = None,
        global_step: int | None = None,
        phase: str = "idle",
        metrics: Mapping[str, Any] | None = None,
    ) -> int:
        self.upsert_run(run_id)
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO metrics(run_id, epoch, global_step, phase, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    epoch,
                    global_step,
                    phase,
                    _json_dumps(metrics),
                    time.time(),
                ),
            )
            return int(cur.lastrowid)

    def record_event(
        self,
        run_id: str,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        phase: str | None = None,
        epoch: int | None = None,
        global_step: int | None = None,
    ) -> int:
        self.upsert_run(run_id)
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO events(
                    run_id, event_type, phase, epoch, global_step, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    event_type,
                    phase,
                    epoch,
                    global_step,
                    _json_dumps(payload),
                    time.time(),
                ),
            )
            return int(cur.lastrowid)

    def upsert_checkpoint(
        self,
        *,
        path: str | Path,
        sha256: str,
        run_id: str | None = None,
        epoch: int | None = None,
        global_step: int | None = None,
        is_loadable: bool = False,
        model_heads: list[str] | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> int:
        run_key = run_id or "unassigned"
        self.upsert_run(run_key)
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO checkpoints(
                    run_id, path, sha256, epoch, global_step, is_loadable,
                    model_heads_json, payload_json, indexed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    run_id=excluded.run_id,
                    sha256=excluded.sha256,
                    epoch=excluded.epoch,
                    global_step=excluded.global_step,
                    is_loadable=excluded.is_loadable,
                    model_heads_json=excluded.model_heads_json,
                    payload_json=excluded.payload_json,
                    indexed_at=excluded.indexed_at
                """,
                (
                    run_key,
                    str(Path(path)),
                    sha256,
                    epoch,
                    global_step,
                    int(is_loadable),
                    _json_dumps(model_heads or []),
                    _json_dumps(payload),
                    time.time(),
                ),
            )
            row = conn.execute(
                "SELECT checkpoint_id FROM checkpoints WHERE path=?",
                (str(Path(path)),),
            ).fetchone()
            return int(row["checkpoint_id"] if row else cur.lastrowid)

    def insert_game(
        self,
        *,
        run_id: str,
        game_id: int | str,
        source: str,
        final_move_history: bytes,
        outcome: float = 0.0,
        epoch: int | None = None,
        checkpoint_id: int | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> int:
        self.upsert_run(run_id)
        move_count = len(final_move_history) // 12
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO games(
                    run_id, external_game_id, source, epoch, checkpoint_id,
                    outcome, move_count, final_history_b64, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(game_id),
                    source,
                    epoch,
                    checkpoint_id,
                    outcome,
                    move_count,
                    encode_bytes(final_move_history),
                    _json_dumps(payload),
                    time.time(),
                ),
            )
            return int(cur.lastrowid)

    def insert_position(
        self,
        game_row_id: int,
        *,
        turn_index: int,
        player: int,
        move_history: bytes,
        root_value: float = 0.0,
        policy_target: Mapping[int, float] | None = None,
        debug: Mapping[str, Any] | None = None,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO positions(
                    game_id, turn_index, player, move_history_b64,
                    root_value, policy_json, debug_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_row_id,
                    turn_index,
                    player,
                    encode_bytes(move_history),
                    root_value,
                    _json_dumps({str(k): v for k, v in (policy_target or {}).items()}),
                    _json_dumps(debug),
                ),
            )
            return int(cur.lastrowid)

    def insert_game_with_positions(
        self,
        *,
        run_id: str,
        game_id: int | str,
        source: str,
        final_move_history: bytes,
        outcome: float = 0.0,
        epoch: int | None = None,
        checkpoint_id: int | None = None,
        payload: Mapping[str, Any] | None = None,
        positions: list[Mapping[str, Any]] | None = None,
    ) -> int:
        """Insert one game and all position rows in a single transaction."""
        self.upsert_run(run_id)
        move_count = len(final_move_history) // 12
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO games(
                    run_id, external_game_id, source, epoch, checkpoint_id,
                    outcome, move_count, final_history_b64, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(game_id),
                    source,
                    epoch,
                    checkpoint_id,
                    outcome,
                    move_count,
                    encode_bytes(final_move_history),
                    _json_dumps(payload),
                    time.time(),
                ),
            )
            game_row_id = int(cur.lastrowid)
            conn.executemany(
                """
                INSERT INTO positions(
                    game_id, turn_index, player, move_history_b64,
                    root_value, policy_json, debug_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        game_row_id,
                        int(pos["turn_index"]),
                        int(pos["player"]),
                        encode_bytes(pos["move_history"]),
                        float(pos.get("root_value", 0.0)),
                        _json_dumps({str(k): v for k, v in pos.get("policy_target", {}).items()}),
                        _json_dumps(pos.get("debug", {})),
                    )
                    for pos in (positions or [])
                ],
            )
            return game_row_id

    def save_axis_preset(
        self,
        *,
        name: str,
        prototype_id: str,
        parameters: Mapping[str, Any],
        payload: Mapping[str, Any] | None = None,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO axis_presets(name, prototype_id, parameters_json, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    name,
                    prototype_id,
                    _json_dumps(parameters),
                    _json_dumps(payload),
                    time.time(),
                ),
            )
            return int(cur.lastrowid)

    def rows(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [_row_to_dict(row) for row in conn.execute(query, params)]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in list(data):
        if key.endswith("_json") and isinstance(data[key], str):
            data[key] = _json_loads(data[key])
        elif key.endswith("_b64") and isinstance(data[key], str):
            data[key] = decode_bytes(data[key])
    return data


def _apply_v1(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            output_dir TEXT NOT NULL DEFAULT '',
            config_json TEXT NOT NULL DEFAULT '{}',
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            epoch INTEGER,
            global_step INTEGER,
            phase TEXT NOT NULL,
            metrics_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_metrics_run_epoch ON metrics(run_id, epoch, global_step);
        CREATE INDEX IF NOT EXISTS idx_metrics_created ON metrics(created_at);

        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            event_type TEXT NOT NULL,
            phase TEXT,
            epoch INTEGER,
            global_step INTEGER,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_events_run_type ON events(run_id, event_type, created_at);

        CREATE TABLE IF NOT EXISTS checkpoints (
            checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            path TEXT NOT NULL UNIQUE,
            sha256 TEXT NOT NULL,
            epoch INTEGER,
            global_step INTEGER,
            is_loadable INTEGER NOT NULL DEFAULT 0,
            model_heads_json TEXT NOT NULL DEFAULT '[]',
            payload_json TEXT NOT NULL DEFAULT '{}',
            indexed_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_checkpoints_run_epoch ON checkpoints(run_id, epoch, global_step);

        CREATE TABLE IF NOT EXISTS games (
            game_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            external_game_id TEXT NOT NULL,
            source TEXT NOT NULL,
            epoch INTEGER,
            checkpoint_id INTEGER REFERENCES checkpoints(checkpoint_id) ON DELETE SET NULL,
            outcome REAL NOT NULL DEFAULT 0.0,
            move_count INTEGER NOT NULL DEFAULT 0,
            final_history_b64 TEXT NOT NULL DEFAULT '',
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_games_run_epoch ON games(run_id, epoch, created_at);
        CREATE INDEX IF NOT EXISTS idx_games_source ON games(source);

        CREATE TABLE IF NOT EXISTS positions (
            position_id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
            turn_index INTEGER NOT NULL,
            player INTEGER NOT NULL,
            move_history_b64 TEXT NOT NULL DEFAULT '',
            root_value REAL NOT NULL DEFAULT 0.0,
            policy_json TEXT NOT NULL DEFAULT '{}',
            debug_json TEXT NOT NULL DEFAULT '{}',
            UNIQUE(game_id, turn_index)
        );
        CREATE INDEX IF NOT EXISTS idx_positions_game_turn ON positions(game_id, turn_index);

        CREATE TABLE IF NOT EXISTS eval_runs (
            eval_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            side_a TEXT NOT NULL,
            side_b TEXT NOT NULL,
            num_games INTEGER NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL,
            completed_at REAL
        );

        CREATE TABLE IF NOT EXISTS eval_games (
            eval_game_id INTEGER PRIMARY KEY AUTOINCREMENT,
            eval_run_id INTEGER NOT NULL REFERENCES eval_runs(eval_run_id) ON DELETE CASCADE,
            game_index INTEGER NOT NULL,
            winner INTEGER NOT NULL,
            moves INTEGER NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS arena_matches (
            match_id TEXT PRIMARY KEY,
            run_id TEXT REFERENCES runs(run_id) ON DELETE SET NULL,
            status TEXT NOT NULL,
            side_a TEXT NOT NULL,
            side_b TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS play_sessions (
            session_id TEXT PRIMARY KEY,
            run_id TEXT REFERENCES runs(run_id) ON DELETE SET NULL,
            status TEXT NOT NULL,
            current_player INTEGER NOT NULL DEFAULT 0,
            move_history_b64 TEXT NOT NULL DEFAULT '',
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS axis_presets (
            preset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            prototype_id TEXT NOT NULL,
            parameters_json TEXT NOT NULL DEFAULT '{}',
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            path TEXT NOT NULL,
            media_type TEXT,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_artifacts_run_kind ON artifacts(run_id, kind);
        """
    )
