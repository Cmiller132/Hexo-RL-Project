"""Reusable dashboard test-state generation."""

from __future__ import annotations

import random
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from hexorl.dashboard.db import DashboardStore
from hexorl.dashboard.play import create_session, session_payload
from hexorl.dashboard.replay import Move, encode_move_history


@dataclass(frozen=True)
class ClassicalFixtureConfig:
    count: int = 8
    examples_per_move_count: int = 1
    move_counts: tuple[int, ...] = (8, 16, 24, 32, 40, 48)
    time_ms: int = 10
    max_depth: int = 3
    near_radius: int = 8
    noise_level: float = 0.08
    random_move_prob: float = 0.04
    opening_random_moves: int = 2
    seed: int = 0
    workers: int = 1


def list_axis_fixtures(store: DashboardStore, *, limit: int = 200) -> list[dict[str, Any]]:
    rows = store.rows(
        "SELECT * FROM play_sessions ORDER BY updated_at DESC LIMIT ?",
        (max(1, min(limit, 1000)),),
    )
    fixtures: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload_json", {})
        if payload.get("mode") != "axis_fixture":
            continue
        move_history = row.get("move_history_b64", b"")
        fixtures.append(
            {
                "session_id": row["session_id"],
                "status": row["status"],
                "updated_at": row["updated_at"],
                "move_count": len(move_history) // 12,
                "payload": payload,
            }
        )
    return fixtures


def generate_classical_fixtures(
    store: DashboardStore,
    config: ClassicalFixtureConfig | None = None,
) -> list[dict[str, Any]]:
    """Generate varied Rust-classical self-play positions as play sessions."""
    config = config or ClassicalFixtureConfig()
    if importlib.util.find_spec("_engine") is None:  # pragma: no cover - depends on local build artifacts.
        raise RuntimeError("Rust _engine extension is required for classical fixtures")

    move_counts = tuple(max(0, int(v)) for v in config.move_counts) or (16,)
    targets = _fixture_targets(move_counts, config.count, config.examples_per_move_count)
    specs = [(index, target_moves, example_index, config) for index, (target_moves, example_index) in enumerate(targets)]
    worker_count = max(1, min(int(config.workers), len(specs)))
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            generated = list(pool.map(_generate_fixture_record, specs))
    else:
        generated = [_generate_fixture_record(spec) for spec in specs]

    fixtures: list[dict[str, Any]] = []
    for history, payload, move_count in generated:
        session = create_session(store, payload=payload, move_history=history)
        loaded = session_payload(store, session.session_id)
        fixtures.append(
            {
                "session_id": session.session_id,
                "status": session.status,
                "move_count": move_count,
                "payload": payload,
                "position": loaded["position"],
            }
        )
    return fixtures


def _generate_fixture_record(
    spec: tuple[int, int, int, ClassicalFixtureConfig],
) -> tuple[bytes, dict[str, Any], int]:
    index, target_moves, example_index, config = spec
    import _engine  # type: ignore

    fixture_seed = int(config.seed) + index * 1009 + target_moves * 17
    rng = random.Random(fixture_seed)
    game = _engine.HexGame()
    moves: list[Move] = []
    searches: list[dict[str, Any]] = []

    while len(moves) < target_moves and not bool(_engine_value(game, "is_over")):
        player = int(_engine_value(game, "current_player"))
        q, r, search = _choose_classical_move(game, config, rng, len(moves))
        game.place(int(q), int(r))
        moves.append((player, int(q), int(r)))
        searches.append(search)

    history = encode_move_history(moves)
    payload = {
        "mode": "axis_fixture",
        "source": "rust_classical_selfplay",
        "label": f"Classical {len(moves)}m #{example_index + 1}",
        "fixture_index": index,
        "example_index": example_index,
        "target_moves": target_moves,
        "actual_moves": len(moves),
        "seed": config.seed,
        "fixture_seed": fixture_seed,
        "time_ms": config.time_ms,
        "max_depth": config.max_depth,
        "near_radius": config.near_radius,
        "noise_level": config.noise_level,
        "random_move_prob": config.random_move_prob,
        "opening_random_moves": config.opening_random_moves,
        "workers": config.workers,
        "winner": _optional_int(_engine_value(game, "winner")),
        "is_over": bool(_engine_value(game, "is_over")),
        "searches": searches[-8:],
    }
    return history, payload, len(moves)


def _choose_classical_move(
    game: Any,
    config: ClassicalFixtureConfig,
    rng: random.Random,
    move_index: int,
) -> tuple[int, int, dict[str, Any]]:
    legal = list(_legal_moves(game, config.near_radius))
    use_random = (
        move_index < config.opening_random_moves
        or (config.random_move_prob > 0 and rng.random() < config.random_move_prob)
    )
    if use_random and legal:
        q, r = rng.choice(legal)
        return int(q), int(r), {"source": "python_random_legal", "nodes": 0, "depth": 0, "score": 0.0}

    try:
        result = game.classical_search(
            time_ms=max(1, int(config.time_ms)),
            max_depth=max(1, int(config.max_depth)),
            near_radius=max(1, int(config.near_radius)),
            noise_level=max(0.0, float(config.noise_level)),
        )
        q, r, score, depth, nodes = result
        return (
            int(q),
            int(r),
            {
                "source": "rust_classical_search",
                "score": float(score),
                "depth": int(depth),
                "nodes": int(nodes),
            },
        )
    except Exception:
        if not legal:
            raise
        q, r = rng.choice(legal)
        return int(q), int(r), {"source": "fallback_random_legal", "nodes": 0, "depth": 0, "score": 0.0}


def _legal_moves(game: Any, near_radius: int) -> list[tuple[int, int]]:
    try:
        moves = game.legal_moves_near(max(1, int(near_radius)))
        if moves:
            return [(int(q), int(r)) for q, r in moves]
    except Exception:
        pass
    return [(int(q), int(r)) for q, r in game.legal_moves()]


def _fixture_targets(
    move_counts: tuple[int, ...],
    count: int,
    examples_per_move_count: int,
) -> list[tuple[int, int]]:
    if examples_per_move_count > 0:
        return [
            (target_moves, example_index)
            for target_moves in move_counts
            for example_index in range(examples_per_move_count)
        ]
    return [(move_counts[index % len(move_counts)], index // len(move_counts)) for index in range(max(1, count))]


def _engine_value(game: Any, name: str) -> Any:
    value = getattr(game, name)
    return value() if callable(value) else value


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
