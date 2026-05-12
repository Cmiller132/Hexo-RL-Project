#!/usr/bin/env python3
"""Analyze Hexo game histories for random-looking play quality.

The dashboard metrics can hide obvious behavioral failures.  This script works
directly from dashboard games or fixed-classical JSONL evidence and reports
spatial scatter, tactical misses, and shallow-classical move agreement.
"""

from __future__ import annotations

import argparse
import base64
import json
import sqlite3
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Python" / "src"))

from hexorl.action_contract.tactical_oracle import scan_tactical_oracle_from_history  # noqa: E402

try:
    import _engine  # noqa: E402

    HAS_ENGINE = True
except ImportError:
    _engine = None
    HAS_ENGINE = False


Move = tuple[int, int, int]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", action="append", default=[], help="dashboard.sqlite3 to analyze.")
    parser.add_argument("--jsonl", action="append", default=[], help="fixed-classical JSONL evidence to analyze.")
    parser.add_argument("--source", default="", help="Optional dashboard games.source filter.")
    parser.add_argument("--latest", type=int, default=64, help="Latest dashboard games per DB.")
    parser.add_argument("--jsonl-limit", type=int, default=64, help="Rows per JSONL evidence file.")
    parser.add_argument("--max-move-index", type=int, default=96, help="Only analyze moves up to this index; negative means all.")
    parser.add_argument("--sample-every", type=int, default=1, help="Analyze every Nth move after the opening.")
    parser.add_argument("--classical-time-ms", type=int, default=10)
    parser.add_argument("--classical-depth", type=int, default=3)
    parser.add_argument("--classical-near-radius", type=int, default=2)
    parser.add_argument("--output", default="runs/game_quality/game_quality_summary.json")
    args = parser.parse_args()

    records = list(_load_records(args))
    summaries = [_analyze_record(record, args) for record in records]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        groups[str(summary["label"])].append(summary)

    payload = {
        "records": len(records),
        "engine_available": HAS_ENGINE,
        "groups": {label: _aggregate(rows) for label, rows in sorted(groups.items())},
        "worst_games": sorted(
            summaries,
            key=lambda row: (
                float(row.get("far_move_frac") or 0.0),
                float(row.get("classical_miss_frac") or 0.0),
                float(row.get("tactical_miss_frac") or 0.0),
            ),
            reverse=True,
        )[:12],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_records(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    for raw in args.jsonl:
        path = Path(raw)
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if idx >= max(1, int(args.jsonl_limit)):
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                yield {
                    "label": path.parent.name,
                    "source_path": str(path),
                    "game_id": row.get("game_index", idx),
                    "source": "fixed_classical",
                    "epoch": _epoch_from_path(path),
                    "history": _history_from_row(row),
                    "move_count": int(row.get("moves") or len(row.get("move_history") or [])),
                    "outcome": row.get("outcome"),
                    "winner": row.get("winner"),
                    "payload": row,
                }
    for raw in args.db:
        db = Path(raw)
        if not db.exists():
            continue
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        filters: list[str] = []
        params: list[Any] = []
        if args.source:
            filters.append("source = ?")
            params.append(args.source)
        where = "WHERE " + " AND ".join(filters) if filters else ""
        query = f"""
            SELECT game_id, run_id, source, epoch, outcome, move_count, final_history_b64, payload_json, created_at
            FROM games
            {where}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(max(1, int(args.latest)))
        for row in con.execute(query, tuple(params)):
            item = dict(row)
            payload = json.loads(item.get("payload_json") or "{}")
            yield {
                "label": db.parent.name,
                "source_path": str(db),
                "game_id": item.get("game_id"),
                "source": item.get("source"),
                "epoch": item.get("epoch"),
                "history": base64.b64decode(item.get("final_history_b64") or ""),
                "move_count": int(item.get("move_count") or 0),
                "outcome": item.get("outcome"),
                "winner": payload.get("winner"),
                "payload": payload,
            }


def _history_from_row(row: dict[str, Any]) -> bytes:
    if row.get("final_history_b64"):
        return base64.b64decode(row["final_history_b64"])
    moves = row.get("move_history") or []
    out = bytearray()
    for move in moves:
        out.extend(struct.pack("<iii", int(move["player"]), int(move["q"]), int(move["r"])))
    return bytes(out)


def _decode_moves(history: bytes) -> list[Move]:
    return [struct.unpack_from("<iii", history, off) for off in range(0, len(history), 12)]


def _encode_moves(moves: list[Move]) -> bytes:
    out = bytearray()
    for player, q, r in moves:
        out.extend(struct.pack("<iii", int(player), int(q), int(r)))
    return bytes(out)


def _analyze_record(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    moves = _decode_moves(record["history"])
    stones: list[Move] = []
    counters: Counter[str] = Counter()
    distance_sum = 0.0
    own_distance_sum = 0.0
    classical_dist_sum = 0.0
    per_player: dict[int, Counter[str]] = defaultdict(Counter)

    max_move_index = int(args.max_move_index)
    sample_every = max(1, int(args.sample_every))
    for idx, (player, q, r) in enumerate(moves):
        if max_move_index >= 0 and idx > max_move_index:
            break
        if idx > 3 and (idx % sample_every) != 0:
            stones.append((player, q, r))
            continue
        any_dist = _nearest_distance((q, r), [(sq, sr) for _sp, sq, sr in stones])
        own_dist = _nearest_distance((q, r), [(sq, sr) for sp, sq, sr in stones if sp == player])
        if idx > 0:
            counters["distance_samples"] += 1
            distance_sum += any_dist
            if any_dist > 2:
                counters["far_gt2"] += 1
            if any_dist > 4:
                counters["far_gt4"] += 1
            if any_dist > 8:
                counters["far_gt8"] += 1
            per_player[player]["distance_samples"] += 1
            if any_dist > 4:
                per_player[player]["far_gt4"] += 1
        if own_dist < 10**8:
            counters["own_distance_samples"] += 1
            own_distance_sum += own_dist
            if own_dist > 4:
                counters["own_far_gt4"] += 1

        history_before = _encode_moves(stones)
        tactical = _tactical_hit(history_before, (q, r))
        if tactical["available"]:
            counters["tactical_available"] += 1
            per_player[player]["tactical_available"] += 1
            if tactical["hit"]:
                counters["tactical_hit"] += 1
                per_player[player]["tactical_hit"] += 1
            else:
                counters["tactical_miss"] += 1
                per_player[player]["tactical_miss"] += 1
        if tactical["open_four_available"]:
            counters["open_four_available"] += 1
            if tactical["open_four_hit"]:
                counters["open_four_hit"] += 1

        classical = _classical_hit(history_before, (q, r), args)
        if classical["available"]:
            counters["classical_available"] += 1
            per_player[player]["classical_available"] += 1
            classical_dist_sum += classical["distance"]
            if classical["hit"]:
                counters["classical_hit"] += 1
                per_player[player]["classical_hit"] += 1
            else:
                counters["classical_miss"] += 1
                per_player[player]["classical_miss"] += 1
        stones.append((player, q, r))

    return {
        "label": record["label"],
        "source_path": record["source_path"],
        "game_id": record["game_id"],
        "source": record["source"],
        "epoch": record["epoch"],
        "moves": len(moves),
        "outcome": record.get("outcome"),
        "winner": record.get("winner"),
        "mean_nearest_any": _div(distance_sum, counters["distance_samples"]),
        "far_move_frac": _div(counters["far_gt4"], counters["distance_samples"]),
        "very_far_move_frac": _div(counters["far_gt8"], counters["distance_samples"]),
        "mean_nearest_own": _div(own_distance_sum, counters["own_distance_samples"]),
        "own_far_move_frac": _div(counters["own_far_gt4"], counters["own_distance_samples"]),
        "tactical_available": counters["tactical_available"],
        "tactical_hit_frac": _div(counters["tactical_hit"], counters["tactical_available"]),
        "tactical_miss_frac": _div(counters["tactical_miss"], counters["tactical_available"]),
        "open_four_available": counters["open_four_available"],
        "open_four_hit_frac": _div(counters["open_four_hit"], counters["open_four_available"]),
        "classical_hit_frac": _div(counters["classical_hit"], counters["classical_available"]),
        "classical_miss_frac": _div(counters["classical_miss"], counters["classical_available"]),
        "mean_classical_distance": _div(classical_dist_sum, counters["classical_available"]),
        "per_player": {
            str(player): {
                "far_move_frac": _div(counts["far_gt4"], counts["distance_samples"]),
                "classical_hit_frac": _div(counts["classical_hit"], counts["classical_available"]),
                "tactical_hit_frac": _div(counts["tactical_hit"], counts["tactical_available"]),
                "tactical_available": counts["tactical_available"],
            }
            for player, counts in sorted(per_player.items())
        },
    }


def _tactical_hit(history: bytes, actual: tuple[int, int]) -> dict[str, Any]:
    if not HAS_ENGINE:
        return {"available": False, "hit": False, "open_four_available": False, "open_four_hit": False}
    try:
        oracle = scan_tactical_oracle_from_history(history)
    except Exception:
        return {"available": False, "hit": False, "open_four_available": False, "open_four_hit": False}
    win = set(map(tuple, oracle.win_now_cells))
    forced = set(map(tuple, oracle.forced_block_cells))
    open_four = set(map(tuple, oracle.open_four_cells))
    open_five = set(map(tuple, oracle.open_five_cells))
    cover = set(map(tuple, oracle.cover_cells))
    critical = win | forced | open_four | open_five | cover
    return {
        "available": bool(critical),
        "hit": actual in critical,
        "open_four_available": bool(open_four),
        "open_four_hit": actual in open_four,
    }


def _classical_hit(history: bytes, actual: tuple[int, int], args: argparse.Namespace) -> dict[str, Any]:
    if not HAS_ENGINE or int(args.classical_time_ms) <= 0:
        return {"available": False, "hit": False, "distance": 0}
    try:
        game = _engine.HexGame()
        for _player, q, r in _decode_moves(history):
            game.place(int(q), int(r))
        q, r, _score, _depth, _nodes = game.classical_search(
            time_ms=int(args.classical_time_ms),
            max_depth=int(args.classical_depth),
            near_radius=int(args.classical_near_radius),
            noise_level=0.0,
        )
        qr = (int(q), int(r))
        return {"available": True, "hit": actual == qr, "distance": _hex_distance(actual, qr)}
    except Exception:
        return {"available": False, "hit": False, "distance": 0}


def _nearest_distance(qr: tuple[int, int], others: list[tuple[int, int]]) -> int:
    if not others:
        return 10**9
    return min(_hex_distance(qr, other) for other in others)


def _hex_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    dq = int(a[0]) - int(b[0])
    dr = int(a[1]) - int(b[1])
    ds = -(dq + dr)
    return max(abs(dq), abs(dr), abs(ds))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = [
        "moves",
        "mean_nearest_any",
        "far_move_frac",
        "very_far_move_frac",
        "mean_nearest_own",
        "own_far_move_frac",
        "tactical_hit_frac",
        "tactical_miss_frac",
        "open_four_hit_frac",
        "classical_hit_frac",
        "classical_miss_frac",
        "mean_classical_distance",
    ]
    return {
        "games": len(rows),
        "avg": {key: sum(float(row.get(key) or 0.0) for row in rows) / len(rows) for key in keys},
        "tactical_available_games": sum(1 for row in rows if int(row.get("tactical_available") or 0) > 0),
        "open_four_available_games": sum(1 for row in rows if int(row.get("open_four_available") or 0) > 0),
        "outcomes": dict(Counter(str(row.get("outcome")) for row in rows)),
    }


def _div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _epoch_from_path(path: Path) -> int | None:
    name = path.name
    marker = "epoch_"
    if marker not in name:
        return None
    try:
        return int(name.split(marker, 1)[1].split("_", 1)[0])
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
