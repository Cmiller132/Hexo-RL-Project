#!/usr/bin/env python3
"""Export dashboard game rows as PNG match snapshots.

Examples:
  python scripts/render_match_png.py --db runs/dashboard.sqlite3 --latest 4
  python scripts/render_match_png.py --suite-root runs/phase3 --run-id trial_0007 --latest 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Python" / "src"))

from hexorl.dashboard.db import DashboardStore  # noqa: E402
from hexorl.dashboard.render import (  # noqa: E402
    MatchSnapshotOptions,
    snapshot_filename,
    write_match_snapshot_png,
)
from hexorl.dashboard.replay import get_replay_position  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="runs/dashboard.sqlite3", help="Dashboard SQLite DB for a single run.")
    parser.add_argument("--suite-root", default="", help="Phase/autotune suite root containing trials/*/dashboard.sqlite3.")
    parser.add_argument("--run-id", default="", help="Run/trial id to filter.")
    parser.add_argument("--game-id", type=int, default=None, help="Specific dashboard games.game_id to render.")
    parser.add_argument("--latest", type=int, default=1, help="Number of newest matching games to render.")
    parser.add_argument("--source", default="", help="Optional games.source filter.")
    parser.add_argument("--out", default="runs/match_snapshots", help="Output PNG file or directory.")
    parser.add_argument("--turn-index", type=int, default=-1, help="-1 renders the final position.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--context-rings", type=int, default=2)
    parser.add_argument("--show-legal", action="store_true")
    parser.add_argument("--near-radius", type=int, default=8)
    parser.add_argument("--fit", choices=["played", "all"], default="played")
    parser.add_argument("--no-numbers", action="store_true")
    args = parser.parse_args()

    rows = _select_rows(args)
    if not rows:
        raise SystemExit("No matching games found.")

    out = Path(args.out)
    single_file = len(rows) == 1 and out.suffix.lower() == ".png"
    if not single_file:
        out.mkdir(parents=True, exist_ok=True)

    turn = None if args.turn_index < 0 else args.turn_index
    written: list[Path] = []
    for row in rows:
        legal_moves = None
        if args.show_legal:
            try:
                legal_moves = get_replay_position(
                    row["final_history_b64"],
                    turn_index=turn,
                    near_radius=max(1, min(int(args.near_radius), 64)),
                    constrain_threats=False,
                ).legal_moves
            except Exception:
                legal_moves = None
        title = f"{row.get('source') or 'match'} epoch {row.get('epoch') if row.get('epoch') is not None else '-'}"
        target = out if single_file else out / snapshot_filename(row, turn_index=turn)
        write_match_snapshot_png(
            target,
            row["final_history_b64"],
            options=MatchSnapshotOptions(
                width=args.width,
                height=args.height,
                turn_index=turn,
                context_rings=args.context_rings,
                show_numbers=not args.no_numbers,
                show_legal=args.show_legal,
                fit=args.fit,
                title=title,
            ),
            legal_moves=legal_moves,
            metadata=row,
        )
        written.append(target)

    for path in written:
        print(path)


def _select_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for store in _stores(args):
        where = []
        params: list[Any] = []
        if args.game_id is not None:
            where.append("game_id=?")
            params.append(args.game_id)
        if args.run_id:
            where.append("run_id=?")
            params.append(args.run_id)
        if args.source:
            where.append("source=?")
            params.append(args.source)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        limit = 1 if args.game_id is not None else max(1, int(args.latest))
        try:
            rows = store.rows(
                f"SELECT * FROM games {clause} ORDER BY created_at DESC LIMIT ?",
                tuple(params + [limit]),
            )
        except Exception:
            continue
        for row in rows:
            row["source_db"] = str(store.path)
            candidates.append(row)
    candidates.sort(key=lambda row: float(row.get("created_at") or 0.0), reverse=True)
    return candidates[: (1 if args.game_id is not None else max(1, int(args.latest)))]


def _stores(args: argparse.Namespace) -> list[DashboardStore]:
    suite_root = Path(args.suite_root).expanduser() if args.suite_root else None
    if suite_root:
        if args.run_id:
            db = suite_root / "trials" / args.run_id / "dashboard.sqlite3"
            return [DashboardStore(db)] if db.exists() else []
        trials = suite_root / "trials"
        if not trials.exists():
            return []
        return [
            DashboardStore(path)
            for path in sorted(trials.glob("*/dashboard.sqlite3"))
            if path.exists()
        ]
    return [DashboardStore(Path(args.db).expanduser())]


if __name__ == "__main__":
    main()
