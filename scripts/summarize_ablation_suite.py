"""Summarize results from scripts/run_ablation_suite.py."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_root", nargs="?", default="runs/ablations_priority_20260427")
    parser.add_argument("--csv", dest="csv_path", default="")
    args = parser.parse_args()

    suite_root = Path(args.suite_root)
    summary_path = suite_root / "suite_summary.jsonl"
    rows = _load_latest_rows(summary_path)
    if not rows:
        print(f"No epoch rows found in {summary_path}")
        return

    table = [_flatten(row) for row in sorted(rows.values(), key=lambda r: r["ablation"])]
    _print_table(table)
    if args.csv_path:
        _write_csv(Path(args.csv_path), table)


def _load_latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "epoch_complete":
                continue
            name = row.get("ablation")
            if not name:
                continue
            if int(row.get("local_epoch", 0)) >= int(latest.get(name, {}).get("local_epoch", 0)):
                latest[name] = row
    return latest


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    train = row.get("train") or {}
    selfplay = row.get("selfplay") or {}
    return {
        "ablation": row.get("ablation"),
        "epoch": row.get("local_epoch"),
        "loss": _round(train.get("loss_total"), 4),
        "policy": _round(train.get("loss_policy"), 4),
        "value": _round(train.get("loss_value"), 4),
        "top1": _round(train.get("policy_top1_acc"), 4),
        "train_bps": _round(train.get("batches_per_sec"), 2),
        "games_min": _round(selfplay.get("games_per_min"), 2),
        "pos_min": _round(selfplay.get("positions_per_min"), 1),
        "buffer": row.get("buffer_size"),
        "full_pct": _round(row.get("full_search_pct"), 2),
        "elapsed_min": _round((row.get("epoch_elapsed_s") or 0.0) / 60.0, 2),
    }


def _round(value: Any, places: int) -> Any:
    if value is None:
        return ""
    return round(float(value), places)


def _print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        "ablation",
        "epoch",
        "loss",
        "policy",
        "value",
        "top1",
        "train_bps",
        "games_min",
        "pos_min",
        "buffer",
        "full_pct",
        "elapsed_min",
    ]
    widths = {
        col: max(len(col), *(len(str(row.get(col, ""))) for row in rows))
        for col in columns
    }
    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
