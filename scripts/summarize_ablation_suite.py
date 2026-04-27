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
    parser.add_argument("--md", dest="md_path", default="")
    args = parser.parse_args()

    suite_root = Path(args.suite_root)
    summary_path = suite_root / "suite_summary.jsonl"
    rows = _load_latest_rows(summary_path)
    if not rows:
        print(f"No epoch rows found in {summary_path}")
        if args.md_path:
            _write_markdown(
                Path(args.md_path),
                suite_root,
                [],
                note="No completed epoch summaries have been written yet.",
            )
        return

    table = [_flatten(row) for row in sorted(rows.values(), key=lambda r: r["ablation"])]
    _print_table(table)
    if args.csv_path:
        _write_csv(Path(args.csv_path), table)
    if args.md_path:
        _write_markdown(Path(args.md_path), suite_root, table)


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


def _write_markdown(
    path: Path,
    suite_root: Path,
    rows: list[dict[str, Any]],
    *,
    note: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "ablation",
        "epoch",
        "loss",
        "top1",
        "train_bps",
        "games_min",
        "pos_min",
        "buffer",
        "elapsed_min",
    ]
    lines = [
        "# Hexo-RL Long Ablation Results",
        "",
        f"Suite root: `{suite_root}`",
        "",
        "This document is generated from the suite JSONL summaries while the run is in progress. "
        "Treat partially completed ablations as early signals, not final conclusions.",
        "",
    ]
    if note:
        lines.extend(["## Current Status", "", note, ""])
    if rows:
        best_loss = min(rows, key=lambda r: _numeric(r.get("loss"), float("inf")))
        best_speed = max(rows, key=lambda r: _numeric(r.get("games_min"), float("-inf")))
        lines.extend(
            [
                "## Current Leaders",
                "",
                f"- Lowest latest loss: `{best_loss['ablation']}` at epoch {best_loss['epoch']} with loss `{best_loss['loss']}`.",
                f"- Fastest latest self-play: `{best_speed['ablation']}` at `{best_speed['games_min']}` games/min.",
                "",
                "## Latest Metrics",
                "",
                "| " + " | ".join(columns) + " |",
                "| " + " | ".join("---" for _ in columns) + " |",
            ]
        )
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Compare ablations primarily after they reach the same epoch count; early epochs are dominated by bootstrap and replay composition.",
            "- Throughput should be judged alongside loss and evaluation, because faster search settings may produce weaker targets.",
            "- Model-size variants are expected to change both training speed and MCTS/inference throughput, so wall-clock progress matters as much as per-epoch loss.",
            "",
            "## Improvement Ideas To Revisit",
            "",
            "- Add sparse policy transfer from inference to MCTS so workers receive priors only for legal moves rather than full 1089-logit vectors.",
            "- Add optional bucketed inference batches for CUDA graph or compile-friendly static shapes, then ablate padding cost versus compile speedup.",
            "- Keep train compile only if the multi-epoch ablation shows the warmup cost amortizes cleanly.",
            "- Add checkpoint-vs-checkpoint arenas between ablations once several variants complete, not only model-vs-classical smoke eval.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _numeric(value: Any, default: float) -> float:
    if value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
