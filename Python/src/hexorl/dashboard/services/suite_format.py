"""Suite formatting and summary helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def tail_lines(path: Path, *, limit: int) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except OSError:
        return []


def last_progress(lines: list[str]) -> dict[str, Any] | None:
    pattern = re.compile(r"Progress:\s+([0-9.]+)%\s+\|\s+Games:\s+(\d+)\s+\(([0-9.]+)/min\)\s+\|\s+Buffer:\s+(\d+)\s+\|\s+Workers:\s+(\d+)/(\d+)")
    for line in reversed(lines):
        if match := pattern.search(line):
            return {"progress_pct": float(match.group(1)), "games_done": int(match.group(2)), "games_per_min": float(match.group(3)), "buffer_positions": int(match.group(4)), "workers_alive": int(match.group(5)), "workers_total": int(match.group(6))}
    return None


def architecture_summary(model: dict[str, Any], family: dict[str, Any] | None = None) -> str:
    family = family or {}
    arch = str(model.get("architecture") or family.get("architecture") or "").lower()
    channels = model.get("channels") or family.get("channels")
    blocks = model.get("blocks") or family.get("blocks")
    heads = model.get("heads") or []
    if arch in {"graph", "graph_hybrid_0"}:
        return f"Graph hybrid 0, {channels} channels, {blocks} residual blocks, {model.get('graph_token_budget', '?')} {model.get('graph_token_set', 'tokens')}, {model.get('graph_layers', '?')} graph layers, heads: {len(heads)}."
    if arch == "restnet":
        return f"ResTNet hybrid trunk, {channels} channels, {blocks} blocks, attention at {model.get('attention_positions') or []}, heads: {len(heads)}."
    return f"CNN residual trunk, {channels} channels, {blocks} blocks, heads: {len(heads)}."


def model_summary_from_trial(family: dict[str, Any], static: dict[str, Any]) -> str:
    arch = str(family.get("architecture") or "")
    if arch in {"graph", "graph_hybrid_0"}:
        return f"graph_hybrid_0 {static.get('graph_token_budget', '?')} tokens x {static.get('graph_layers', '?')} layers"
    return f"{arch or 'model'} {family.get('channels', '?')}x{family.get('blocks', '?')}"


def worker_summary(selfplay: dict[str, Any]) -> str:
    if selfplay.get("workers_alive") is None and selfplay.get("workers_total") is None:
        return ""
    return f"{selfplay.get('workers_alive') or 0}/{selfplay.get('workers_total') or 0}"


def per_second(positions_per_min: Any) -> float | None:
    try:
        return float(positions_per_min) / 60.0
    except (TypeError, ValueError):
        return None


def event_positions_per_second(event: dict[str, Any]) -> float | None:
    return per_second(event.get("positions_per_min")) or per_second(event.get("selected_positions_per_min")) or per_second((event.get("selfplay") or {}).get("positions_per_min")) or per_second((event.get("throughput") or {}).get("selfplay_positions_per_min"))


def event_blurb(event: str) -> str:
    return {"runtime_sweep_start": "Runtime sweep is testing worker/batch settings", "runtime_sweep_result": "Runtime sweep recorded a probe result", "runtime_sweep_selected": "Runtime sweep selected the fastest stable setting", "trial_epoch_complete": "Epoch finished; metrics and checkpoint were written", "trial_evaluated": "Evaluation finished; scheduler score updated", "trial_pruned": "Trial was pruned by a hard gate or scheduler decision", "pbt_generation_start": "PBT generation started", "stage_start": "Autotune stage started"}.get(event, event.replace("_", " "))
