"""Suite/autotune dashboard data services."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hexorl.dashboard.db import DashboardStore
from hexorl.dashboard.services.suite_format import (
    architecture_summary,
    event_blurb,
    event_positions_per_second,
    last_progress,
    model_summary_from_trial,
    per_second,
    tail_lines,
    worker_summary,
)
from hexorl.dashboard.services.suite_io import jsonl_tail, mtime, read_json, tail_jsonl


def suite_trial_dirs(run_root: Path) -> list[Path]:
    trials = run_root / "trials"
    if not trials.exists():
        return []
    return sorted(path for path in trials.iterdir() if (path / "dashboard.sqlite3").exists())


def suite_store_for_run(run_root: Path | None, run_id: str | None) -> DashboardStore | None:
    if run_root is None or not run_id:
        return None
    direct = run_root / "trials" / run_id / "dashboard.sqlite3"
    if direct.exists():
        return DashboardStore(direct)
    for trial_dir in suite_trial_dirs(run_root):
        db = trial_dir / "dashboard.sqlite3"
        try:
            rows = DashboardStore(db).rows("SELECT run_id FROM runs WHERE run_id=? LIMIT 1", (run_id,))
        except Exception:
            continue
        if rows:
            return DashboardStore(db)
    return None


def suite_runs(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trial_state = {trial["trial_id"]: trial for trial in suite_state(run_root).get("trials", [])}
    for trial_dir in suite_trial_dirs(run_root):
        db = trial_dir / "dashboard.sqlite3"
        try:
            run_rows = DashboardStore(db).rows("SELECT * FROM runs ORDER BY updated_at DESC")
        except Exception:
            continue
        for row in run_rows:
            trial_id = str(row.get("run_id") or trial_dir.name)
            state = trial_state.get(trial_id, {})
            row.update({"trial_id": trial_id, "source_db": str(db), "name": trial_id})
            row["payload_json"] = {
                **dict(row.get("payload_json") or {}),
                "family": (state.get("family") or {}).get("name"),
                "pruned": state.get("pruned"),
                "last_score": state.get("last_score"),
            }
            rows.append(row)
    return sorted(rows, key=lambda row: float(row.get("updated_at") or 0.0), reverse=True)


def suite_games(run_root: Path, *, run_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    trial_dirs = [run_root / "trials" / run_id] if run_id else suite_trial_dirs(run_root)
    rows: list[dict[str, Any]] = []
    for trial_dir in trial_dirs:
        db = trial_dir / "dashboard.sqlite3"
        if not db.exists():
            continue
        try:
            store = DashboardStore(db)
            query = "SELECT * FROM games WHERE run_id=? ORDER BY created_at DESC LIMIT ?" if run_id else "SELECT * FROM games ORDER BY created_at DESC LIMIT ?"
            params: tuple[Any, ...] = (run_id, limit) if run_id else (limit,)
            next_rows = store.rows(query, params)
        except Exception:
            continue
        for row in next_rows:
            row.update({"trial_id": trial_dir.name, "source_db": str(db)})
            rows.append(row)
    return sorted(rows, key=lambda row: float(row.get("created_at") or 0.0), reverse=True)[:limit]


def suite_checkpoints(run_root: Path, *, run_id: str | None = None) -> list[dict[str, Any]]:
    trial_dirs = [run_root / "trials" / run_id] if run_id else suite_trial_dirs(run_root)
    rows: list[dict[str, Any]] = []
    score_by_trial = suite_score_by_trial(run_root)
    for trial_dir in trial_dirs:
        db = trial_dir / "dashboard.sqlite3"
        if not db.exists():
            continue
        try:
            store = DashboardStore(db)
            query = "SELECT * FROM checkpoints WHERE run_id=? ORDER BY indexed_at DESC" if run_id else "SELECT * FROM checkpoints ORDER BY indexed_at DESC"
            next_rows = store.rows(query, (run_id,) if run_id else ())
        except Exception:
            continue
        for row in next_rows:
            trial_id = str(row.get("run_id") or trial_dir.name)
            row.update({"trial_id": trial_id, "source_db": str(db), "score": score_by_trial.get(trial_id)})
            row["scheduler_score"] = row["score"]
            rows.append(row)
    return sorted(rows, key=lambda row: (_sort_float(row.get("score")), int(row.get("epoch") or -1), float(row.get("indexed_at") or 0.0)), reverse=True)


def suite_best_checkpoints(run_root: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    best_by_path: dict[str, dict[str, Any]] = {}
    for row in suite_checkpoints(run_root):
        path = str(row.get("path") or "")
        if not path:
            continue
        current = best_by_path.get(path)
        rank_key = (_sort_float(row.get("score")), int(row.get("epoch") or -1))
        if current is None or rank_key > (_sort_float(current.get("score")), int(current.get("epoch") or -1)):
            best_by_path[path] = row
    ranked = sorted(best_by_path.values(), key=lambda row: (_sort_float(row.get("score")), int(row.get("epoch") or -1), float(row.get("indexed_at") or 0.0)), reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked[:limit]


def suite_trials(run_root: Path) -> list[dict[str, Any]]:
    state_trials = {trial["trial_id"]: trial for trial in suite_state(run_root).get("trials", [])}
    score_by_trial = suite_score_by_trial(run_root)
    rows: list[dict[str, Any]] = []
    for trial_dir in suite_trial_dirs(run_root):
        trial_id = trial_dir.name
        state = dict(state_trials.get(trial_id) or {})
        trial_json = read_json(trial_dir / "trial.json")
        latest = read_json(trial_dir / "LATEST.json")
        family = state.get("family") or trial_json.get("family") or {}
        static = state.get("static") or trial_json.get("static") or latest.get("static") or {}
        latest_selfplay = latest.get("selfplay") or {}
        latest_train = latest.get("train") or {}
        score = score_by_trial.get(trial_id, state.get("last_score"))
        if score is None:
            score_rows = jsonl_tail(trial_dir / "scores.jsonl", limit=1)
            score = score_rows[-1].get("scheduler_score") if score_rows else None
        counts = _trial_counts(trial_dir)
        rows.append({
            "trial_id": trial_id,
            "family": family.get("name") or latest.get("family") or "",
            "architecture": family.get("architecture") or "",
            "model_summary": model_summary_from_trial(family, static),
            "stage": latest.get("stage") or state.get("stage") or trial_json.get("stage") or "",
            "epoch": state.get("epoch") or latest.get("epoch") or trial_json.get("epoch") or 0,
            "score": finite_or_none(score),
            "pruned": bool(state.get("pruned") or trial_json.get("pruned") or False),
            "prune_reason": state.get("prune_reason") or trial_json.get("prune_reason") or "",
            "checkpoint_path": state.get("checkpoint_path") or latest.get("checkpoint_path") or "",
            "games": counts["games"],
            "positions": counts["positions"],
            "checkpoints": counts["checkpoints"],
            "metrics": counts["metrics"],
            "selfplay_positions_per_min": latest_selfplay.get("positions_per_min"),
            "positions_per_sec": per_second(latest_selfplay.get("positions_per_min")),
            "workers": worker_summary(latest_selfplay),
            "epoch_elapsed_s": latest.get("epoch_elapsed_s"),
            "loss_total": latest_train.get("loss_total"),
            "policy_top1_acc": latest_train.get("policy_top1_acc"),
            "sparse_policy_top1_acc": latest_train.get("sparse_policy_top1_acc"),
            "pair_policy_top1_acc": latest_train.get("pair_policy_top1_acc"),
            "runtime_sweep": state.get("runtime_sweep") or trial_json.get("runtime_sweep") or {},
            "updated_at": max(mtime(trial_dir / "LATEST.json"), mtime(trial_dir / "dashboard.sqlite3")),
        })
    return sorted(rows, key=lambda row: (not bool(row.get("pruned")), _sort_float(row.get("score")), float(row.get("updated_at") or 0.0)), reverse=True)


def suite_status(run_root: Path) -> dict[str, Any]:
    manifest = read_json(run_root / "manifest.json")
    state = suite_state(run_root)
    events = jsonl_tail(run_root / "events.jsonl", limit=100)
    trials = suite_trials(run_root)
    latest_stage = _latest_suite_stage(state, events, trials)
    best = next((trial for trial in trials if not trial.get("pruned") and trial.get("score") is not None), None)
    activity = _supervisor_activity(run_root, trials, events, manifest)
    return {
        "enabled": True,
        "run_root": str(run_root),
        "latest_stage": latest_stage,
        "trial_count": len(trials),
        "live_trial_count": sum(1 for trial in trials if not trial.get("pruned")),
        "total_games": sum(int(trial.get("games") or 0) for trial in trials),
        "total_positions": sum(int(trial.get("positions") or 0) for trial in trials),
        "best_trial_id": best.get("trial_id") if best else None,
        "best_score": best.get("score") if best else None,
        "manifest": manifest,
        "state_elapsed_s": state.get("elapsed_s"),
        "host": manifest.get("host", {}),
        "args": manifest.get("args", {}),
        "last_event": events[-1] if events else None,
        "current_activity": activity,
        "current_trial_id": activity.get("trial_id"),
        "current_model": activity.get("model"),
        "current_positions_per_sec": activity.get("positions_per_sec"),
        "current_stage": activity.get("stage") or latest_stage,
        "last_event_name": (events[-1] if events else {}).get("event"),
        "last_event_time": (events[-1] if events else {}).get("time"),
    }


def suite_trial_detail(run_root: Path, trial_id: str) -> dict[str, Any]:
    trial_dir = run_root / "trials" / trial_id
    if not trial_dir.exists():
        return {}
    state = next((trial for trial in suite_state(run_root).get("trials", []) if trial.get("trial_id") == trial_id), {})
    trial_json = read_json(trial_dir / "trial.json")
    latest = read_json(trial_dir / "LATEST.json")
    checkpoints = suite_checkpoints(run_root, run_id=trial_id)
    checkpoint_path = state.get("checkpoint_path") or latest.get("checkpoint_path") or (checkpoints[0].get("path") if checkpoints else "")
    checkpoint = _checkpoint_metadata(Path(checkpoint_path)) if checkpoint_path else {}
    cfg = checkpoint.get("cfg") or {}
    model_metadata = checkpoint.get("model_metadata") or cfg.get("model") or {}
    family = state.get("family") or trial_json.get("family") or {}
    static = state.get("static") or trial_json.get("static") or latest.get("static") or {}
    architecture = model_metadata or {
        "architecture": family.get("architecture"),
        "channels": family.get("channels"),
        "blocks": family.get("blocks"),
        "heads": trial_json.get("heads"),
        "graph_token_set": static.get("graph_token_set"),
        "graph_token_budget": static.get("graph_token_budget"),
        "graph_layers": static.get("graph_layers"),
        "candidate_budget": static.get("candidate_budget"),
        "sparse_prior_stage": static.get("sparse_prior_stage"),
    }
    return {
        "trial_id": trial_id,
        "trial_dir": str(trial_dir),
        "trial": trial_json,
        "state": state,
        "latest": latest,
        "checkpoint_metadata": checkpoint,
        "config": cfg,
        "model_metadata": model_metadata,
        "architecture": architecture,
        "architecture_summary": architecture_summary(architecture, family),
        "current_activity": _trial_activity(jsonl_tail(trial_dir / "events.jsonl", limit=1), latest),
    }


def suite_trial_scores(run_root: Path, trial_id: str) -> list[dict[str, Any]]:
    return jsonl_tail(run_root / "trials" / trial_id / "scores.jsonl", limit=10000)


def suite_trial_events(run_root: Path, trial_id: str, *, limit: int = 1000) -> list[dict[str, Any]]:
    return jsonl_tail(run_root / "trials" / trial_id / "events.jsonl", limit=limit)


def suite_trial_loss_curve(run_root: Path, trial_id: str, *, limit: int = 5000) -> list[dict[str, Any]]:
    store = suite_store_for_run(run_root, trial_id)
    if store is None:
        return []
    rows = store.rows("SELECT epoch, global_step, phase, metrics_json, created_at FROM metrics WHERE run_id=? ORDER BY created_at ASC LIMIT ?", (trial_id, max(1, min(limit, 10000))))
    return [{**{k: row[k] for k in ("epoch", "global_step", "phase", "created_at")}, **dict(row.get("metrics_json") or {})} for row in rows]


def suite_trial_runtime_sweep(run_root: Path, trial_id: str) -> dict[str, Any]:
    trial_dir = run_root / "trials" / trial_id
    state = next((trial for trial in suite_state(run_root).get("trials", []) if trial.get("trial_id") == trial_id), {})
    return state.get("runtime_sweep") or read_json(trial_dir / "trial.json").get("runtime_sweep") or read_json(trial_dir / "LATEST.json").get("runtime_sweep") or {}


def suite_manifest(run_root: Path) -> dict[str, Any]:
    return {"run_root": str(run_root), "manifest": read_json(run_root / "manifest.json"), "manifest_path": str(run_root / "manifest.json")}


def suite_family_space(run_root: Path) -> dict[str, Any]:
    from hexorl.tuning.family_spaces import all_family_spaces, valid_recipe_examples

    trials = suite_trials(run_root)
    spawned: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        spawned.setdefault(str(trial.get("family") or ""), []).append({"trial_id": trial.get("trial_id"), "score": trial.get("score"), "stage": trial.get("stage"), "pruned": trial.get("pruned")})
    return {
        "families": [space.to_manifest() for space in all_family_spaces().values()],
        "recipes": [recipe.to_manifest() for recipe in valid_recipe_examples().values()],
        "spawned_trials": spawned,
    }


def suite_scheduler(run_root: Path) -> dict[str, Any]:
    manifest = read_json(run_root / "manifest.json")
    state = suite_state(run_root)
    scheduler = dict(manifest.get("scheduler") or state.get("scheduler") or {})
    events = [row for row in jsonl_tail(run_root / "events.jsonl", limit=1000) if "scheduler" in str(row.get("event", "")) or row.get("decision")]
    return {
        "current_stage": str(state.get("stage") or state.get("current_stage") or ""),
        "planned_stages": list(scheduler.get("stages") or manifest.get("stages") or []),
        "scheduler": scheduler,
        "budget": dict(manifest.get("budget") or scheduler.get("budget") or {}),
        "decisions": events,
        "state": state,
    }


def suite_runtime_sweep(run_root: Path) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for trial in suite_trials(run_root):
        payload = suite_trial_runtime_sweep(run_root, str(trial["trial_id"]))
        rows = payload.get("probes") or payload.get("results") or payload.get("history") or []
        for row in rows if isinstance(rows, list) else []:
            probes.append({"trial_id": trial["trial_id"], "family": trial.get("family"), "architecture": trial.get("architecture"), **dict(row)})
        choice = payload.get("selected") or payload.get("choice")
        if isinstance(choice, dict):
            selected.append({"trial_id": trial["trial_id"], "family": trial.get("family"), **choice})
    return {"probes": probes, "selected": selected}


def suite_state(run_root: Path) -> dict[str, Any]:
    return read_json(run_root / "state.json")


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number in (float("inf"), float("-inf")) or number != number else number


def _trial_counts(trial_dir: Path) -> dict[str, int]:
    counts = {"games": 0, "positions": 0, "checkpoints": 0, "metrics": 0}
    db = trial_dir / "dashboard.sqlite3"
    if not db.exists():
        return counts
    try:
        store = DashboardStore(db)
        for key in counts:
            rows = store.rows(f"SELECT COUNT(*) AS n FROM {key}")
            counts[key] = int(rows[0]["n"] if rows else 0)
    except Exception:
        pass
    return counts


def suite_score_by_trial(run_root: Path) -> dict[str, float | None]:
    scores = {str(trial["trial_id"]): finite_or_none(trial.get("last_score")) for trial in suite_state(run_root).get("trials", []) if trial.get("trial_id")}
    for trial_dir in suite_trial_dirs(run_root):
        if scores.get(trial_dir.name) is not None:
            continue
        for row in reversed(jsonl_tail(trial_dir / "scores.jsonl", limit=32)):
            score = finite_or_none(row.get("scheduler_score")) or finite_or_none(row.get("score"))
            if score is not None:
                scores[trial_dir.name] = score
                break
    return scores


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "error": "checkpoint_not_found"}
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    cfg = checkpoint.get("cfg_json") or {}
    state = checkpoint.get("model_state_dict") or {}
    return {"path": str(path), "epoch": checkpoint.get("epoch"), "global_step": checkpoint.get("global_step"), "cfg": cfg, "model_metadata": checkpoint.get("model_metadata") or cfg.get("model") or {}, "action_contract_metadata": checkpoint.get("action_contract_metadata", {}), "model_parameter_tensors": len(state) if hasattr(state, "__len__") else None}


def _latest_suite_stage(state: dict[str, Any], events: list[dict[str, Any]], trials: list[dict[str, Any]]) -> str:
    for key in ("stage", "current_stage", "latest_stage"):
        if state.get(key):
            return str(state[key])
    return str(next((row["stage"] for row in reversed(events) if row.get("stage")), "") or next((row["stage"] for row in trials if row.get("stage")), ""))


def _supervisor_activity(run_root: Path, trials: list[dict[str, Any]], events: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    trial_by_id = {str(trial.get("trial_id")): trial for trial in trials if trial.get("trial_id")}
    latest_trial_id = next((str(event.get("trial_id")) for event in reversed(events) if event.get("trial_id")), "")
    trial = trial_by_id.get(latest_trial_id, {})
    progress = last_progress(tail_lines(run_root / "supervisor.log", limit=120))
    latest_event = events[-1] if events else {}
    max_game_moves = int((manifest.get("args") or {}).get("max_game_moves") or 0)
    positions_per_sec = (float(progress["games_per_min"]) * max_game_moves / 60.0) if progress and max_game_moves > 0 else (event_positions_per_second(latest_event) or trial.get("positions_per_sec"))
    return {"trial_id": latest_trial_id or None, "model": trial.get("family") or latest_event.get("family") or None, "architecture": trial.get("architecture") or None, "stage": latest_event.get("stage") or trial.get("stage") or "", "action": f"Self-play running, {progress['progress_pct']:.1f}% of current epoch" if progress else event_blurb(str(latest_event.get("event") or "Waiting for supervisor events")), "positions_per_sec": positions_per_sec, "progress": progress, "last_log_line": (tail_lines(run_root / "supervisor.log", limit=1) or [""])[-1], "log_tail": tail_lines(run_root / "supervisor.log", limit=20), "last_event": latest_event}


def _trial_activity(events: list[dict[str, Any]], latest: dict[str, Any]) -> dict[str, Any]:
    event = events[-1] if events else {}
    latest_train = latest.get("train") or {}
    return {"event": event.get("event_type") or event.get("event") or "", "phase": event.get("phase") or latest.get("stage") or "", "epoch": latest.get("epoch") or latest_train.get("epoch"), "positions_per_sec": per_second((latest.get("selfplay") or {}).get("positions_per_min")), "loss_total": latest_train.get("loss_total"), "updated_at": latest.get("epoch_elapsed_s")}


def _sort_float(value: Any) -> float:
    return finite_or_none(value) if finite_or_none(value) is not None else float("-inf")
