"""Append-only repairs for Phase 1 scout scorecards from dashboard metrics."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from hexorl.eval.scorecard import ScorecardRecord, append_scorecard, read_scorecards


TRAIN_COMPONENT_KEYS = (
    "loss_total",
    "loss_policy_place",
    "loss_value",
    "pair_policy_weight_mean",
    "batches_per_sec",
    "elapsed_s",
    "graph_peak_cuda_allocated_mb",
    "graph_microbatch_oom_retries",
    "graph_microbatch_nonfinite_retries",
)

BUFFER_COMPONENT_KEYS = (
    "avg_missing_target_policy_mass",
    "avg_target_policy_mass_outside_window",
    "avg_candidate_recall_mcts_top1",
    "avg_candidate_recall_mcts_top4",
    "avg_candidate_recall_mcts_top8",
    "avg_candidate_recall_winning_move",
    "avg_candidate_recall_forced_block",
    "avg_candidate_recall_two_placement_cover",
    "critical_overflow_count",
)

PAIR_BUFFER_COMPONENT_KEYS = (
    "pair_prior_hit_frac",
    "pair_fallback_prior_use",
    "pair_fallback_prior_use_on_mcts_top1",
    "pair_fallback_prior_use_on_mcts_top4",
    "pair_fallback_prior_use_on_mcts_top8",
)


@dataclass(frozen=True)
class CandidateScorecardRepair:
    candidate_id: str
    scorecard_path: str
    appended_rows: int
    skipped_rows: int
    missing_dashboard_epochs: tuple[int, ...]
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["missing_dashboard_epochs"] = list(self.missing_dashboard_epochs)
        return payload


@dataclass(frozen=True)
class ScorecardRepairSummary:
    run_dir: str
    candidates: tuple[CandidateScorecardRepair, ...]
    dry_run: bool = False

    @property
    def appended_rows(self) -> int:
        return sum(row.appended_rows for row in self.candidates)

    @property
    def missing_dashboard_epochs(self) -> dict[str, tuple[int, ...]]:
        return {
            row.candidate_id: row.missing_dashboard_epochs
            for row in self.candidates
            if row.missing_dashboard_epochs
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "dry_run": self.dry_run,
            "appended_rows": self.appended_rows,
            "missing_dashboard_epochs": {
                candidate_id: list(epochs)
                for candidate_id, epochs in self.missing_dashboard_epochs.items()
            },
            "candidates": [row.to_dict() for row in self.candidates],
        }


def repair_phase1_scorecards_from_dashboard(
    run_dir: str | Path,
    *,
    candidate_ids: Iterable[str] | None = None,
    dry_run: bool = False,
) -> ScorecardRepairSummary:
    """Append corrected scorecard rows with train components from dashboard.sqlite3.

    This preserves the append-only scorecard contract. Existing rows are never
    edited or deleted; when a row lacks dashboard-derived train components, a
    corrected copy is appended with repair metadata and the dashboard DB added
    to evidence paths.
    """

    root = Path(run_dir)
    candidates_root = root / "candidates"
    requested = {str(candidate_id) for candidate_id in candidate_ids or ()}
    candidate_dirs = [
        path
        for path in sorted(candidates_root.iterdir())
        if path.is_dir() and (not requested or path.name in requested)
    ]
    repairs = tuple(
        repair_candidate_scorecards_from_dashboard(path, dry_run=dry_run)
        for path in candidate_dirs
        if (path / "scorecards.jsonl").exists()
    )
    return ScorecardRepairSummary(run_dir=str(root), candidates=repairs, dry_run=bool(dry_run))


def repair_candidate_scorecards_from_dashboard(
    candidate_dir: str | Path,
    *,
    dry_run: bool = False,
) -> CandidateScorecardRepair:
    candidate_path = Path(candidate_dir)
    candidate_id = candidate_path.name
    scorecard_path = candidate_path / "scorecards.jsonl"
    dashboard_path = candidate_path / "dashboard.sqlite3"
    records = read_scorecards(scorecard_path)
    if not dashboard_path.exists():
        return CandidateScorecardRepair(
            candidate_id=candidate_id,
            scorecard_path=str(scorecard_path),
            appended_rows=0,
            skipped_rows=len(records),
            missing_dashboard_epochs=tuple(sorted({record.epoch for record in records})),
            dry_run=bool(dry_run),
        )

    train_metrics = _dashboard_train_metrics_by_epoch(dashboard_path)
    pair_strategy_mode = _candidate_pair_strategy_mode(candidate_path)
    latest_by_epoch: dict[int, ScorecardRecord] = {}
    for record in records:
        current = latest_by_epoch.get(record.epoch)
        if current is None or _record_sort_key(record) >= _record_sort_key(current):
            latest_by_epoch[record.epoch] = record

    appended = 0
    skipped = 0
    missing: list[int] = []
    repaired_at = datetime.now(timezone.utc).isoformat()
    for epoch, record in sorted(latest_by_epoch.items()):
        if _has_dashboard_train_components(record):
            skipped += 1
            continue
        metric = train_metrics.get(epoch)
        if metric is None:
            missing.append(epoch)
            continue
        repaired = _repaired_record(
            record,
            metric,
            dashboard_path=dashboard_path,
            pair_strategy_mode=pair_strategy_mode,
            repaired_at=repaired_at,
        )
        if not dry_run:
            append_scorecard(scorecard_path, repaired)
        appended += 1

    return CandidateScorecardRepair(
        candidate_id=candidate_id,
        scorecard_path=str(scorecard_path),
        appended_rows=appended,
        skipped_rows=skipped,
        missing_dashboard_epochs=tuple(missing),
        dry_run=bool(dry_run),
    )


def _dashboard_train_metrics_by_epoch(dashboard_path: Path) -> dict[int, dict[str, Any]]:
    con = sqlite3.connect(dashboard_path)
    try:
        rows = con.execute(
            """
            select metric_id, epoch, global_step, metrics_json, created_at
            from metrics
            where phase = 'train' and epoch is not null
            order by metric_id asc
            """
        ).fetchall()
    finally:
        con.close()

    metrics: dict[int, dict[str, Any]] = {}
    for metric_id, epoch, global_step, metrics_json, created_at in rows:
        payload = json.loads(metrics_json)
        metrics[int(epoch)] = {
            "metric_id": int(metric_id),
            "epoch": int(epoch),
            "global_step": int(global_step or 0),
            "created_at": created_at,
            "payload": payload,
        }
    return metrics


def _repaired_record(
    record: ScorecardRecord,
    metric: Mapping[str, Any],
    *,
    dashboard_path: Path,
    pair_strategy_mode: str,
    repaired_at: str,
) -> dict[str, Any]:
    payload = record.to_dict()
    components = dict(record.component_metrics)
    metric_payload = dict(metric.get("payload", {}))
    train = metric_payload.get("train", {})
    buffer = metric_payload.get("buffer", {})
    if not isinstance(train, Mapping):
        train = {}
    if not isinstance(buffer, Mapping):
        buffer = {}
    loss_total = _finite_float(train, "loss_total", "loss")
    components["train_loss"] = loss_total
    components["loss_total"] = loss_total
    for key in TRAIN_COMPONENT_KEYS:
        if key == "loss_total":
            continue
        components[key] = _finite_float(train, key)
    for key in BUFFER_COMPONENT_KEYS:
        components[key] = _finite_float(buffer, key)
    for key in PAIR_BUFFER_COMPONENT_KEYS:
        components[key] = 0.0 if pair_strategy_mode == "none" else _finite_float(buffer, key)
    payload["component_metrics"] = components
    evidence_paths = list(record.evidence_paths)
    dashboard = str(dashboard_path)
    if dashboard not in evidence_paths:
        evidence_paths.append(dashboard)
    payload["evidence_paths"] = evidence_paths
    payload["created_at"] = repaired_at
    metadata = dict(record.metadata)
    metadata["scorecard_repair"] = {
        "source": "dashboard_train_metric",
        "dashboard_path": dashboard,
        "metric_id": int(metric.get("metric_id", 0) or 0),
        "global_step": int(metric.get("global_step", 0) or 0),
        "original_created_at": record.created_at,
        "repaired_at": repaired_at,
        "reason": "populate_train_components_from_dashboard",
    }
    payload["metadata"] = metadata
    return payload


def _has_dashboard_train_components(record: ScorecardRecord) -> bool:
    repair = record.metadata.get("scorecard_repair")
    if isinstance(repair, Mapping) and repair.get("source") == "dashboard_train_metric":
        return True
    return "loss_total" in record.component_metrics and "loss_policy_place" in record.component_metrics


def _candidate_pair_strategy_mode(candidate_path: Path) -> str:
    manifest_path = candidate_path / "candidate_manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            mode = str(payload.get("pair_strategy_mode", "")).strip()
            if mode:
                return mode
        except (OSError, json.JSONDecodeError):
            pass
    parts = candidate_path.name.rsplit("__", 2)
    if len(parts) >= 2:
        return parts[-2]
    return "none"


def _finite_float(values: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        value = values.get(key)
        if value is None:
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result):
            return result
    return 0.0


def _record_sort_key(record: ScorecardRecord) -> tuple[int, str]:
    return (max(record.completed_epochs, record.epoch), record.created_at)
