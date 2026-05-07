"""Phase 2 scout review and promotion report helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from hexorl.eval.scorecard import ScorecardRecord, read_scorecards


@dataclass(frozen=True)
class Phase2CandidateRank:
    candidate_id: str
    rank: int
    classical_survival_lcb: float
    completed_epochs: int
    scorecard_path: str
    study_id: str = ""
    trial_id: str | int = ""
    config_hash: str = ""
    checkpoint_lineage: dict[str, Any] = field(default_factory=dict)
    supporting_metrics: dict[str, float] = field(default_factory=dict)
    evidence_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class Phase2ExcludedCandidate:
    candidate_id: str
    reason: str
    completed_epochs: int
    scorecard_path: str = ""
    status: str = ""
    hard_gate_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class Phase2PromotionReport:
    created_at: str
    min_epoch_floor: int
    scalar_name: str
    ranked: tuple[Phase2CandidateRank, ...]
    excluded: tuple[Phase2ExcludedCandidate, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ranked"] = [asdict(item) for item in self.ranked]
        payload["excluded"] = [asdict(item) for item in self.excluded]
        return payload


def rank_phase2_survivors(
    scorecards: Iterable[ScorecardRecord | Mapping[str, Any]],
    *,
    min_epoch_floor: int = 12,
    scalar_name: str = "classical_survival_lcb",
) -> Phase2PromotionReport:
    """Rank epoch-floor survivors and exclude quarantine, hard-fail, and non-floor rows."""

    latest_by_candidate: dict[str, ScorecardRecord] = {}
    for item in scorecards:
        record = item if isinstance(item, ScorecardRecord) else ScorecardRecord.from_mapping(item)
        current = latest_by_candidate.get(record.candidate_id)
        if current is None or _record_sort_key(record) > _record_sort_key(current):
            latest_by_candidate[record.candidate_id] = record

    eligible: list[ScorecardRecord] = []
    excluded: list[Phase2ExcludedCandidate] = []
    for record in latest_by_candidate.values():
        reason = _phase2_exclusion_reason(record, min_epoch_floor=min_epoch_floor, scalar_name=scalar_name)
        if reason:
            excluded.append(_excluded(record, reason))
        else:
            eligible.append(record)

    eligible.sort(key=lambda record: (float(record.scalar_score), record.candidate_id), reverse=True)
    ranked = tuple(
        Phase2CandidateRank(
            candidate_id=record.candidate_id,
            rank=idx,
            classical_survival_lcb=float(record.scalar_score),
            completed_epochs=record.completed_epochs,
            scorecard_path=str(record.metadata.get("scorecard_path", "")),
            study_id=record.study_id,
            trial_id=record.trial_id,
            config_hash=record.config_hash,
            checkpoint_lineage=dict(record.checkpoint_lineage),
            supporting_metrics={
                key: float(record.component_metrics[key])
                for key in (
                    "classical_win_rate",
                    "classical_draw_rate",
                    "generated_positions_per_second",
                    "strength_per_generated_position",
                    "strength_per_wall_clock_second",
                    "pair_overhead",
                    "pair_head_quality",
                    "tactical_suite_score",
                    "outside_window_robustness",
                    "illegal_or_crash_rate",
                )
                if key in record.component_metrics
            },
            evidence_paths=tuple(record.evidence_paths),
        )
        for idx, record in enumerate(eligible, start=1)
    )
    return Phase2PromotionReport(
        created_at=datetime.now(timezone.utc).isoformat(),
        min_epoch_floor=min_epoch_floor,
        scalar_name=scalar_name,
        ranked=ranked,
        excluded=tuple(sorted(excluded, key=lambda item: item.candidate_id)),
        metadata={
            "eligible_count": len(ranked),
            "excluded_count": len(excluded),
            "promotion_semantics": "epoch_floor_classical_survival_lcb_desc",
        },
    )


def build_phase2_promotion_report_from_scorecard_files(
    scorecard_paths: Iterable[Path | str],
    *,
    min_epoch_floor: int = 12,
    scalar_name: str = "classical_survival_lcb",
) -> Phase2PromotionReport:
    records: list[ScorecardRecord] = []
    for path in scorecard_paths:
        path_obj = Path(path)
        for record in read_scorecards(path_obj):
            metadata = dict(record.metadata)
            metadata.setdefault("scorecard_path", str(path_obj))
            records.append(
                ScorecardRecord.from_mapping({**record.to_dict(), "metadata": metadata})
            )
    return rank_phase2_survivors(
        records,
        min_epoch_floor=min_epoch_floor,
        scalar_name=scalar_name,
    )


def write_phase2_promotion_report(path: Path | str, report: Phase2PromotionReport) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _phase2_exclusion_reason(
    record: ScorecardRecord,
    *,
    min_epoch_floor: int,
    scalar_name: str,
) -> str:
    if record.status in {"quarantined", "ready_for_retest", "retesting"}:
        return "runtime_quarantine"
    if record.status in {"failed", "hard_failed"}:
        return "hard_failed"
    if not record.hard_pass:
        return "hard_failed"
    if max(record.completed_epochs, record.epoch) < min_epoch_floor:
        return "below_epoch_floor"
    if record.scalar_name != scalar_name:
        return "wrong_scalar"
    if "classical_survival_lcb" not in record.component_metrics:
        return "missing_classical_survival_lcb"
    if not record.evidence_paths:
        return "missing_evidence"
    if not _has_fixed_classical_evidence(record):
        return "missing_classical_evidence"
    if float(record.component_metrics.get("illegal_or_crash_rate", 0.0) or 0.0) > 0.0:
        return "hard_failed"
    return ""


def _excluded(record: ScorecardRecord, reason: str) -> Phase2ExcludedCandidate:
    failures = record.hard_gates.get("failures", ())
    if isinstance(failures, str):
        failures = (failures,)
    return Phase2ExcludedCandidate(
        candidate_id=record.candidate_id,
        reason=reason,
        completed_epochs=record.completed_epochs,
        scorecard_path=str(record.metadata.get("scorecard_path", "")),
        status=record.status,
        hard_gate_failures=tuple(str(item) for item in failures),
    )


def _record_sort_key(record: ScorecardRecord) -> tuple[int, str]:
    return (max(record.completed_epochs, record.epoch), record.created_at)


def _has_fixed_classical_evidence(record: ScorecardRecord) -> bool:
    games = float(record.component_metrics.get("classical_survival_games", 0.0) or 0.0)
    if games <= 0.0:
        return False
    metadata_lcb = record.metadata.get("classical_survival_lcb")
    if isinstance(metadata_lcb, Mapping):
        metadata_games = float(metadata_lcb.get("games", 0.0) or 0.0)
        if metadata_games > 0.0:
            return True
    return any("fixed_classical" in str(path) for path in record.evidence_paths)
