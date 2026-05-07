"""Final champion selection from persisted Hexo scorecard evidence."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from hexorl.eval.scorecard import ScorecardRecord, read_scorecards


DEFAULT_CHAMPION_SCALAR = "classical_survival_lcb"


@dataclass(frozen=True)
class ChampionCandidateRank:
    candidate_id: str
    rank: int
    scalar_name: str
    scalar_score: float
    completed_epochs: int
    checkpoint_lineage: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=dict)
    scorecard_paths: tuple[str, ...] = ()
    evidence_paths: tuple[str, ...] = ()
    optuna_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChampionRejectedCandidate:
    candidate_id: str
    reason: str
    scalar_name: str = ""
    scalar_score: float | None = None
    completed_epochs: int = 0
    scorecard_paths: tuple[str, ...] = ()
    gate_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChampionSelectionReport:
    created_at: str
    scalar_name: str
    selected: ChampionCandidateRank | None
    runner_up: ChampionCandidateRank | None
    ranked: tuple[ChampionCandidateRank, ...]
    rejected: tuple[ChampionRejectedCandidate, ...]
    runner_up_comparison: dict[str, Any]
    reproduction_command: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected"] = asdict(self.selected) if self.selected is not None else None
        payload["runner_up"] = asdict(self.runner_up) if self.runner_up is not None else None
        payload["ranked"] = [asdict(item) for item in self.ranked]
        payload["rejected"] = [asdict(item) for item in self.rejected]
        return payload


def select_champion_from_scorecards(
    scorecards: Iterable[ScorecardRecord | Mapping[str, Any]],
    *,
    reproduction_command: str,
    min_completed_epochs: int = 12,
    scalar_name: str = DEFAULT_CHAMPION_SCALAR,
) -> ChampionSelectionReport:
    """Select the final champion from Hexo scorecards and hard gates.

    Optuna values are carried into the report for traceability only. Ranking is
    intentionally based on persisted Hexo scorecard scalar evidence after hard
    gates and epoch-floor checks.
    """

    latest_by_candidate: dict[str, ScorecardRecord] = {}
    scorecard_paths_by_candidate: dict[str, set[str]] = {}
    for item in scorecards:
        record = item if isinstance(item, ScorecardRecord) else ScorecardRecord.from_mapping(item)
        candidate_paths = scorecard_paths_by_candidate.setdefault(record.candidate_id, set())
        scorecard_path = str(record.metadata.get("scorecard_path", ""))
        if scorecard_path:
            candidate_paths.add(scorecard_path)
        current = latest_by_candidate.get(record.candidate_id)
        if current is None or _record_sort_key(record) > _record_sort_key(current):
            latest_by_candidate[record.candidate_id] = record

    eligible: list[ScorecardRecord] = []
    rejected: list[ChampionRejectedCandidate] = []
    for record in latest_by_candidate.values():
        reason = _champion_rejection_reason(
            record,
            min_completed_epochs=min_completed_epochs,
            scalar_name=scalar_name,
        )
        if reason:
            rejected.append(_rejected(record, reason, scorecard_paths_by_candidate))
        else:
            eligible.append(record)

    eligible.sort(
        key=lambda record: (
            float(record.scalar_score),
            int(max(record.completed_epochs, record.epoch)),
            record.candidate_id,
        ),
        reverse=True,
    )
    ranked = tuple(
        _rank(idx, record, scorecard_paths_by_candidate)
        for idx, record in enumerate(eligible, start=1)
    )
    selected = ranked[0] if ranked else None
    runner_up = ranked[1] if len(ranked) > 1 else None
    return ChampionSelectionReport(
        created_at=datetime.now(timezone.utc).isoformat(),
        scalar_name=scalar_name,
        selected=selected,
        runner_up=runner_up,
        ranked=ranked,
        rejected=tuple(sorted(rejected, key=lambda item: item.candidate_id)),
        runner_up_comparison=_runner_up_comparison(selected, runner_up),
        reproduction_command=str(reproduction_command),
        metadata={
            "selection_semantics": "hexo_scorecard_scalar_after_hard_gates",
            "optuna_value_role": "trace_only_not_ranking_authority",
            "min_completed_epochs": int(min_completed_epochs),
            "eligible_count": len(ranked),
            "rejected_count": len(rejected),
        },
    )


def build_champion_selection_report_from_scorecard_files(
    scorecard_paths: Iterable[Path | str],
    *,
    reproduction_command: str,
    min_completed_epochs: int = 12,
    scalar_name: str = DEFAULT_CHAMPION_SCALAR,
) -> ChampionSelectionReport:
    records: list[ScorecardRecord] = []
    for path in scorecard_paths:
        path_obj = Path(path)
        for record in read_scorecards(path_obj):
            payload = record.to_dict()
            metadata = dict(payload.get("metadata", {}))
            metadata.setdefault("scorecard_path", str(path_obj))
            payload["metadata"] = metadata
            records.append(ScorecardRecord.from_mapping(payload))
    return select_champion_from_scorecards(
        records,
        reproduction_command=reproduction_command,
        min_completed_epochs=min_completed_epochs,
        scalar_name=scalar_name,
    )


def write_champion_selection_report(path: Path | str, report: ChampionSelectionReport) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _champion_rejection_reason(
    record: ScorecardRecord,
    *,
    min_completed_epochs: int,
    scalar_name: str,
) -> str:
    if record.status in {"quarantined", "ready_for_retest", "retesting"}:
        return "runtime_quarantine"
    if record.status in {"failed", "hard_failed"}:
        return "hard_failed"
    if not record.hard_gates:
        return "missing_gates"
    if not record.hard_pass:
        return "hard_failed"
    if max(record.completed_epochs, record.epoch) < int(min_completed_epochs):
        return "below_epoch_floor"
    if record.scalar_name != scalar_name:
        return "wrong_scalar"
    if not math.isfinite(float(record.scalar_score)):
        return "invalid_scalar"
    if float(record.component_metrics.get("illegal_or_crash_rate", 0.0) or 0.0) > 0.0:
        return "hard_failed"
    if not record.evidence_paths:
        return "missing_evidence"
    if not _has_fixed_classical_evidence(record):
        return "missing_classical_evidence"
    if not record.checkpoint_lineage or not record.checkpoint_lineage.get("checkpoint_path"):
        return "missing_checkpoint_lineage"
    return ""


def _rank(
    rank: int,
    record: ScorecardRecord,
    scorecard_paths_by_candidate: Mapping[str, set[str]],
) -> ChampionCandidateRank:
    optuna_value = record.metadata.get("optuna_value")
    if optuna_value is not None:
        optuna_value = float(optuna_value)
    return ChampionCandidateRank(
        candidate_id=record.candidate_id,
        rank=rank,
        scalar_name=record.scalar_name,
        scalar_score=float(record.scalar_score),
        completed_epochs=max(record.completed_epochs, record.epoch),
        checkpoint_lineage=dict(record.checkpoint_lineage),
        gates=dict(record.hard_gates),
        scorecard_paths=tuple(sorted(scorecard_paths_by_candidate.get(record.candidate_id, set()))),
        evidence_paths=tuple(record.evidence_paths),
        optuna_value=optuna_value,
        metadata={
            "study_id": record.study_id,
            "trial_id": record.trial_id,
            "config_hash": record.config_hash,
            "checkpoint": record.checkpoint_lineage.get("checkpoint_path", ""),
        },
    )


def _rejected(
    record: ScorecardRecord,
    reason: str,
    scorecard_paths_by_candidate: Mapping[str, set[str]],
) -> ChampionRejectedCandidate:
    failures = record.hard_gates.get("failures", ())
    if isinstance(failures, str):
        failures = (failures,)
    return ChampionRejectedCandidate(
        candidate_id=record.candidate_id,
        reason=reason,
        scalar_name=record.scalar_name,
        scalar_score=float(record.scalar_score),
        completed_epochs=max(record.completed_epochs, record.epoch),
        scorecard_paths=tuple(sorted(scorecard_paths_by_candidate.get(record.candidate_id, set()))),
        gate_failures=tuple(str(item) for item in failures),
    )


def _runner_up_comparison(
    selected: ChampionCandidateRank | None,
    runner_up: ChampionCandidateRank | None,
) -> dict[str, Any]:
    if selected is None:
        return {"status": "no_eligible_champion"}
    if runner_up is None:
        return {"status": "single_eligible_champion", "selected": selected.candidate_id}
    return {
        "status": "compared",
        "selected": selected.candidate_id,
        "runner_up": runner_up.candidate_id,
        "scalar_delta": selected.scalar_score - runner_up.scalar_score,
        "completed_epoch_delta": selected.completed_epochs - runner_up.completed_epochs,
    }


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
