"""Phase 3 scorecard formulas, hard gates, and scout scorecard artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


HEALTH_WARMUP_WEIGHTS = {
    "policy_target_quality": 0.45,
    "value_calibration_score": 0.35,
    "outside_window_robustness": 0.20,
}

PRE_CLASSICAL_WEIGHTS = {
    "tactical_suite_score": 0.30,
    "outside_window_robustness": 0.25,
    "policy_target_quality": 0.25,
    "value_calibration_score": 0.20,
}

STRENGTH_WEIGHTS = {
    "league_lcb": 0.40,
    "outside_window_robustness": 0.20,
    "tactical_suite_score": 0.15,
    "classical_survival_score": 0.10,
    "value_calibration_score": 0.10,
    "policy_target_quality": 0.05,
}

SCHEDULER_PENALTIES = {
    "epoch_seconds": 0.10,
    "truncation_rate": 0.10,
    "illegal_or_crash_rate": 0.20,
    "fallback_prior_use_on_mcts_topk": 0.10,
    "pair_fallback_prior_use_on_mcts_topk": 0.10,
}

MILESTONE_K_ZERO_SENTINELS = (
    "illegal_move_rate",
    "post_terminal_move_attempts",
    "replay_mismatch_rate",
    "d6_mismatch_rate",
    "legal_mask_mismatch_rate",
    "oracle_threat_mismatch_rate",
    "missing_legal_action_rows",
    "pair_mask_violation_rate",
)

MILESTONE_K_REQUIRED_STATUS = {
    "target_leakage_check_status": "pass",
}

CLASSICAL_CONFIDENCE_Z = {
    "normal_90": 1.6448536269514722,
    "normal_95": 1.959963984540054,
    "normal_99": 2.5758293035489004,
}


@dataclass(frozen=True)
class ScorecardResult:
    mode: str
    score: float
    base_score: float
    hard_pass: bool
    hard_failures: tuple[str, ...] = ()
    components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ClassicalGameEvidence:
    """Persisted fixed-classical game row used to recompute survival LCB."""

    outcome: str
    moves: int
    max_moves: int
    illegal_or_crash_penalty: float
    confidence_method: str
    opponent_id: str
    seed: int
    candidate_id: str = ""
    checkpoint_id: str = ""
    evidence_path: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ClassicalGameEvidence":
        return cls(
            outcome=str(payload["outcome"]),
            moves=int(payload["moves"]),
            max_moves=int(payload["max_moves"]),
            illegal_or_crash_penalty=float(payload.get("illegal_or_crash_penalty", 0.0) or 0.0),
            confidence_method=str(payload.get("confidence_method", "normal_95")),
            opponent_id=str(payload["opponent_id"]),
            seed=int(payload["seed"]),
            candidate_id=str(payload.get("candidate_id", "")),
            checkpoint_id=str(payload.get("checkpoint_id", "")),
            evidence_path=str(payload.get("evidence_path", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClassicalSurvivalLCB:
    score: float
    mean: float
    lower_confidence_bound: float
    games: int
    confidence_method: str
    z_value: float
    opponent_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    illegal_or_crash_penalty_total: float
    per_game_scores: tuple[float, ...]

    def to_components(self) -> dict[str, float]:
        return {
            "classical_survival_lcb": self.score,
            "classical_survival_mean": self.mean,
            "classical_survival_games": float(self.games),
            "classical_illegal_or_crash_penalty_total": self.illegal_or_crash_penalty_total,
        }


@dataclass(frozen=True)
class ScorecardRecord:
    """Append-only scorecard row for one evaluated checkpoint."""

    candidate_id: str
    scalar_score: float
    component_metrics: dict[str, float]
    hard_gates: dict[str, Any]
    study_id: str = ""
    trial_id: str | int = ""
    config_hash: str = ""
    checkpoint_lineage: dict[str, Any] = field(default_factory=dict)
    evidence_paths: tuple[str, ...] = ()
    scalar_name: str = "classical_survival_lcb"
    epoch: int = 0
    completed_epochs: int = 0
    status: str = "healthy"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def hard_pass(self) -> bool:
        if self.status in {"quarantined", "failed", "hard_failed"}:
            return False
        failures = self.hard_gates.get("failures", ())
        if failures:
            return False
        hard_pass = self.hard_gates.get("hard_pass")
        if hard_pass is not None:
            return bool(hard_pass)
        for key, value in self.hard_gates.items():
            if key.endswith("_pass") and value is False:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_paths"] = list(self.evidence_paths)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ScorecardRecord":
        data = dict(payload)
        scalar_name = str(data.get("scalar_name", "classical_survival_lcb"))
        if "scalar_score" not in data:
            data["scalar_score"] = data.get(scalar_name, 0.0)
        data["scalar_score"] = float(data["scalar_score"])
        data["component_metrics"] = {
            str(key): float(value) for key, value in dict(data.get("component_metrics", {})).items()
        }
        data["hard_gates"] = dict(data.get("hard_gates", {}))
        data["checkpoint_lineage"] = dict(data.get("checkpoint_lineage", {}))
        data["evidence_paths"] = tuple(str(path) for path in data.get("evidence_paths", ()))
        data["epoch"] = int(data.get("epoch", 0) or 0)
        data["completed_epochs"] = int(data.get("completed_epochs", data["epoch"]) or 0)
        data["schema_version"] = int(data.get("schema_version", 1) or 1)
        data["created_at"] = str(data.get("created_at") or "")
        metadata = dict(data.get("metadata", {}))
        known_fields = {item.name for item in fields(cls)}
        extra_fields = {key: value for key, value in data.items() if key not in known_fields}
        if extra_fields:
            metadata = {**metadata, "extra_fields": {**dict(metadata.get("extra_fields", {})), **extra_fields}}
        data["metadata"] = metadata
        return cls(**{key: value for key, value in data.items() if key in known_fields})


def compute_phase3_scorecard(
    metrics: dict[str, float],
    *,
    epoch: int,
    classical_threshold_epoch: int = 12,
    candidate_model: bool = False,
    fallback_prior_baseline: float | None = None,
) -> ScorecardResult:
    """Compute the documented Phase 3 scorecard from z-normalized metrics."""

    metrics = _with_prior_fallback_aliases(metrics)
    hard_failures = candidate_hard_gate_failures(
        metrics,
        candidate_model=candidate_model,
        fallback_prior_baseline=fallback_prior_baseline,
    )
    if epoch < 8:
        mode = "health_warmup"
        base_score = _weighted(metrics, HEALTH_WARMUP_WEIGHTS)
        score = base_score
        components = dict(HEALTH_WARMUP_WEIGHTS)
    elif epoch < classical_threshold_epoch:
        mode = "pre_classical_strategy"
        base_score = _weighted(metrics, PRE_CLASSICAL_WEIGHTS)
        score = base_score
        components = dict(PRE_CLASSICAL_WEIGHTS)
    else:
        mode = "scheduler_strength"
        strength_score = _weighted(metrics, STRENGTH_WEIGHTS)
        penalty = _weighted(metrics, SCHEDULER_PENALTIES)
        base_score = strength_score
        score = strength_score - penalty
        components = {**STRENGTH_WEIGHTS, **{key: -value for key, value in SCHEDULER_PENALTIES.items()}}
    return ScorecardResult(
        mode=mode,
        score=score,
        base_score=base_score,
        hard_pass=not hard_failures,
        hard_failures=tuple(hard_failures),
        components=components,
    )


def should_prune_phase3_trial(result: ScorecardResult, *, epoch: int) -> bool:
    """Return whether a scheduler may prune from this scorecard alone.

    Before epoch 8 the spec treats trials as health checks only, so ordinary
    low scores cannot prune a trial. Hard sentinel failures still can.
    """

    if epoch < 8:
        return not result.hard_pass
    return not result.hard_pass or result.score < 0.0


def candidate_hard_gate_failures(
    metrics: dict[str, float],
    *,
    candidate_model: bool,
    fallback_prior_baseline: float | None = None,
) -> list[str]:
    failures: list[str] = milestone_k_hard_gate_failures(metrics)
    if metrics.get("illegal_or_crash_rate", 0.0) > 0.0:
        failures.append("illegal_or_crash_rate")
    if metrics.get("critical_overflow_count", 0.0) != 0.0:
        failures.append("critical_overflow_count")
    if not candidate_model:
        return failures
    thresholds = {
        "candidate_discovery_winning_move": 0.995,
        "candidate_discovery_forced_block": 0.995,
        "candidate_discovery_two_placement_cover": 0.990,
    }
    for key, threshold in thresholds.items():
        if metrics.get(key, 0.0) < threshold:
            failures.append(key)
    if metrics.get("missing_target_policy_mass", 0.0) > 0.010:
        failures.append("missing_target_policy_mass")
    if fallback_prior_baseline is not None:
        current = metrics.get("fallback_prior_use_on_mcts_topk", 0.0)
        if current >= fallback_prior_baseline:
            failures.append("fallback_prior_use_on_mcts_topk")
    return failures


def milestone_k_hard_gate_failures(metrics: dict[str, float | str]) -> list[str]:
    """Return non-negotiable bug-sentinel failures for checkpoint promotion."""
    failures: list[str] = []
    for key in MILESTONE_K_ZERO_SENTINELS:
        if float(metrics.get(key, 0.0) or 0.0) != 0.0:
            failures.append(key)
    for key, expected in MILESTONE_K_REQUIRED_STATUS.items():
        actual = metrics.get(key, expected)
        if str(actual).lower() != expected:
            failures.append(key)
    return failures


def final_score_from_league_lcb(rating: Any) -> float:
    """Champion-selection helper: consume LCB, not raw mean."""

    if isinstance(rating, dict):
        return float(rating["lcb"])
    return float(rating.lcb)


def classical_survival_lcb(
    evidence: Iterable[ClassicalGameEvidence | Mapping[str, Any]],
    *,
    confidence_method: str | None = None,
) -> ClassicalSurvivalLCB:
    """Recompute the fixed-classical survival scalar from persisted game rows."""

    rows = [
        row if isinstance(row, ClassicalGameEvidence) else ClassicalGameEvidence.from_mapping(row)
        for row in evidence
    ]
    if not rows:
        raise ValueError("classical_survival_lcb requires at least one game evidence row")
    methods = {row.confidence_method for row in rows}
    method = confidence_method or (methods.pop() if len(methods) == 1 else "")
    if not method:
        raise ValueError("mixed confidence methods require an explicit confidence_method")
    if method not in CLASSICAL_CONFIDENCE_Z:
        raise ValueError(f"unsupported classical survival confidence method {method!r}")
    z_value = CLASSICAL_CONFIDENCE_Z[method]
    scores = tuple(_classical_game_survival_score(row) for row in rows)
    mean = sum(scores) / len(scores)
    if len(scores) > 1:
        variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
        stderr = math.sqrt(variance) / math.sqrt(len(scores))
    else:
        stderr = 0.0
    lcb = mean - z_value * stderr
    return ClassicalSurvivalLCB(
        score=lcb,
        mean=mean,
        lower_confidence_bound=lcb,
        games=len(scores),
        confidence_method=method,
        z_value=z_value,
        opponent_ids=tuple(sorted({row.opponent_id for row in rows})),
        seeds=tuple(row.seed for row in rows),
        illegal_or_crash_penalty_total=sum(row.illegal_or_crash_penalty for row in rows),
        per_game_scores=scores,
    )


def load_classical_game_evidence(path: Path | str) -> list[ClassicalGameEvidence]:
    return [
        ClassicalGameEvidence.from_mapping(payload)
        for payload in _read_jsonl(path)
    ]


def append_scorecard(path: Path | str, record: ScorecardRecord | Mapping[str, Any]) -> ScorecardRecord:
    """Append one scorecard JSON line without rewriting existing rows."""

    scorecard = record if isinstance(record, ScorecardRecord) else ScorecardRecord.from_mapping(record)
    _append_jsonl(Path(path), scorecard.to_dict())
    return scorecard


def read_scorecards(path: Path | str) -> list[ScorecardRecord]:
    return [ScorecardRecord.from_mapping(payload) for payload in _read_jsonl(path)]


def build_classical_scorecard_record(
    *,
    candidate_id: str,
    evidence_path: Path | str,
    component_metrics: Mapping[str, float] | None = None,
    hard_gates: Mapping[str, Any] | None = None,
    study_id: str = "",
    trial_id: str | int = "",
    config_hash: str = "",
    checkpoint_lineage: Mapping[str, Any] | None = None,
    epoch: int = 0,
    completed_epochs: int | None = None,
    status: str = "healthy",
    metadata: Mapping[str, Any] | None = None,
) -> ScorecardRecord:
    evidence_file = Path(evidence_path)
    lcb = classical_survival_lcb(load_classical_game_evidence(evidence_file))
    components = dict(component_metrics or {})
    components.update(lcb.to_components())
    return ScorecardRecord(
        candidate_id=candidate_id,
        scalar_name="classical_survival_lcb",
        scalar_score=lcb.score,
        component_metrics=components,
        hard_gates=dict(hard_gates or {"hard_pass": True, "failures": []}),
        study_id=study_id,
        trial_id=trial_id,
        config_hash=config_hash,
        checkpoint_lineage=dict(checkpoint_lineage or {}),
        evidence_paths=(str(evidence_file),),
        epoch=int(epoch),
        completed_epochs=int(completed_epochs if completed_epochs is not None else epoch),
        status=status,
        metadata={**dict(metadata or {}), "classical_survival_lcb": asdict(lcb)},
    )


def _weighted(metrics: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights[key] * float(metrics.get(key, 0.0)) for key in weights)


def _with_prior_fallback_aliases(metrics: dict[str, float]) -> dict[str, float]:
    """Normalize top-k fallback telemetry names used by replay and scorecards."""
    out = dict(metrics)
    if "fallback_prior_use_on_mcts_topk" not in out:
        for key in (
            "fallback_prior_use_on_mcts_top4",
            "fallback_prior_use_on_mcts_top8",
            "fallback_prior_use_on_mcts_top1",
            "fallback_prior_use",
        ):
            if key in out:
                out["fallback_prior_use_on_mcts_topk"] = out[key]
                break
    if "pair_fallback_prior_use_on_mcts_topk" not in out:
        for key in (
            "pair_fallback_prior_use_on_mcts_top4",
            "pair_fallback_prior_use_on_mcts_top8",
            "pair_fallback_prior_use_on_mcts_top1",
            "pair_fallback_prior_use",
        ):
            if key in out:
                out["pair_fallback_prior_use_on_mcts_topk"] = out[key]
                break
    return out


def _classical_game_survival_score(row: ClassicalGameEvidence) -> float:
    if row.max_moves <= 0:
        raise ValueError("classical game evidence max_moves must be positive")
    if row.moves < 0:
        raise ValueError("classical game evidence moves must be non-negative")
    survival = min(float(row.moves) / float(row.max_moves), 1.0)
    outcome = row.outcome.lower()
    if outcome in {"win", "model_win", "candidate_win"}:
        raw = 1.15
    elif outcome in {"draw", "max_moves", "max_move", "survived"}:
        raw = 1.0
    elif outcome in {"loss", "classical_win", "model_loss", "candidate_loss"}:
        raw = survival
    else:
        raise ValueError(f"unsupported classical game outcome {row.outcome!r}")
    return raw - max(float(row.illegal_or_crash_penalty), 0.0)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    file_path = Path(path)
    if not file_path.exists():
        return rows
    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{file_path}:{line_number} is not a JSON object")
        rows.append(payload)
    return rows
