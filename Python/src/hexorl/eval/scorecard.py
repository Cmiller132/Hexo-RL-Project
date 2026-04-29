"""Phase 3 scorecard formulas and hard gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
}


@dataclass(frozen=True)
class ScorecardResult:
    mode: str
    score: float
    base_score: float
    hard_pass: bool
    hard_failures: tuple[str, ...] = ()
    components: dict[str, float] = field(default_factory=dict)


def compute_phase3_scorecard(
    metrics: dict[str, float],
    *,
    epoch: int,
    classical_threshold_epoch: int = 12,
    candidate_model: bool = False,
    fallback_prior_baseline: float | None = None,
) -> ScorecardResult:
    """Compute the documented Phase 3 scorecard from z-normalized metrics."""

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
    failures: list[str] = []
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


def final_score_from_league_lcb(rating: Any) -> float:
    """Champion-selection helper: consume LCB, not raw mean."""

    if isinstance(rating, dict):
        return float(rating["lcb"])
    return float(rating.lcb)


def _weighted(metrics: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights[key] * float(metrics.get(key, 0.0)) for key in weights)
