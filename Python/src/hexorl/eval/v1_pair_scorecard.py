"""Deterministic V1 pair-action eval and scorecard schema gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


V1_PAIR_SCORECARD_SCHEMA_VERSION = 1
V1_PAIR_CANDIDATE_ID = "global_pair_biaffine_0:sampled_joint_pair_v1"
V1_PAIR_REQUIRED_BASELINES = (
    "global_xattn_0:none",
    "global_graph768_champion:none",
    "fair_sequential_dag_neural_baseline",
)
V1_PAIR_REQUIRED_METRICS = (
    "candidate_generation_latency_p50_ms",
    "candidate_generation_latency_p95_ms",
    "pair_scores_per_second",
    "inference_latency_p50_ms",
    "inference_latency_p95_ms",
    "neural_calls_per_expanded_full_turn_node",
    "queue_backpressure_ratio",
    "gpu_utilization",
    "candidate_recall_best_search_pair",
    "candidate_recall_audit_topk_mass",
    "regret_weighted_candidate_recall",
    "candidate_recall_equivalence_class",
    "tactical_candidate_support_recall",
    "tactical_suite_accuracy",
    "normalized_pair_target_entropy",
)
V1_PAIR_REQUIRED_EQUAL_WALL_CLOCK_FIELDS = (
    "hardware_profile",
    "candidate_budget",
    "batching_protocol",
    "opponent_checkpoints",
    "confidence_interval",
    "arena_stopping_protocol",
    "wall_clock_seconds_per_game",
    "games_per_pairing",
)


@dataclass(frozen=True)
class V1PairScorecardGateResult:
    hard_pass: bool
    failures: tuple[str, ...]
    schema_only: bool
    strength_claimed: bool

    def to_hard_gates(self) -> dict[str, Any]:
        return {
            "hard_pass": self.hard_pass,
            "failures": list(self.failures),
            "schema_only": self.schema_only,
            "strength_claimed": self.strength_claimed,
        }


def validate_v1_pair_scorecard_payload(payload: Mapping[str, Any]) -> V1PairScorecardGateResult:
    """Validate deterministic V1 metric/evidence fields before scorecard use."""

    failures: list[str] = []
    if int(payload.get("schema_version", 0) or 0) != V1_PAIR_SCORECARD_SCHEMA_VERSION:
        failures.append("schema_version")
    if str(payload.get("candidate_id", "")) != V1_PAIR_CANDIDATE_ID:
        failures.append("candidate_id")

    baselines = tuple(str(item) for item in payload.get("side_by_side_baselines", ()))
    for baseline in V1_PAIR_REQUIRED_BASELINES:
        if baseline not in baselines:
            failures.append(f"missing_baseline:{baseline}")

    metrics = dict(payload.get("metrics", {}))
    for metric in V1_PAIR_REQUIRED_METRICS:
        value = metrics.get(metric)
        if value is None:
            failures.append(f"missing_metric:{metric}")
            continue
        if not isinstance(value, (int, float)) or not _finite(float(value)):
            failures.append(f"non_finite_metric:{metric}")

    equal_wall_clock = dict(payload.get("equal_wall_clock", {}))
    for field in V1_PAIR_REQUIRED_EQUAL_WALL_CLOCK_FIELDS:
        if field not in equal_wall_clock:
            failures.append(f"missing_equal_wall_clock_field:{field}")
    strength_claimed = bool(equal_wall_clock.get("strength_claimed", False))
    schema_only = not strength_claimed
    if strength_claimed:
        if not equal_wall_clock.get("evidence_artifact_paths"):
            failures.append("equal_wall_clock_evidence_artifact_paths")
        if float(equal_wall_clock.get("games_per_pairing", 0.0) or 0.0) <= 0.0:
            failures.append("equal_wall_clock_games_per_pairing")
        comparisons = dict(equal_wall_clock.get("baseline_comparisons", {}))
        for baseline in V1_PAIR_REQUIRED_BASELINES:
            if baseline not in comparisons:
                failures.append(f"missing_equal_wall_clock_comparison:{baseline}")
    else:
        blocker = str(equal_wall_clock.get("schema_only_blocker", "")).strip()
        if not blocker:
            failures.append("schema_only_blocker")

    return V1PairScorecardGateResult(
        hard_pass=not failures,
        failures=tuple(failures),
        schema_only=schema_only,
        strength_claimed=strength_claimed,
    )


def v1_pair_scorecard_payload_template(
    *,
    metrics: Mapping[str, float] | None = None,
    schema_only_blocker: str,
) -> dict[str, Any]:
    """Return a deterministic schema-gated V1 scorecard payload skeleton."""

    metric_values = {name: 0.0 for name in V1_PAIR_REQUIRED_METRICS}
    metric_values.update({str(key): float(value) for key, value in dict(metrics or {}).items()})
    return {
        "schema_version": V1_PAIR_SCORECARD_SCHEMA_VERSION,
        "candidate_id": V1_PAIR_CANDIDATE_ID,
        "side_by_side_baselines": list(V1_PAIR_REQUIRED_BASELINES),
        "metrics": metric_values,
        "equal_wall_clock": {
            "strength_claimed": False,
            "schema_only_blocker": schema_only_blocker,
            "hardware_profile": "required_before_strength_claim",
            "candidate_budget": 0,
            "batching_protocol": "required_before_strength_claim",
            "opponent_checkpoints": [],
            "confidence_interval": "required_before_strength_claim",
            "arena_stopping_protocol": "required_before_strength_claim",
            "wall_clock_seconds_per_game": 0.0,
            "games_per_pairing": 0,
            "baseline_comparisons": {},
            "evidence_artifact_paths": [],
        },
    }


def required_v1_pair_metric_names() -> tuple[str, ...]:
    return V1_PAIR_REQUIRED_METRICS


def required_v1_pair_baselines() -> tuple[str, ...]:
    return V1_PAIR_REQUIRED_BASELINES


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
