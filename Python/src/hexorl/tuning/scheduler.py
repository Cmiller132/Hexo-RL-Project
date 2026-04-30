"""Typed autotune scheduler decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hexorl.tuning.scoring import ScoreComponents


@dataclass(frozen=True)
class SchedulerDecision:
    trial_id: str
    action: str
    reason_code: str
    score_components: dict[str, float]
    validation_results: tuple[dict[str, Any], ...]
    runtime_budget: dict[str, Any]
    progress_signals: dict[str, Any]
    trace_ids: tuple[str, ...]
    likely_owner: str

    def to_log(self) -> dict[str, Any]:
        return self.__dict__.copy()


class AutotuneScheduler:
    def decide(
        self,
        trial_id: str,
        score: ScoreComponents,
        *,
        validation_results: tuple[dict[str, Any], ...],
        runtime_budget: dict[str, Any],
        progress_signals: dict[str, Any],
        trace_ids: tuple[str, ...],
    ) -> SchedulerDecision:
        failures = [item for item in validation_results if not item.get("ok", False)]
        if failures:
            return SchedulerDecision(
                trial_id,
                "reject",
                "validation_failed",
                score.to_manifest(),
                validation_results,
                runtime_budget,
                progress_signals,
                trace_ids,
                failures[0].get("owner", "recipe validation"),
            )
        if progress_signals.get("stalled"):
            return SchedulerDecision(
                trial_id,
                "abort",
                "watchdog_no_progress",
                score.to_manifest(),
                validation_results,
                runtime_budget,
                progress_signals,
                trace_ids,
                str(progress_signals.get("likely_owner", "runtime scheduler")),
            )
        action = "promote" if score.total >= 1.0 else "early_stop"
        return SchedulerDecision(
            trial_id,
            action,
            "score_threshold",
            score.to_manifest(),
            validation_results,
            runtime_budget,
            progress_signals,
            trace_ids,
            "scheduler/scoring.py",
        )
