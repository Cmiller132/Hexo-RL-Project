"""Named autotune score components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScoreComponents:
    quality: float
    throughput: float
    stability: float
    resource_use: float
    validation_failures: float = 0.0
    stall_penalty: float = 0.0
    budget_penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.quality
            + self.throughput
            + self.stability
            + self.resource_use
            - self.validation_failures
            - self.stall_penalty
            - self.budget_penalty
        )

    def to_manifest(self) -> dict[str, float]:
        payload = self.__dict__.copy()
        payload["total"] = self.total
        return payload


def score_trial(metrics: dict[str, float], *, validation_failure_count: int = 0, stall_count: int = 0) -> ScoreComponents:
    return ScoreComponents(
        quality=float(metrics.get("win_rate", metrics.get("quality", 0.0))),
        throughput=float(metrics.get("positions_per_sec", 0.0)) / 1000.0,
        stability=1.0 - min(1.0, float(metrics.get("loss_std", 0.0))),
        resource_use=1.0 - min(1.0, float(metrics.get("idle_fraction", 0.0))),
        validation_failures=float(validation_failure_count) * 2.0,
        stall_penalty=float(stall_count) * 1.5,
        budget_penalty=max(0.0, float(metrics.get("memory_fraction", 0.0)) - 0.9) * 5.0,
    )


def score_report(score: ScoreComponents) -> dict[str, Any]:
    return {"score_components": score.to_manifest(), "selected_metric": "total"}
