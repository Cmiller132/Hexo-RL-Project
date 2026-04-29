"""Persisted ASHA rung table with deterministic replay."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_ASHA_RESOURCES = (8, 12, 14)


@dataclass(frozen=True)
class TrialObservation:
    trial_id: str
    resource: int
    score: float
    completed_epochs: int
    wall_time_seconds: float
    selfplay_positions: int
    hard_failure: bool = False
    failure_reason: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def comparable_resource(self) -> int:
        return int(self.resource)


@dataclass
class ASHARungTable:
    resources: tuple[int, ...] = DEFAULT_ASHA_RESOURCES
    promotion_fraction: float = 0.5
    observations: list[TrialObservation] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.resources = tuple(int(resource) for resource in self.resources)
        if not self.resources:
            raise ValueError("ASHA resources cannot be empty")
        if any(resource <= 0 for resource in self.resources):
            raise ValueError("ASHA resources must be positive")
        if self.resources != tuple(sorted(self.resources)):
            raise ValueError("ASHA resources must be sorted")
        if not (0.0 < self.promotion_fraction <= 1.0):
            raise ValueError("promotion_fraction must be in (0, 1]")

    @classmethod
    def default(cls) -> "ASHARungTable":
        return cls(resources=DEFAULT_ASHA_RESOURCES)

    def record(self, observation: TrialObservation) -> None:
        if observation.resource not in self.resources:
            raise ValueError(f"resource {observation.resource} is not an ASHA rung")
        if not observation.hard_failure and observation.completed_epochs < observation.resource:
            raise ValueError("completed_epochs must reach the reported rung resource")
        self.observations.append(observation)

    def rung_observations(self, resource: int) -> list[TrialObservation]:
        return [
            observation
            for observation in self.observations
            if observation.comparable_resource() == int(resource)
        ]

    def promoted_trials(self, resource: int) -> list[str]:
        ranked = [
            observation
            for observation in self.rung_observations(resource)
            if not observation.hard_failure
        ]
        ranked.sort(key=lambda observation: (-observation.score, observation.trial_id))
        if not ranked:
            return []
        keep = max(1, math.ceil(len(ranked) * self.promotion_fraction))
        return [observation.trial_id for observation in ranked[:keep]]

    def quarantine_trials(self, resource: int | None = None) -> list[str]:
        observations = (
            self.observations if resource is None else self.rung_observations(resource)
        )
        return [
            observation.trial_id
            for observation in observations
            if observation.hard_failure
        ]

    def decision_for(self, resource: int) -> dict[str, Any]:
        observations = self.rung_observations(resource)
        promoted = set(self.promoted_trials(resource))
        quarantined = set(self.quarantine_trials(resource))
        pruned = sorted(
            observation.trial_id
            for observation in observations
            if observation.trial_id not in promoted and observation.trial_id not in quarantined
        )
        decision = {
            "resource": int(resource),
            "promoted": sorted(promoted),
            "pruned": pruned,
            "quarantined": sorted(quarantined),
            "mode": "asha_same_resource",
            "promotion_semantics": "same_resource_rung_successive_halving",
        }
        self.decisions.append(decision)
        return decision

    def replay_decisions(self) -> list[dict[str, Any]]:
        return [
            {
                "resource": int(resource),
                "promoted": sorted(self.promoted_trials(resource)),
                "pruned": sorted(
                    observation.trial_id
                    for observation in self.rung_observations(resource)
                    if observation.trial_id not in self.promoted_trials(resource)
                    and not observation.hard_failure
                ),
                "quarantined": sorted(self.quarantine_trials(resource)),
                "mode": "asha_same_resource",
                "promotion_semantics": "same_resource_rung_successive_halving",
            }
            for resource in self.resources
            if self.rung_observations(resource)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "resources": list(self.resources),
            "promotion_fraction": self.promotion_fraction,
            "observations": [asdict(observation) for observation in self.observations],
            "decisions": self.decisions,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ASHARungTable":
        table = cls(
            resources=tuple(payload["resources"]),
            promotion_fraction=float(payload.get("promotion_fraction", 0.5)),
        )
        table.observations = [
            TrialObservation(**observation)
            for observation in payload.get("observations", [])
        ]
        table.decisions = list(payload.get("decisions", []))
        return table

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "ASHARungTable":
        return cls.from_dict(json.loads(Path(path).read_text()))
