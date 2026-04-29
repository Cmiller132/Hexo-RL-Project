"""PB2-style GP-bandit scheduler for continuous dynamic knobs."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PB2Observation:
    trial_id: str
    epoch: int
    params: dict[str, float]
    score: float
    compatible_group: str = "default"


@dataclass
class PB2Scheduler:
    bounds: dict[str, tuple[float, float]]
    uncertainty_weight: float = 0.25
    length_scale: float = 0.35
    noise: float = 1e-6
    parameter_conditions: dict[str, dict[str, Any]] = field(default_factory=dict)
    observations: list[PB2Observation] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name, (low, high) in self.bounds.items():
            if high <= low:
                raise ValueError(f"invalid bounds for {name}")
        if self.uncertainty_weight < 0.0:
            raise ValueError("uncertainty_weight must be non-negative")
        if self.length_scale <= 0.0:
            raise ValueError("length_scale must be positive")
        if self.noise <= 0.0:
            raise ValueError("noise must be positive")

    def observe(self, observation: PB2Observation) -> None:
        self.observations.append(observation)

    def active_param_names(self, context: dict[str, Any] | None = None) -> list[str]:
        context = context or {}
        names: list[str] = []
        for name in sorted(self.bounds):
            condition = self.parameter_conditions.get(name)
            if condition is None:
                names.append(name)
                continue
            key = condition["key"]
            values = set(condition["values"])
            if context.get(key) in values:
                names.append(name)
        return names

    def fit_response_model(
        self,
        compatible_group: str | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        names = self.active_param_names(context)
        observations = [
            observation
            for observation in self.observations
            if compatible_group is None or observation.compatible_group == compatible_group
        ]
        observations = [
            observation
            for observation in observations
            if all(name in observation.params for name in names)
        ]
        if not observations:
            raise ValueError("PB2 requires at least one observation")
        scores = [observation.score for observation in observations]
        score_mean = sum(scores) / len(scores)
        score_scale = _std(scores) or 1.0
        epoch_min = min(observation.epoch for observation in observations)
        epoch_max = max(observation.epoch for observation in observations)
        rows = [
            [
                _normalize_range(observation.epoch, epoch_min, epoch_max),
                *[self._normalize_param(name, observation.params[name]) for name in names],
            ]
            for observation in observations
        ]
        targets = [(score - score_mean) / score_scale for score in scores]
        kernel = _kernel_matrix(rows, self.length_scale, self.noise)
        alpha = _solve_linear(kernel, targets)
        residuals = [target - _gp_mean(alpha, rows, row, self.length_scale) for row, target in zip(rows, targets, strict=True)]
        residual_std = _std(residuals) or 1e-6
        return {
            "param_names": names,
            "rows": rows,
            "targets": targets,
            "epochs": [observation.epoch for observation in observations],
            "epoch_min": epoch_min,
            "epoch_max": epoch_max,
            "score_mean": score_mean,
            "score_scale": score_scale,
            "weights": alpha,
            "kernel": "rbf_time_parameter",
            "length_scale": self.length_scale,
            "noise": self.noise,
            "residual_std": residual_std,
            "compatible_group": compatible_group,
            "context": dict(context or {}),
        }

    def propose(
        self,
        current_params: dict[str, float],
        *,
        seed: int = 0,
        compatible_group: str | None = None,
        candidates: int = 64,
        context: dict[str, Any] | None = None,
        epoch: int | None = None,
    ) -> dict[str, Any]:
        model = self.fit_response_model(compatible_group, context=context)
        rng = random.Random(seed)
        names = model["param_names"]
        candidate_records: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        proposal_epoch = int(epoch if epoch is not None else model["epoch_max"] + 1)
        for _ in range(candidates):
            proposal = dict(current_params)
            clamped: dict[str, bool] = {}
            for name in names:
                low, high = self.bounds[name]
                span = high - low
                value = current_params[name] + rng.gauss(0.0, 0.20 * span)
                clamped_value = min(max(value, low), high)
                proposal[name] = clamped_value
                clamped[name] = clamped_value != value
            row = [
                _normalize_range(proposal_epoch, model["epoch_min"], model["epoch_max"]),
                *[self._normalize_param(name, proposal[name]) for name in names],
            ]
            predicted_z, posterior_std = _gp_predict(model, row)
            uncertainty = max(posterior_std, model["residual_std"] * _distance_to_nearest(row, model["rows"]))
            acquisition = predicted_z + self.uncertainty_weight * uncertainty
            record = {
                "params": proposal,
                "predicted_z": predicted_z,
                "uncertainty": uncertainty,
                "acquisition": acquisition,
                "clamped": clamped,
            }
            candidate_records.append(record)
            if best is None or record["acquisition"] > best["acquisition"]:
                best = record
        assert best is not None
        event = {
            "source_method": "pb2",
            "seed": seed,
            "fit_inputs": model,
            "current_params": dict(current_params),
            "proposed_mutations": candidate_records,
            "accepted_mutation": best,
            "final_values": dict(best["params"]),
            "rejected_mutations": [
                {**record, "reason": "lower_acquisition"}
                for record in candidate_records
                if record is not best
            ],
            "compatible_group": compatible_group,
            "context": dict(context or {}),
            "source_method_detail": "gp_ucb_response_model",
        }
        self.events.append(event)
        return event

    @staticmethod
    def replay_proposal(event: dict[str, Any]) -> dict[str, float]:
        if event.get("source_method") != "pb2":
            raise ValueError("event is not a PB2 event")
        return dict(event["final_values"])

    def _normalize_param(self, name: str, value: float) -> float:
        low, high = self.bounds[name]
        if high <= low:
            raise ValueError(f"invalid bounds for {name}")
        return (float(value) - low) / (high - low)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bounds": {name: list(bounds) for name, bounds in self.bounds.items()},
            "uncertainty_weight": self.uncertainty_weight,
            "length_scale": self.length_scale,
            "noise": self.noise,
            "parameter_conditions": self.parameter_conditions,
            "observations": [observation.__dict__ for observation in self.observations],
            "events": self.events,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PB2Scheduler":
        scheduler = cls(
            {name: tuple(bounds) for name, bounds in payload["bounds"].items()},
            uncertainty_weight=payload["uncertainty_weight"],
            length_scale=payload.get("length_scale", 0.35),
            noise=payload.get("noise", 1e-6),
            parameter_conditions=payload.get("parameter_conditions", {}),
        )
        scheduler.observations = [
            PB2Observation(**observation)
            for observation in payload.get("observations", [])
        ]
        scheduler.events = list(payload.get("events", []))
        return scheduler

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "PB2Scheduler":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _kernel_matrix(rows: list[list[float]], length_scale: float, noise: float) -> list[list[float]]:
    matrix = [[_rbf_kernel(left, right, length_scale) for right in rows] for left in rows]
    for i in range(len(matrix)):
        matrix[i][i] += noise
    return matrix


def _rbf_kernel(left: list[float], right: list[float], length_scale: float) -> float:
    dist2 = sum((a - b) ** 2 for a, b in zip(left, right, strict=True))
    return math.exp(-0.5 * dist2 / max(length_scale * length_scale, 1e-12))


def _gp_mean(alpha: list[float], rows: list[list[float]], row: list[float], length_scale: float) -> float:
    weights = [_rbf_kernel(row, other, length_scale) for other in rows]
    return sum(weight * value for weight, value in zip(weights, alpha, strict=True))


def _gp_predict(model: dict[str, Any], row: list[float]) -> tuple[float, float]:
    rows = model["rows"]
    length_scale = float(model["length_scale"])
    noise = float(model["noise"])
    kernel = _kernel_matrix(rows, length_scale, noise)
    k_star = [_rbf_kernel(row, other, length_scale) for other in rows]
    alpha = list(model["weights"])
    mean = sum(weight * value for weight, value in zip(k_star, alpha, strict=True))
    v = _solve_linear(kernel, k_star)
    variance = max(0.0, 1.0 + noise - sum(left * right for left, right in zip(k_star, v, strict=True)))
    return mean, math.sqrt(variance)


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector, strict=True)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        if abs(pivot_value) < 1e-12:
            continue
        for item in range(col, n + 1):
            augmented[col][item] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            for item in range(col, n + 1):
                augmented[row][item] -= factor * augmented[col][item]
    return [augmented[row][n] for row in range(n)]


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _distance_to_nearest(row: list[float], rows: list[list[float]]) -> float:
    return min(
        math.sqrt(sum((left - right) ** 2 for left, right in zip(row, other, strict=True)))
        for other in rows
    )


def _normalize_range(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return (float(value) - float(low)) / (float(high) - float(low))
