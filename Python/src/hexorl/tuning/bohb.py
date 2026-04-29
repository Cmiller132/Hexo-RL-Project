"""BOHB Hyperband brackets and TPE-style good/bad density sampling."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import pstdev
from typing import Any


@dataclass(frozen=True)
class HyperbandBracket:
    bracket_id: int
    initial_configs: int
    budgets: tuple[int, ...]
    eta: int


@dataclass
class SearchSpace:
    parameters: dict[str, dict[str, Any]]

    def active_parameters(self, config: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
        active: dict[str, dict[str, Any]] = {}
        config = config or {}
        for name, spec in self.parameters.items():
            condition = spec.get("condition")
            if condition is None:
                active[name] = spec
                continue
            key = condition["key"]
            values = set(condition["values"])
            if config.get(key) in values:
                active[name] = spec
        return active

    def sample_random(self, rng: random.Random, base: dict[str, Any] | None = None) -> dict[str, Any]:
        config = dict(base or {})
        changed = True
        while changed:
            changed = False
            for name, spec in self.active_parameters(config).items():
                if name in config:
                    continue
                config[name] = _sample_spec(spec, rng)
                changed = True
        return self.prune_inactive(config)

    def prune_inactive(self, config: dict[str, Any]) -> dict[str, Any]:
        active = self.active_parameters(config)
        return {name: config[name] for name in active if name in config}


@dataclass
class BOHBSampler:
    search_space: SearchSpace
    min_resource: int = 8
    max_resource: int = 14
    eta: int = 2
    warmup_points: int = 6
    top_fraction: float = 0.33
    random_fraction: float = 0.25
    model_candidate_count: int = 64
    observations: list[dict[str, Any]] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.min_resource <= 0 or self.max_resource <= 0:
            raise ValueError("BOHB resources must be positive")
        if self.max_resource < self.min_resource:
            raise ValueError("max_resource must be >= min_resource")
        if self.eta < 2:
            raise ValueError("eta must be >= 2")
        if self.warmup_points < 1:
            raise ValueError("warmup_points must be positive")
        if not (0.0 < self.top_fraction < 1.0):
            raise ValueError("top_fraction must be in (0, 1)")
        if not (0.0 <= self.random_fraction <= 1.0):
            raise ValueError("random_fraction must be in [0, 1]")
        if self.model_candidate_count < 1:
            raise ValueError("model_candidate_count must be positive")

    def create_brackets(self) -> list[HyperbandBracket]:
        s_max = int(math.floor(math.log(self.max_resource / self.min_resource, self.eta)))
        total_budget = (s_max + 1) * self.max_resource
        brackets: list[HyperbandBracket] = []
        for s in range(s_max, -1, -1):
            n = int(math.ceil((total_budget / self.max_resource) * (self.eta**s) / (s + 1)))
            r = self.max_resource / (self.eta**s)
            budgets = tuple(
                max(self.min_resource, int(round(r * (self.eta**i))))
                for i in range(s + 1)
            )
            brackets.append(HyperbandBracket(s, n, budgets, self.eta))
        return brackets

    def observe(
        self,
        config: dict[str, Any],
        score: float,
        *,
        valid: bool = True,
        budget: int | None = None,
        status: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.observations.append(
            {
                "config": self.search_space.prune_inactive(dict(config)),
                "score": float(score),
                "valid": bool(valid),
                "budget": int(self.max_resource if budget is None else budget),
                "status": status or ("completed" if valid else "invalid"),
                "reason": reason,
            }
        )

    def density_model(self, budget: int | None = None) -> dict[str, Any]:
        budget = self._model_budget() if budget is None else int(budget)
        valid = [
            observation
            for observation in self.observations
            if observation["valid"] and int(observation.get("budget", self.max_resource)) == budget
        ]
        if len(valid) < self.warmup_points:
            valid = [observation for observation in self.observations if observation["valid"]]
        valid.sort(key=lambda observation: observation["score"], reverse=True)
        split = max(1, int(math.ceil(len(valid) * self.top_fraction)))
        good = valid[:split]
        bad = valid[split:] or valid[:split]
        return {
            "good": _fit_density(self.search_space, [item["config"] for item in good]),
            "bad": _fit_density(self.search_space, [item["config"] for item in bad]),
            "valid_points": len(valid),
            "invalid_points": sum(1 for observation in self.observations if not observation["valid"]),
            "budget": budget,
            "top_fraction": self.top_fraction,
        }

    def sample(self, *, seed: int = 0) -> dict[str, Any]:
        rng = random.Random(seed)
        use_model = (
            len([item for item in self.observations if item["valid"]]) >= self.warmup_points
            and rng.random() >= self.random_fraction
        )
        if not use_model:
            config = self.search_space.sample_random(rng)
            source = "random"
            model_state = None
            candidate_scores: list[dict[str, Any]] = []
        else:
            model_state = self.density_model()
            candidates = [
                _sample_from_density(self.search_space, model_state["good"], rng)
                for _ in range(self.model_candidate_count)
            ]
            scored = [
                {
                    "config": candidate,
                    "l_over_g": _density_ratio(
                        self.search_space,
                        candidate,
                        model_state["good"],
                        model_state["bad"],
                    ),
                }
                for candidate in candidates
            ]
            scored.sort(key=lambda item: (-float(item["l_over_g"]), json.dumps(item["config"], sort_keys=True)))
            config = dict(scored[0]["config"])
            candidate_scores = scored[: min(8, len(scored))]
            source = "good_bad_density"
        record = {
            "config": config,
            "seed": seed,
            "source": source,
            "density_model": model_state,
            "candidate_scores": candidate_scores,
            "brackets": [asdict(bracket) for bracket in self.create_brackets()],
        }
        self.samples.append(record)
        return record

    def to_dict(self) -> dict[str, Any]:
        return {
            "search_space": self.search_space.parameters,
            "min_resource": self.min_resource,
            "max_resource": self.max_resource,
            "eta": self.eta,
            "warmup_points": self.warmup_points,
            "top_fraction": self.top_fraction,
            "random_fraction": self.random_fraction,
            "model_candidate_count": self.model_candidate_count,
            "observations": self.observations,
            "samples": self.samples,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BOHBSampler":
        sampler = cls(
            SearchSpace(payload["search_space"]),
            min_resource=payload["min_resource"],
            max_resource=payload["max_resource"],
            eta=payload["eta"],
            warmup_points=payload["warmup_points"],
            top_fraction=payload["top_fraction"],
            random_fraction=payload["random_fraction"],
            model_candidate_count=payload.get("model_candidate_count", 64),
        )
        sampler.observations = list(payload.get("observations", []))
        sampler.samples = list(payload.get("samples", []))
        return sampler

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "BOHBSampler":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @staticmethod
    def replay_sample(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "config": dict(record["config"]),
            "source": record["source"],
            "seed": record["seed"],
            "density_model": record.get("density_model"),
        }

    def _model_budget(self) -> int:
        valid_budgets = sorted(
            {
                int(observation.get("budget", self.max_resource))
                for observation in self.observations
                if observation.get("valid")
            },
            reverse=True,
        )
        for budget in valid_budgets:
            count = sum(
                1
                for observation in self.observations
                if observation.get("valid") and int(observation.get("budget", self.max_resource)) == budget
            )
            if count >= self.warmup_points:
                return budget
        return valid_budgets[0] if valid_budgets else self.max_resource


def _sample_spec(spec: dict[str, Any], rng: random.Random) -> Any:
    if spec["type"] == "categorical":
        return rng.choice(list(spec["choices"]))
    if spec["type"] == "int":
        return rng.randint(int(spec["low"]), int(spec["high"]))
    if spec["type"] == "float":
        return rng.uniform(float(spec["low"]), float(spec["high"]))
    raise ValueError(f"unknown parameter type {spec['type']}")


def _fit_density(search_space: SearchSpace, configs: list[dict[str, Any]]) -> dict[str, Any]:
    density: dict[str, Any] = {}
    for name, spec in search_space.parameters.items():
        values = [config[name] for config in configs if name in config]
        if not values:
            continue
        if spec["type"] in {"int", "float"}:
            mean = sum(float(value) for value in values) / len(values)
            std = pstdev(float(value) for value in values) if len(values) > 1 else 0.0
            density[name] = {"type": spec["type"], "mean": mean, "std": max(std, 1e-9)}
        else:
            counts: list[dict[str, Any]] = []
            for value in values:
                for item in counts:
                    if item["value"] == value:
                        item["count"] += 1
                        break
                else:
                    counts.append({"value": value, "count": 1})
            density[name] = {"type": "categorical", "counts": counts}
    return density


def _sample_from_density(
    search_space: SearchSpace,
    density: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    changed = True
    while changed:
        changed = False
        for name, spec in search_space.active_parameters(config).items():
            if name in config:
                continue
            dist = density.get(name)
            if dist is None:
                config[name] = _sample_spec(spec, rng)
            elif dist["type"] == "categorical":
                choices = [item["value"] for item in dist["counts"]]
                weights = [item["count"] for item in dist["counts"]]
                config[name] = rng.choices(choices, weights=weights, k=1)[0]
            else:
                std = max(float(dist["std"]), (float(spec["high"]) - float(spec["low"])) * 0.05)
                value = rng.gauss(float(dist["mean"]), std)
                value = min(max(value, float(spec["low"])), float(spec["high"]))
                config[name] = int(round(value)) if spec["type"] == "int" else value
            changed = True
    return search_space.prune_inactive(config)


def _density_ratio(
    search_space: SearchSpace,
    config: dict[str, Any],
    good_density: dict[str, Any],
    bad_density: dict[str, Any],
) -> float:
    log_good = _log_density(search_space, config, good_density)
    log_bad = _log_density(search_space, config, bad_density)
    return math.exp(max(-50.0, min(50.0, log_good - log_bad)))


def _log_density(
    search_space: SearchSpace,
    config: dict[str, Any],
    density: dict[str, Any],
) -> float:
    score = 0.0
    for name, spec in search_space.active_parameters(config).items():
        if name not in config:
            continue
        dist = density.get(name)
        if dist is None:
            score += _random_log_probability(spec)
            continue
        if dist["type"] == "categorical":
            total = sum(int(item["count"]) for item in dist["counts"])
            choices = len(spec.get("choices", [])) or max(1, len(dist["counts"]))
            count = 0
            for item in dist["counts"]:
                if item["value"] == config[name]:
                    count = int(item["count"])
                    break
            score += math.log((count + 1.0) / (total + choices))
        else:
            mean = float(dist["mean"])
            std = max(float(dist["std"]), (float(spec["high"]) - float(spec["low"])) * 0.02, 1e-9)
            z = (float(config[name]) - mean) / std
            score += -0.5 * z * z - math.log(std)
    return score


def _random_log_probability(spec: dict[str, Any]) -> float:
    if spec["type"] == "categorical":
        return -math.log(max(1, len(spec.get("choices", []))))
    width = max(float(spec["high"]) - float(spec["low"]), 1e-12)
    return -math.log(width)
