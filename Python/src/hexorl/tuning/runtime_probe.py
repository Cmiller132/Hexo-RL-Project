"""Reusable runtime probe and calibration primitives.

The probe calibrates only compute/runtime knobs. It does not own architecture,
pair strategy, search semantics, graph tokens, loss weights, or schedules.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping

from hexorl.tuning.debug_bundle import write_runtime_failure_debug_bundle
from hexorl.tuning.quarantine import CandidateQuarantineRecord, classify_runtime_bottleneck


RUNTIME_ONLY_FIELDS = (
    ("selfplay", "num_workers"),
    ("selfplay", "batch_size_per_worker"),
    ("inference", "max_batch_size"),
    ("inference", "max_wait_us"),
)


@dataclass(frozen=True)
class RuntimeKnobs:
    selfplay_workers: int
    batch_size_per_worker: int
    inference_max_batch_size: int
    inference_max_wait_us: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RuntimeKnobs":
        return cls(
            selfplay_workers=int(payload.get("selfplay_workers", payload.get("workers"))),
            batch_size_per_worker=int(payload["batch_size_per_worker"]),
            inference_max_batch_size=int(payload.get("inference_max_batch_size", payload.get("max_batch_size"))),
            inference_max_wait_us=int(payload.get("inference_max_wait_us", payload.get("max_wait_us"))),
        )

    def to_legacy_candidate(self) -> dict[str, int]:
        return {
            "workers": self.selfplay_workers,
            "batch_size_per_worker": self.batch_size_per_worker,
            "max_batch_size": self.inference_max_batch_size,
            "max_wait_us": self.inference_max_wait_us,
        }


@dataclass(frozen=True)
class RuntimeProbeIdentity:
    candidate_id: str
    architecture_id: str
    heads: tuple[str, ...]
    pair_mode: str
    pair_row_cap: int
    full_sims: int
    pcr_sims: int
    graph_token_set: str
    graph_token_budget: int
    graph_layers: int
    host_profile: Mapping[str, Any]
    config_hash: str
    code_hash: str
    architecture_contract_version: str = ""
    recipe_schema_version: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)
    optuna_trial_number: int | None = field(default=None, compare=False, repr=False)

    def cache_payload(self) -> dict[str, Any]:
        return {
            "architecture_id": self.architecture_id,
            "heads": list(self.heads),
            "pair_mode": self.pair_mode,
            "pair_row_cap": int(self.pair_row_cap),
            "full_sims": int(self.full_sims),
            "pcr_sims": int(self.pcr_sims),
            "graph_token_set": self.graph_token_set,
            "graph_token_budget": int(self.graph_token_budget),
            "graph_layers": int(self.graph_layers),
            "host_profile": _jsonable(self.host_profile),
            "config_hash": self.config_hash,
            "code_hash": self.code_hash,
            "architecture_contract_version": self.architecture_contract_version,
            "recipe_schema_version": self.recipe_schema_version,
            "extra": _jsonable(self.extra),
        }

    def cache_key(self) -> str:
        return stable_hash(self.cache_payload())


@dataclass(frozen=True)
class RuntimeProbeResult:
    candidate: RuntimeKnobs
    ok: bool
    positions: int
    elapsed_s: float
    positions_per_second: float = 0.0
    score: float = 0.0
    memory: Mapping[str, Any] = field(default_factory=dict)
    gpu_before: Mapping[str, Any] = field(default_factory=dict)
    gpu_after: Mapping[str, Any] = field(default_factory=dict)
    selfplay: Mapping[str, Any] = field(default_factory=dict)
    replay_memory: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""

    def __post_init__(self) -> None:
        pps = float(self.positions_per_second)
        if pps <= 0.0 and self.positions > 0 and self.elapsed_s > 0.0:
            pps = float(self.positions) / max(float(self.elapsed_s), 1e-6)
            object.__setattr__(self, "positions_per_second", pps)
        if float(self.score) <= 0.0:
            object.__setattr__(self, "score", pps)

    @property
    def positions_per_min(self) -> float:
        return float(self.positions_per_second) * 60.0

    @property
    def memory_safe(self) -> bool:
        return not bool((self.memory or {}).get("unsafe"))

    @property
    def safe(self) -> bool:
        return bool(self.ok and self.positions > 0 and self.positions_per_second > 0.0 and self.memory_safe)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_legacy_candidate(),
            "ok": self.ok,
            "positions": int(self.positions),
            "elapsed_s": float(self.elapsed_s),
            "positions_per_second": float(self.positions_per_second),
            "positions_per_min": self.positions_per_min,
            "score": float(self.score),
            "memory": _jsonable(self.memory),
            "gpu_before": _jsonable(self.gpu_before),
            "gpu_after": _jsonable(self.gpu_after),
            "selfplay": _jsonable(self.selfplay),
            "replay_memory": _jsonable(self.replay_memory),
            "error": self.error,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RuntimeProbeResult":
        candidate = payload.get("candidate")
        if not isinstance(candidate, RuntimeKnobs):
            if not isinstance(candidate, Mapping):
                raise ValueError("runtime probe result requires a candidate mapping")
            candidate = RuntimeKnobs.from_mapping(candidate)
        pps = float(
            payload.get(
                "positions_per_second",
                float(payload.get("positions_per_min", 0.0) or 0.0) / 60.0,
            )
            or 0.0
        )
        return cls(
            candidate=candidate,
            ok=bool(payload.get("ok")),
            positions=int(payload.get("positions", 0) or 0),
            elapsed_s=float(payload.get("elapsed_s", 0.0) or 0.0),
            positions_per_second=pps,
            score=float(payload.get("score", 0.0) or 0.0),
            memory=payload.get("memory") or {},
            gpu_before=payload.get("gpu_before") or {},
            gpu_after=payload.get("gpu_after") or {},
            selfplay=payload.get("selfplay") or {},
            replay_memory=payload.get("replay_memory") or {},
            error=str(payload.get("error", "") or ""),
        )


@dataclass
class RuntimeProbeDecision:
    status: str
    identity: RuntimeProbeIdentity
    results: list[RuntimeProbeResult]
    selected: RuntimeKnobs | None = None
    cache_hit: bool = False
    quarantine: CandidateQuarantineRecord | None = None
    debug_bundle_path: Path | None = None

    @property
    def quarantined(self) -> bool:
        return self.status == "quarantined"


class RuntimeCalibrationCache:
    def __init__(self, path: Path | str | None = None, entries: Mapping[str, Any] | None = None):
        self.path = Path(path) if path is not None else None
        self.entries: dict[str, dict[str, Any]] = {
            str(key): dict(value) for key, value in (entries or {}).items()
        }

    @classmethod
    def load(cls, path: Path | str) -> "RuntimeCalibrationCache":
        cache = cls(path)
        cache.reload()
        return cache

    def reload(self) -> None:
        if self.path is None or not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        entries = payload.get("entries", payload) if isinstance(payload, Mapping) else {}
        self.entries = {str(key): dict(value) for key, value in entries.items() if isinstance(value, Mapping)}

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "entries": self.entries}
        self.path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def get_valid(self, identity: RuntimeProbeIdentity, *, speed_threshold: float) -> dict[str, Any] | None:
        entry = self.entries.get(identity.cache_key())
        if not entry:
            return None
        if entry.get("identity") != identity.cache_payload():
            return None
        selected = entry.get("selected_result")
        if not isinstance(selected, Mapping):
            return None
        selected_result = RuntimeProbeResult.from_mapping(selected)
        if not result_passes_speed_gate(selected_result, speed_threshold):
            return None
        results = [
            RuntimeProbeResult.from_mapping(row)
            for row in entry.get("results", [])
            if isinstance(row, Mapping)
        ]
        best = select_best_runtime_result(results, speed_threshold=speed_threshold)
        if best is not None and best.candidate != selected_result.candidate:
            return None
        return entry

    def store(
        self,
        identity: RuntimeProbeIdentity,
        *,
        selected: RuntimeProbeResult,
        results: Iterable[RuntimeProbeResult],
    ) -> None:
        self.entries[identity.cache_key()] = {
            "schema_version": 1,
            "identity": identity.cache_payload(),
            "identity_key": identity.cache_key(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selected": selected.candidate.to_legacy_candidate(),
            "selected_result": selected.to_dict(),
            "results": [result.to_dict() for result in results],
        }
        self.save()


ProbeRunner = Callable[[RuntimeKnobs, int], RuntimeProbeResult | Mapping[str, Any]]


class RuntimeProbe:
    def __init__(
        self,
        *,
        identity: RuntimeProbeIdentity,
        candidates: Iterable[RuntimeKnobs | Mapping[str, Any]],
        runner: ProbeRunner,
        cache: RuntimeCalibrationCache | None = None,
        speed_threshold: float = 2.0,
        debug_bundle_root: Path | str | None = None,
        repro_command: str | Iterable[str] | None = None,
        dashboard_links: Mapping[str, Any] | None = None,
    ):
        self.identity = identity
        self.candidates = [
            item if isinstance(item, RuntimeKnobs) else RuntimeKnobs.from_mapping(item)
            for item in candidates
        ]
        self.runner = runner
        self.cache = cache
        self.speed_threshold = float(speed_threshold)
        self.debug_bundle_root = Path(debug_bundle_root) if debug_bundle_root is not None else None
        self.repro_command = repro_command
        self.dashboard_links = dashboard_links or {}

    def run(self) -> RuntimeProbeDecision:
        if self.cache is not None:
            cached = self.cache.get_valid(self.identity, speed_threshold=self.speed_threshold)
            if cached is not None:
                selected_result = RuntimeProbeResult.from_mapping(cached["selected_result"])
                results = [
                    RuntimeProbeResult.from_mapping(row)
                    for row in cached.get("results", [])
                    if isinstance(row, Mapping)
                ]
                return RuntimeProbeDecision(
                    status="selected",
                    identity=self.identity,
                    selected=selected_result.candidate,
                    results=results,
                    cache_hit=True,
                )

        results: list[RuntimeProbeResult] = []
        for index, candidate in enumerate(self.candidates):
            try:
                row = self.runner(candidate, index)
                result = row if isinstance(row, RuntimeProbeResult) else RuntimeProbeResult.from_mapping(row)
            except Exception as exc:
                result = RuntimeProbeResult(
                    candidate=candidate,
                    ok=False,
                    positions=0,
                    elapsed_s=0.0,
                    error=f"{type(exc).__name__}:{exc}",
                )
            results.append(result)

        selected = select_best_runtime_result(results, speed_threshold=self.speed_threshold)
        if selected is not None:
            if self.cache is not None:
                self.cache.store(self.identity, selected=selected, results=results)
            return RuntimeProbeDecision(
                status="selected",
                identity=self.identity,
                selected=selected.candidate,
                results=results,
            )

        reason = speed_quarantine_reason(results, self.speed_threshold)
        category = classify_runtime_bottleneck([result.to_dict() for result in results])
        quarantine = CandidateQuarantineRecord.quarantined(
            candidate_id=self.identity.candidate_id,
            reason=reason,
            reason_category=category,
            evidence={
                "cache_key": self.identity.cache_key(),
                "speed_threshold_positions_per_second": self.speed_threshold,
                "runtime_probe_results": [result.to_dict() for result in results],
            },
            config_hash=self.identity.config_hash,
            code_hash=self.identity.code_hash,
        )
        bundle_path = None
        if self.debug_bundle_root is not None:
            bundle_path = write_runtime_failure_debug_bundle(
                self.debug_bundle_root,
                candidate_id=self.identity.candidate_id,
                reason=reason,
                repro_command=self.repro_command,
                runtime_telemetry={
                    "identity": self.identity.cache_payload(),
                    "speed_threshold_positions_per_second": self.speed_threshold,
                    "reason_category": category,
                },
                runtime_probe_results=[result.to_dict() for result in results],
                dashboard_links=self.dashboard_links,
            )
            quarantine.add_evidence("debug_bundle", {"path": str(bundle_path)})
        return RuntimeProbeDecision(
            status="quarantined",
            identity=self.identity,
            results=results,
            quarantine=quarantine,
            debug_bundle_path=bundle_path,
        )


def result_passes_speed_gate(result: RuntimeProbeResult, speed_threshold: float) -> bool:
    return result.safe and float(result.positions_per_second) > float(speed_threshold)


def select_best_runtime_result(
    results: Iterable[RuntimeProbeResult],
    *,
    speed_threshold: float = 2.0,
) -> RuntimeProbeResult | None:
    valid = [result for result in results if result_passes_speed_gate(result, speed_threshold)]
    if not valid:
        return None
    return max(valid, key=lambda result: (float(result.score), float(result.positions_per_second)))


def speed_quarantine_reason(results: list[RuntimeProbeResult], speed_threshold: float) -> str:
    if not results:
        return f"runtime_probe_speed_quarantine:no_probe_candidates_above_{speed_threshold:g}_positions_per_second"
    memory_unsafe = [result for result in results if not result.memory_safe]
    safe_rows = [result for result in results if result.safe]
    if memory_unsafe and not safe_rows:
        return "runtime_probe_speed_quarantine:all_probe_candidates_memory_unsafe"
    best_safe = max((result.positions_per_second for result in safe_rows), default=0.0)
    return (
        "runtime_probe_speed_quarantine:"
        f"best_safe_{best_safe:.3f}_positions_per_second_not_above_{speed_threshold:g}"
    )


def apply_runtime_knobs(config: Any, knobs: RuntimeKnobs) -> Any:
    """Apply only runtime knobs to a config-like object and return it."""

    before = semantic_config_hash(config)
    _set_nested(config, ("selfplay", "num_workers"), knobs.selfplay_workers)
    _set_nested(config, ("selfplay", "batch_size_per_worker"), knobs.batch_size_per_worker)
    _set_nested(config, ("inference", "max_batch_size"), knobs.inference_max_batch_size)
    _set_nested(config, ("inference", "max_wait_us"), knobs.inference_max_wait_us)
    after = semantic_config_hash(config)
    if before != after:
        raise RuntimeError("runtime knob application changed semantic config fields")
    return config


def runtime_knobs_from_config(config: Any) -> RuntimeKnobs:
    return RuntimeKnobs(
        selfplay_workers=int(_get_nested(config, ("selfplay", "num_workers"))),
        batch_size_per_worker=int(_get_nested(config, ("selfplay", "batch_size_per_worker"))),
        inference_max_batch_size=int(_get_nested(config, ("inference", "max_batch_size"))),
        inference_max_wait_us=int(_get_nested(config, ("inference", "max_wait_us"))),
    )


def identity_from_config(
    *,
    candidate_id: str,
    config: Any,
    host_profile: Mapping[str, Any],
    code_hash: str,
    config_hash: str | None = None,
    architecture_contract_version: str = "",
    recipe_schema_version: str = "",
    optuna_trial_number: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> RuntimeProbeIdentity:
    model = _get_nested(config, ("model",), default={})
    selfplay = _get_nested(config, ("selfplay",), default={})
    return RuntimeProbeIdentity(
        candidate_id=candidate_id,
        architecture_id=str(_get_field(model, "architecture", "unknown")),
        heads=tuple(str(head) for head in (_get_field(model, "heads", ()) or ())),
        pair_mode=str(_get_field(model, "pair_strategy", "none")),
        pair_row_cap=int(_get_field(model, "pair_strategy_max_pairs", 0) or 0),
        full_sims=int(_get_field(selfplay, "mcts_simulations", 0) or 0),
        pcr_sims=int(_get_field(selfplay, "pcr_low_sims", 0) or 0),
        graph_token_set=str(_get_field(model, "graph_token_set", "")),
        graph_token_budget=int(_get_field(model, "graph_token_budget", 0) or 0),
        graph_layers=int(_get_field(model, "graph_layers", 0) or 0),
        host_profile=dict(host_profile),
        config_hash=config_hash or semantic_config_hash(config),
        code_hash=code_hash,
        architecture_contract_version=architecture_contract_version,
        recipe_schema_version=recipe_schema_version,
        optuna_trial_number=optuna_trial_number,
        extra=extra or {},
    )


def semantic_config_hash(config: Any) -> str:
    payload = _jsonable(config)
    for path in RUNTIME_ONLY_FIELDS:
        _remove_nested(payload, path)
    return stable_hash(payload)


def stable_hash(payload: Any) -> str:
    blob = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _jsonable(payload: Any) -> Any:
    if hasattr(payload, "model_dump"):
        return _jsonable(payload.model_dump())
    if is_dataclass(payload):
        return _jsonable(asdict(payload))
    if isinstance(payload, Mapping):
        return {str(key): _jsonable(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_jsonable(value) for value in payload]
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, float) and not math.isfinite(payload):
        return str(payload)
    if hasattr(payload, "__dict__") and not isinstance(payload, type):
        return _jsonable(vars(payload))
    return payload


def _get_nested(config: Any, path: tuple[str, ...], default: Any = None) -> Any:
    current = config
    for key in path:
        current = _get_field(current, key, default)
        if current is default:
            return default
    return current


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set_nested(config: Any, path: tuple[str, ...], value: Any) -> None:
    parent = _get_nested(config, path[:-1])
    key = path[-1]
    if isinstance(parent, MutableMapping):
        parent[key] = value
    else:
        setattr(parent, key, value)


def _remove_nested(payload: Any, path: tuple[str, ...]) -> None:
    current = payload
    for key in path[:-1]:
        if not isinstance(current, MutableMapping) or key not in current:
            return
        current = current[key]
    if isinstance(current, MutableMapping):
        current.pop(path[-1], None)
