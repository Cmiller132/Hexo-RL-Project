"""Typed runtime sweep specifications and no-progress watchdogs."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HostProfile:
    cpu_count: int
    memory_gb: float
    gpu_name: str = "none"
    os: str = platform.platform()
    python: str = platform.python_version()

    @classmethod
    def local(cls) -> "HostProfile":
        return cls(cpu_count=os.cpu_count() or 1, memory_gb=8.0)

    def to_manifest(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class WatchdogSpec:
    selfplay_progress_s: float = 30.0
    inference_response_s: float = 5.0
    training_batch_s: float = 20.0
    evaluation_progress_s: float = 20.0
    artifact_write_s: float = 10.0

    def validate(self) -> None:
        if min(self.__dict__.values()) <= 0:
            raise ValueError("watchdog thresholds must be positive")


@dataclass(frozen=True)
class RuntimeSpec:
    selfplay_workers: int
    rust_threads: int
    torch_threads: int
    dataloader_workers: int
    inference_max_batch: int
    microbatch_wait_ms: float
    leaf_batch_size: int
    record_queue_capacity: int
    replay_prefetch: int
    train_batch_size: int
    compile_model: bool = False
    memory_fraction: float = 0.75
    watchdogs: WatchdogSpec = WatchdogSpec()

    def validate(self, host: HostProfile) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        self.watchdogs.validate()
        if self.selfplay_workers <= 0 or self.rust_threads <= 0 or self.torch_threads <= 0:
            failures.append(_failure("runtime_budget", "worker/thread counts must be positive"))
        if self.selfplay_workers + self.rust_threads + self.torch_threads > host.cpu_count * 2:
            failures.append(_failure("runtime_budget", "CPU thread budget exceeds host oversubscription policy"))
        if self.inference_max_batch <= 0 or self.leaf_batch_size <= 0 or self.train_batch_size <= 0:
            failures.append(_failure("runtime_budget", "batch sizes must be positive"))
        if self.record_queue_capacity < self.selfplay_workers:
            failures.append(_failure("backpressure", "record queue capacity must cover active self-play workers"))
        if not 0.1 <= self.memory_fraction <= 0.95:
            failures.append(_failure("memory_budget", "memory_fraction must be in [0.1, 0.95]"))
        return failures

    def to_manifest(self) -> dict[str, Any]:
        payload = self.__dict__.copy()
        payload["watchdogs"] = self.watchdogs.__dict__.copy()
        return payload


def default_runtime_spec(host: HostProfile | None = None) -> RuntimeSpec:
    host = host or HostProfile.local()
    workers = max(1, min(4, host.cpu_count // 2))
    return RuntimeSpec(
        selfplay_workers=workers,
        rust_threads=max(1, host.cpu_count // 2),
        torch_threads=max(1, host.cpu_count // 4),
        dataloader_workers=max(0, min(4, host.cpu_count // 4)),
        inference_max_batch=32,
        microbatch_wait_ms=2.0,
        leaf_batch_size=16,
        record_queue_capacity=max(64, workers * 8),
        replay_prefetch=4,
        train_batch_size=64,
    )


def simulate_no_progress(spec: RuntimeSpec, subsystem: str, *, trace_id: str = "phase08-sim") -> dict[str, Any]:
    thresholds = spec.watchdogs.__dict__
    key = {
        "selfplay": "selfplay_progress_s",
        "inference": "inference_response_s",
        "training": "training_batch_s",
        "evaluation": "evaluation_progress_s",
        "artifact": "artifact_write_s",
    }[subsystem]
    return {
        "event": "watchdog_abort",
        "subsystem": subsystem,
        "threshold_s": thresholds[key],
        "trace_id": trace_id,
        "likely_owner": {
            "selfplay": "selfplay worker/game runner",
            "inference": "inference adapter/transport",
            "training": "train adapter/dataloader",
            "evaluation": "eval arena/policy provider",
            "artifact": "autotune reporting/manifests",
        }[subsystem],
        "action": "inspect trace ids, queue depth, scheduler decision, and last progress counter",
    }


def _failure(kind: str, message: str) -> dict[str, Any]:
    return {"kind": kind, "message": message, "owner": "tuning/runtime_sweep.py"}
