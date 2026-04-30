"""Read-only dashboard contract inspection dispatcher."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from hexorl.dashboard.inspection_services import (
    contract_catalog,
    default_inspector_services,
    fact_payload,
    trace_payload,
)


class InspectorService(Protocol):
    name: str

    def inspect(self, request: "InspectionRequest", inspector: "ContractInspector") -> dict[str, Any]: ...


@dataclass(frozen=True)
class InspectionRequest:
    history: bytes = b""
    policy_target: tuple[tuple[int, int, float], ...] = ()
    pair_policy_target: tuple[tuple[tuple[int, int], tuple[int, int], float], ...] = ()
    model_family: str = "dense_cnn"
    recipe_id: str = "dashboard-default"
    recipe_hash: str = ""
    checkpoint_manifest: dict[str, Any] | None = None
    trace: dict[str, Any] | None = None
    model_output: dict[str, Any] | None = None
    replay_identity: dict[str, Any] | None = None
    autotune_report: dict[str, Any] | None = None
    compare_to: dict[str, Any] | None = None


class ContractInspector:
    """Dispatcher over focused read-only dashboard inspector services."""

    def __init__(self, services: tuple[InspectorService, ...] | None = None) -> None:
        self._services: dict[str, InspectorService] = {}
        for service in services or default_inspector_services():
            self.register(service.name, service)

    def register(self, name: str, service: InspectorService) -> None:
        if name in self._services:
            raise ValueError(f"dashboard inspector already registered: {name}")
        self._services[name] = service

    def views(self) -> tuple[str, ...]:
        return tuple(sorted(self._services))

    def inspect(self, view: str, **kwargs: Any) -> dict[str, Any]:
        request = InspectionRequest(**kwargs)
        try:
            service = self._services[view]
        except KeyError as exc:
            raise KeyError(f"unknown dashboard inspector view: {view}") from exc
        started = time.monotonic()
        payload = service.inspect(request, self)
        payload.setdefault("view", view)
        payload.setdefault("facts", fact_payload(request.history, request))
        payload.setdefault("trace", trace_payload(request.trace))
        payload["inspector"] = {
            "dispatcher": "ContractInspector",
            "service": service.__class__.__name__,
            "elapsed_ms": (time.monotonic() - started) * 1000.0,
        }
        return payload


def required_view_names() -> tuple[str, ...]:
    return (
        "history",
        "legal-table",
        "tactical",
        "candidates",
        "pairs",
        "graph",
        "d6",
        "model-input",
        "model-output",
        "trace",
        "replay",
        "checkpoint",
        "recipe",
        "autotune",
    )


__all__ = [
    "ContractInspector",
    "InspectionRequest",
    "InspectorService",
    "contract_catalog",
    "required_view_names",
]
