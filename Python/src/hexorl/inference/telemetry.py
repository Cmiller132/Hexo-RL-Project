"""Structured inference telemetry records."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InferenceTelemetry:
    request_id: str
    trace_id: str
    operation_name: str
    transport_state: str
    queue_depth: int
    batch_size: int
    wait_ms: float
    heartbeat_age_ms: float
    adapter_name: str
    status: str = "ok"
    error_code: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def timeout_message(
    *,
    request_id: str,
    trace_id: str,
    operation_name: str,
    queue_depth: int,
    heartbeat_age_ms: float,
    transport_state: str,
    timeout_ms: float,
) -> str:
    return (
        "inference response timed out "
        f"request_id={request_id} trace_id={trace_id} operation_name={operation_name} "
        f"timeout_ms={timeout_ms:.0f} queue_depth={queue_depth} "
        f"heartbeat_age_ms={heartbeat_age_ms:.1f} transport_state={transport_state}"
    )
