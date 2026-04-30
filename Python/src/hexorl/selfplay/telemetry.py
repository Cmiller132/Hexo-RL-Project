"""Structured self-play telemetry and debug bundle contracts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol

from hexorl.contracts.identity import stable_digest
from hexorl.contracts.validation import ContractValidationError


REQUIRED_CONTRACT_TRACE_SPANS = (
    "history_parse_ms",
    "engine_replay_ms",
    "legal_table_ms",
    "tactical_oracle_ms",
    "candidate_build_ms",
    "pair_table_build_ms",
    "graph_token_build_ms",
    "graph_relation_build_ms",
    "graph_tensorize_ms",
    "ipc_pack_ms",
    "ipc_wait_ms",
    "queue_wait_ms",
    "collate_ms",
    "model_forward_ms",
    "scatter_ms",
    "decode_ms",
    "pair_chunk_count",
    "pair_chunk_forward_ms",
)

REQUIRED_EVENT_TYPES = (
    "selfplay_worker_heartbeat",
    "selfplay_phase_transition",
    "selfplay_no_progress",
    "selfplay_game_summary",
    "policy_eval_timing",
    "pair_strategy_summary",
    "contract_validation_failure",
    "inference_protocol_mismatch",
    "selfplay_position_debug_bundle",
    "selfplay_mutation_guard_failure",
    "selfplay_resource_profile",
    "selfplay_backpressure",
    "selfplay_batching_summary",
)

DEBUG_FAILURE_OWNERS = (
    "engine replay/legal",
    "engine invariant hook",
    "PyO3 protocol decode",
    "contract validation",
    "D6 transform",
    "candidate builder",
    "pair table builder",
    "graph semantic builder",
    "graph tensorizer",
    "inference protocol/transport",
    "model forward/output validation",
    "policy provider row mapping",
    "pair strategy",
    "EngineAdapter/MCTS",
    "MCTS token lifecycle",
    "move application",
    "record writer/replay encoding",
)


@dataclass(frozen=True)
class ContractTrace:
    trace_id: str
    history_hash: str
    model_family: str
    phase: str
    legal_count: int
    candidate_count: int
    pair_rows_total: int
    pair_rows_scored: int
    graph_token_count: int
    graph_relation_count: int
    timings_ms: Mapping[str, float]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        timings = {key: float(value) for key, value in dict(self.timings_ms).items()}
        missing = [key for key in REQUIRED_CONTRACT_TRACE_SPANS if key not in timings]
        if missing:
            raise ContractValidationError(
                f"ContractTrace missing timing spans: {', '.join(missing)}",
                owner="ContractTrace",
            )
        object.__setattr__(self, "legal_count", int(self.legal_count))
        object.__setattr__(self, "candidate_count", int(self.candidate_count))
        object.__setattr__(self, "pair_rows_total", int(self.pair_rows_total))
        object.__setattr__(self, "pair_rows_scored", int(self.pair_rows_scored))
        object.__setattr__(self, "graph_token_count", int(self.graph_token_count))
        object.__setattr__(self, "graph_relation_count", int(self.graph_relation_count))
        object.__setattr__(self, "timings_ms", MappingProxyType(timings))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))

    @classmethod
    def from_context(
        cls,
        context,
        *,
        timings_ms: Mapping[str, float] | None = None,
        pair_rows_scored: int = 0,
        warnings: tuple[str, ...] = (),
    ) -> "ContractTrace":
        timings = _complete_timings(timings_ms)
        graph_batch = getattr(context, "graph_batch", None)
        graph_token_count = int(getattr(getattr(graph_batch, "token_qr", ()), "shape", (0,))[0]) if graph_batch is not None else 0
        graph_relation_count = int(getattr(getattr(graph_batch, "edge_index", ()), "shape", (0, 0))[1]) if graph_batch is not None else 0
        candidate_table = getattr(context, "candidate_table", None)
        pair_table = getattr(context, "pair_table", None)
        return cls(
            trace_id=str(context.trace_id),
            history_hash=str(context.history_hash),
            model_family=str(context.model_family),
            phase=str(context.phase),
            legal_count=int(context.legal_table.rows.shape[0]),
            candidate_count=0 if candidate_table is None else int(candidate_table.rows.shape[0]),
            pair_rows_total=0 if pair_table is None else int(pair_table.possible_pair_count),
            pair_rows_scored=int(pair_rows_scored),
            graph_token_count=graph_token_count,
            graph_relation_count=graph_relation_count,
            timings_ms=timings,
            warnings=warnings,
        )

    def to_event_payload(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "history_hash": self.history_hash,
            "model_family": self.model_family,
            "phase": self.phase,
            "legal_count": self.legal_count,
            "candidate_count": self.candidate_count,
            "pair_rows_total": self.pair_rows_total,
            "pair_rows_scored": self.pair_rows_scored,
            "graph_token_count": self.graph_token_count,
            "graph_relation_count": self.graph_relation_count,
            "timings_ms": dict(self.timings_ms),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class SelfPlayDebugBundle:
    owner_subsystem: str
    run_id: str
    game_id: int
    move_index: int
    seed: int
    phase: str
    sections: Mapping[str, Mapping[str, object]]
    validation_failures: tuple[str, ...] = ()
    mutation_guard_failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.owner_subsystem not in DEBUG_FAILURE_OWNERS:
            raise ContractValidationError(
                f"unknown debug owner subsystem {self.owner_subsystem!r}",
                owner="SelfPlayDebugBundle",
            )
        required = (
            "engine",
            "contract",
            "d6",
            "model_input",
            "raw_output",
            "policy",
            "pair",
            "mcts",
            "replay",
        )
        missing = [key for key in required if key not in self.sections]
        if missing:
            raise ContractValidationError(
                f"debug bundle missing sections: {', '.join(missing)}",
                owner="SelfPlayDebugBundle",
            )
        object.__setattr__(self, "sections", MappingProxyType({k: MappingProxyType(dict(v)) for k, v in self.sections.items()}))
        object.__setattr__(self, "validation_failures", tuple(self.validation_failures))
        object.__setattr__(self, "mutation_guard_failures", tuple(self.mutation_guard_failures))

    @property
    def bundle_hash(self) -> str:
        return stable_digest(
            (
                "SelfPlayDebugBundle",
                self.owner_subsystem,
                self.run_id,
                self.game_id,
                self.move_index,
                self.phase,
                tuple(sorted((key, tuple(sorted(value.items()))) for key, value in self.sections.items())),
            )
        )

    def to_event_payload(self) -> dict[str, object]:
        return {
            "owner_subsystem": self.owner_subsystem,
            "run_id": self.run_id,
            "game_id": int(self.game_id),
            "move_index": int(self.move_index),
            "seed": int(self.seed),
            "phase": self.phase,
            "sections": {key: dict(value) for key, value in self.sections.items()},
            "validation_failures": list(self.validation_failures),
            "mutation_guard_failures": list(self.mutation_guard_failures),
            "bundle_hash": self.bundle_hash,
        }


@dataclass
class SelfPlayMutationGuard:
    owner_subsystem: str
    payload_name: str
    initial_hash: str

    def check(self, current_hash: str) -> None:
        if str(current_hash) != self.initial_hash:
            raise ContractValidationError(
                f"{self.payload_name} mutated after validation",
                owner=self.owner_subsystem,
            )


class SelfPlayTelemetrySink(Protocol):
    def emit(self, event_type: str, payload: Mapping[str, object]) -> None: ...


class InMemorySelfPlayTelemetrySink:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def emit(self, event_type: str, payload: Mapping[str, object]) -> None:
        if event_type not in REQUIRED_EVENT_TYPES:
            raise ContractValidationError(
                f"unknown self-play telemetry event {event_type!r}",
                owner="SelfPlayTelemetrySink",
            )
        event = {
            "event": event_type,
            "timestamp_unix_ms": int(time.time() * 1000),
            **dict(payload),
        }
        self.events.append(event)


class LoggingSelfPlayTelemetrySink:
    def __init__(self, logger) -> None:
        self.logger = logger

    def emit(self, event_type: str, payload: Mapping[str, object]) -> None:
        if event_type not in REQUIRED_EVENT_TYPES:
            raise ContractValidationError(
                f"unknown self-play telemetry event {event_type!r}",
                owner="SelfPlayTelemetrySink",
            )
        self.logger.info("%s %s", event_type, dict(payload))


def heartbeat_payload(
    *,
    worker_id: int,
    process_id: int,
    run_id: str,
    game_id: int,
    phase: str,
    move_index: int,
    positions_completed: int,
    model_family: str,
    recipe_id: str,
    policy_provider: str,
    pair_strategy: str,
    no_progress_duration_ms: float = 0.0,
    warning_count: int = 0,
    last_warning: str = "",
    **extra: object,
) -> dict[str, object]:
    payload = {
        "worker_id": int(worker_id),
        "process_id": int(process_id),
        "run_id": str(run_id),
        "game_id": int(game_id),
        "current_phase": str(phase),
        "move_index": int(move_index),
        "positions_completed_since_last_heartbeat": int(positions_completed),
        "last_successful_inference_request_id": str(extra.pop("last_successful_inference_request_id", "")),
        "last_engine_operation": str(extra.pop("last_engine_operation", "")),
        "legal_count": int(extra.pop("legal_count", 0)),
        "candidate_count": int(extra.pop("candidate_count", 0)),
        "pair_count": int(extra.pop("pair_count", 0)),
        "token_count": int(extra.pop("token_count", 0)),
        "relation_count": int(extra.pop("relation_count", 0)),
        "active_model_family": str(model_family),
        "recipe_id": str(recipe_id),
        "policy_provider": str(policy_provider),
        "pair_strategy": str(pair_strategy),
        "pair_rows_possible": int(extra.pop("pair_rows_possible", 0)),
        "pair_rows_scored": int(extra.pop("pair_rows_scored", 0)),
        "root_generation": int(extra.pop("root_generation", 0)),
        "batch_generation": int(extra.pop("batch_generation", 0)),
        "ffi_protocol_version": str(extra.pop("ffi_protocol_version", "")),
        "legal_byte_hash": str(extra.pop("legal_byte_hash", "")),
        "history_byte_hash": str(extra.pop("history_byte_hash", "")),
        "inference_slot_sequence": int(extra.pop("inference_slot_sequence", 0)),
        "rust_mcts_error_code": str(extra.pop("rust_mcts_error_code", "")),
        "forbidden_fallback_attempted": bool(extra.pop("forbidden_fallback_attempted", False)),
        "recent_timing_summary": dict(extra.pop("recent_timing_summary", {})),
        "warning_count": int(warning_count),
        "last_warning": str(last_warning),
        "no_progress_duration_ms": float(no_progress_duration_ms),
    }
    payload.update(extra)
    return payload


def no_progress_payload(
    *,
    phase: str,
    elapsed_ms: float,
    last_completed_position: int,
    last_ipc_send_ms: float = 0.0,
    last_ipc_receive_ms: float = 0.0,
    last_engine_operation: str = "",
    last_engine_operation_ms: float = 0.0,
    last_rust_error_code: str = "",
    last_policy_request_id: str = "",
    last_policy_wait_ms: float = 0.0,
    last_record_writer_operation: str = "",
    last_record_writer_ms: float = 0.0,
    queue_depth: int = 0,
    transport_state: str = "",
    suggested_owner: str = "engine replay/legal",
) -> dict[str, object]:
    return {
        "phase": str(phase),
        "elapsed_ms": float(elapsed_ms),
        "last_completed_position": int(last_completed_position),
        "last_ipc_send_ms": float(last_ipc_send_ms),
        "last_ipc_receive_ms": float(last_ipc_receive_ms),
        "last_engine_operation": str(last_engine_operation),
        "last_engine_operation_ms": float(last_engine_operation_ms),
        "last_rust_error_code": str(last_rust_error_code),
        "last_policy_request_id": str(last_policy_request_id),
        "last_policy_wait_ms": float(last_policy_wait_ms),
        "last_record_writer_operation": str(last_record_writer_operation),
        "last_record_writer_ms": float(last_record_writer_ms),
        "queue_depth": int(queue_depth),
        "transport_state": str(transport_state),
        "suggested_next_subsystem": str(suggested_owner),
    }


def _complete_timings(timings_ms: Mapping[str, float] | None) -> dict[str, float]:
    timings = {key: float(value) for key, value in dict(timings_ms or {}).items()}
    for key in REQUIRED_CONTRACT_TRACE_SPANS:
        timings.setdefault(key, 0.0)
    return timings
