"""Replay record writing boundary for self-play."""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Any

from hexorl.contracts.identity import stable_digest
from hexorl.contracts.validation import ContractValidationError
from hexorl.selfplay.records import COMPACT_VERSION_V2, GameRecord
from hexorl.selfplay.telemetry import SelfPlayTelemetrySink


@dataclass(frozen=True)
class RecordWriteResult:
    ok: bool
    game_id: int
    positions_written: int
    record_hash: str
    schema_version: int
    elapsed_ms: float
    error: str = ""
    backpressure_events: int = 0


class SelfPlayRecordWriter:
    """Validates and writes complete self-play records outside the worker."""

    def write(self, record: GameRecord, *, run_request) -> RecordWriteResult:
        raise NotImplementedError


class QueueSelfPlayRecordWriter(SelfPlayRecordWriter):
    def __init__(
        self,
        output_queue,
        *,
        lookahead_horizons: list[int],
        lookahead_lambdas: list[float],
        telemetry_sink: SelfPlayTelemetrySink,
        rgsc_service: Any = None,
        put_timeout_s: float = 0.5,
        max_backpressure_events: int = 60,
    ) -> None:
        self.output_queue = output_queue
        self.lookahead_horizons = list(lookahead_horizons)
        self.lookahead_lambdas = list(lookahead_lambdas)
        self.telemetry_sink = telemetry_sink
        self.rgsc_service = rgsc_service
        self.put_timeout_s = float(put_timeout_s)
        self.max_backpressure_events = int(max_backpressure_events)
        self.last_operation = "idle"
        self.last_elapsed_ms = 0.0

    def write(self, record: GameRecord, *, run_request) -> RecordWriteResult:
        t0 = time.monotonic()
        self.last_operation = "validate"
        try:
            validate_game_record(record)
            if self.rgsc_service is not None:
                restart_idx = getattr(record, "rgsc_restart_entry_index", None)
                refreshes_before = int(getattr(self.rgsc_service, "refreshes", 0))
                inserted = self.rgsc_service.observe_game(
                    record,
                    restart_entry_index=restart_idx,
                )
                record.rgsc_prb_inserted = bool(inserted)
                record.rgsc_metrics = {
                    "rgsc_prb_size": float(len(self.rgsc_service.prb)),
                    "rgsc_restart_attempts": 1.0 if getattr(record, "rgsc_restart_attempted", False) else 0.0,
                    "rgsc_restart_successes": 1.0 if getattr(record, "rgsc_restart_used", False) else 0.0,
                    "rgsc_restart_rejections": 1.0
                    if getattr(record, "rgsc_restart_attempted", False) and not getattr(record, "rgsc_restart_used", False)
                    else 0.0,
                    "rgsc_prb_insertions": 1.0 if inserted else 0.0,
                    "rgsc_prb_refreshes": float(int(getattr(self.rgsc_service, "refreshes", 0)) - refreshes_before),
                    "rgsc_last_ema_delta": float(getattr(self.rgsc_service, "last_ema_delta", 0.0)),
                    "rgsc_last_staleness": float(getattr(self.rgsc_service, "last_staleness", 0.0)),
                    "rgsc_tree_node_insertions": float(getattr(record, "rgsc_tree_node_insertions", 0)),
                }
                record.rgsc_prb_snapshot = self.rgsc_service.snapshot_entries()

            self.last_operation = "queue_put"
            backpressure_events = 0
            while True:
                try:
                    self.output_queue.put(record, timeout=self.put_timeout_s)
                    break
                except queue.Full:
                    backpressure_events += 1
                    self.telemetry_sink.emit(
                        "selfplay_backpressure",
                        {
                            "source": "record_writer",
                            "game_id": int(record.game_id),
                            "queue_depth": _queue_size(self.output_queue),
                            "backpressure_events": backpressure_events,
                            "bounded": True,
                        },
                    )
                    if backpressure_events >= self.max_backpressure_events:
                        raise ContractValidationError(
                            "record writer queue remained full past bounded retry budget",
                            owner="record writer/replay encoding",
                        )
            elapsed = (time.monotonic() - t0) * 1000.0
            self.last_elapsed_ms = elapsed
            result = RecordWriteResult(
                ok=True,
                game_id=int(record.game_id),
                positions_written=len(record.positions),
                record_hash=record_hash(record),
                schema_version=COMPACT_VERSION_V2,
                elapsed_ms=elapsed,
                backpressure_events=backpressure_events,
            )
            return result
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000.0
            self.last_elapsed_ms = elapsed
            self.telemetry_sink.emit(
                "contract_validation_failure",
                {
                    "owner_subsystem": "record writer/replay encoding",
                    "game_id": int(getattr(record, "game_id", -1)),
                    "error": str(exc),
                },
            )
            return RecordWriteResult(
                ok=False,
                game_id=int(getattr(record, "game_id", -1)),
                positions_written=0,
                record_hash="",
                schema_version=COMPACT_VERSION_V2,
                elapsed_ms=elapsed,
                error=str(exc),
            )


class InMemorySelfPlayRecordWriter(SelfPlayRecordWriter):
    def __init__(self, *, telemetry_sink: SelfPlayTelemetrySink) -> None:
        self.telemetry_sink = telemetry_sink
        self.records: list[GameRecord] = []
        self.last_operation = "idle"
        self.last_elapsed_ms = 0.0

    def write(self, record: GameRecord, *, run_request) -> RecordWriteResult:
        t0 = time.monotonic()
        self.last_operation = "memory_write"
        validate_game_record(record)
        self.records.append(record)
        elapsed = (time.monotonic() - t0) * 1000.0
        self.last_elapsed_ms = elapsed
        return RecordWriteResult(
            ok=True,
            game_id=int(record.game_id),
            positions_written=len(record.positions),
            record_hash=record_hash(record),
            schema_version=COMPACT_VERSION_V2,
            elapsed_ms=elapsed,
        )


def validate_game_record(record: GameRecord) -> None:
    if not isinstance(record, GameRecord):
        raise ContractValidationError("SelfPlayRecordWriter accepts only GameRecord", owner="record writer/replay encoding")
    if int(record.game_id) < 0:
        raise ContractValidationError("GameRecord game_id must be non-negative", owner="record writer/replay encoding")
    if int(record.game_length) != len(record.positions):
        raise ContractValidationError("GameRecord game_length does not match positions", owner="record writer/replay encoding")
    for index, pos in enumerate(record.positions):
        if int(pos.game_id) != int(record.game_id):
            raise ContractValidationError("PositionRecord game_id disagrees with GameRecord", owner="record writer/replay encoding")
        if int(pos.turn_index) != index and int(pos.turn_index) < 0:
            raise ContractValidationError("PositionRecord turn_index is invalid", owner="record writer/replay encoding")
        if pos.outcome is None:
            raise ContractValidationError("PositionRecord outcome must be assigned before write", owner="record writer/replay encoding")
        if not isinstance(pos.move_history, (bytes, bytearray)):
            raise ContractValidationError("PositionRecord move_history must be bytes", owner="record writer/replay encoding")


def record_hash(record: GameRecord) -> str:
    payload = (
        "GameRecord",
        COMPACT_VERSION_V2,
        int(record.game_id),
        float(record.outcome),
        tuple(
            (
                bytes(pos.move_history),
                tuple(pos.policy_target_v2),
                tuple(pos.pair_policy_target_v2),
                float(pos.root_value),
                int(pos.player),
                int(pos.turn_index),
            )
            for pos in record.positions
        ),
        bytes(record.final_move_history),
        str(record.terminal_reason),
    )
    return stable_digest(payload)


def _queue_size(output_queue) -> int:
    try:
        return int(output_queue.qsize())
    except Exception:
        return -1
