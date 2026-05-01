"""Inference batching and backpressure helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class InferenceBatchingProfile:
    max_batch_size: int
    max_wait_us: int
    ready_workers: int
    total_positions: int
    backpressure_policy: str = "bounded_max_batch_skip_overflow"

    def to_markdown(self) -> str:
        return (
            f"- max_batch_size: {self.max_batch_size}\n"
            f"- max_wait_us: {self.max_wait_us}\n"
            f"- ready_workers: {self.ready_workers}\n"
            f"- total_positions: {self.total_positions}\n"
            f"- backpressure_policy: {self.backpressure_policy}\n"
        )


def would_exceed_batch(total: int, count: int, max_batch_size: int) -> bool:
    return int(total) + int(count) > int(max_batch_size)


@dataclass(frozen=True)
class ReadyRequest:
    worker_id: int
    count: int
    operation_code: int
    protocol_version: int = 1
    request_schema_version: int = 1
    layout_hash: str = ""
    enqueued_monotonic_s: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class SelectedBatch:
    worker_ids: list[int]
    operation_code: int
    total_positions: int
    queue_depth: int
    high_watermark_hit: bool
    retryable_backpressure: bool
    wait_ms: float


class BatchingPolicy:
    """Testable owner for compatible-kind batching, fairness, and backpressure."""

    def __init__(
        self,
        *,
        max_batch_size: int,
        max_wait_us: int,
        high_watermark: float = 0.9,
        low_watermark: float = 0.5,
    ) -> None:
        self.max_batch_size = int(max_batch_size)
        self.max_wait_us = int(max_wait_us)
        self.high_watermark = float(high_watermark)
        self.low_watermark = float(low_watermark)
        if self.max_batch_size <= 0:
            raise ValueError("batching policy requires positive max_batch_size")
        if not 0.0 < self.low_watermark <= self.high_watermark <= 1.0:
            raise ValueError("batching watermarks must satisfy 0 < low <= high <= 1")

    def select_batch(self, ready: list[ReadyRequest], *, max_total: int | None = None) -> SelectedBatch:
        if not ready:
            return SelectedBatch([], 0, 0, 0, False, False, 0.0)
        limit = self.max_batch_size if max_total is None else min(int(max_total), self.max_batch_size)
        groups: dict[tuple[int, int, int, str], list[ReadyRequest]] = {}
        for request in ready:
            key = (
                int(request.operation_code),
                int(request.protocol_version),
                int(request.request_schema_version),
                str(request.layout_hash),
            )
            groups.setdefault(key, []).append(request)
        selected_group = min(
            groups.values(),
            key=lambda group: min(item.enqueued_monotonic_s for item in group),
        )
        selected: list[ReadyRequest] = []
        total = 0
        retryable = False
        for request in selected_group:
            count = int(request.count)
            if count <= 0:
                continue
            if would_exceed_batch(total, count, limit):
                retryable = True
                if selected:
                    continue
            selected.append(request)
            total += count
            if total >= limit:
                retryable = retryable or len(selected) < len(selected_group)
                break
        oldest = min((item.enqueued_monotonic_s for item in selected), default=time.monotonic())
        fill_rate = total / float(self.max_batch_size)
        return SelectedBatch(
            worker_ids=[item.worker_id for item in selected],
            operation_code=int(selected[0].operation_code) if selected else 0,
            total_positions=int(total),
            queue_depth=sum(int(item.count) for item in ready if int(item.count) > 0),
            high_watermark_hit=fill_rate >= self.high_watermark,
            retryable_backpressure=retryable,
            wait_ms=max(0.0, (time.monotonic() - oldest) * 1000.0),
        )
