"""Server-side inference scheduling loop."""

from __future__ import annotations

import asyncio
import time

from hexorl.inference.server.batching import BatchingPolicy, ReadyRequest
from hexorl.inference.server.collation import ServerCollator
from hexorl.inference.server.execution import ServerExecutor
from hexorl.inference.server.metrics import ServerMetrics
from hexorl.inference.server.scatter import ServerScatterer
from hexorl.inference.control import (
    CTL_ENQUEUED_NS,
    CTL_LAYOUT_HASH,
    CTL_OPCODE,
    CTL_STATUS,
    STATUS_READY,
    read_all_dyn_dims,
)


class InferenceScheduler:
    def __init__(
        self,
        *,
        queue,
        num_workers: int,
        max_batch: int,
        max_wait_us: int,
        stop_event,
        batching_policy: BatchingPolicy,
        collator: ServerCollator,
        executor: ServerExecutor,
        scatterer: ServerScatterer,
        metrics: ServerMetrics,
        weight_poll,
        device,
        fp16: bool,
        manifest,
    ) -> None:
        self.queue = queue
        self.num_workers = int(num_workers)
        self.max_batch = int(max_batch)
        self.max_wait_us = int(max_wait_us)
        self.stop_event = stop_event
        self.batching_policy = batching_policy
        self.collator = collator
        self.executor = executor
        self.scatterer = scatterer
        self.metrics = metrics
        self.weight_poll = weight_poll
        self.device = device
        self.fp16 = bool(fp16)
        self.manifest = manifest

    async def run(self) -> None:
        print(
            f"[inference-server] Started on {self.device}, "
            f"fp16={self.fp16}, max_batch={self.max_batch}, "
            f"workers={self.num_workers}",
            flush=True,
        )
        wait_s = self.max_wait_us / 1_000_000.0
        while not self.stop_event.is_set():
            self.weight_poll()
            if not self.any_worker_ready():
                await asyncio.sleep(wait_s)
                continue
            await asyncio.sleep(wait_s)
            selected = self.select_ready_batch(max_total=self.max_batch)
            ready_workers = selected.worker_ids
            if not ready_workers:
                continue
            operation_name = self.manifest.operation_name_for_code(selected.operation_code)
            build_t0 = time.monotonic()
            collated = self.collator.collate(ready_workers, operation_name)
            self.metrics.total_build_ms += (time.monotonic() - build_t0) * 1000.0
            if collated.total_count <= 0:
                continue
            for worker_id in ready_workers:
                self.queue.get_slot(worker_id).req_ready.clear()
            outputs = self.executor.forward(collated)
            scatter_t0 = time.monotonic()
            self.scatterer.scatter(collated=collated, outputs=outputs)
            self.metrics.total_scatter_ms += (time.monotonic() - scatter_t0) * 1000.0
            for worker_id in ready_workers:
                self.queue.get_slot(worker_id).res_ready.set()
            self.metrics.record_batch(collated.total_count)
        for line in self.metrics.summary():
            print(line, flush=True)

    def select_ready_batch(self, max_total: int | None = None):
        ready: list[ReadyRequest] = []
        for worker_id in range(self.num_workers):
            slot = self.queue.get_slot(worker_id)
            if slot.req_ready.is_set() and int(slot.control[CTL_STATUS]) == STATUS_READY:
                dims = read_all_dyn_dims(slot.control, ("B",))
                count = int(dims.get("B", 1))
                if count > 0:
                    enqueued_ns = int(slot.control[CTL_ENQUEUED_NS])
                    ready.append(
                        ReadyRequest(
                            worker_id=worker_id,
                            count=count,
                            operation_code=int(slot.control[CTL_OPCODE]),
                            layout_hash=str(int(slot.control[CTL_LAYOUT_HASH])),
                            enqueued_monotonic_s=enqueued_ns / 1_000_000_000 if enqueued_ns else time.monotonic(),
                        )
                    )
        return self.batching_policy.select_batch(ready, max_total=max_total or self.max_batch)

    def any_worker_ready(self) -> bool:
        for worker_id in range(self.num_workers):
            slot = self.queue.get_slot(worker_id)
            if slot.req_ready.is_set() and int(slot.control[CTL_STATUS]) == STATUS_READY:
                return True
        return False


__all__ = ["InferenceScheduler"]
