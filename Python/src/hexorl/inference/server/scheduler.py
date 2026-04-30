"""Server-side inference scheduling loop."""

from __future__ import annotations

import asyncio
import time

import numpy as np

from hexorl.inference.protocol import InferenceRequestKind, REQUEST_CODE_TO_KIND
from hexorl.inference.server.batching import BatchingPolicy, ReadyRequest
from hexorl.inference.server.collation import ServerCollator
from hexorl.inference.server.execution import ServerExecutor
from hexorl.inference.server.metrics import ServerMetrics
from hexorl.inference.server.scatter import ServerScatterer


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
            ready_workers = self.drain_ready_workers(max_total=self.max_batch)
            if not ready_workers:
                continue

            build_t0 = time.monotonic()
            if self.is_graph_request(ready_workers):
                collated = self.collator.collate_graph(ready_workers)
            else:
                collated = self.collator.collate_dense(ready_workers)
            self.metrics.total_build_ms += (time.monotonic() - build_t0) * 1000.0

            if collated.total_count <= 0:
                continue

            for worker_id in ready_workers:
                self.queue.get_slot(worker_id).req_ready.clear()

            if collated.is_graph:
                if collated.graph_inputs is None:
                    raise RuntimeError("graph batch missing graph inputs")
                outputs = self.executor.forward_graph(collated.graph_inputs)
                scatter_t0 = time.monotonic()
                self.scatterer.scatter_graph(ready_workers=ready_workers, outputs=outputs)
            else:
                if collated.dense_tensor is None:
                    raise RuntimeError("dense batch missing tensor")
                outputs = self.executor.forward_dense(collated.dense_tensor, collated.sparse_inputs)
                scatter_t0 = time.monotonic()
                self.scatterer.scatter_dense(
                    ready_workers=ready_workers,
                    per_worker_counts=collated.per_worker_counts,
                    outputs=outputs,
                )
            self.metrics.total_scatter_ms += (time.monotonic() - scatter_t0) * 1000.0

            for worker_id in ready_workers:
                self.queue.get_slot(worker_id).res_ready.set()

            self.metrics.record_batch(collated.total_count)

        for line in self.metrics.summary():
            print(line, flush=True)

    def drain_ready_workers(self, max_total: int | None = None) -> list[int]:
        if max_total is None:
            max_total = self.max_batch
        ready: list[ReadyRequest] = []
        for worker_id in range(self.num_workers):
            slot = self.queue.get_slot(worker_id)
            if slot.req_ready.is_set():
                count = int(slot.req_count[0])
                if count > 0:
                    kind_code = int(getattr(slot, "req_kind", np.array([0], dtype=np.uint8))[0])
                    ready.append(ReadyRequest(worker_id=worker_id, count=count, request_kind_code=kind_code))
                else:
                    slot.req_ready.clear()
        return self.batching_policy.select_batch(ready, max_total=max_total).worker_ids

    def is_graph_request(self, ready_workers: list[int]) -> bool:
        if not ready_workers:
            return False
        slot = self.queue.get_slot(ready_workers[0])
        kind = REQUEST_CODE_TO_KIND.get(int(getattr(slot, "req_kind", np.array([0], dtype=np.uint8))[0]))
        return kind in (
            InferenceRequestKind.GLOBAL_GRAPH_POLICY_VALUE,
            InferenceRequestKind.GRAPH_PAIR_POLICY_VALUE,
        )

    def any_worker_ready(self) -> bool:
        for worker_id in range(self.num_workers):
            if self.queue.get_slot(worker_id).req_ready.is_set():
                return True
        return False


__all__ = ["InferenceScheduler"]
