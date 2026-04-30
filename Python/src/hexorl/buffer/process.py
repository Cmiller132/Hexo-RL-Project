"""Dedicated replay-buffer owner process.

The training pipeline can use an in-process ``RingBuffer`` for small tests, but
production self-play should isolate buffer ownership. This process accepts
completed games or position batches, computes targets when needed, and serves
sampled records back to consumers.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
from dataclasses import dataclass
from typing import Optional

import numpy as np

from hexorl.buffer.ring import RingBuffer, replay_feature_flags
from hexorl.buffer.targets import process_game_record
from hexorl.config import Config
from hexorl.models.factory import model_uses_global_graph


@dataclass
class BufferRequest:
    op: str
    payload: object = None
    response: object = None


class BufferProcess:
    """Process wrapper that owns a ``RingBuffer`` and serializes access."""

    def __init__(self, cfg: Config, capacity: Optional[int] = None):
        self.cfg = cfg
        self.capacity = capacity or cfg.buffer.capacity
        self._ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        self.requests: mp.Queue = self._ctx.Queue()
        self._process: Optional[mp.Process] = None

    def start(self):
        if self._process is not None:
            raise RuntimeError("BufferProcess already started")
        self._process = self._ctx.Process(target=self._run, name="hexorl-buffer", daemon=False)
        self._process.start()

    def stop(self):
        if self._process is None:
            return
        self.requests.put(BufferRequest("stop"))
        self._process.join(timeout=10.0)
        if self._process.exitcode is None:
            self._process.terminate()
        self._process = None

    def append_game(self, game_record):
        self.requests.put(BufferRequest("append_game", game_record))

    def append_positions(self, positions):
        self.requests.put(BufferRequest("append_positions", list(positions)))

    def sample_records(self, n: int, timeout: float = 10.0):
        parent, child = self._ctx.Pipe(duplex=False)
        self.requests.put(BufferRequest("sample", int(n), child))
        if not parent.poll(timeout):
            raise TimeoutError("Timed out waiting for buffer sample response")
        return parent.recv()

    def stats(self, timeout: float = 10.0) -> dict:
        parent, child = self._ctx.Pipe(duplex=False)
        self.requests.put(BufferRequest("stats", response=child))
        if not parent.poll(timeout):
            raise TimeoutError("Timed out waiting for buffer stats response")
        return parent.recv()

    def _run(self):
        ring = RingBuffer(
            capacity=self.capacity,
            max_policy_entries=self.cfg.selfplay.policy_target_top_k,
            max_policy_v2_entries=min(
                max(self.cfg.selfplay.policy_target_top_k, self.cfg.model.candidate_budget),
                512,
            ),
            recency_decay=self.cfg.buffer.recency_decay,
            num_lookahead=len(self.cfg.buffer.lookahead_horizons),
            **replay_feature_flags(
                self.cfg.model.heads,
                architecture=self.cfg.model.architecture,
                sparse_policy=self.cfg.model.sparse_policy,
                graph=model_uses_global_graph(self.cfg),
            ),
        )

        while True:
            try:
                req: BufferRequest = self.requests.get(timeout=0.5)
            except queue.Empty:
                continue

            if req.op == "stop":
                break
            if req.op == "append_game":
                process_game_record(
                    req.payload,
                    lookahead_horizons=self.cfg.buffer.lookahead_horizons,
                    lookahead_lambdas=self.cfg.buffer.lookahead_lambdas,
                )
                ring.extend(req.payload.positions)
            elif req.op == "append_positions":
                ring.extend(req.payload)
            elif req.op == "sample":
                indices = ring.sample_indices(
                    int(req.payload),
                    recency_decay=self.cfg.buffer.recency_decay,
                    pcr_weight=self.cfg.buffer.pcr_weight,
                )
                records = ring.get_batch(np.asarray(indices, dtype=np.int64))
                if req.response is not None:
                    req.response.send(records)
            elif req.op == "stats" and req.response is not None:
                req.response.send(ring.stats)
