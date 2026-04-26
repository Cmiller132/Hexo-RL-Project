"""GPU inference server — the central process for neural network evaluation.

One server process owns the GPU + model weights. N self-play workers
write leaf-batch tensors into shared-memory request slots; the server
batches across workers, runs one forward pass, and writes results back.

Architecture: §5 of SYSTEM_DESIGN.md — KataGo-style NNEvaluator pattern.
"""

import asyncio
import time
import signal
import multiprocessing as mp
import numpy as np
import torch
from typing import List, Optional, Tuple

from hexorl.config import Config
from hexorl.model.network import from_config, HexNet
from hexorl.inference.shm_queue import (
    InferenceQueue,
    WorkerSlots,
    create_inference_queue,
    connect_inference_queue,
    BOARD_AREA,
    NUM_CHANNELS,
    BOARD_SIZE,
)


class InferenceServer:
    """GPU inference server process.

    Usage:
        server = InferenceServer(cfg, num_workers=24)
        server.start()       # Launches in background process
        ...
        server.stop()        # Signals shutdown
        server.join()        # Waits for graceful exit
    """

    def __init__(self, cfg: Config, num_workers: int):
        self.cfg = cfg
        self.num_workers = num_workers
        self.max_batch = cfg.inference.max_batch_size
        self.max_wait_us = cfg.inference.max_wait_us
        self.fp16 = cfg.inference.fp16

        self._process: Optional[mp.Process] = None
        self._stop_event = mp.Event()
        self._ready_event = mp.Event()
        self._model: Optional[torch.nn.Module] = None
        self._device: Optional[torch.device] = None

        self.n_batches = 0
        self.n_positions = 0
        self.total_forward_ms = 0.0

    def __getstate__(self):
        """Exclude non-picklable attributes for spawn-mode transport."""
        state = self.__dict__.copy()
        for key in ("_owned_queue", "_queue", "_model", "_device", "_process"):
            state.pop(key, None)
        return state

    # ── Process management ────────────────────────────────────────────────

    def start(self):
        """Launch the inference server in a background process.

        Creates shared-memory queue in the main process (owns lifecycle),
        then spawns the child. The child reconnects to the same queue
        via named shared memory — safe in both fork and spawn modes.
        """
        if self._process is not None:
            raise RuntimeError("Server already running")

        # Create queue in main process — this allocates named SharedMemory
        # segments and SharedEvent bytes that persist until unlinked.
        _queue = create_inference_queue(self.num_workers, self.max_batch)
        self._owned_queue = _queue

        self._process = mp.Process(
            target=self._run,
            name="hexorl-inference-server",
            daemon=False,
        )
        self._process.start()
        if not self._ready_event.wait(timeout=30.0):
            raise RuntimeError("Inference server failed to start within 30s")

    def stop(self):
        """Signal the server to shut down gracefully."""
        self._stop_event.set()

    def join(self, timeout: Optional[float] = None):
        """Wait for the server process to exit."""
        if self._process is not None:
            self._process.join(timeout=timeout)
            if self._process.exitcode is None:
                self._process.terminate()
            self._process = None

    def close(self):
        """Release shared-memory resources (main process side)."""
        if hasattr(self, "_owned_queue") and self._owned_queue is not None:
            self._owned_queue.close()
            self._owned_queue = None

    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    # ── Main loop (runs in the child process) ─────────────────────────────

    def _run(self):
        """Server entry point — called in the spawned process."""
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        self._model = from_config(self.cfg, device=self._device)

        # Reconnect to the queue created in start().
        self._queue = connect_inference_queue(self.num_workers, self.max_batch)

        self._ready_event.set()

        try:
            asyncio.run(self._event_loop())
        except Exception as e:
            print(f"[inference-server] Fatal error: {e}", flush=True)
            raise
        finally:
            if self._queue is not None:
                self._queue.close()

    # ── Adaptive batching event loop ──────────────────────────────────────

    async def _event_loop(self):
        """Main asyncio loop — adaptive batching across workers.

        Algorithm (§5.3 of SYSTEM_DESIGN.md):
          while running:
              drain ready requests (up to max_batch total)
              if batch_size > 0:
                  upload → forward → download (with stream pipelining)
                  signal res_ready for drained workers
              else:
                  await any-doorbell with short timeout
        """
        print(f"[inference-server] Started on {self._device}, "
              f"fp16={self.fp16}, max_batch={self.max_batch}, "
              f"workers={self.num_workers}", flush=True)

        loop = asyncio.get_event_loop()

        while not self._stop_event.is_set():
            ready_workers = self._drain_ready_workers()

            if ready_workers:
                batch_tensor, per_worker_counts, total_count = self._build_batch(ready_workers)

                if total_count > 0:
                    policies, values = await loop.run_in_executor(
                        None, self._forward, batch_tensor
                    )

                    self._scatter_results(ready_workers, per_worker_counts, policies, values)

                    for worker_id in ready_workers:
                        slot = self._queue.get_slot(worker_id)
                        slot.req_ready.clear()
                        slot.res_ready.set()

                    self.n_batches += 1
                    self.n_positions += total_count
            else:
                await asyncio.sleep(0.0)
                await asyncio.sleep(self.max_wait_us / 1_000_000.0)

        print(f"[inference-server] Shutting down. "
              f"Batches: {self.n_batches}, Positions: {self.n_positions}", flush=True)

    # ── Worker drain ──────────────────────────────────────────────────────

    def _drain_ready_workers(self) -> List[int]:
        """Collect worker IDs whose req_ready event is set.

        Returns list of ready worker IDs (may be empty).
        """
        ready = []
        for i in range(self.num_workers):
            slot = self._queue.get_slot(i)
            if slot.req_ready.is_set():
                count = int(slot.req_count[0])
                if count > 0:
                    ready.append(i)
                else:
                    slot.req_ready.clear()

        return ready

    # ── Batch building ────────────────────────────────────────────────────

    def _build_batch(
        self, ready_workers: List[int]
    ) -> Tuple[torch.Tensor, List[int], int]:
        """Concatenate tensors from all ready workers into one batch.

        Returns:
            batch_tensor: (total_count, 13, 33, 33) on the correct device.
            per_worker_counts: list of counts per ready worker.
            total_count: sum of all counts.
        """
        tensors = []
        counts = []
        for worker_id in ready_workers:
            slot = self._queue.get_slot(worker_id)
            c = int(slot.req_count[0])
            if c > 0:
                worker_tensor = np.array(slot.req_tensor[:c], copy=True)
                tensors.append(worker_tensor)
                counts.append(c)

        if not tensors:
            return torch.empty(0), [], 0

        batch = np.concatenate(tensors, axis=0)
        total = batch.shape[0]
        batch_tensor = torch.from_numpy(batch).to(self._device)

        return batch_tensor, counts, total

    # ── Forward pass ──────────────────────────────────────────────────────

    def _forward(
        self, batch_tensor: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the model forward pass on a batch.

        Args:
            batch_tensor: (total_count, 13, 33, 33) on device.

        Returns:
            policies: (total_count, BOARD_AREA) float32 numpy array.
            values:   (total_count,) float32 numpy array.
        """
        t0 = time.monotonic()

        with torch.no_grad():
            if self.fp16 and self._device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = self._model(batch_tensor)
            else:
                out = self._model(batch_tensor)

        p = out["policy"].float()
        v = HexNet.bins_to_value(out["value"]).float()

        policies = p.cpu().numpy()
        values = v.cpu().numpy()

        elapsed = (time.monotonic() - t0) * 1000.0
        self.total_forward_ms += elapsed

        return policies, values

    # ── Result scattering ─────────────────────────────────────────────────

    def _scatter_results(
        self,
        ready_workers: List[int],
        per_worker_counts: List[int],
        policies: np.ndarray,
        values: np.ndarray,
    ):
        """Distribute flat policy/value arrays back to per-worker slots."""
        offset = 0
        for worker_id, count in zip(ready_workers, per_worker_counts):
            slot = self._queue.get_slot(worker_id)
            slot.res_policy[:count] = policies[offset:offset + count]
            slot.res_value[:count] = values[offset:offset + count]
            offset += count

    # ── Stats ─────────────────────────────────────────────────────────────

    @property
    def positions_per_sec(self) -> float:
        """Positions processed per second (cumulative)."""
        if self.total_forward_ms <= 0:
            return 0.0
        return self.n_positions / (self.total_forward_ms / 1000.0)
