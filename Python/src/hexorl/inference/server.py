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
from hexorl.runtime import configure_torch_runtime
from hexorl.inference.shm_queue import (
    create_inference_queue,
    connect_inference_queue,
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

    def __init__(
        self,
        cfg: Config,
        num_workers: int,
        initial_state_dict: Optional[dict] = None,
    ):
        self.cfg = cfg
        self.num_workers = num_workers
        self.max_batch = cfg.inference.max_batch_size
        self.max_wait_us = cfg.inference.max_wait_us
        self.fp16 = cfg.inference.fp16

        self._mp_ctx = mp.get_context("spawn")
        self._process: Optional[mp.Process] = None
        self._stop_event = self._mp_ctx.Event()
        self._ready_event = self._mp_ctx.Event()
        self._weight_queue = self._mp_ctx.Queue(maxsize=2)
        self._initial_state_dict = self._state_to_cpu(initial_state_dict)
        self._model: Optional[torch.nn.Module] = None
        self._device: Optional[torch.device] = None
        self._forward_stream = None
        self._host_batch_tensor: Optional[torch.Tensor] = None
        self._host_batch_np: Optional[np.ndarray] = None

        self.n_batches = 0
        self.n_positions = 0
        self.total_forward_ms = 0.0

    @staticmethod
    def _state_to_cpu(state_dict: Optional[dict]) -> Optional[dict]:
        if state_dict is None:
            return None
        if state_dict and all(str(k).startswith("_orig_mod.") for k in state_dict):
            state_dict = {
                str(k).removeprefix("_orig_mod."): v
                for k, v in state_dict.items()
            }
        return {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in state_dict.items()
        }

    def __getstate__(self):
        """Exclude non-picklable attributes for spawn-mode transport."""
        state = self.__dict__.copy()
        for key in ("_owned_queue", "_queue", "_model", "_device", "_process", "_mp_ctx"):
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

        self._process = self._mp_ctx.Process(
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
        self.close()

    def close(self):
        """Release shared-memory resources (main process side)."""
        if hasattr(self, "_owned_queue") and self._owned_queue is not None:
            self._owned_queue.close()
            self._owned_queue = None

    def update_weights(self, state_dict: dict):
        """Queue a model state update for hot-swap in the inference process.

        Tensors are moved to CPU before crossing the process boundary. The child
        process loads them onto its inference device between batches.
        """
        cpu_state = self._state_to_cpu(state_dict)
        if cpu_state is None:
            return
        while self._weight_queue.full():
            try:
                self._weight_queue.get_nowait()
            except Exception:
                break
        self._weight_queue.put(cpu_state)

    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    # ── Main loop (runs in the child process) ─────────────────────────────

    def _run(self):
        """Server entry point — called in the spawned process."""
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        configure_torch_runtime(self.cfg)

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        self._model = from_config(self.cfg, device=self._device)
        if self._device.type == "cuda" and getattr(self.cfg.runtime, "channels_last", True):
            self._model = self._model.to(memory_format=torch.channels_last)
        compile_inference = getattr(self.cfg.runtime, "compile_inference", None)
        if compile_inference is None:
            compile_inference = getattr(self.cfg.runtime, "compile_model", False)
        if self._device.type == "cuda" and compile_inference:
            try:
                self._model = torch.compile(
                    self._model,
                    mode=getattr(self.cfg.runtime, "compile_mode", "reduce-overhead"),
                )
            except Exception as exc:
                print(f"[inference-server] torch.compile disabled: {exc}", flush=True)
        if self._initial_state_dict is not None:
            initial = {
                k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                for k, v in self._initial_state_dict.items()
            }
            self._model.load_state_dict(initial, strict=False)
            self._model.eval()
        if self._device.type == "cuda":
            self._forward_stream = torch.cuda.Stream(priority=-1)

        # Reconnect to the queue created in start().
        self._queue = connect_inference_queue(self.num_workers, self.max_batch)
        self._prepare_host_batch()

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

        wait_s = self.max_wait_us / 1_000_000.0

        while not self._stop_event.is_set():
            self._poll_weight_updates()
            # Poll until at least one worker is ready, then wait for more.
            if not self._any_worker_ready():
                await asyncio.sleep(wait_s)
                continue

            # At least one worker ready — wait max_wait_us for more to arrive.
            await asyncio.sleep(wait_s)

            ready_workers = self._drain_ready_workers(max_total=self.max_batch)
            if not ready_workers:
                continue

            batch_tensor, per_worker_counts, total_count = self._build_batch(ready_workers)

            if total_count > 0:
                # Clear req_ready before the forward pass (P1-1 fix).
                for worker_id in ready_workers:
                    self._queue.get_slot(worker_id).req_ready.clear()

                policies, values = self._forward(batch_tensor)

                self._scatter_results(ready_workers, per_worker_counts, policies, values)

                for worker_id in ready_workers:
                    self._queue.get_slot(worker_id).res_ready.set()

                self.n_batches += 1
                self.n_positions += total_count

        print(f"[inference-server] Shutting down. "
              f"Batches: {self.n_batches}, Positions: {self.n_positions}", flush=True)

    # ── Worker drain ──────────────────────────────────────────────────────

    def _drain_ready_workers(self, max_total: Optional[int] = None) -> List[int]:
        """Collect worker IDs whose req_ready event is set.

        Stops accumulating when the cumulative position count reaches max_total.
        Returns list of ready worker IDs (may be empty).
        """
        if max_total is None:
            max_total = self.max_batch

        ready = []
        total = 0
        for i in range(self.num_workers):
            if total >= max_total:
                break
            slot = self._queue.get_slot(i)
            if slot.req_ready.is_set():
                count = int(slot.req_count[0])
                if count > 0:
                    if total + count > max_total and ready:
                        continue
                    ready.append(i)
                    total += count
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
        counts = []
        total = 0
        for worker_id in ready_workers:
            slot = self._queue.get_slot(worker_id)
            c = int(slot.req_count[0])
            if c > 0:
                if self._host_batch_np is None:
                    raise RuntimeError("host batch buffer was not initialized")
                np.copyto(self._host_batch_np[total:total + c], slot.req_tensor[:c])
                total += c
                counts.append(c)

        if total == 0:
            return torch.empty(0), [], 0

        batch_tensor = self._host_batch_tensor[:total].to(self._device, non_blocking=True)
        if self._device.type == "cuda" and getattr(self.cfg.runtime, "channels_last", True):
            batch_tensor = batch_tensor.contiguous(memory_format=torch.channels_last)

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

        with torch.inference_mode():
            if self._device.type == "cuda" and self._forward_stream is not None:
                with torch.cuda.stream(self._forward_stream):
                    if self.fp16:
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            out = self._model(batch_tensor)
                    else:
                        out = self._model(batch_tensor)
                self._forward_stream.synchronize()
            elif self.fp16 and self._device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = self._model(batch_tensor)
            else:
                out = self._model(batch_tensor)

        p = self._sanitize_policy_logits(out["policy"])
        value_logits = self._sanitize_value_logits(out["value"])
        v = torch.nan_to_num(
            HexNet.bins_to_value(value_logits).float(),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp_(-1.0, 1.0)

        policies = p.cpu().numpy()
        values = v.cpu().numpy()

        elapsed = (time.monotonic() - t0) * 1000.0
        self.total_forward_ms += elapsed

        return policies, values

    @staticmethod
    def _sanitize_policy_logits(logits: torch.Tensor) -> torch.Tensor:
        """Keep policy logits finite before handing them to Rust MCTS."""
        return torch.nan_to_num(
            logits.float(),
            nan=0.0,
            posinf=80.0,
            neginf=-80.0,
        ).clamp_(-80.0, 80.0)

    @staticmethod
    def _sanitize_value_logits(logits: torch.Tensor) -> torch.Tensor:
        """Keep value logits finite so softmax cannot produce NaNs."""
        return torch.nan_to_num(
            logits.float(),
            nan=0.0,
            posinf=80.0,
            neginf=-80.0,
        ).clamp_(-80.0, 80.0)

    def _prepare_host_batch(self):
        """Allocate a reusable staging buffer for request gathering."""
        pin = self._device is not None and self._device.type == "cuda"
        try:
            self._host_batch_tensor = torch.empty(
                (self.max_batch, 13, 33, 33),
                dtype=torch.float32,
                pin_memory=pin,
            )
            self._host_batch_np = self._host_batch_tensor.numpy()
        except Exception:
            self._host_batch_np = np.empty((self.max_batch, 13, 33, 33), dtype=np.float32)
            self._host_batch_tensor = torch.from_numpy(self._host_batch_np)

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

    def _any_worker_ready(self) -> bool:
        """Return True if at least one worker slot has req_ready set."""
        for i in range(self.num_workers):
            if self._queue.get_slot(i).req_ready.is_set():
                return True
        return False

    def _poll_weight_updates(self):
        """Apply the newest queued model weights, dropping stale updates."""
        latest = None
        while True:
            try:
                latest = self._weight_queue.get_nowait()
            except Exception:
                break
        if latest is not None and self._model is not None:
            latest = {
                k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                for k, v in latest.items()
            }
            self._model.load_state_dict(latest, strict=False)
            self._model.eval()

    @property
    def positions_per_sec(self) -> float:
        """Positions processed per second (cumulative)."""
        if self.total_forward_ms <= 0:
            return 0.0
        return self.n_positions / (self.total_forward_ms / 1000.0)
