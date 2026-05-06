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
from hexorl.inference.adapters import (
    decode_dense_outputs,
    decode_global_graph_outputs,
    sanitize_policy_logits,
    sanitize_value_logits,
)
from hexorl.inference.protocol import (
    GRAPH_HEAD_OPP,
    GRAPH_HEAD_PAIR_FIRST,
    GRAPH_HEAD_PAIR_JOINT,
    GRAPH_HEAD_PAIR_SECOND,
    GRAPH_HEAD_REGRET,
)
from hexorl.models.assembly import bins_to_value, from_config, load_model_state
from hexorl.models.registry import resolve_model_spec
from hexorl.runtime import configure_torch_runtime
from hexorl.inference.shm_queue import (
    CANDIDATE_FEATURES,
    GRAPH_SCHEMA_VERSION,
    MAX_GRAPH_ACTIONS,
    MAX_GRAPH_PAIRS,
    MAX_GRAPH_TOKENS,
    MAX_PAIR_CANDIDATES,
    RELATION_SCHEMA_VERSION,
    create_inference_queue,
    connect_inference_queue,
)


def _graph_pair_request_is_second_placement(graph_inputs: dict[str, torch.Tensor]) -> bool:
    first = graph_inputs.get("pair_first_indices")
    legal = graph_inputs.get("legal_token_indices")
    if first is None or legal is None or first.numel() == 0 or legal.numel() == 0:
        return False
    valid_first = first[first >= 0]
    if valid_first.numel() == 0:
        return False
    return not bool(torch.isin(valid_first, legal[legal >= 0]).all().item())


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
        resolved = resolve_model_spec(cfg)
        self._global_graph_mode = resolved.global_graph
        missing_heads = sorted(set(resolved.selfplay_required_outputs) - set(resolved.outputs))
        if missing_heads:
            raise ValueError(
                "InferenceServer requires model heads for self-play inference: "
                f"{missing_heads}"
            )
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
        self.total_build_ms = 0.0
        self.total_forward_ms = 0.0
        self.total_model_ms = 0.0
        self.total_postprocess_ms = 0.0
        self.total_download_ms = 0.0
        self.total_scatter_ms = 0.0
        self.min_batch = 0
        self.max_batch_seen = 0

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
        timeout_s = max(1.0, float(getattr(self.cfg.runtime, "inference_start_timeout_s", 30.0)))
        if not self._ready_event.wait(timeout=timeout_s):
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5.0)
            self.close()
            self._process = None
            raise RuntimeError(f"Inference server failed to start within {timeout_s:g}s")

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
            load_model_state(self._model, initial, allow_partial=False)
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

            graph_request = self._ready_workers_graph_mode(ready_workers)
            if graph_request is not None:
                ready_workers = graph_request

            build_t0 = time.monotonic()
            if self._is_graph_request(ready_workers):
                graph_inputs, graph_row_sources, total_count = self._build_graph_inputs(ready_workers)
                batch_tensor = None
                sparse_inputs = None
            else:
                batch_tensor, per_worker_counts, total_count = self._build_batch(ready_workers)
                graph_inputs = None
                sparse_inputs = self._build_sparse_inputs(ready_workers, per_worker_counts, total_count)
            self.total_build_ms += (time.monotonic() - build_t0) * 1000.0

            if total_count > 0:
                # Clear req_ready before the forward pass (P1-1 fix).
                for worker_id in ready_workers:
                    self._queue.get_slot(worker_id).req_ready.clear()

                if graph_inputs is not None:
                    (
                        graph_place,
                        values,
                        graph_opp,
                        graph_pair_first,
                        graph_pair_joint,
                        graph_pair_second,
                        graph_regret,
                    ) = self._forward_graph(graph_inputs)
                    sparse_logits = pair_logits = policies = None
                else:
                    policies, values, sparse_logits, pair_logits, regret_rank = self._forward(batch_tensor, sparse_inputs)
                    graph_place = graph_opp = graph_pair_first = graph_pair_joint = graph_pair_second = None

                scatter_t0 = time.monotonic()
                if graph_inputs is not None:
                    self._scatter_graph_results(
                        graph_row_sources,
                        values,
                        graph_place,
                        graph_opp,
                        graph_pair_first,
                        graph_pair_joint,
                        graph_pair_second,
                        graph_regret,
                    )
                else:
                    self._scatter_results(
                        ready_workers,
                        per_worker_counts,
                        policies,
                        values,
                        sparse_logits,
                        pair_logits,
                        regret_rank,
                    )
                self.total_scatter_ms += (time.monotonic() - scatter_t0) * 1000.0

                for worker_id in ready_workers:
                    self._queue.get_slot(worker_id).res_ready.set()

                self.n_batches += 1
                self.n_positions += total_count
                self.min_batch = (
                    total_count
                    if self.min_batch == 0
                    else min(self.min_batch, total_count)
                )
                self.max_batch_seen = max(self.max_batch_seen, total_count)

        avg_batch = self.n_positions / max(self.n_batches, 1)
        print(
            "[inference-server] Shutting down. "
            f"Batches: {self.n_batches}, Positions: {self.n_positions}, "
            f"Avg batch: {avg_batch:.1f}, Min batch: {self.min_batch}, "
            f"Max batch: {self.max_batch_seen}",
            flush=True,
        )
        print(
            "[inference-server] Timing ms total: "
            f"build={self.total_build_ms:.1f}, "
            f"forward={self.total_forward_ms:.1f}, "
            f"model={self.total_model_ms:.1f}, "
            f"postprocess={self.total_postprocess_ms:.1f}, "
            f"download={self.total_download_ms:.1f}, "
            f"scatter={self.total_scatter_ms:.1f}",
            flush=True,
        )

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

    def _is_graph_request(self, ready_workers: List[int]) -> bool:
        if not ready_workers:
            return False
        slot = self._queue.get_slot(ready_workers[0])
        return bool(getattr(slot, "req_mode", np.array([0], dtype=np.uint8))[0] == 1)

    def _ready_workers_graph_mode(self, ready_workers: List[int]) -> Optional[List[int]]:
        """Keep a drained batch homogeneous when graph and dense workers race."""
        modes = [
            int(getattr(self._queue.get_slot(worker_id), "req_mode", np.array([0], dtype=np.uint8))[0])
            for worker_id in ready_workers
        ]
        if not modes or all(mode == modes[0] for mode in modes):
            return None
        target = modes[0]
        return [
            worker_id
            for worker_id, mode in zip(ready_workers, modes)
            if int(mode) == int(target)
        ]

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

    def _build_graph_inputs(
        self,
        ready_workers: List[int],
    ) -> Tuple[dict[str, torch.Tensor], List[tuple[int, int]], int]:
        if not self._global_graph_mode:
            raise RuntimeError("graph IPC request received by a non-global-graph inference server")
        row_meta: list[tuple[int, int, np.ndarray]] = []
        for worker_id in ready_workers:
            slot = self._queue.get_slot(worker_id)
            request_count = int(slot.req_count[0])
            if request_count <= 0:
                continue
            if request_count > int(getattr(slot, "req_graph_batch_meta").shape[0]):
                raise ValueError("graph IPC request count exceeds per-worker graph batch capacity")
            meta = np.array(slot.req_graph_meta, copy=True)
            if int(meta[0]) != GRAPH_SCHEMA_VERSION or int(meta[1]) != RELATION_SCHEMA_VERSION:
                raise ValueError(
                    "graph IPC schema mismatch: "
                    f"got ({int(meta[0])}, {int(meta[1])}), "
                    f"expected ({GRAPH_SCHEMA_VERSION}, {RELATION_SCHEMA_VERSION})"
                )
            token_count, legal_count, opp_count, pair_count = map(int, meta[2:6])
            if token_count > MAX_GRAPH_TOKENS or legal_count > MAX_GRAPH_ACTIONS:
                raise ValueError("graph request exceeds shared-memory token/legal capacity")
            if opp_count > MAX_GRAPH_ACTIONS or pair_count > MAX_GRAPH_PAIRS:
                raise ValueError("graph request exceeds shared-memory opponent/pair capacity")
            batch_meta = np.array(slot.req_graph_batch_meta[:request_count], copy=True)
            if request_count == 1 and not batch_meta[0].any():
                batch_meta[0] = (token_count, legal_count, opp_count, pair_count, 0, 0, 0, 0)
            for local_row in range(request_count):
                row_meta.append((worker_id, local_row, batch_meta[local_row]))
        total = len(row_meta)
        if total == 0:
            return {}, [], 0
        max_t = max(int(meta[0]) for _worker_id, _local_row, meta in row_meta)
        max_a = max(int(meta[1]) for _worker_id, _local_row, meta in row_meta)
        max_o = max(int(meta[2]) for _worker_id, _local_row, meta in row_meta)
        max_p = max(int(meta[3]) for _worker_id, _local_row, meta in row_meta)

        token_features = np.zeros((total, max_t, self._queue.get_slot(ready_workers[0]).req_graph_token_features.shape[1]), dtype=np.float32)
        token_type = np.zeros((total, max_t), dtype=np.int64)
        token_qr = np.zeros((total, max_t, 2), dtype=np.int32)
        token_mask = np.zeros((total, max_t), dtype=np.bool_)
        legal_token_indices = np.full((total, max_a), -1, dtype=np.int64)
        legal_mask = np.zeros((total, max_a), dtype=np.bool_)
        opp_legal_qr = np.zeros((total, max_o, 2), dtype=np.int32)
        opp_legal_mask = np.zeros((total, max_o), dtype=np.bool_)
        pair_token_indices = np.full((total, max_p), -1, dtype=np.int64)
        pair_first_indices = np.full((total, max_p), -1, dtype=np.int64)
        pair_second_indices = np.full((total, max_p), -1, dtype=np.int64)
        relation_type = np.zeros((total, max_t, max_t), dtype=np.int16)
        relation_bias = np.zeros((total, 1, max_t, max_t), dtype=np.float32)

        row_sources: list[tuple[int, int]] = []
        for row, (worker_id, local_row, meta) in enumerate(row_meta):
            slot = self._queue.get_slot(worker_id)
            t, a, o, p, token_off, legal_off, opp_off, pair_off = map(int, meta[:8])
            if token_off + t > MAX_GRAPH_TOKENS or legal_off + a > MAX_GRAPH_ACTIONS:
                raise ValueError("graph batch row exceeds packed token/legal capacity")
            if opp_off + o > MAX_GRAPH_ACTIONS or pair_off + p > MAX_GRAPH_PAIRS:
                raise ValueError("graph batch row exceeds packed opponent/pair capacity")
            token_slice = slice(token_off, token_off + t)
            legal_slice = slice(legal_off, legal_off + a)
            opp_slice = slice(opp_off, opp_off + o)
            pair_slice = slice(pair_off, pair_off + p)
            token_features[row, :t] = slot.req_graph_token_features[token_slice]
            token_type[row, :t] = slot.req_graph_token_type[token_slice].astype(np.int64)
            token_qr[row, :t] = slot.req_graph_token_qr[token_slice]
            token_mask[row, :t] = slot.req_graph_token_mask[token_slice].astype(bool)
            legal_token_indices[row, :a] = slot.req_graph_legal_token_indices[legal_slice]
            legal_mask[row, :a] = slot.req_graph_legal_mask[legal_slice].astype(bool)
            if o:
                opp_legal_qr[row, :o] = slot.req_graph_opp_legal_qr[opp_slice]
                opp_legal_mask[row, :o] = slot.req_graph_opp_legal_mask[opp_slice].astype(bool)
            if p:
                pair_token_indices[row, :p] = slot.req_graph_pair_token_indices[pair_slice]
                pair_first_indices[row, :p] = slot.req_graph_pair_first_indices[pair_slice]
                pair_second_indices[row, :p] = slot.req_graph_pair_second_indices[pair_slice]
            relation_type[row, :t, :t] = slot.req_graph_relation_type[token_slice, token_slice]
            relation_bias[row, :, :t, :t] = slot.req_graph_relation_bias[:, token_slice, token_slice]
            row_sources.append((worker_id, local_row))

        return (
            {
                "token_features": torch.from_numpy(token_features).to(self._device, non_blocking=True),
                "token_type": torch.from_numpy(token_type).to(self._device, non_blocking=True),
                "token_qr": torch.from_numpy(token_qr).to(self._device, non_blocking=True),
                "token_mask": torch.from_numpy(token_mask).to(self._device, non_blocking=True),
                "legal_token_indices": torch.from_numpy(legal_token_indices).to(self._device, non_blocking=True),
                "legal_mask": torch.from_numpy(legal_mask).to(self._device, non_blocking=True),
                "opp_legal_qr": torch.from_numpy(opp_legal_qr).to(self._device, non_blocking=True),
                "opp_legal_mask": torch.from_numpy(opp_legal_mask).to(self._device, non_blocking=True),
                "pair_token_indices": torch.from_numpy(pair_token_indices).to(self._device, non_blocking=True),
                "pair_first_indices": torch.from_numpy(pair_first_indices).to(self._device, non_blocking=True),
                "pair_second_indices": torch.from_numpy(pair_second_indices).to(self._device, non_blocking=True),
                "relation_type": torch.from_numpy(relation_type).to(self._device, non_blocking=True),
                "relation_bias": torch.from_numpy(relation_bias).to(self._device, non_blocking=True),
            },
            row_sources,
            total,
        )

    def _build_sparse_inputs(
        self,
        ready_workers: List[int],
        per_worker_counts: List[int],
        total_count: int,
    ) -> Optional[dict[str, torch.Tensor]]:
        if total_count <= 0:
            return None
        max_k = 0
        for worker_id, count in zip(ready_workers, per_worker_counts):
            slot = self._queue.get_slot(worker_id)
            if count > 0:
                counts = slot.req_candidate_count[:count]
                max_k = max(max_k, int(counts.max()) if counts.size else 0)
        max_p = 0
        for worker_id, count in zip(ready_workers, per_worker_counts):
            slot = self._queue.get_slot(worker_id)
            if count > 0 and getattr(slot, "req_pair_count", None) is not None:
                counts = slot.req_pair_count[:count]
                max_p = max(max_p, int(counts.max()) if counts.size else 0)
        if max_k <= 0 and max_p <= 0:
            return None

        max_k = max(max_k, 1)
        indices = np.full((total_count, max_k), -1, dtype=np.int64)
        features = np.zeros((total_count, max_k, CANDIDATE_FEATURES), dtype=np.float32)
        mask = np.zeros((total_count, max_k), dtype=np.bool_)
        max_p = max(max_p, 0)
        pair_indices = (
            np.full((total_count, max_p, 2), -1, dtype=np.int64)
            if max_p > 0
            else None
        )
        pair_mask = (
            np.zeros((total_count, max_p), dtype=np.bool_)
            if max_p > 0
            else None
        )
        offset = 0
        for worker_id, count in zip(ready_workers, per_worker_counts):
            slot = self._queue.get_slot(worker_id)
            for row in range(count):
                k = int(slot.req_candidate_count[row])
                if k > 0:
                    kk = min(k, max_k)
                    indices[offset + row, :kk] = slot.req_candidate_indices[row, :kk]
                    features[offset + row, :kk] = slot.req_candidate_features[row, :kk]
                    mask[offset + row, :kk] = slot.req_candidate_mask[row, :kk].astype(bool)
                if pair_indices is not None and pair_mask is not None:
                    p = int(slot.req_pair_count[row])
                    if p > 0:
                        pp = min(p, max_p, MAX_PAIR_CANDIDATES)
                        pair_indices[offset + row, :pp] = slot.req_pair_indices[row, :pp]
                        pair_mask[offset + row, :pp] = slot.req_pair_mask[row, :pp].astype(bool)
            offset += count
        out = {
            "candidate_indices": torch.from_numpy(indices).to(self._device, non_blocking=True),
            "candidate_features": torch.from_numpy(features).to(self._device, non_blocking=True),
            "candidate_mask": torch.from_numpy(mask).to(self._device, non_blocking=True),
        }
        if pair_indices is not None and pair_mask is not None:
            out["pair_candidate_indices"] = torch.from_numpy(pair_indices).to(
                self._device, non_blocking=True
            )
            out["pair_candidate_mask"] = torch.from_numpy(pair_mask).to(
                self._device, non_blocking=True
            )
        return out

    # ── Forward pass ──────────────────────────────────────────────────────

    def _forward(
        self,
        batch_tensor: torch.Tensor,
        sparse_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Run the model forward pass on a batch.

        Args:
            batch_tensor: (total_count, 13, 33, 33) on device.

        Returns:
            policies: (total_count, BOARD_AREA) float32 numpy array.
            values:   (total_count,) float32 numpy array.
        """
        t0 = time.monotonic()

        model_t0 = time.monotonic()
        with torch.inference_mode():
            if self._device.type == "cuda" and self._forward_stream is not None:
                with torch.cuda.stream(self._forward_stream):
                    if self.fp16:
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            out = self._model(batch_tensor, **sparse_inputs) if sparse_inputs else self._model(batch_tensor)
                    else:
                        out = self._model(batch_tensor, **sparse_inputs) if sparse_inputs else self._model(batch_tensor)
                self._forward_stream.synchronize()
            elif self.fp16 and self._device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = self._model(batch_tensor, **sparse_inputs) if sparse_inputs else self._model(batch_tensor)
            else:
                out = self._model(batch_tensor, **sparse_inputs) if sparse_inputs else self._model(batch_tensor)
        model_ms = (time.monotonic() - model_t0) * 1000.0

        post_t0 = time.monotonic()
        decoded = decode_dense_outputs(
            out,
            value_decoder=bins_to_value,
            sparse_requested=sparse_inputs is not None,
        )
        post_ms = (time.monotonic() - post_t0) * 1000.0

        download_t0 = time.monotonic()
        policies = decoded.policy
        values = decoded.value
        sparse = decoded.sparse_policy
        pair = decoded.pair_policy
        regret = decoded.regret_rank
        download_ms = (time.monotonic() - download_t0) * 1000.0

        elapsed = (time.monotonic() - t0) * 1000.0
        self.total_forward_ms += elapsed
        self.total_model_ms += model_ms
        self.total_postprocess_ms += post_ms
        self.total_download_ms += download_ms

        return policies, values, sparse, pair, regret

    def _forward_graph(
        self,
        graph_inputs: dict[str, torch.Tensor],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Run the global graph model and return keyed graph logits."""
        t0 = time.monotonic()
        model_t0 = time.monotonic()
        with torch.inference_mode():
            if self._device.type == "cuda" and self._forward_stream is not None:
                with torch.cuda.stream(self._forward_stream):
                    if self.fp16:
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            out = self._model(**graph_inputs)
                    else:
                        out = self._model(**graph_inputs)
                self._forward_stream.synchronize()
            elif self.fp16 and self._device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = self._model(**graph_inputs)
            else:
                out = self._model(**graph_inputs)
        model_ms = (time.monotonic() - model_t0) * 1000.0

        post_t0 = time.monotonic()
        decoded = decode_global_graph_outputs(
            out,
            graph_inputs,
            value_decoder=bins_to_value,
        )
        post_ms = (time.monotonic() - post_t0) * 1000.0

        download_t0 = time.monotonic()
        place_np = decoded.policy_place
        values_np = decoded.value
        opp_np = decoded.opp_policy
        pair_first_np = decoded.pair_first
        pair_joint_np = decoded.pair_joint
        pair_second_np = decoded.pair_second
        regret_np = decoded.regret_rank
        download_ms = (time.monotonic() - download_t0) * 1000.0

        elapsed = (time.monotonic() - t0) * 1000.0
        self.total_forward_ms += elapsed
        self.total_model_ms += model_ms
        self.total_postprocess_ms += post_ms
        self.total_download_ms += download_ms
        return place_np, values_np, opp_np, pair_first_np, pair_joint_np, pair_second_np, regret_np

    @staticmethod
    def _sanitize_policy_logits(logits: torch.Tensor) -> torch.Tensor:
        """Keep policy logits finite before handing them to Rust MCTS."""
        return sanitize_policy_logits(logits)

    @staticmethod
    def _sanitize_value_logits(logits: torch.Tensor) -> torch.Tensor:
        """Keep value logits finite so softmax cannot produce NaNs."""
        return sanitize_value_logits(logits)

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
        sparse_logits: Optional[np.ndarray] = None,
        pair_logits: Optional[np.ndarray] = None,
        regret_rank: Optional[np.ndarray] = None,
    ):
        """Distribute flat policy/value arrays back to per-worker slots."""
        offset = 0
        for worker_id, count in zip(ready_workers, per_worker_counts):
            slot = self._queue.get_slot(worker_id)
            slot.res_policy[:count] = policies[offset:offset + count]
            slot.res_value[:count] = values[offset:offset + count]
            if getattr(slot, "res_regret_rank", None) is not None:
                slot.res_regret_rank[:count] = 0.0
                if regret_rank is not None:
                    slot.res_regret_rank[:count] = regret_rank[offset:offset + count]
            if sparse_logits is not None:
                k = sparse_logits.shape[1]
                slot.res_sparse_logits[:count, :k] = sparse_logits[offset:offset + count]
            if pair_logits is not None:
                p = pair_logits.shape[1]
                slot.res_pair_logits[:count, :p] = pair_logits[offset:offset + count]
            offset += count

    def _scatter_graph_results(
        self,
        row_sources: List[tuple[int, int]],
        values: np.ndarray,
        place_logits: np.ndarray,
        opp_logits: Optional[np.ndarray],
        pair_first_logits: Optional[np.ndarray],
        pair_joint_logits: Optional[np.ndarray],
        pair_second_logits: Optional[np.ndarray],
        regret_rank: Optional[np.ndarray] = None,
    ):
        """Scatter graph logits using each worker's keyed legal/pair counts."""
        head_flags = 0
        if opp_logits is not None:
            head_flags |= GRAPH_HEAD_OPP
        if pair_first_logits is not None:
            head_flags |= GRAPH_HEAD_PAIR_FIRST
        if pair_joint_logits is not None:
            head_flags |= GRAPH_HEAD_PAIR_JOINT
        if pair_second_logits is not None:
            head_flags |= GRAPH_HEAD_PAIR_SECOND
        if regret_rank is not None:
            head_flags |= GRAPH_HEAD_REGRET

        worker_totals: dict[int, list[int]] = {}
        for row, (worker_id, local_row) in enumerate(row_sources):
            slot = self._queue.get_slot(worker_id)
            meta = np.array(slot.req_graph_batch_meta[local_row], copy=True)
            if local_row == 0 and not meta.any():
                token_count, legal_count, opp_count, pair_count = map(int, slot.req_graph_meta[2:6])
                token_off = legal_off = opp_off = pair_off = 0
            else:
                token_count, legal_count, opp_count, pair_count, token_off, legal_off, opp_off, pair_off = map(int, meta[:8])
            if legal_count > place_logits.shape[1]:
                raise ValueError("graph place logits shorter than legal row table")
            if pair_count and pair_joint_logits is None and pair_second_logits is None:
                raise ValueError("graph model did not return any pair logits for a pair request")
            slot.res_value[local_row] = float(values[row])
            totals = worker_totals.setdefault(worker_id, [0, 0, 0, 0])
            totals[0] += token_count
            totals[1] += legal_count
            totals[2] += opp_count
            totals[3] += pair_count
            slot.res_graph_meta[:] = (
                GRAPH_SCHEMA_VERSION,
                RELATION_SCHEMA_VERSION,
                totals[1],
                totals[2],
                totals[3],
                totals[0],
                MAX_GRAPH_TOKENS,
                head_flags,
            )
            if local_row == 0:
                slot.res_graph_place_logits.fill(0.0)
                slot.res_graph_opp_logits.fill(0.0)
                slot.res_graph_pair_first_logits.fill(0.0)
                slot.res_graph_pair_logits.fill(0.0)
                slot.res_graph_pair_second_logits.fill(0.0)
            if getattr(slot, "res_graph_regret_rank", None) is not None:
                slot.res_graph_regret_rank[local_row] = (
                    float(regret_rank[row])
                    if regret_rank is not None and row < len(regret_rank)
                    else 0.0
                )
            slot.res_graph_place_logits[legal_off : legal_off + legal_count] = place_logits[row, :legal_count]
            if opp_count and opp_logits is not None:
                slot.res_graph_opp_logits[opp_off : opp_off + opp_count] = opp_logits[row, :opp_count]
            if legal_count and pair_first_logits is not None:
                slot.res_graph_pair_first_logits[legal_off : legal_off + legal_count] = pair_first_logits[row, :legal_count]
            if pair_count and pair_joint_logits is not None:
                slot.res_graph_pair_logits[pair_off : pair_off + pair_count] = pair_joint_logits[row, :pair_count]
            if pair_count and pair_second_logits is not None:
                slot.res_graph_pair_second_logits[pair_off : pair_off + pair_count] = pair_second_logits[row, :pair_count]
            slot.req_mode[0] = 0

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
            load_model_state(self._model, latest, allow_partial=False)
            self._model.eval()

    @property
    def positions_per_sec(self) -> float:
        """Positions processed per second (cumulative)."""
        if self.total_forward_ms <= 0:
            return 0.0
        return self.n_positions / (self.total_forward_ms / 1000.0)
