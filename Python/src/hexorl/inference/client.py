"""Inference client — worker-side interface to the GPU inference server.

Each self-play worker creates one InferenceClient that connects to the
shared-memory slots allocated by the server. The submit() method is the
only Python call in the MCTS inner loop.
"""

import time
import numpy as np
from typing import Optional
from hexorl.inference.shm_queue import (
    MAX_CANDIDATES,
    MAX_PAIR_CANDIDATES,
    InferenceQueue,
    connect_inference_queue,
)


class InferenceClient:
    """Worker-side inference client.

    Usage:
        client = InferenceClient(worker_id=3, num_workers=24, max_batch_size=128)
        client.connect()
        ...
        policies, values = client.submit(tensor, count)
        ...
        client.disconnect()
    """

    def __init__(
        self,
        worker_id: int,
        num_workers: int,
        max_batch_size: int,
        timeout_ms: float = 1000.0,
    ):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.max_batch = max_batch_size
        self.timeout_ms = timeout_ms
        self._queue: Optional[InferenceQueue] = None
        self._slot = None
        self._connected = False

        # Stats
        self.n_submits = 0
        self.total_wait_ms = 0.0

    def connect(self):
        """Connect to the server's shared-memory queue."""
        if self._connected:
            return
        self._queue = connect_inference_queue(self.num_workers, self.max_batch)
        self._slot = self._queue.get_slot(self.worker_id)
        self._connected = True

    def disconnect(self):
        """Disconnect and release shared-memory references."""
        if self._queue is not None:
            self._queue.close()
            self._queue = None
        self._slot = None
        self._connected = False

    def submit(
        self,
        tensor: np.ndarray,
        count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Submit a batch of leaves for GPU evaluation.

        Args:
            tensor: float32 array of shape (count, 13, 33, 33).
            count: Number of leaves in the batch.

        Returns:
            (policies, values):
                policies: float32 array of shape (count, 1089) — policy logits.
                values:   float32 array of shape (count,) — value scalars.
        """
        if not self._connected:
            raise RuntimeError("InferenceClient not connected. Call connect() first.")

        if count > self.max_batch:
            raise ValueError(
                f"Batch count {count} exceeds max_batch {self.max_batch}"
            )
        if count <= 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

        self._slot.res_ready.clear()

        # 1. Write tensor data to the shared-memory request slot.
        req_view = self._slot.req_tensor[:count]          # (count, 13, 33, 33)
        np.copyto(req_view, tensor.reshape(count, 13, 33, 33))

        # 2. Set the batch count.
        if getattr(self._slot, "req_candidate_count", None) is not None:
            self._slot.req_candidate_count[:count] = 0
        if getattr(self._slot, "req_pair_count", None) is not None:
            self._slot.req_pair_count[:count] = 0
        self._slot.req_count[0] = count

        # 3. Signal the server.
        self._slot.req_ready.set()

        # 4. Wait for the response.
        t0 = time.monotonic()
        if not self._slot.res_ready.wait(timeout=self.timeout_ms / 1000.0):
            raise TimeoutError(
                f"Inference server did not respond within {self.timeout_ms:.0f}ms"
            )
        elapsed = (time.monotonic() - t0) * 1000.0
        self.total_wait_ms += elapsed
        self.n_submits += 1

        self._slot.res_ready.clear()

        # 5. Read results from shared memory.
        policies = self._slot.res_policy[:count]   # (count, 1089)
        values = self._slot.res_value[:count]      # (count,)

        # Flatten policies to 1D — expand_and_backprop expects flat array.
        policies = policies.ravel()  # (count * 1089,)

        return policies, values

    def submit_sparse(
        self,
        tensor: np.ndarray,
        count: int,
        candidate_indices: np.ndarray,
        candidate_features: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Submit a batch with optional candidate/action-keyed sparse inputs."""
        if not self._connected:
            raise RuntimeError("InferenceClient not connected. Call connect() first.")
        if count > self.max_batch:
            raise ValueError(f"Batch count {count} exceeds max_batch {self.max_batch}")
        if count <= 0:
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty((0, 0), dtype=np.float32),
            )
        k = int(candidate_indices.shape[1])
        if k > MAX_CANDIDATES:
            raise ValueError(f"Candidate count {k} exceeds MAX_CANDIDATES {MAX_CANDIDATES}")

        self._slot.res_ready.clear()
        np.copyto(self._slot.req_tensor[:count], tensor.reshape(count, 13, 33, 33))
        self._slot.req_candidate_count[:count] = k
        if getattr(self._slot, "req_pair_count", None) is not None:
            self._slot.req_pair_count[:count] = 0
        self._slot.req_candidate_indices[:count, :k] = candidate_indices[:count, :k]
        self._slot.req_candidate_features[:count, :k] = candidate_features[:count, :k]
        self._slot.req_candidate_mask[:count, :k] = candidate_mask[:count, :k].astype(np.uint8)
        if k < MAX_CANDIDATES:
            self._slot.req_candidate_count[count:] = 0
            self._slot.req_candidate_mask[:count, k:] = 0
        self._slot.req_count[0] = count
        self._slot.req_ready.set()

        t0 = time.monotonic()
        if not self._slot.res_ready.wait(timeout=self.timeout_ms / 1000.0):
            raise TimeoutError(
                f"Inference server did not respond within {self.timeout_ms:.0f}ms"
            )
        elapsed = (time.monotonic() - t0) * 1000.0
        self.total_wait_ms += elapsed
        self.n_submits += 1
        self._slot.res_ready.clear()

        policies = self._slot.res_policy[:count].ravel()
        values = self._slot.res_value[:count]
        sparse = np.array(self._slot.res_sparse_logits[:count, :k], copy=True)
        self._slot.req_candidate_count[:count] = 0
        return policies, values, sparse

    def submit_sparse_pair(
        self,
        tensor: np.ndarray,
        count: int,
        candidate_indices: np.ndarray,
        candidate_features: np.ndarray,
        candidate_mask: np.ndarray,
        pair_candidate_indices: np.ndarray,
        pair_candidate_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Submit sparse action rows plus candidate-pair rows and return pair logits."""
        if not self._connected:
            raise RuntimeError("InferenceClient not connected. Call connect() first.")
        if count > self.max_batch:
            raise ValueError(f"Batch count {count} exceeds max_batch {self.max_batch}")
        if count <= 0:
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty((0, 0), dtype=np.float32),
                np.empty((0, 0), dtype=np.float32),
            )
        k = int(candidate_indices.shape[1])
        p = int(pair_candidate_indices.shape[1])
        if k > MAX_CANDIDATES:
            raise ValueError(f"Candidate count {k} exceeds MAX_CANDIDATES {MAX_CANDIDATES}")
        if p > MAX_PAIR_CANDIDATES:
            raise ValueError(
                f"Pair candidate count {p} exceeds MAX_PAIR_CANDIDATES {MAX_PAIR_CANDIDATES}"
            )
        if pair_candidate_indices.shape[:2] != pair_candidate_mask.shape[:2]:
            raise ValueError("pair_candidate_indices and pair_candidate_mask must share (B, P)")
        if pair_candidate_indices.shape[2] != 2:
            raise ValueError("pair_candidate_indices must have shape (B, P, 2)")

        self._slot.res_ready.clear()
        np.copyto(self._slot.req_tensor[:count], tensor.reshape(count, 13, 33, 33))
        self._slot.req_candidate_count[:count] = k
        self._slot.req_candidate_indices[:count, :k] = candidate_indices[:count, :k]
        self._slot.req_candidate_features[:count, :k] = candidate_features[:count, :k]
        self._slot.req_candidate_mask[:count, :k] = candidate_mask[:count, :k].astype(np.uint8)
        self._slot.req_pair_count[:count] = p
        self._slot.req_pair_indices[:count, :p] = pair_candidate_indices[:count, :p]
        self._slot.req_pair_mask[:count, :p] = pair_candidate_mask[:count, :p].astype(np.uint8)
        if k < MAX_CANDIDATES:
            self._slot.req_candidate_mask[:count, k:] = 0
        if p < MAX_PAIR_CANDIDATES:
            self._slot.req_pair_mask[:count, p:] = 0
        self._slot.req_count[0] = count
        self._slot.req_ready.set()

        t0 = time.monotonic()
        if not self._slot.res_ready.wait(timeout=self.timeout_ms / 1000.0):
            raise TimeoutError(
                f"Inference server did not respond within {self.timeout_ms:.0f}ms"
            )
        elapsed = (time.monotonic() - t0) * 1000.0
        self.total_wait_ms += elapsed
        self.n_submits += 1
        self._slot.res_ready.clear()

        policies = self._slot.res_policy[:count].ravel()
        values = self._slot.res_value[:count]
        sparse = np.array(self._slot.res_sparse_logits[:count, :k], copy=True)
        pair = np.array(self._slot.res_pair_logits[:count, :p], copy=True)
        self._slot.req_candidate_count[:count] = 0
        self._slot.req_pair_count[:count] = 0
        return policies, values, sparse, pair

    @property
    def avg_wait_ms(self) -> float:
        """Average wait time in milliseconds."""
        if self.n_submits == 0:
            return 0.0
        return self.total_wait_ms / self.n_submits

    def __del__(self):
        self.disconnect()
