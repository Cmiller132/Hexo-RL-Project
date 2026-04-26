"""Inference client — worker-side interface to the GPU inference server.

Each self-play worker creates one InferenceClient that connects to the
shared-memory slots allocated by the server. The submit() method is the
only Python call in the MCTS inner loop.
"""

import time
import numpy as np
from typing import Optional
from hexorl.inference.shm_queue import InferenceQueue, connect_inference_queue, BOARD_AREA


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
        policies = np.array(self._slot.res_policy[:count], copy=True)   # (count, 1089)
        values = np.array(self._slot.res_value[:count], copy=True)      # (count,)

        # Flatten policies to 1D — expand_and_backprop expects flat array.
        policies = policies.ravel()  # (count * 1089,)

        return policies, values

    @property
    def avg_wait_ms(self) -> float:
        """Average wait time in milliseconds."""
        if self.n_submits == 0:
            return 0.0
        return self.total_wait_ms / self.n_submits

    def __del__(self):
        self.disconnect()
