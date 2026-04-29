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
    MAX_GRAPH_ACTIONS,
    MAX_GRAPH_PAIRS,
    MAX_GRAPH_TOKENS,
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
        if getattr(self._slot, "req_mode", None) is not None:
            self._slot.req_mode[0] = 0
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
        if getattr(self._slot, "req_mode", None) is not None:
            self._slot.req_mode[0] = 0
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
        if getattr(self._slot, "req_mode", None) is not None:
            self._slot.req_mode[0] = 0
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

    def submit_regret_rank(self, tensor: np.ndarray, count: int) -> np.ndarray:
        """Submit dense/crop positions and return regret-rank logits."""
        policies, _values = self.submit(tensor, count)
        _ = policies
        if getattr(self._slot, "res_regret_rank", None) is None:
            raise RuntimeError("inference queue does not expose regret-rank responses")
        return np.array(self._slot.res_regret_rank[:count], copy=True)

    def submit_graph(self, graph_batch) -> dict[str, np.ndarray | dict[str, object]]:
        """Submit one padded global graph request and return keyed logits.

        The graph IPC slot is deliberately single-position per worker. Graph
        MCTS prioritizes exact global action identity and capacity failures over
        dense fallback throughput.
        """
        if not self._connected:
            raise RuntimeError("InferenceClient not connected. Call connect() first.")
        if self._slot.req_mode is None:
            raise RuntimeError("inference queue does not expose graph IPC slots")
        from hexorl.graph.batch import validate_graph_ipc_capacity

        validate_graph_ipc_capacity(graph_batch)

        token_count = int(np.asarray(graph_batch.token_features).shape[0])
        legal_count = int(np.asarray(graph_batch.legal_qr).shape[0])
        opp_count = int(np.asarray(graph_batch.opp_legal_qr).shape[0])
        pair_count = int(np.asarray(graph_batch.pair_token_indices).shape[0])
        if token_count > MAX_GRAPH_TOKENS:
            raise ValueError(f"graph token count {token_count} exceeds MAX_GRAPH_TOKENS {MAX_GRAPH_TOKENS}")
        if legal_count > MAX_GRAPH_ACTIONS:
            raise ValueError(f"graph legal row count {legal_count} exceeds MAX_GRAPH_ACTIONS {MAX_GRAPH_ACTIONS}")
        if opp_count > MAX_GRAPH_ACTIONS:
            raise ValueError(f"graph opponent legal row count {opp_count} exceeds MAX_GRAPH_ACTIONS {MAX_GRAPH_ACTIONS}")
        if pair_count > MAX_GRAPH_PAIRS:
            raise ValueError(f"graph pair row count {pair_count} exceeds MAX_GRAPH_PAIRS {MAX_GRAPH_PAIRS}")

        self._slot.res_ready.clear()
        self._slot.req_mode[0] = 1
        self._slot.req_count[0] = 1
        self._slot.req_candidate_count[:1] = 0
        self._slot.req_pair_count[:1] = 0
        self._slot.req_graph_meta[:] = (
            int(graph_batch.schema_version),
            int(graph_batch.relation_schema_version),
            token_count,
            legal_count,
            opp_count,
            pair_count,
            MAX_GRAPH_TOKENS,
            MAX_GRAPH_ACTIONS,
        )

        self._slot.req_graph_token_features.fill(0.0)
        self._slot.req_graph_token_type.fill(0)
        self._slot.req_graph_token_qr.fill(0)
        self._slot.req_graph_token_mask.fill(0)
        self._slot.req_graph_legal_token_indices.fill(-1)
        self._slot.req_graph_legal_qr.fill(0)
        self._slot.req_graph_legal_mask.fill(0)
        self._slot.req_graph_opp_legal_qr.fill(0)
        self._slot.req_graph_opp_legal_mask.fill(0)
        self._slot.req_graph_pair_token_indices.fill(-1)
        self._slot.req_graph_pair_first_indices.fill(-1)
        self._slot.req_graph_pair_second_indices.fill(-1)
        self._slot.req_graph_relation_type.fill(0)
        self._slot.req_graph_relation_bias.fill(0.0)

        self._slot.req_graph_token_features[:token_count] = np.asarray(graph_batch.token_features, dtype=np.float32)
        self._slot.req_graph_token_type[:token_count] = np.asarray(graph_batch.token_type, dtype=np.int16)
        self._slot.req_graph_token_qr[:token_count] = np.asarray(graph_batch.token_qr, dtype=np.int32)
        self._slot.req_graph_token_mask[:token_count] = np.asarray(graph_batch.token_mask, dtype=np.uint8)
        self._slot.req_graph_legal_token_indices[:legal_count] = np.asarray(graph_batch.legal_token_indices, dtype=np.int64)
        self._slot.req_graph_legal_qr[:legal_count] = np.asarray(graph_batch.legal_qr, dtype=np.int32)
        self._slot.req_graph_legal_mask[:legal_count] = np.asarray(graph_batch.legal_mask, dtype=np.uint8)
        if opp_count:
            self._slot.req_graph_opp_legal_qr[:opp_count] = np.asarray(graph_batch.opp_legal_qr, dtype=np.int32)
            self._slot.req_graph_opp_legal_mask[:opp_count] = np.asarray(graph_batch.opp_legal_mask, dtype=np.uint8)
        if pair_count:
            self._slot.req_graph_pair_token_indices[:pair_count] = np.asarray(graph_batch.pair_token_indices, dtype=np.int64)
            self._slot.req_graph_pair_first_indices[:pair_count] = np.asarray(graph_batch.pair_first_indices, dtype=np.int64)
            self._slot.req_graph_pair_second_indices[:pair_count] = np.asarray(graph_batch.pair_second_indices, dtype=np.int64)
        self._slot.req_graph_relation_type[:token_count, :token_count] = np.asarray(
            graph_batch.relation_type,
            dtype=np.int16,
        )
        self._slot.req_graph_relation_bias[:, :token_count, :token_count] = np.asarray(
            graph_batch.relation_bias,
            dtype=np.float32,
        )

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
        self._slot.req_mode[0] = 0

        meta = {
            "schema_version": int(self._slot.res_graph_meta[0]),
            "relation_schema_version": int(self._slot.res_graph_meta[1]),
            "legal_count": int(self._slot.res_graph_meta[2]),
            "opp_legal_count": int(self._slot.res_graph_meta[3]),
            "pair_count": int(self._slot.res_graph_meta[4]),
            "prior_source": "global_graph",
            "legal_qr": np.array(self._slot.req_graph_legal_qr[:legal_count], copy=True),
            "legal_mask": np.array(self._slot.req_graph_legal_mask[:legal_count].astype(bool), copy=True),
        }
        return {
            "policy_place": np.array(self._slot.res_graph_place_logits[:legal_count], copy=True),
            "opp_policy": np.array(self._slot.res_graph_opp_logits[:opp_count], copy=True),
            "policy_pair_first": np.array(self._slot.res_graph_pair_first_logits[:legal_count], copy=True),
            "policy_pair_joint": np.array(self._slot.res_graph_pair_logits[:pair_count], copy=True),
            "policy_pair_second": np.array(self._slot.res_graph_pair_second_logits[:pair_count], copy=True),
            "regret_rank": np.array(
                getattr(self._slot, "res_graph_regret_rank", np.zeros(1, dtype=np.float32))[:1],
                copy=True,
            ),
            "value": np.array(self._slot.res_value[:1], copy=True),
            "metadata": meta,
        }

    @property
    def avg_wait_ms(self) -> float:
        """Average wait time in milliseconds."""
        if self.n_submits == 0:
            return 0.0
        return self.total_wait_ms / self.n_submits

    def __del__(self):
        self.disconnect()
