"""Worker-side typed inference client."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from hexorl.inference.adapters import (
    DensePolicyValueAdapter,
    GlobalGraphPolicyValueAdapter,
    PairScoringAdapter,
    SparsePolicyValueAdapter,
)
from hexorl.inference.protocol import (
    InferenceProtocolManifest,
    InferenceRequestKind,
    default_protocol_manifest,
    make_request,
)
from hexorl.inference.shm_queue import InferenceQueue, connect_inference_queue
from hexorl.inference.client.handshake import load_declared_server_manifest, negotiate_client_handshake
from hexorl.inference.client.static_response import zero_count_response
from hexorl.inference.client.transport import ShmTransport, TransportState


class InferenceClient:
    """Worker-side inference client.

    Shared memory remains the physical transport, but all runtime requests now
    cross through a typed protocol envelope and one transport lifecycle.
    """

    def __init__(
        self,
        worker_id: int,
        num_workers: int,
        max_batch_size: int,
        timeout_ms: float = 1000.0,
        manifest: InferenceProtocolManifest | None = None,
        server_manifest: InferenceProtocolManifest | None = None,
    ):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.max_batch = max_batch_size
        self.timeout_ms = timeout_ms
        self._manifest_explicit = manifest is not None
        self.manifest = manifest or default_protocol_manifest(
            max_batch_size=max_batch_size,
            timeout_ms=timeout_ms,
        )
        self.server_manifest = server_manifest
        self._queue: Optional[InferenceQueue] = None
        self._slot = None
        self._transport: ShmTransport | None = None
        self._connected = False
        self._adapters = self._build_adapters()

        self.n_submits = 0
        self.total_wait_ms = 0.0
        self.handshake = None

    def _build_adapters(self):
        return {
            InferenceRequestKind.DENSE_POLICY_VALUE: DensePolicyValueAdapter(self.manifest),
            InferenceRequestKind.REGRET_RANK_POLICY_VALUE: DensePolicyValueAdapter(self.manifest),
            InferenceRequestKind.SPARSE_POLICY_VALUE: SparsePolicyValueAdapter(self.manifest),
            InferenceRequestKind.PAIR_SCORING: PairScoringAdapter(self.manifest),
            InferenceRequestKind.SPARSE_PAIR_POLICY_VALUE: PairScoringAdapter(self.manifest),
            InferenceRequestKind.GLOBAL_GRAPH_POLICY_VALUE: GlobalGraphPolicyValueAdapter(self.manifest),
            InferenceRequestKind.GRAPH_PAIR_POLICY_VALUE: GlobalGraphPolicyValueAdapter(self.manifest),
        }

    def connect(self):
        if self._connected:
            return
        self._queue = connect_inference_queue(self.num_workers, self.max_batch)
        self._slot = self._queue.get_slot(self.worker_id)
        self._transport = ShmTransport(
            worker_id=self.worker_id,
            slot=self._slot,
            timeout_ms=self.timeout_ms,
        )
        self._transport.state = TransportState.HANDSHAKING
        server_manifest = load_declared_server_manifest(
            num_workers=self.num_workers,
            max_batch_size=self.max_batch,
            explicit_manifest=self.server_manifest,
        )
        if not self._manifest_explicit:
            self.manifest = server_manifest
            self._adapters = self._build_adapters()
        self.handshake = negotiate_client_handshake(
            client_manifest=self.manifest,
            server_manifest=server_manifest,
        )
        self._transport.mark_ready()
        self._connected = True

    def disconnect(self):
        if self._transport is not None:
            self._transport.close()
            self._transport = None
        if self._queue is not None:
            self._queue.close()
            self._queue = None
        self._slot = None
        self._connected = False

    def submit(self, tensor: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
        return self.evaluate_dense(tensor, count)

    def evaluate_dense(
        self,
        tensor: np.ndarray,
        count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        response = self._request(
            InferenceRequestKind.DENSE_POLICY_VALUE,
            {"tensor": tensor, "count": int(count)},
        )
        return response.head_outputs["policy"], response.head_outputs["value"]

    def evaluate_sparse(
        self,
        tensor: np.ndarray,
        count: int,
        candidate_indices: np.ndarray,
        candidate_features: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        response = self._request(
            InferenceRequestKind.SPARSE_POLICY_VALUE,
            {
                "tensor": tensor,
                "count": int(count),
                "candidate_indices": candidate_indices,
                "candidate_features": candidate_features,
                "candidate_mask": candidate_mask,
            },
        )
        heads = response.head_outputs
        return heads["policy"], heads["value"], heads.get("sparse_policy", np.empty((count, 0), dtype=np.float32))

    def evaluate_pair_scoring(
        self,
        tensor: np.ndarray,
        count: int,
        candidate_indices: np.ndarray,
        candidate_features: np.ndarray,
        candidate_mask: np.ndarray,
        pair_candidate_indices: np.ndarray,
        pair_candidate_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        response = self._request(
            InferenceRequestKind.PAIR_SCORING,
            {
                "tensor": tensor,
                "count": int(count),
                "candidate_indices": candidate_indices,
                "candidate_features": candidate_features,
                "candidate_mask": candidate_mask,
                "pair_candidate_indices": pair_candidate_indices,
                "pair_candidate_mask": pair_candidate_mask,
            },
        )
        heads = response.head_outputs
        return (
            heads["policy"],
            heads["value"],
            heads.get("sparse_policy", np.empty((count, 0), dtype=np.float32)),
            heads.get("pair_policy", np.empty((count, 0), dtype=np.float32)),
        )

    def evaluate_regret_rank(self, tensor: np.ndarray, count: int) -> np.ndarray:
        rank, _value = self.evaluate_regret_heads(tensor, count)
        return rank

    def evaluate_regret_heads(self, tensor: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
        response = self._request(
            InferenceRequestKind.REGRET_RANK_POLICY_VALUE,
            {"tensor": tensor, "count": int(count)},
        )
        if "regret_rank" not in response.head_outputs:
            raise RuntimeError("inference response does not expose regret-rank head")
        if "regret_value" not in response.head_outputs:
            raise RuntimeError("inference response does not expose regret-value head")
        return response.head_outputs["regret_rank"], response.head_outputs["regret_value"]

    def evaluate_global_graph(self, graph_batch) -> dict[str, np.ndarray | dict[str, object]]:
        response = self._request(
            InferenceRequestKind.GLOBAL_GRAPH_POLICY_VALUE,
            {"graph_batch": graph_batch},
        )
        return dict(response.head_outputs)

    def evaluate_graph_pair_policy(self, graph_batch) -> dict[str, np.ndarray | dict[str, object]]:
        response = self._request(
            InferenceRequestKind.GRAPH_PAIR_POLICY_VALUE,
            {"graph_batch": graph_batch},
        )
        return dict(response.head_outputs)

    def _request(self, kind: InferenceRequestKind, payload: dict[str, object]):
        if not self._connected or self._transport is None:
            raise RuntimeError("InferenceClient not connected. Call connect() first.")
        if int(payload.get("count", 1)) <= 0 and kind != InferenceRequestKind.GLOBAL_GRAPH_POLICY_VALUE:
            return zero_count_response(kind.value)
        request = make_request(
            kind=kind,
            manifest=self.manifest,
            payload=payload,
            deadline_monotonic_s=time.monotonic() + self.timeout_ms / 1000.0,
            slot_generation=self._transport.generation,
        )
        response = self._transport.round_trip(request, self._adapters[kind])
        self.total_wait_ms += float(response.telemetry.get("wait_ms", 0.0))
        self.n_submits += 1
        return response

    @property
    def avg_wait_ms(self) -> float:
        if self.n_submits == 0:
            return 0.0
        return self.total_wait_ms / self.n_submits

    def __del__(self):
        self.disconnect()
