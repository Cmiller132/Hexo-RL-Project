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
    load_server_manifest,
    make_request,
    negotiate_protocol,
)
from hexorl.inference.shm_queue import InferenceQueue, connect_inference_queue
from hexorl.inference.shm_transport import ShmTransport, TransportState


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
        server_manifest = self.server_manifest or load_server_manifest(
            num_workers=self.num_workers,
            max_batch_size=self.max_batch,
        )
        if not self._manifest_explicit:
            self.manifest = server_manifest
            self._adapters = self._build_adapters()
        selected_kind = self._select_request_kind(server_manifest)
        self.handshake = negotiate_protocol(
            client_manifest=self.manifest,
            server_manifest=server_manifest,
            request_kind=selected_kind,
            required_heads=tuple(self.manifest.heads),
        )
        self._transport.mark_ready()
        self._connected = True

    def _select_request_kind(self, server_manifest: InferenceProtocolManifest) -> InferenceRequestKind:
        client_kinds = [InferenceRequestKind(kind) for kind in self.manifest.request_kind]
        server_kinds = set(server_manifest.request_kind)
        for kind in client_kinds:
            if kind.value in server_kinds:
                return kind
        raise RuntimeError(
            "no compatible inference request kind between client and server: "
            f"client={list(self.manifest.request_kind)} server={list(server_manifest.request_kind)}"
        )

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
        response = self._request(
            InferenceRequestKind.REGRET_RANK_POLICY_VALUE,
            {"tensor": tensor, "count": int(count)},
        )
        if "regret_rank" not in response.head_outputs:
            raise RuntimeError("inference response does not expose regret-rank head")
        return response.head_outputs["regret_rank"]

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
            empty = np.empty(0, dtype=np.float32)
            if kind == InferenceRequestKind.SPARSE_POLICY_VALUE:
                return _StaticResponse({"policy": empty, "value": empty, "sparse_policy": np.empty((0, 0), dtype=np.float32)})
            if kind == InferenceRequestKind.PAIR_SCORING:
                return _StaticResponse({
                    "policy": empty,
                    "value": empty,
                    "sparse_policy": np.empty((0, 0), dtype=np.float32),
                    "pair_policy": np.empty((0, 0), dtype=np.float32),
                })
            return _StaticResponse({"policy": empty, "value": empty})
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


class _StaticResponse:
    def __init__(self, heads: dict[str, np.ndarray]):
        self.head_outputs = heads
