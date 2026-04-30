"""Shared-memory transport for typed inference requests."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

import numpy as np

from hexorl.inference.adapters.base import InferenceAdapter
from hexorl.inference.protocol import (
    InferenceRequest,
    InferenceRequestKind,
    InferenceResponse,
    REQUEST_KIND_TO_CODE,
)
from hexorl.inference.shm_queue import (
    MAX_CANDIDATES,
    MAX_GRAPH_ACTIONS,
    MAX_GRAPH_PAIRS,
    MAX_GRAPH_TOKENS,
    MAX_PAIR_CANDIDATES,
)
from hexorl.inference.telemetry import InferenceTelemetry, timeout_message


class TransportState(str, Enum):
    CREATED = "created"
    HANDSHAKING = "handshaking"
    READY = "ready"
    DRAINING = "draining"
    CLOSED = "closed"
    FAILED = "failed"


class ShmTransport:
    """Single lifecycle owner for request packing, signaling, wait, and decode."""

    def __init__(self, *, worker_id: int, slot: Any, timeout_ms: float):
        self.worker_id = int(worker_id)
        self.slot = slot
        self.timeout_ms = float(timeout_ms)
        self.state = TransportState.CREATED
        self.generation = 0
        self.last_heartbeat_monotonic_s = time.monotonic()

    def mark_ready(self) -> None:
        if self.state == TransportState.CLOSED:
            raise RuntimeError("cannot ready a closed inference transport")
        self.state = TransportState.READY
        self.last_heartbeat_monotonic_s = time.monotonic()

    def close(self) -> None:
        self.state = TransportState.CLOSED

    def round_trip(self, request: InferenceRequest, adapter: InferenceAdapter) -> InferenceResponse:
        adapter.validate_request(request)
        if self.state not in (TransportState.READY, TransportState.HANDSHAKING):
            raise RuntimeError(f"inference transport is not ready: {self.state.value}")
        self.state = TransportState.DRAINING
        t0 = time.monotonic()
        self._write_request(request)
        self.slot.req_ready.set()
        if not self.slot.res_ready.wait(timeout=max(self.timeout_ms, 1.0) / 1000.0):
            self.state = TransportState.FAILED
            heartbeat_age = (time.monotonic() - self.last_heartbeat_monotonic_s) * 1000.0
            raise TimeoutError(
                timeout_message(
                    request_id=request.request_id,
                    trace_id=request.trace_id,
                    request_kind=request.request_kind.value,
                    queue_depth=int(self.slot.req_count[0]),
                    heartbeat_age_ms=heartbeat_age,
                    transport_state=self.state.value,
                    timeout_ms=self.timeout_ms,
                )
            )
        wait_ms = (time.monotonic() - t0) * 1000.0
        self.last_heartbeat_monotonic_s = time.monotonic()
        self.slot.res_ready.clear()
        response = self._read_response(request, wait_ms=wait_ms, adapter=adapter)
        adapter.assert_response(response)
        self.state = TransportState.READY
        self.generation += 1
        return response

    def _write_request(self, request: InferenceRequest) -> None:
        self.slot.res_ready.clear()
        self.slot.req_kind[0] = REQUEST_KIND_TO_CODE[request.request_kind]
        if request.request_kind in (
            InferenceRequestKind.GLOBAL_GRAPH_POLICY_VALUE,
            InferenceRequestKind.GRAPH_PAIR_POLICY_VALUE,
        ):
            self._write_graph(request.payload["graph_batch"])
            return
        payload = request.payload
        count = int(payload["count"])
        tensor = np.asarray(payload["tensor"], dtype=np.float32).reshape(count, 13, 33, 33)
        np.copyto(self.slot.req_tensor[:count], tensor)
        self.slot.req_count[0] = count
        self.slot.req_candidate_count[:count] = 0
        self.slot.req_pair_count[:count] = 0
        if request.request_kind in (
            InferenceRequestKind.SPARSE_POLICY_VALUE,
            InferenceRequestKind.PAIR_SCORING,
            InferenceRequestKind.SPARSE_PAIR_POLICY_VALUE,
        ):
            self._write_sparse(payload, count)
        if request.request_kind in (InferenceRequestKind.PAIR_SCORING, InferenceRequestKind.SPARSE_PAIR_POLICY_VALUE):
            self._write_pairs(payload, count)

    def _write_sparse(self, payload: Any, count: int) -> None:
        candidate_indices = np.asarray(payload["candidate_indices"], dtype=np.int64)
        candidate_features = np.asarray(payload["candidate_features"], dtype=np.float32)
        candidate_mask = np.asarray(payload["candidate_mask"], dtype=np.uint8)
        k = int(candidate_indices.shape[1])
        if k > MAX_CANDIDATES:
            raise ValueError(f"candidate count {k} exceeds MAX_CANDIDATES {MAX_CANDIDATES}")
        self.slot.req_candidate_count[:count] = k
        self.slot.req_candidate_indices[:count, :k] = candidate_indices[:count, :k]
        self.slot.req_candidate_features[:count, :k] = candidate_features[:count, :k]
        self.slot.req_candidate_mask[:count, :k] = candidate_mask[:count, :k]
        if k < MAX_CANDIDATES:
            self.slot.req_candidate_mask[:count, k:] = 0

    def _write_pairs(self, payload: Any, count: int) -> None:
        pair_indices = np.asarray(payload["pair_candidate_indices"], dtype=np.int64)
        pair_mask = np.asarray(payload["pair_candidate_mask"], dtype=np.uint8)
        p = int(pair_indices.shape[1])
        if p > MAX_PAIR_CANDIDATES:
            raise ValueError(f"pair count {p} exceeds MAX_PAIR_CANDIDATES {MAX_PAIR_CANDIDATES}")
        self.slot.req_pair_count[:count] = p
        self.slot.req_pair_indices[:count, :p] = pair_indices[:count, :p]
        self.slot.req_pair_mask[:count, :p] = pair_mask[:count, :p]
        if p < MAX_PAIR_CANDIDATES:
            self.slot.req_pair_mask[:count, p:] = 0

    def _write_graph(self, graph_batch: Any) -> None:
        from hexorl.graph.tensorize import validate_graph_ipc_capacity

        validate_graph_ipc_capacity(graph_batch)
        token_count = int(np.asarray(graph_batch.token_features).shape[0])
        legal_count = int(np.asarray(graph_batch.legal_qr).shape[0])
        opp_count = int(np.asarray(graph_batch.opp_legal_qr).shape[0])
        pair_count = int(np.asarray(graph_batch.pair_token_indices).shape[0])
        if token_count > MAX_GRAPH_TOKENS:
            raise ValueError(f"graph token count {token_count} exceeds MAX_GRAPH_TOKENS {MAX_GRAPH_TOKENS}")
        if legal_count > MAX_GRAPH_ACTIONS or opp_count > MAX_GRAPH_ACTIONS:
            raise ValueError("graph action count exceeds shared-memory capacity")
        if pair_count > MAX_GRAPH_PAIRS:
            raise ValueError(f"graph pair row count {pair_count} exceeds MAX_GRAPH_PAIRS {MAX_GRAPH_PAIRS}")
        self.slot.req_count[0] = 1
        self.slot.req_candidate_count[:1] = 0
        self.slot.req_pair_count[:1] = 0
        self.slot.req_graph_meta[:] = (
            int(graph_batch.schema_version),
            int(graph_batch.relation_schema_version),
            token_count,
            legal_count,
            opp_count,
            pair_count,
            MAX_GRAPH_TOKENS,
            MAX_GRAPH_ACTIONS,
        )
        self.slot.req_graph_token_features.fill(0.0)
        self.slot.req_graph_token_type.fill(0)
        self.slot.req_graph_token_qr.fill(0)
        self.slot.req_graph_token_mask.fill(0)
        self.slot.req_graph_legal_token_indices.fill(-1)
        self.slot.req_graph_legal_qr.fill(0)
        self.slot.req_graph_legal_mask.fill(0)
        self.slot.req_graph_opp_legal_qr.fill(0)
        self.slot.req_graph_opp_legal_mask.fill(0)
        self.slot.req_graph_pair_token_indices.fill(-1)
        self.slot.req_graph_pair_first_indices.fill(-1)
        self.slot.req_graph_pair_second_indices.fill(-1)
        self.slot.req_graph_relation_type.fill(0)
        self.slot.req_graph_relation_bias.fill(0.0)
        self.slot.req_graph_token_features[:token_count] = np.asarray(graph_batch.token_features, dtype=np.float32)
        self.slot.req_graph_token_type[:token_count] = np.asarray(graph_batch.token_type, dtype=np.int16)
        self.slot.req_graph_token_qr[:token_count] = np.asarray(graph_batch.token_qr, dtype=np.int32)
        self.slot.req_graph_token_mask[:token_count] = np.asarray(graph_batch.token_mask, dtype=np.uint8)
        self.slot.req_graph_legal_token_indices[:legal_count] = np.asarray(graph_batch.legal_token_indices, dtype=np.int64)
        self.slot.req_graph_legal_qr[:legal_count] = np.asarray(graph_batch.legal_qr, dtype=np.int32)
        self.slot.req_graph_legal_mask[:legal_count] = np.asarray(graph_batch.legal_mask, dtype=np.uint8)
        if opp_count:
            self.slot.req_graph_opp_legal_qr[:opp_count] = np.asarray(graph_batch.opp_legal_qr, dtype=np.int32)
            self.slot.req_graph_opp_legal_mask[:opp_count] = np.asarray(graph_batch.opp_legal_mask, dtype=np.uint8)
        if pair_count:
            self.slot.req_graph_pair_token_indices[:pair_count] = np.asarray(graph_batch.pair_token_indices, dtype=np.int64)
            self.slot.req_graph_pair_first_indices[:pair_count] = np.asarray(graph_batch.pair_first_indices, dtype=np.int64)
            self.slot.req_graph_pair_second_indices[:pair_count] = np.asarray(graph_batch.pair_second_indices, dtype=np.int64)
        self.slot.req_graph_relation_type[:token_count, :token_count] = np.asarray(graph_batch.relation_type, dtype=np.int16)
        self.slot.req_graph_relation_bias[:, :token_count, :token_count] = np.asarray(graph_batch.relation_bias, dtype=np.float32)

    def _read_response(self, request: InferenceRequest, *, wait_ms: float, adapter: InferenceAdapter) -> InferenceResponse:
        count = int(self.slot.req_count[0])
        heads: dict[str, Any] = {}
        if request.request_kind in (
            InferenceRequestKind.GLOBAL_GRAPH_POLICY_VALUE,
            InferenceRequestKind.GRAPH_PAIR_POLICY_VALUE,
        ):
            legal_count = int(self.slot.res_graph_meta[2])
            opp_count = int(self.slot.res_graph_meta[3])
            pair_count = int(self.slot.res_graph_meta[4])
            heads = {
                "policy_place": np.array(self.slot.res_graph_place_logits[:legal_count], copy=True),
                "opp_policy": np.array(self.slot.res_graph_opp_logits[:opp_count], copy=True),
                "policy_pair_first": np.array(self.slot.res_graph_pair_first_logits[:legal_count], copy=True),
                "policy_pair_joint": np.array(self.slot.res_graph_pair_logits[:pair_count], copy=True),
                "policy_pair_second": np.array(self.slot.res_graph_pair_second_logits[:pair_count], copy=True),
                "regret_rank": np.array(self.slot.res_graph_regret_rank[:1], copy=True),
                "value": np.array(self.slot.res_value[:1], copy=True),
                "metadata": {
                    "schema_version": int(self.slot.res_graph_meta[0]),
                    "relation_schema_version": int(self.slot.res_graph_meta[1]),
                    "legal_count": legal_count,
                    "opp_legal_count": opp_count,
                    "pair_count": pair_count,
                    "prior_source": "global_graph",
                    "legal_qr": np.array(self.slot.req_graph_legal_qr[:legal_count], copy=True),
                    "legal_mask": np.array(self.slot.req_graph_legal_mask[:legal_count].astype(bool), copy=True),
                },
            }
        else:
            k = int(np.max(self.slot.req_candidate_count[:count])) if count else 0
            p = int(np.max(self.slot.req_pair_count[:count])) if count else 0
            heads = {
                "policy": np.array(self.slot.res_policy[:count], copy=True).ravel(),
                "value": np.array(self.slot.res_value[:count], copy=True),
            }
            if k:
                heads["sparse_policy"] = np.array(self.slot.res_sparse_logits[:count, :k], copy=True)
            if p:
                heads["pair_policy"] = np.array(self.slot.res_pair_logits[:count, :p], copy=True)
            if request.request_kind == InferenceRequestKind.REGRET_RANK_POLICY_VALUE:
                heads["regret_rank"] = np.array(self.slot.res_regret_rank[:count], copy=True)
        telemetry = InferenceTelemetry(
            request_id=request.request_id,
            trace_id=request.trace_id,
            request_kind=request.request_kind.value,
            transport_state=self.state.value,
            queue_depth=count,
            batch_size=count,
            wait_ms=wait_ms,
            heartbeat_age_ms=0.0,
            adapter_name=adapter.name,
        ).to_dict()
        return InferenceResponse(
            request_id=request.request_id,
            trace_id=request.trace_id,
            protocol_version=request.protocol_version,
            request_kind=request.request_kind,
            response_schema_version=request.response_schema_version,
            manifest_hash=request.manifest_hash,
            status="ok",
            response_generation=self.generation + 1,
            output_contract=request.output_contract,
            head_outputs=heads,
            telemetry=telemetry,
        )
