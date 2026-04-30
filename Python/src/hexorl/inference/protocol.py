"""Typed inference protocol contracts for the shared-memory boundary."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np

from hexorl.inference.shm_queue import (
    BOARD_AREA,
    BOARD_SIZE,
    MAX_CANDIDATES,
    MAX_GRAPH_ACTIONS,
    MAX_GRAPH_PAIRS,
    MAX_GRAPH_TOKENS,
    MAX_PAIR_CANDIDATES,
    NUM_CHANNELS,
)


PROTOCOL_VERSION = 1
REQUEST_SCHEMA_VERSION = 1
RESPONSE_SCHEMA_VERSION = 1


class InferenceProtocolMismatch(RuntimeError):
    """Raised before enqueue when negotiated protocol metadata does not match."""


class InferencePayloadValidationError(ValueError):
    """Raised when a typed request payload violates its declared contract."""


class InferenceOutputValidationError(RuntimeError):
    """Raised when model outputs fail response contract validation."""


class InferenceRequestKind(str, Enum):
    DENSE_POLICY_VALUE = "dense_policy_value"
    SPARSE_POLICY_VALUE = "sparse_policy_value"
    GLOBAL_GRAPH_POLICY_VALUE = "global_graph_policy_value"
    PAIR_SCORING = "pair_scoring"
    SPARSE_PAIR_POLICY_VALUE = "sparse_pair_policy_value"
    GRAPH_PAIR_POLICY_VALUE = "graph_pair_policy_value"
    REGRET_RANK_POLICY_VALUE = "regret_rank_policy_value"


REQUEST_KIND_TO_CODE: dict[InferenceRequestKind, int] = {
    InferenceRequestKind.DENSE_POLICY_VALUE: 1,
    InferenceRequestKind.SPARSE_POLICY_VALUE: 2,
    InferenceRequestKind.GLOBAL_GRAPH_POLICY_VALUE: 3,
    InferenceRequestKind.PAIR_SCORING: 4,
    InferenceRequestKind.SPARSE_PAIR_POLICY_VALUE: 5,
    InferenceRequestKind.GRAPH_PAIR_POLICY_VALUE: 6,
    InferenceRequestKind.REGRET_RANK_POLICY_VALUE: 7,
}
REQUEST_CODE_TO_KIND = {code: kind for kind, code in REQUEST_KIND_TO_CODE.items()}


@dataclass(frozen=True)
class InferenceProtocolManifest:
    protocol_version: int
    request_kind: tuple[str, ...]
    request_schema_version: int
    response_schema_version: int
    model_family: str
    model_spec_version: str
    input_contract: str
    output_contract: str
    action_contract: str
    graph_schema_version: int
    relation_schema_version: int
    candidate_contract_version: int
    pair_action_contract_version: int
    ffi_protocol_version: int
    legal_row_encoding: str
    history_row_encoding: str
    pair_row_encoding: str
    heads: tuple[str, ...]
    adapter_name: str
    adapter_version: int
    transport: str
    max_batch_size: int
    max_legal_rows: int
    max_candidate_rows: int
    max_pair_rows: int
    max_graph_tokens: int
    max_graph_relations: int
    timeout_ms: float
    heartbeat_interval_ms: float
    created_by_git_sha: str
    config_hash: str

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def supports(self, kind: InferenceRequestKind) -> bool:
        return kind.value in self.request_kind


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def ndarray_hash(array: np.ndarray | None) -> str:
    if array is None:
        return "none"
    arr = np.asarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(json.dumps(arr.shape).encode("utf-8"))
    h.update(np.ascontiguousarray(arr).view(np.uint8))
    return h.hexdigest()


def default_protocol_manifest(
    *,
    max_batch_size: int,
    timeout_ms: float,
    heads: tuple[str, ...] = ("policy", "value"),
    adapter_name: str = "hexorl-shm",
    model_family: str = "hexorl",
    model_spec_version: str = "v2",
    config_hash: str = "unconfigured",
) -> InferenceProtocolManifest:
    from hexorl.contracts.candidates import CANDIDATE_SCHEMA_VERSION
    from hexorl.graph.semantic_builder import GRAPH_SCHEMA_VERSION, RELATION_SCHEMA_VERSION

    kinds = tuple(kind.value for kind in InferenceRequestKind)
    return InferenceProtocolManifest(
        protocol_version=PROTOCOL_VERSION,
        request_kind=kinds,
        request_schema_version=REQUEST_SCHEMA_VERSION,
        response_schema_version=RESPONSE_SCHEMA_VERSION,
        model_family=model_family,
        model_spec_version=model_spec_version,
        input_contract=f"dense:{NUM_CHANNELS}x{BOARD_SIZE}x{BOARD_SIZE}",
        output_contract=f"policy:{BOARD_AREA};value:scalar",
        action_contract="board_index:v1",
        graph_schema_version=GRAPH_SCHEMA_VERSION,
        relation_schema_version=RELATION_SCHEMA_VERSION,
        candidate_contract_version=CANDIDATE_SCHEMA_VERSION,
        pair_action_contract_version=1,
        ffi_protocol_version=1,
        legal_row_encoding="rust-legal-row-u64-v1",
        history_row_encoding="rust-compact-history-row-v1",
        pair_row_encoding="candidate-index-pair-v1",
        heads=tuple(heads),
        adapter_name=adapter_name,
        adapter_version=1,
        transport="shared_memory_v2",
        max_batch_size=int(max_batch_size),
        max_legal_rows=MAX_GRAPH_ACTIONS,
        max_candidate_rows=MAX_CANDIDATES,
        max_pair_rows=max(MAX_PAIR_CANDIDATES, MAX_GRAPH_PAIRS),
        max_graph_tokens=MAX_GRAPH_TOKENS,
        max_graph_relations=MAX_GRAPH_TOKENS * MAX_GRAPH_TOKENS,
        timeout_ms=float(timeout_ms),
        heartbeat_interval_ms=max(1.0, min(float(timeout_ms) / 4.0, 250.0)),
        created_by_git_sha=os.environ.get("HEXO_GIT_SHA", "unknown"),
        config_hash=config_hash,
    )


@dataclass(frozen=True)
class InferenceHandshake:
    client_manifest_hash: str
    server_manifest_hash: str
    request_kind: InferenceRequestKind
    accepted: bool


def negotiate_protocol(
    *,
    client_manifest: InferenceProtocolManifest,
    server_manifest: InferenceProtocolManifest,
    request_kind: InferenceRequestKind,
) -> InferenceHandshake:
    if client_manifest.protocol_version != server_manifest.protocol_version:
        raise InferenceProtocolMismatch(
            "inference protocol version mismatch: "
            f"client={client_manifest.protocol_version} server={server_manifest.protocol_version}"
        )
    if client_manifest.request_schema_version != server_manifest.request_schema_version:
        raise InferenceProtocolMismatch(
            "inference request schema mismatch: "
            f"client={client_manifest.request_schema_version} server={server_manifest.request_schema_version}"
        )
    if client_manifest.response_schema_version != server_manifest.response_schema_version:
        raise InferenceProtocolMismatch(
            "inference response schema mismatch: "
            f"client={client_manifest.response_schema_version} server={server_manifest.response_schema_version}"
        )
    if not client_manifest.supports(request_kind) or not server_manifest.supports(request_kind):
        raise InferenceProtocolMismatch(f"inference request kind is unsupported: {request_kind.value}")
    client_hash = client_manifest.hash()
    server_hash = server_manifest.hash()
    if client_hash != server_hash:
        raise InferenceProtocolMismatch(
            "inference manifest hash mismatch: "
            f"client={client_hash[:12]} server={server_hash[:12]} kind={request_kind.value}"
        )
    return InferenceHandshake(
        client_manifest_hash=client_hash,
        server_manifest_hash=server_hash,
        request_kind=request_kind,
        accepted=True,
    )


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    trace_id: str
    protocol_version: int
    request_kind: InferenceRequestKind
    request_schema_version: int
    response_schema_version: int
    output_contract: str
    manifest_hash: str
    position_hash: str
    history_hash: str
    legal_hash: str
    pair_hash: str
    adapter_capability: str
    slot_generation: int
    deadline_monotonic_s: float
    payload_schema_version: int
    payload_kind: str
    payload: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    trace_id: str
    protocol_version: int
    request_kind: InferenceRequestKind
    response_schema_version: int
    manifest_hash: str
    status: str
    response_generation: int
    output_contract: str
    head_outputs: Mapping[str, Any] = field(repr=False)
    telemetry: Mapping[str, Any]
    warnings: tuple[str, ...] = ()
    error_code: str | None = None

    def require_ok(self) -> None:
        if self.status != "ok":
            raise InferenceOutputValidationError(
                f"inference response failed request_id={self.request_id} "
                f"kind={self.request_kind.value} error_code={self.error_code}"
            )
        for name, value in self.head_outputs.items():
            if isinstance(value, np.ndarray) and not np.isfinite(value).all():
                raise InferenceOutputValidationError(
                    f"inference response head {name} contains non-finite values "
                    f"request_id={self.request_id} kind={self.request_kind.value}"
                )


def make_request(
    *,
    kind: InferenceRequestKind,
    manifest: InferenceProtocolManifest,
    payload: Mapping[str, Any],
    deadline_monotonic_s: float,
    slot_generation: int,
    trace_id: str | None = None,
) -> InferenceRequest:
    request_id = uuid.uuid4().hex
    position_hash = ndarray_hash(payload.get("tensor"))
    history_hash = ndarray_hash(payload.get("history_rows"))
    legal_hash = ndarray_hash(payload.get("candidate_indices"))
    pair_hash = ndarray_hash(payload.get("pair_candidate_indices"))
    if "graph_batch" in payload:
        graph = payload["graph_batch"]
        legal_hash = ndarray_hash(getattr(graph, "legal_qr", None))
        pair_hash = ndarray_hash(getattr(graph, "pair_token_indices", None))
    return InferenceRequest(
        request_id=request_id,
        trace_id=trace_id or request_id,
        protocol_version=manifest.protocol_version,
        request_kind=kind,
        request_schema_version=manifest.request_schema_version,
        response_schema_version=manifest.response_schema_version,
        output_contract=manifest.output_contract,
        manifest_hash=manifest.hash(),
        position_hash=position_hash,
        history_hash=history_hash,
        legal_hash=legal_hash,
        pair_hash=pair_hash,
        adapter_capability=kind.value,
        slot_generation=int(slot_generation),
        deadline_monotonic_s=float(deadline_monotonic_s),
        payload_schema_version=manifest.request_schema_version,
        payload_kind=kind.value,
        payload=payload,
    )
