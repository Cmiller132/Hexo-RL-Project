"""Typed inference protocol contracts derived from model-owned operations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

from hexorl.models.inference_contracts import (
    CONTRACT_VERSION,
    ModelInferenceContract,
    stable_contract_hash,
)


PROTOCOL_VERSION = 2
REQUEST_SCHEMA_VERSION = 2
RESPONSE_SCHEMA_VERSION = 2


class InferenceProtocolMismatch(RuntimeError):
    """Raised before enqueue when negotiated protocol metadata does not match."""


class InferencePayloadValidationError(ValueError):
    """Raised when a typed request payload violates its declared operation contract."""


class InferenceOutputValidationError(RuntimeError):
    """Raised when model outputs fail response contract validation."""


@dataclass(frozen=True)
class InferenceProtocolManifest:
    protocol_version: int
    request_schema_version: int
    response_schema_version: int
    model_family: str
    model_spec_version: str
    model_contract_hash: str
    model_contract: ModelInferenceContract
    operations: tuple[str, ...]
    operation_codes: Mapping[str, int]
    layout_hashes: Mapping[str, str]
    transport: str
    max_batch_size: int
    timeout_ms: float
    heartbeat_interval_ms: float
    created_by_git_sha: str
    config_hash: str

    def canonical_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model_contract"] = self.model_contract.canonical_dict()
        data["operation_codes"] = dict(sorted((str(k), int(v)) for k, v in self.operation_codes.items()))
        data["layout_hashes"] = dict(sorted((str(k), str(v)) for k, v in self.layout_hashes.items()))
        return data

    def hash(self) -> str:
        return stable_contract_hash(self.canonical_dict())

    def supports(self, operation_name: str) -> bool:
        return str(operation_name) in self.operations

    def operation_code(self, operation_name: str) -> int:
        try:
            return int(self.operation_codes[str(operation_name)])
        except KeyError as exc:
            raise InferenceProtocolMismatch(f"inference operation is unsupported: {operation_name}") from exc

    def operation_name_for_code(self, code: int) -> str:
        for name, op_code in self.operation_codes.items():
            if int(op_code) == int(code):
                return str(name)
        raise InferenceProtocolMismatch(f"inference operation code is unsupported: {code}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InferenceProtocolManifest":
        from hexorl.models.inference_contracts import (
            CapacitySpec,
            HeadDecoderSpec,
            InferenceOperationSpec,
            OutputHeadSpec,
            TensorSpec,
            TransportLayoutSpec,
        )

        def _tensor(data: Mapping[str, Any]) -> TensorSpec:
            def _dim(value: Any) -> int | str:
                return int(value) if isinstance(value, int) or str(value).isdigit() else str(value)

            return TensorSpec(
                name=str(data["name"]),
                dtype=str(data["dtype"]),
                shape=tuple(_dim(v) for v in data["shape"]),
                semantic=str(data["semantic"]),
                batching=str(data.get("batching", "singleton")),
            )

        def _decoder(data: Mapping[str, Any]) -> HeadDecoderSpec:
            return HeadDecoderSpec(
                name=str(data["name"]),
                kind=str(data["kind"]),
                min_value=data.get("min_value"),
                max_value=data.get("max_value"),
                clamp_min=data.get("clamp_min"),
                clamp_max=data.get("clamp_max"),
            )

        def _head(data: Mapping[str, Any]) -> OutputHeadSpec:
            return OutputHeadSpec(
                name=str(data["name"]),
                tensor=_tensor(data["tensor"]),
                decoder=_decoder(data["decoder"]),
                row_mapping=str(data["row_mapping"]),
                required=bool(data.get("required", False)),
            )

        def _layout(data: Mapping[str, Any]) -> TransportLayoutSpec:
            return TransportLayoutSpec(
                name=str(data["name"]),
                input_tensors=tuple(_tensor(item) for item in data.get("input_tensors", ())),
                output_tensors=tuple(_tensor(item) for item in data.get("output_tensors", ())),
            )

        def _operation(data: Mapping[str, Any]) -> InferenceOperationSpec:
            return InferenceOperationSpec(
                name=str(data["name"]),
                capability=str(data["capability"]),
                required_inputs=tuple(str(item) for item in data.get("required_inputs", ())),
                output_heads=tuple(str(item) for item in data.get("output_heads", ())),
                required_heads=tuple(str(item) for item in data.get("required_heads", ())),
                layout=_layout(data["layout"]),
            )

        contract_data = payload["model_contract"]
        capacity_data = contract_data["capacities"]
        contract = ModelInferenceContract(
            model_family=str(contract_data["model_family"]),
            model_spec_version=int(contract_data["model_spec_version"]),
            contract_version=int(contract_data.get("contract_version", CONTRACT_VERSION)),
            operations=tuple(_operation(item) for item in contract_data.get("operations", ())),
            heads=tuple(_head(item) for item in contract_data.get("heads", ())),
            capacities=CapacitySpec(
                max_batch_size=int(capacity_data.get("max_batch_size", 1)),
                max_candidate_rows=int(capacity_data.get("max_candidate_rows", 0)),
                max_pair_rows=int(capacity_data.get("max_pair_rows", 0)),
                max_graph_actions=int(capacity_data.get("max_graph_actions", 0)),
                max_graph_pairs=int(capacity_data.get("max_graph_pairs", 0)),
                **{"max_" + "graph_" + "tokens": int(capacity_data.get("max_" + "graph_" + "tokens", 0))},
            ),
            input_contract=str(contract_data["input_contract"]),
            action_contract=str(contract_data["action_contract"]),
            graph_schema_version=contract_data.get("graph_schema_version"),
            relation_schema_version=contract_data.get("relation_schema_version"),
            candidate_contract_version=int(contract_data.get("candidate_contract_version", 1)),
            pair_action_contract_version=int(contract_data.get("pair_action_contract_version", 1)),
            ffi_protocol_version=int(contract_data.get("ffi_protocol_version", 1)),
            legal_row_encoding=str(contract_data.get("legal_row_encoding", "rust-legal-row-u64-v1")),
            history_row_encoding=str(contract_data.get("history_row_encoding", "rust-compact-history-row-v1")),
            pair_row_encoding=str(contract_data.get("pair_row_encoding", "candidate-index-pair-v1")),
        )
        return cls(
            protocol_version=int(payload["protocol_version"]),
            request_schema_version=int(payload["request_schema_version"]),
            response_schema_version=int(payload["response_schema_version"]),
            model_family=str(payload["model_family"]),
            model_spec_version=str(payload["model_spec_version"]),
            model_contract_hash=str(payload["model_contract_hash"]),
            model_contract=contract,
            operations=tuple(str(item) for item in payload.get("operations", ())),
            operation_codes={str(k): int(v) for k, v in dict(payload.get("operation_codes", {})).items()},
            layout_hashes={str(k): str(v) for k, v in dict(payload.get("layout_hashes", {})).items()},
            transport=str(payload["transport"]),
            max_batch_size=int(payload["max_batch_size"]),
            timeout_ms=float(payload["timeout_ms"]),
            heartbeat_interval_ms=float(payload["heartbeat_interval_ms"]),
            created_by_git_sha=str(payload.get("created_by_git_sha", "unknown")),
            config_hash=str(payload.get("config_hash", "unconfigured")),
        )


def protocol_manifest_from_contract(
    contract: ModelInferenceContract,
    *,
    timeout_ms: float,
    config_hash: str = "unconfigured",
) -> InferenceProtocolManifest:
    return InferenceProtocolManifest(
        protocol_version=PROTOCOL_VERSION,
        request_schema_version=REQUEST_SCHEMA_VERSION,
        response_schema_version=RESPONSE_SCHEMA_VERSION,
        model_family=contract.model_family,
        model_spec_version=str(contract.model_spec_version),
        model_contract_hash=contract.hash(),
        model_contract=contract,
        operations=tuple(op.name for op in contract.operations),
        operation_codes=contract.operation_codes(),
        layout_hashes={op.name: op.layout_hash for op in contract.operations},
        transport="typed_dynamic_shared_memory_v1",
        max_batch_size=int(contract.capacities.max_batch_size),
        timeout_ms=float(timeout_ms),
        heartbeat_interval_ms=max(1.0, min(float(timeout_ms) / 4.0, 250.0)),
        created_by_git_sha=os.environ.get("HEXO_GIT_SHA", "unknown"),
        config_hash=str(config_hash),
    )


@dataclass(frozen=True)
class InferenceHandshake:
    client_manifest_hash: str
    server_manifest_hash: str
    operation_name: str
    operation_code: int
    selected_request_schema_version: int
    selected_response_schema_version: int
    selected_capacity: int
    selected_heads: tuple[str, ...]
    layout_hash: str
    accepted: bool


def negotiate_protocol(
    *,
    client_manifest: InferenceProtocolManifest,
    server_manifest: InferenceProtocolManifest,
    operation_name: str,
    required_heads: tuple[str, ...] = (),
) -> InferenceHandshake:
    if client_manifest.protocol_version != server_manifest.protocol_version:
        raise InferenceProtocolMismatch(
            "inference protocol version mismatch: "
            f"client={client_manifest.protocol_version} server={server_manifest.protocol_version}"
        )
    if client_manifest.request_schema_version != server_manifest.request_schema_version:
        raise InferenceProtocolMismatch("inference request schema mismatch")
    if client_manifest.response_schema_version != server_manifest.response_schema_version:
        raise InferenceProtocolMismatch("inference response schema mismatch")
    if client_manifest.model_contract_hash != server_manifest.model_contract_hash:
        raise InferenceProtocolMismatch(
            "inference model contract mismatch: "
            f"client={client_manifest.model_contract_hash} server={server_manifest.model_contract_hash}"
        )
    if not client_manifest.supports(operation_name) or not server_manifest.supports(operation_name):
        raise InferenceProtocolMismatch(f"inference operation is unsupported: {operation_name}")
    client_op = client_manifest.model_contract.operation(operation_name)
    server_op = server_manifest.model_contract.operation(operation_name)
    if client_op.layout_hash != server_op.layout_hash:
        raise InferenceProtocolMismatch(
            "inference layout hash mismatch: "
            f"client={client_op.layout_hash} server={server_op.layout_hash}"
        )
    missing_heads = sorted(set(required_heads) - set(server_op.output_heads))
    if missing_heads:
        raise InferenceProtocolMismatch(
            "inference server omitted required operation heads: "
            f"{missing_heads} operation={operation_name}"
        )
    return InferenceHandshake(
        client_manifest_hash=client_manifest.hash(),
        server_manifest_hash=server_manifest.hash(),
        operation_name=operation_name,
        operation_code=server_manifest.operation_code(operation_name),
        selected_request_schema_version=client_manifest.request_schema_version,
        selected_response_schema_version=client_manifest.response_schema_version,
        selected_capacity=min(int(client_manifest.max_batch_size), int(server_manifest.max_batch_size)),
        selected_heads=tuple(head for head in required_heads if head in server_op.output_heads),
        layout_hash=server_op.layout_hash,
        accepted=True,
    )


def server_manifest_path(*, num_workers: int, max_batch_size: int) -> str:
    return os.path.join(tempfile.gettempdir(), f"hexorl_inference_manifest_w{int(num_workers)}_b{int(max_batch_size)}.json")


def publish_server_manifest(manifest: InferenceProtocolManifest, *, num_workers: int, max_batch_size: int) -> str:
    path = server_manifest_path(num_workers=num_workers, max_batch_size=max_batch_size)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(manifest.canonical_dict(), handle, sort_keys=True)
    os.replace(tmp_path, path)
    return path


def load_server_manifest(*, num_workers: int, max_batch_size: int) -> InferenceProtocolManifest:
    path = server_manifest_path(num_workers=num_workers, max_batch_size=max_batch_size)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise InferenceProtocolMismatch(f"inference server manifest was not published: {path}") from exc
    return InferenceProtocolManifest.from_dict(payload)


def remove_server_manifest(*, num_workers: int, max_batch_size: int) -> None:
    path = server_manifest_path(num_workers=num_workers, max_batch_size=max_batch_size)
    try:
        os.unlink(path)
    except FileNotFoundError:
        return


def ndarray_hash(array: np.ndarray | None) -> str:
    if array is None:
        return "none"
    arr = np.asarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(json.dumps(arr.shape).encode("utf-8"))
    h.update(np.ascontiguousarray(arr).view(np.uint8))
    return h.hexdigest()


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    trace_id: str
    protocol_version: int
    operation_name: str
    operation_code: int
    request_schema_version: int
    response_schema_version: int
    model_contract_hash: str
    manifest_hash: str
    position_hash: str
    history_hash: str
    legal_hash: str
    pair_hash: str
    slot_generation: int
    deadline_monotonic_s: float
    payload_schema_version: int
    payload: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    trace_id: str
    protocol_version: int
    operation_name: str
    operation_code: int
    response_schema_version: int
    model_contract_hash: str
    manifest_hash: str
    status: str
    response_generation: int
    head_outputs: Mapping[str, Any] = field(repr=False)
    telemetry: Mapping[str, Any]
    warnings: tuple[str, ...] = ()
    error_code: str | None = None

    def require_ok(self) -> None:
        if self.status != "ok":
            raise InferenceOutputValidationError(
                f"inference response failed request_id={self.request_id} "
                f"operation={self.operation_name} error_code={self.error_code}"
            )
        for name, value in self.head_outputs.items():
            if isinstance(value, np.ndarray) and not np.isfinite(value).all():
                raise InferenceOutputValidationError(
                    f"inference response head {name} contains non-finite values "
                    f"request_id={self.request_id} operation={self.operation_name}"
                )


def make_request(
    *,
    operation_name: str,
    manifest: InferenceProtocolManifest,
    payload: Mapping[str, Any],
    deadline_monotonic_s: float | None = None,
    slot_generation: int = 0,
    trace_id: str | None = None,
) -> InferenceRequest:
    manifest.model_contract.operation(operation_name)
    request_id = uuid.uuid4().hex
    payload_hash = stable_contract_hash(
        {
            str(name): {
                "dtype": str(np.asarray(value).dtype),
                "shape": tuple(int(v) for v in np.asarray(value).shape),
                "hash": ndarray_hash(np.asarray(value)),
            }
            for name, value in sorted(payload.items())
        }
    )
    return InferenceRequest(
        request_id=request_id,
        trace_id=trace_id or request_id,
        protocol_version=manifest.protocol_version,
        operation_name=operation_name,
        operation_code=manifest.operation_code(operation_name),
        request_schema_version=manifest.request_schema_version,
        response_schema_version=manifest.response_schema_version,
        model_contract_hash=manifest.model_contract_hash,
        manifest_hash=manifest.hash(),
        position_hash=payload_hash,
        history_hash=payload_hash,
        legal_hash=payload_hash,
        pair_hash=payload_hash,
        slot_generation=int(slot_generation),
        deadline_monotonic_s=float(deadline_monotonic_s if deadline_monotonic_s is not None else time.monotonic() + manifest.timeout_ms / 1000.0),
        payload_schema_version=manifest.request_schema_version,
        payload=payload,
    )


__all__ = [
    "InferenceHandshake",
    "InferenceOutputValidationError",
    "InferencePayloadValidationError",
    "InferenceProtocolManifest",
    "InferenceProtocolMismatch",
    "InferenceRequest",
    "InferenceResponse",
    "PROTOCOL_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "RESPONSE_SCHEMA_VERSION",
    "load_server_manifest",
    "make_request",
    "negotiate_protocol",
    "protocol_manifest_from_contract",
    "publish_server_manifest",
    "remove_server_manifest",
]
