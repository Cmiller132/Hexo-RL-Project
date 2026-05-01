"""Client-side contract-walking shared-memory transport."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

import numpy as np

from hexorl.inference.control import (
    CTL_CONTRACT_HASH,
    CTL_DEADLINE_NS,
    CTL_ENQUEUED_NS,
    CTL_GENERATION,
    CTL_LAYOUT_HASH,
    CTL_OPCODE,
    CTL_STATUS,
    STATUS_DRAINING,
    STATUS_OK,
    STATUS_READY,
    hash_word,
    read_all_dyn_dims,
    write_dyn_dims,
)
from hexorl.inference.protocol import InferenceOutputValidationError, InferenceRequest, InferenceResponse
from hexorl.inference.telemetry import InferenceTelemetry, timeout_message


class TransportState(str, Enum):
    CREATED = "created"
    HANDSHAKING = "handshaking"
    READY = "ready"
    DRAINING = "draining"
    CLOSED = "closed"
    FAILED = "failed"


class ShmTransport:
    def __init__(self, *, worker_id: int, slot: Any, timeout_ms: float, manifest: Any):
        self.worker_id = int(worker_id)
        self.slot = slot
        self.timeout_ms = float(timeout_ms)
        self.manifest = manifest
        self.state = TransportState.CREATED
        self.generation = int(self.slot.control[CTL_GENERATION])
        self.last_heartbeat_monotonic_s = time.monotonic()

    def mark_ready(self) -> None:
        if self.state == TransportState.CLOSED:
            raise RuntimeError("cannot ready a closed inference transport")
        self.state = TransportState.READY
        self.last_heartbeat_monotonic_s = time.monotonic()

    def close(self) -> None:
        self.state = TransportState.CLOSED

    def round_trip(self, request: InferenceRequest) -> InferenceResponse:
        if self.state not in (TransportState.READY, TransportState.HANDSHAKING):
            raise RuntimeError(f"inference transport is not ready: {self.state.value}")
        operation = self.manifest.model_contract.operation(request.operation_name)
        self._validate_payload(operation, request.payload)
        self.state = TransportState.DRAINING
        t0 = time.monotonic()
        dims = self._write_request(request, operation)
        self.slot.req_ready.set()
        if not self.slot.res_ready.wait(timeout=max(self.timeout_ms, 1.0) / 1000.0):
            self.state = TransportState.FAILED
            raise TimeoutError(self._timeout_text(request, dims))
        wait_ms = (time.monotonic() - t0) * 1000.0
        self.last_heartbeat_monotonic_s = time.monotonic()
        self.slot.res_ready.clear()
        response = self._read_response(request, operation, wait_ms=wait_ms)
        response.require_ok()
        self.state = TransportState.READY
        self.generation += 1
        return response

    def _validate_payload(self, operation: Any, payload: MappingLike) -> None:
        missing = sorted(set(operation.required_inputs) - set(payload.keys()))
        if missing:
            raise ValueError(f"inference payload missing required tensors for {operation.name}: {missing}")

    def _write_request(self, request: InferenceRequest, operation: Any) -> dict[str, int]:
        slot = self.slot
        slot.clear_response_payload()
        slot.clear_request_payload()
        dims: dict[str, int] = {}
        for spec in operation.layout.input_tensors:
            arr = np.asarray(request.payload[spec.name])
            arr = _normalize_request_array(arr, spec)
            _merge_dims(dims, _dims_from_shape(spec, arr.shape))
            _copy_into(slot.request_tensor(spec.name), arr, spec, dims)
        dims.setdefault("B", 1)
        slot.control[CTL_GENERATION] = self.generation + 1
        slot.control[CTL_OPCODE] = int(request.operation_code)
        slot.control[CTL_STATUS] = STATUS_DRAINING
        slot.control[CTL_LAYOUT_HASH] = hash_word(operation.layout_hash)
        slot.control[CTL_CONTRACT_HASH] = hash_word(request.model_contract_hash)
        slot.control[CTL_ENQUEUED_NS] = time.monotonic_ns()
        slot.control[CTL_DEADLINE_NS] = int(request.deadline_monotonic_s * 1_000_000_000)
        write_dyn_dims(slot.control, dims)
        slot.control[CTL_STATUS] = STATUS_READY
        return dims

    def _read_response(self, request: InferenceRequest, operation: Any, *, wait_ms: float) -> InferenceResponse:
        slot = self.slot
        if int(slot.control[CTL_STATUS]) != STATUS_OK:
            raise InferenceOutputValidationError(f"inference operation failed in server: {request.operation_name}")
        dims = read_all_dyn_dims(slot.control, _operation_dims(operation, self.manifest.model_contract))
        heads = {}
        for head_name in operation.output_heads:
            if slot.has_response_tensor(head_name):
                head = self.manifest.model_contract.head(head_name)
                heads[head_name] = np.array(_slice_view(slot.response_tensor(head_name), head.tensor, dims), copy=True)
        telemetry = InferenceTelemetry(
            request_id=request.request_id,
            trace_id=request.trace_id,
            operation_name=request.operation_name,
            transport_state=self.state.value,
            queue_depth=int(dims.get("B", 1)),
            batch_size=int(dims.get("B", 1)),
            wait_ms=wait_ms,
            heartbeat_age_ms=0.0,
            adapter_name="contract_arena",
        ).to_dict()
        telemetry["operation_code"] = int(request.operation_code)
        telemetry["layout_hash"] = operation.layout_hash
        telemetry["model_contract_hash"] = request.model_contract_hash
        return InferenceResponse(
            request_id=request.request_id,
            trace_id=request.trace_id,
            protocol_version=request.protocol_version,
            operation_name=request.operation_name,
            operation_code=request.operation_code,
            response_schema_version=request.response_schema_version,
            model_contract_hash=request.model_contract_hash,
            manifest_hash=request.manifest_hash,
            status="ok",
            response_generation=int(slot.control[CTL_GENERATION]),
            head_outputs=heads,
            telemetry=telemetry,
        )

    def _timeout_text(self, request: InferenceRequest, dims: dict[str, int]) -> str:
        heartbeat_age = (time.monotonic() - self.last_heartbeat_monotonic_s) * 1000.0
        return timeout_message(
            request_id=request.request_id,
            trace_id=request.trace_id,
            operation_name=request.operation_name,
            queue_depth=int(dims.get("B", 1)),
            heartbeat_age_ms=heartbeat_age,
            transport_state=self.state.value,
            timeout_ms=self.timeout_ms,
        )


MappingLike = dict[str, Any]


def _normalize_request_array(arr: np.ndarray, spec: Any) -> np.ndarray:
    if spec.batching == "stack_over_b" and arr.ndim == len(spec.shape) - 1:
        arr = arr.reshape((1, *arr.shape))
    return np.ascontiguousarray(arr)


def _dims_from_shape(spec: Any, shape: tuple[int, ...]) -> dict[str, int]:
    if len(shape) != len(spec.shape):
        raise ValueError(f"tensor {spec.name} rank mismatch: expected {spec.shape} got {shape}")
    dims: dict[str, int] = {}
    for declared, actual in zip(spec.shape, shape):
        if isinstance(declared, str):
            if declared in dims and dims[declared] != int(actual):
                raise ValueError(f"dimension {declared} has conflicting values")
            dims[declared] = int(actual)
        elif int(declared) != int(actual):
            raise ValueError(f"tensor {spec.name} dimension mismatch: expected {declared} got {actual}")
    return dims


def _merge_dims(target: dict[str, int], incoming: dict[str, int]) -> None:
    for name, value in incoming.items():
        if name in target and target[name] != int(value):
            raise ValueError(f"dimension {name} has conflicting values")
        target[name] = int(value)


def _copy_into(view: np.ndarray, arr: np.ndarray, spec: Any, dims: dict[str, int]) -> None:
    view[_slices_for(spec, dims)] = arr.astype(view.dtype, copy=False)


def _slice_view(view: np.ndarray, spec: Any, dims: dict[str, int]) -> np.ndarray:
    return view[_slices_for(spec, dims)]


def _slices_for(spec: Any, dims: dict[str, int]) -> tuple[slice, ...]:
    out = []
    for dim in spec.shape:
        out.append(slice(0, dims.get(dim, 0) if isinstance(dim, str) else int(dim)))
    return tuple(out)


def _operation_dims(operation: Any, contract: Any) -> tuple[str, ...]:
    names: list[str] = []
    for spec in operation.layout.input_tensors:
        names.extend(spec.dynamic_dims())
    for head_name in operation.output_heads:
        names.extend(contract.head(head_name).tensor.dynamic_dims())
    return tuple(dict.fromkeys(names))


__all__ = ["ShmTransport", "TransportState"]
