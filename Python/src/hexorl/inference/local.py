"""In-process evaluator using the same model-owned inference contract."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from hexorl.inference.protocol import InferenceResponse, make_request
from hexorl.inference.server.outputs import decode_outputs
from hexorl.inference.telemetry import InferenceTelemetry


class LocalEvaluator:
    def __init__(self, model: torch.nn.Module, *, manifest, device: torch.device | None = None):
        self.model = model
        self.manifest = manifest
        self.device = device or _model_device(model)
        self.model.eval()

    def evaluate(self, op: str, payload: dict[str, np.ndarray]) -> InferenceResponse:
        operation = self.manifest.model_contract.operation(op)
        request = make_request(operation_name=op, manifest=self.manifest, payload=payload)
        dims, inputs = self._inputs(operation, payload)
        t0 = time.monotonic()
        with torch.inference_mode():
            raw = self.model.forward(**inputs)
        decoded = decode_outputs(raw, operation=operation, contract=self.manifest.model_contract)
        heads = {name: _slice_output(np.asarray(decoded[name]), self.manifest.model_contract.head(name).tensor, dims) for name in operation.output_heads if name in decoded}
        elapsed = (time.monotonic() - t0) * 1000.0
        return InferenceResponse(
            request_id=request.request_id,
            trace_id=request.trace_id,
            protocol_version=request.protocol_version,
            operation_name=op,
            operation_code=request.operation_code,
            response_schema_version=request.response_schema_version,
            model_contract_hash=request.model_contract_hash,
            manifest_hash=request.manifest_hash,
            status="ok",
            response_generation=0,
            head_outputs=heads,
            telemetry=InferenceTelemetry(request.request_id, request.trace_id, op, "local", int(dims.get("B", 1)), int(dims.get("B", 1)), elapsed, 0.0, "local").to_dict(),
        )

    def close(self) -> None:
        return

    def _inputs(self, operation: Any, payload: dict[str, np.ndarray]) -> tuple[dict[str, int], dict[str, torch.Tensor]]:
        dims: dict[str, int] = {}
        inputs = {}
        for spec in operation.layout.input_tensors:
            arr = np.asarray(payload[spec.name])
            if spec.batching == "stack_over_b" and arr.ndim == len(spec.shape) - 1:
                arr = arr.reshape((1, *arr.shape))
            _merge_dims(dims, _dims(spec, arr.shape))
            model_arr = arr if "B" in spec.dynamic_dims() else arr.reshape((1, *arr.shape))
            inputs[spec.name] = torch.from_numpy(np.ascontiguousarray(model_arr)).to(self.device)
        dims.setdefault("B", 1)
        return dims, inputs


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _dims(spec: Any, shape: tuple[int, ...]) -> dict[str, int]:
    if len(shape) != len(spec.shape):
        raise ValueError(f"tensor {spec.name} rank mismatch")
    out = {}
    for declared, actual in zip(spec.shape, shape):
        if isinstance(declared, str):
            out[declared] = int(actual)
        elif int(declared) != int(actual):
            raise ValueError(f"tensor {spec.name} dimension mismatch")
    return out


def _merge_dims(target: dict[str, int], incoming: dict[str, int]) -> None:
    for name, value in incoming.items():
        if name in target and target[name] != int(value):
            raise ValueError(f"dimension {name} has conflicting values")
        target[name] = int(value)


def _slice_output(data: np.ndarray, spec: Any, dims: dict[str, int]) -> np.ndarray:
    if "B" in spec.dynamic_dims():
        return np.array(data[_slices(spec, dims)], copy=True)
    row_slices = (slice(0, 1), *_slices(spec, dims))
    return np.array(data[row_slices].reshape(tuple(s.stop for s in _slices(spec, dims))), copy=True)


def _slices(spec: Any, dims: dict[str, int]) -> tuple[slice, ...]:
    return tuple(slice(0, dims.get(dim, 0) if isinstance(dim, str) else int(dim)) for dim in spec.shape)


__all__ = ["LocalEvaluator"]
