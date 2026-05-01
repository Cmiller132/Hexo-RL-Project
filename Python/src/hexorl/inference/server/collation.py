"""Server-side contract-walking arena collation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from hexorl.inference.control import read_all_dyn_dims


@dataclass(frozen=True)
class CollatedBatch:
    ready_workers: list[int]
    per_worker_dims: list[dict[str, int]]
    total_count: int
    operation: Any
    model_inputs: dict[str, torch.Tensor]
    metadata: dict[str, Any]


class ServerCollator:
    def __init__(self, *, cfg, queue, device: torch.device, max_batch: int, manifest):
        del cfg, max_batch
        self.queue = queue
        self.device = device
        self.manifest = manifest

    def collate(self, ready_workers: list[int], operation_name: str) -> CollatedBatch:
        operation = self.manifest.model_contract.operation(operation_name)
        known = _operation_dims(operation)
        worker_dims = [read_all_dyn_dims(self.queue.get_slot(worker).control, known) for worker in ready_workers]
        total = sum(max(1, dims.get("B", 1)) for dims in worker_dims)
        inputs = {
            spec.name: torch.from_numpy(self._collate_tensor(ready_workers, worker_dims, spec)).to(self.device, non_blocking=True)
            for spec in operation.layout.input_tensors
        }
        return CollatedBatch(ready_workers, worker_dims, total, operation, inputs, {})

    def _collate_tensor(self, workers: list[int], dims_by_worker: list[dict[str, int]], spec: Any) -> np.ndarray:
        if spec.batching == "stack_over_b":
            return np.concatenate([_slice(self.queue.get_slot(w).request_tensor(spec.name), spec, d) for w, d in zip(workers, dims_by_worker)], axis=0)
        target = _target_shape(spec, dims_by_worker)
        out = _blank(target, self.queue.get_slot(workers[0]).request_tensor(spec.name).dtype)
        offset = 0
        for row, (worker, dims) in enumerate(zip(workers, dims_by_worker)):
            item = _slice(self.queue.get_slot(worker).request_tensor(spec.name), spec, dims)
            if "B" in spec.dynamic_dims():
                b = max(1, dims.get("B", 1))
                out[_assign_slices(spec, dims, offset=offset)] = item
                offset += b
            else:
                out[_assign_slices(spec, dims, offset=row)] = item
        return out


def _operation_dims(operation: Any) -> tuple[str, ...]:
    names: list[str] = []
    for spec in operation.layout.input_tensors:
        names.extend(spec.dynamic_dims())
    return tuple(dict.fromkeys(names))


def _slice(view: np.ndarray, spec: Any, dims: dict[str, int]) -> np.ndarray:
    return np.array(view[_slices(spec, dims)], copy=True)


def _slices(spec: Any, dims: dict[str, int]) -> tuple[slice, ...]:
    return tuple(slice(0, dims.get(dim, 0) if isinstance(dim, str) else int(dim)) for dim in spec.shape)


def _target_shape(spec: Any, dims_by_worker: list[dict[str, int]]) -> tuple[int, ...]:
    shape = []
    for dim in spec.shape:
        if dim == "B":
            shape.append(sum(max(1, dims.get("B", 1)) for dims in dims_by_worker))
        elif isinstance(dim, str):
            shape.append(max((dims.get(dim, 0) for dims in dims_by_worker), default=0))
        else:
            shape.append(int(dim))
    if "B" not in spec.dynamic_dims():
        shape.insert(0, len(dims_by_worker))
    return tuple(shape)


def _assign_slices(spec: Any, dims: dict[str, int], *, offset: int) -> tuple[slice, ...]:
    slices = []
    if "B" not in spec.dynamic_dims():
        slices.append(slice(offset, offset + 1))
    for dim in spec.shape:
        if dim == "B":
            width = max(1, dims.get("B", 1))
            slices.append(slice(offset, offset + width))
        elif isinstance(dim, str):
            slices.append(slice(0, dims.get(dim, 0)))
        else:
            slices.append(slice(0, int(dim)))
    return tuple(slices)


def _blank(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    fill = -1 if np.issubdtype(dtype, np.signedinteger) else 0
    return np.full(shape, fill, dtype=dtype)


__all__ = ["CollatedBatch", "ServerCollator"]
