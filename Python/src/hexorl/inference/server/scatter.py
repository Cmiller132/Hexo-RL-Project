"""Server-side contract-walking response scatter."""

from __future__ import annotations

import numpy as np

from hexorl.inference.control import (
    CTL_CONTRACT_HASH,
    CTL_GENERATION,
    CTL_LAYOUT_HASH,
    CTL_STATUS,
    STATUS_OK,
    hash_word,
    write_dyn_dims,
)


class ServerScatterer:
    def __init__(self, *, queue, manifest):
        self.queue = queue
        self.manifest = manifest

    def scatter(self, *, collated, outputs: dict[str, object]) -> None:
        offset = 0
        for row, (worker_id, dims) in enumerate(zip(collated.ready_workers, collated.per_worker_dims)):
            slot = self.queue.get_slot(worker_id)
            slot.clear_response_payload()
            slot.control[CTL_LAYOUT_HASH] = hash_word(collated.operation.layout_hash)
            slot.control[CTL_CONTRACT_HASH] = hash_word(self.manifest.model_contract_hash)
            slot.control[CTL_GENERATION] = int(slot.control[CTL_GENERATION]) + 1
            write_dyn_dims(slot.control, dims)
            for head_name in collated.operation.output_heads:
                if head_name in outputs and slot.has_response_tensor(head_name):
                    head = self.manifest.model_contract.head(head_name)
                    data = np.asarray(outputs[head_name])
                    slot.response_tensor(head_name)[_target_slices(head.tensor, dims)] = _source_slice(data, head.tensor, dims, row=row, offset=offset)
            slot.control[CTL_STATUS] = STATUS_OK
            offset += max(1, dims.get("B", 1))


def _target_slices(spec, dims: dict[str, int]) -> tuple[slice, ...]:
    return tuple(slice(0, dims.get(dim, 0) if isinstance(dim, str) else int(dim)) for dim in spec.shape)


def _source_slice(data: np.ndarray, spec, dims: dict[str, int], *, row: int, offset: int) -> np.ndarray:
    if "B" in spec.dynamic_dims():
        b = max(1, dims.get("B", 1))
        return data[offset:offset + b]
    slices = [slice(row, row + 1)]
    for dim in spec.shape:
        slices.append(slice(0, dims.get(dim, 0) if isinstance(dim, str) else int(dim)))
    return np.asarray(data[tuple(slices)]).reshape(tuple(s.stop for s in _target_slices(spec, dims)))


__all__ = ["ServerScatterer"]
