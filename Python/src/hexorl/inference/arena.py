"""Contract-derived shared-memory tensor arena."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np

from hexorl.inference.control import CONTROL_WORDS, CTL_STATUS, STATUS_EMPTY


DTYPES = {"float32": np.float32, "int16": np.int16, "int32": np.int32, "int64": np.int64, "uint8": np.uint8}
logger = logging.getLogger(__name__)


def _align(offset: int, alignment: int = 64) -> int:
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def _shm_name(base: str, worker_id: int) -> str:
    return f"hx_{base}_{worker_id}"


def _create_shm(name: str, size: int) -> SharedMemory:
    try:
        return SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        try:
            old = SharedMemory(name=name, create=False)
            old.close()
            old.unlink()
        except FileNotFoundError:
            logger.debug("stale shared memory disappeared: %s", name)
        return SharedMemory(name=name, create=True, size=size)


class SharedEvent:
    def __init__(self, name: str, create: bool = True):
        self._create = create
        self._shm = _create_shm(name, 1) if create else SharedMemory(name=name, create=False)
        self._buf = self._shm.buf
        if create:
            self.clear()

    def is_set(self) -> bool:
        return self._buf[0] == 1

    def set(self) -> None:
        self._buf[0] = 1

    def clear(self) -> None:
        self._buf[0] = 0

    def wait(self, timeout: float | None = None) -> bool:
        start = time.monotonic()
        while self._buf[0] == 0:
            if timeout is not None and time.monotonic() - start >= timeout:
                return False
            time.sleep(0.0001)
        return True

    def close(self) -> None:
        self._shm.close()
        if self._create:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                logger.debug("event already unlinked")


@dataclass(frozen=True)
class TensorViewSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


@dataclass(frozen=True)
class ArenaLayout:
    request_tensors: dict[str, TensorViewSpec]
    response_tensors: dict[str, TensorViewSpec]
    request_nbytes: int
    response_nbytes: int


def _shape(spec: Any, capacities: Any) -> tuple[int, ...]:
    return tuple(capacities.dim_capacity(dim) if isinstance(dim, str) else int(dim) for dim in spec.shape)


def _pack_specs(tensors: list[Any], capacities: Any) -> tuple[dict[str, TensorViewSpec], int]:
    out: dict[str, TensorViewSpec] = {}
    offset = 0
    for tensor in tensors:
        dtype = np.dtype(DTYPES[str(tensor.dtype)])
        offset = _align(offset, min(max(dtype.itemsize, 1), 64))
        shape = _shape(tensor, capacities)
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        out[str(tensor.name)] = TensorViewSpec(str(tensor.name), str(tensor.dtype), shape, offset, nbytes)
        offset += nbytes
    return out, _align(offset)


def arena_layout_from_manifest(manifest: Any) -> ArenaLayout:
    request_specs: dict[str, Any] = {}
    response_specs: dict[str, Any] = {}
    for operation in manifest.model_contract.operations:
        for tensor in operation.layout.input_tensors:
            request_specs.setdefault(tensor.name, tensor)
        for tensor in operation.layout.output_tensors:
            response_specs.setdefault(tensor.name, tensor)
    capacities = manifest.model_contract.capacities
    request, req_nbytes = _pack_specs(list(request_specs.values()), capacities)
    response, res_nbytes = _pack_specs(list(response_specs.values()), capacities)
    return ArenaLayout(request, response, max(req_nbytes, 1), max(res_nbytes, 1))


class WorkerSlots:
    def __init__(self, worker_id: int, max_batch_size: int, manifest: Any, create: bool = True):
        del max_batch_size
        self.worker_id = int(worker_id)
        self.manifest = manifest
        self.layout = arena_layout_from_manifest(manifest)
        self._create = bool(create)
        self._allocate() if create else self._connect()

    def request_tensor(self, name: str) -> np.ndarray:
        return self._view(self.req_arena, self.layout.request_tensors[name])

    def response_tensor(self, name: str) -> np.ndarray:
        return self._view(self.res_arena, self.layout.response_tensors[name])

    def has_response_tensor(self, name: str) -> bool:
        return name in self.layout.response_tensors

    def clear_request_payload(self) -> None:
        self.req_arena.fill(0)
        self.control[CTL_STATUS] = STATUS_EMPTY

    def clear_response_payload(self) -> None:
        self.res_arena.fill(0)

    def close(self) -> None:
        for event in (self.req_ready, self.res_ready):
            event.close()
        for shm in self._shms:
            try:
                shm.close()
            except BufferError:
                logger.debug("shared memory buffer still exported: %s", shm.name)
            if self._create:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    logger.debug("shared memory already unlinked: %s", shm.name)

    def _allocate(self) -> None:
        self._bind(_create_shm(_shm_name("ctl", self.worker_id), CONTROL_WORDS * 8), _create_shm(_shm_name("qar", self.worker_id), self.layout.request_nbytes), _create_shm(_shm_name("rar", self.worker_id), self.layout.response_nbytes))
        self.control.fill(0)
        self.req_arena.fill(0)
        self.res_arena.fill(0)
        self.req_ready = SharedEvent(_shm_name("qr", self.worker_id), create=True)
        self.res_ready = SharedEvent(_shm_name("rr", self.worker_id), create=True)

    def _connect(self) -> None:
        self._bind(SharedMemory(name=_shm_name("ctl", self.worker_id), create=False), SharedMemory(name=_shm_name("qar", self.worker_id), create=False), SharedMemory(name=_shm_name("rar", self.worker_id), create=False))
        self.req_ready = SharedEvent(_shm_name("qr", self.worker_id), create=False)
        self.res_ready = SharedEvent(_shm_name("rr", self.worker_id), create=False)

    def _bind(self, control: SharedMemory, request: SharedMemory, response: SharedMemory) -> None:
        self.control_shm, self.req_arena_shm, self.res_arena_shm = control, request, response
        self._shms = [control, request, response]
        self.control = np.ndarray((CONTROL_WORDS,), dtype=np.uint64, buffer=control.buf)
        self.req_arena = np.ndarray((self.layout.request_nbytes,), dtype=np.uint8, buffer=request.buf)
        self.res_arena = np.ndarray((self.layout.response_nbytes,), dtype=np.uint8, buffer=response.buf)

    @staticmethod
    def _view(arena: np.ndarray, spec: TensorViewSpec) -> np.ndarray:
        raw = arena[spec.offset:spec.offset + spec.nbytes]
        return np.ndarray(spec.shape, dtype=np.dtype(DTYPES[spec.dtype]), buffer=raw)


class InferenceQueue:
    def __init__(self, num_workers: int, max_batch_size: int, manifest: Any, create: bool = True):
        self.slots = [WorkerSlots(i, max_batch_size, manifest, create=create) for i in range(int(num_workers))]

    def get_slot(self, worker_id: int) -> WorkerSlots:
        return self.slots[int(worker_id)]

    def close(self) -> None:
        for slot in self.slots:
            slot.close()


def create_inference_queue(num_workers: int, max_batch_size: int, manifest: Any) -> InferenceQueue:
    return InferenceQueue(num_workers, max_batch_size, manifest, create=True)


def connect_inference_queue(num_workers: int, max_batch_size: int, manifest: Any) -> InferenceQueue:
    return InferenceQueue(num_workers, max_batch_size, manifest, create=False)

