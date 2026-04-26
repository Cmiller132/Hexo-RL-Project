"""Shared-memory inference queue — KataGo-style NNEvaluator channel.

Each worker gets two fixed-size slots in shared memory:
  req_tensor[i]  — (MAX_BATCH, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE) f32
  req_count[i]   — u32 batch count
  res_policy[i]  — (MAX_BATCH, BOARD_AREA) f32
  res_value[i]   — (MAX_BATCH,) f32

Plus two shared-memory doorbell bytes per worker (spawn-safe):
  req_ready[i]   — worker → server: request ready
  res_ready[i]   — server → worker: response ready
"""

import time as _time
import multiprocessing as mp
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from typing import List, Optional


NUM_CHANNELS = 13
BOARD_SIZE = 33
BOARD_AREA = 33 * 33  # 1089
TENSOR_ELEMENTS = NUM_CHANNELS * BOARD_SIZE * BOARD_SIZE  # 13 * 33 * 33 = 14157


def _shm_name(base: str, worker_id: int) -> str:
    """Generate a unique shared-memory name per worker."""
    return f"hexorl_{base}_{worker_id}"


def _create_shm(name: str, size: int) -> SharedMemory:
    """Create a SharedMemory segment, cleaning up any leftover from crash."""
    try:
        return SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        try:
            existing = SharedMemory(name=name, create=False)
            existing.close()
            existing.unlink()
        except FileNotFoundError:
            pass
        return SharedMemory(name=name, create=True, size=size)


class SharedEvent:
    """A spawn-safe Event backed by a named shared-memory byte.

    Uses a single byte in a SharedMemory segment. Accessible by name
    from any process — works in both fork and spawn start methods.
    """

    def __init__(self, name: str, create: bool = True):
        self._name = name
        self._create = create
        if create:
            self._shm = _create_shm(name, 1)
            self._buf = self._shm.buf
            self._buf[0] = 0
        else:
            self._shm = SharedMemory(name=name, create=False)
            self._buf = self._shm.buf

    def is_set(self) -> bool:
        return self._buf[0] == 1

    def set(self):
        self._buf[0] = 1

    def clear(self):
        self._buf[0] = 0

    def wait(self, timeout: float = None) -> bool:
        """Busy-wait with polling. Suitable for sub-millisecond latency."""
        start = _time.monotonic()
        while self._buf[0] == 0:
            if timeout is not None and _time.monotonic() - start >= timeout:
                return False
            _time.sleep(0.0001)
        return True

    def close(self):
        self._shm.close()
        if self._create:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass


class WorkerSlots:
    """One worker's shared-memory slots (created on the server side)."""

    def __init__(
        self,
        worker_id: int,
        max_batch_size: int,
        create: bool = True,
    ):
        self.worker_id = worker_id
        self.max_batch = max_batch_size
        self._create = create

        self.req_tensor_shm: Optional[SharedMemory] = None
        self.req_tensor: Optional[np.ndarray] = None

        self.req_count_shm: Optional[SharedMemory] = None
        self.req_count: Optional[np.ndarray] = None

        self.res_policy_shm: Optional[SharedMemory] = None
        self.res_policy: Optional[np.ndarray] = None

        self.res_value_shm: Optional[SharedMemory] = None
        self.res_value: Optional[np.ndarray] = None

        self.req_ready: Optional[SharedEvent] = None
        self.res_ready: Optional[SharedEvent] = None

        if create:
            self._allocate()
        else:
            self._connect()

    def _allocate(self):
        """Create new shared-memory segments and events (server side)."""
        self.req_tensor_shm = _create_shm(
            _shm_name("req_tensor", self.worker_id),
            self.max_batch * TENSOR_ELEMENTS * 4,
        )
        self.req_tensor = np.ndarray(
            (self.max_batch, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype=np.float32,
            buffer=self.req_tensor_shm.buf,
        )

        self.req_count_shm = _create_shm(
            _shm_name("req_count", self.worker_id), 4
        )
        self.req_count = np.ndarray(
            (1,), dtype=np.uint32, buffer=self.req_count_shm.buf
        )
        self.req_count[0] = 0

        self.res_policy_shm = _create_shm(
            _shm_name("res_policy", self.worker_id),
            self.max_batch * BOARD_AREA * 4,
        )
        self.res_policy = np.ndarray(
            (self.max_batch, BOARD_AREA),
            dtype=np.float32,
            buffer=self.res_policy_shm.buf,
        )

        self.res_value_shm = _create_shm(
            _shm_name("res_value", self.worker_id), self.max_batch * 4
        )
        self.res_value = np.ndarray(
            (self.max_batch,), dtype=np.float32, buffer=self.res_value_shm.buf
        )

        self.req_ready = SharedEvent(_shm_name("req_ready", self.worker_id), create=True)
        self.res_ready = SharedEvent(_shm_name("res_ready", self.worker_id), create=True)

    def _connect(self):
        """Connect to existing shared-memory segments (worker side)."""
        self.req_tensor_shm = SharedMemory(
            name=_shm_name("req_tensor", self.worker_id), create=False
        )
        self.req_tensor = np.ndarray(
            (self.max_batch, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype=np.float32,
            buffer=self.req_tensor_shm.buf,
        )

        self.req_count_shm = SharedMemory(
            name=_shm_name("req_count", self.worker_id), create=False
        )
        self.req_count = np.ndarray(
            (1,), dtype=np.uint32, buffer=self.req_count_shm.buf
        )

        self.res_policy_shm = SharedMemory(
            name=_shm_name("res_policy", self.worker_id), create=False
        )
        self.res_policy = np.ndarray(
            (self.max_batch, BOARD_AREA),
            dtype=np.float32,
            buffer=self.res_policy_shm.buf,
        )

        self.res_value_shm = SharedMemory(
            name=_shm_name("res_value", self.worker_id), create=False
        )
        self.res_value = np.ndarray(
            (self.max_batch,), dtype=np.float32, buffer=self.res_value_shm.buf
        )

        self.req_ready = SharedEvent(_shm_name("req_ready", self.worker_id), create=False)
        self.res_ready = SharedEvent(_shm_name("res_ready", self.worker_id), create=False)

    def close(self):
        """Close and unlink all shared memory segments."""
        for attr in ("req_tensor_shm", "req_count_shm", "res_policy_shm", "res_value_shm"):
            shm = getattr(self, attr, None)
            if shm is not None:
                shm.close()
                if self._create:
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass
        for evt in (self.req_ready, self.res_ready):
            if evt is not None:
                evt.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class InferenceQueue:
    """Manages shared-memory slots and events for all workers."""

    def __init__(
        self,
        num_workers: int,
        max_batch_size: int,
        create: bool = True,
    ):
        self.num_workers = num_workers
        self.max_batch_size = max_batch_size
        self._create = create
        self.slots: List[WorkerSlots] = []

        for i in range(num_workers):
            slot = WorkerSlots(worker_id=i, max_batch_size=max_batch_size, create=create)
            self.slots.append(slot)

    def get_slot(self, worker_id: int) -> WorkerSlots:
        return self.slots[worker_id]

    def close(self):
        for slot in self.slots:
            slot.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def create_inference_queue(num_workers: int, max_batch_size: int) -> InferenceQueue:
    """Create shared-memory slots for all workers (called by server)."""
    return InferenceQueue(num_workers, max_batch_size, create=True)


def connect_inference_queue(num_workers: int, max_batch_size: int) -> InferenceQueue:
    """Connect to existing shared-memory slots (called by workers)."""
    return InferenceQueue(num_workers, max_batch_size, create=False)
