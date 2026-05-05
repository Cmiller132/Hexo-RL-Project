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
import contextlib
import numpy as np
import logging
from multiprocessing.shared_memory import SharedMemory
from typing import List, Optional

from hexorl.action_contract.candidates import CANDIDATE_FEATURES
from hexorl.graph.batch import (
    GRAPH_FEATURE_DIM,
    GRAPH_SCHEMA_VERSION,
    RELATION_SCHEMA_VERSION,
)


NUM_CHANNELS = 13
BOARD_SIZE = 33
BOARD_AREA = 33 * 33  # 1089
MAX_CANDIDATES = 512
MAX_PAIR_CANDIDATES = 512
MAX_GRAPH_TOKENS = 4096
MAX_GRAPH_ACTIONS = 4096
MAX_GRAPH_PAIRS = 4096
TENSOR_ELEMENTS = NUM_CHANNELS * BOARD_SIZE * BOARD_SIZE  # 13 * 33 * 33 = 14157
logger = logging.getLogger(__name__)


def _shm_name(base: str, worker_id: int) -> str:
    """Generate a unique shared-memory name per worker."""
    # macOS POSIX shared-memory names are very short in practice. Keep these
    # names compact; server and clients share this mapping.
    aliases = {
        "req_tensor": "rt",
        "req_count": "rc",
        "res_policy": "rp",
        "res_value": "rv",
        "res_regret_rank": "rrr",
        "req_candidate_count": "qcc",
        "req_candidate_indices": "qci",
        "req_candidate_features": "qcf",
        "req_candidate_mask": "qcm",
        "res_sparse_logits": "rsl",
        "req_pair_count": "qpc",
        "req_pair_indices": "qpi",
        "req_pair_mask": "qpm",
        "res_pair_logits": "rpl",
        "req_mode": "qm",
        "req_graph_meta": "qgm",
        "req_graph_token_features": "qgtf",
        "req_graph_token_type": "qgtt",
        "req_graph_token_qr": "qgtq",
        "req_graph_token_mask": "qgtm",
        "req_graph_legal_token_indices": "qgli",
        "req_graph_legal_qr": "qglq",
        "req_graph_legal_mask": "qglm",
        "req_graph_opp_legal_qr": "qgoq",
        "req_graph_opp_legal_mask": "qgom",
        "req_graph_pair_token_indices": "qgpi",
        "req_graph_pair_first_indices": "qgpf",
        "req_graph_pair_second_indices": "qgps",
        "req_graph_relation_type": "qgrt",
        "req_graph_relation_bias": "qgrb",
        "res_graph_meta": "rgm",
        "res_graph_place_logits": "rgpl",
        "res_graph_opp_logits": "rgol",
        "res_graph_pair_first_logits": "rgpf",
        "res_graph_pair_logits": "rgpr",
        "res_graph_pair_second_logits": "rgps",
        "res_graph_regret_rank": "rgrr",
        "req_ready": "qr",
        "res_ready": "rr",
    }
    return f"hx_{aliases.get(base, base)}_{worker_id}"


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
            logger.debug("SharedMemory %s disappeared before cleanup", name)
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
                logger.debug("SharedEvent %s already unlinked", self._name)


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
        self.req_mode_shm: Optional[SharedMemory] = None
        self.req_mode: Optional[np.ndarray] = None

        self.res_policy_shm: Optional[SharedMemory] = None
        self.res_policy: Optional[np.ndarray] = None

        self.res_value_shm: Optional[SharedMemory] = None
        self.res_value: Optional[np.ndarray] = None
        self.res_regret_rank_shm: Optional[SharedMemory] = None
        self.res_regret_rank: Optional[np.ndarray] = None
        self.req_candidate_count_shm: Optional[SharedMemory] = None
        self.req_candidate_count: Optional[np.ndarray] = None
        self.req_candidate_indices_shm: Optional[SharedMemory] = None
        self.req_candidate_indices: Optional[np.ndarray] = None
        self.req_candidate_features_shm: Optional[SharedMemory] = None
        self.req_candidate_features: Optional[np.ndarray] = None
        self.req_candidate_mask_shm: Optional[SharedMemory] = None
        self.req_candidate_mask: Optional[np.ndarray] = None
        self.res_sparse_logits_shm: Optional[SharedMemory] = None
        self.res_sparse_logits: Optional[np.ndarray] = None
        self.req_pair_count_shm: Optional[SharedMemory] = None
        self.req_pair_count: Optional[np.ndarray] = None
        self.req_pair_indices_shm: Optional[SharedMemory] = None
        self.req_pair_indices: Optional[np.ndarray] = None
        self.req_pair_mask_shm: Optional[SharedMemory] = None
        self.req_pair_mask: Optional[np.ndarray] = None
        self.res_pair_logits_shm: Optional[SharedMemory] = None
        self.res_pair_logits: Optional[np.ndarray] = None
        self.req_graph_meta_shm: Optional[SharedMemory] = None
        self.req_graph_meta: Optional[np.ndarray] = None
        self.req_graph_token_features_shm: Optional[SharedMemory] = None
        self.req_graph_token_features: Optional[np.ndarray] = None
        self.req_graph_token_type_shm: Optional[SharedMemory] = None
        self.req_graph_token_type: Optional[np.ndarray] = None
        self.req_graph_token_qr_shm: Optional[SharedMemory] = None
        self.req_graph_token_qr: Optional[np.ndarray] = None
        self.req_graph_token_mask_shm: Optional[SharedMemory] = None
        self.req_graph_token_mask: Optional[np.ndarray] = None
        self.req_graph_legal_token_indices_shm: Optional[SharedMemory] = None
        self.req_graph_legal_token_indices: Optional[np.ndarray] = None
        self.req_graph_legal_qr_shm: Optional[SharedMemory] = None
        self.req_graph_legal_qr: Optional[np.ndarray] = None
        self.req_graph_legal_mask_shm: Optional[SharedMemory] = None
        self.req_graph_legal_mask: Optional[np.ndarray] = None
        self.req_graph_opp_legal_qr_shm: Optional[SharedMemory] = None
        self.req_graph_opp_legal_qr: Optional[np.ndarray] = None
        self.req_graph_opp_legal_mask_shm: Optional[SharedMemory] = None
        self.req_graph_opp_legal_mask: Optional[np.ndarray] = None
        self.req_graph_pair_token_indices_shm: Optional[SharedMemory] = None
        self.req_graph_pair_token_indices: Optional[np.ndarray] = None
        self.req_graph_pair_first_indices_shm: Optional[SharedMemory] = None
        self.req_graph_pair_first_indices: Optional[np.ndarray] = None
        self.req_graph_pair_second_indices_shm: Optional[SharedMemory] = None
        self.req_graph_pair_second_indices: Optional[np.ndarray] = None
        self.req_graph_relation_type_shm: Optional[SharedMemory] = None
        self.req_graph_relation_type: Optional[np.ndarray] = None
        self.req_graph_relation_bias_shm: Optional[SharedMemory] = None
        self.req_graph_relation_bias: Optional[np.ndarray] = None
        self.res_graph_meta_shm: Optional[SharedMemory] = None
        self.res_graph_meta: Optional[np.ndarray] = None
        self.res_graph_place_logits_shm: Optional[SharedMemory] = None
        self.res_graph_place_logits: Optional[np.ndarray] = None
        self.res_graph_opp_logits_shm: Optional[SharedMemory] = None
        self.res_graph_opp_logits: Optional[np.ndarray] = None
        self.res_graph_pair_first_logits_shm: Optional[SharedMemory] = None
        self.res_graph_pair_first_logits: Optional[np.ndarray] = None
        self.res_graph_pair_logits_shm: Optional[SharedMemory] = None
        self.res_graph_pair_logits: Optional[np.ndarray] = None
        self.res_graph_pair_second_logits_shm: Optional[SharedMemory] = None
        self.res_graph_pair_second_logits: Optional[np.ndarray] = None
        self.res_graph_regret_rank_shm: Optional[SharedMemory] = None
        self.res_graph_regret_rank: Optional[np.ndarray] = None

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
        self.req_mode_shm = _create_shm(
            _shm_name("req_mode", self.worker_id), 1
        )
        self.req_mode = np.ndarray((1,), dtype=np.uint8, buffer=self.req_mode_shm.buf)
        self.req_mode[0] = 0

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
        self.res_regret_rank_shm = _create_shm(
            _shm_name("res_regret_rank", self.worker_id), self.max_batch * 4
        )
        self.res_regret_rank = np.ndarray(
            (self.max_batch,), dtype=np.float32, buffer=self.res_regret_rank_shm.buf
        )

        self.req_candidate_count_shm = _create_shm(
            _shm_name("req_candidate_count", self.worker_id), self.max_batch * 2
        )
        self.req_candidate_count = np.ndarray(
            (self.max_batch,), dtype=np.uint16, buffer=self.req_candidate_count_shm.buf
        )
        self.req_candidate_count.fill(0)
        self.req_candidate_indices_shm = _create_shm(
            _shm_name("req_candidate_indices", self.worker_id),
            self.max_batch * MAX_CANDIDATES * 8,
        )
        self.req_candidate_indices = np.ndarray(
            (self.max_batch, MAX_CANDIDATES),
            dtype=np.int64,
            buffer=self.req_candidate_indices_shm.buf,
        )
        self.req_candidate_features_shm = _create_shm(
            _shm_name("req_candidate_features", self.worker_id),
            self.max_batch * MAX_CANDIDATES * CANDIDATE_FEATURES * 4,
        )
        self.req_candidate_features = np.ndarray(
            (self.max_batch, MAX_CANDIDATES, CANDIDATE_FEATURES),
            dtype=np.float32,
            buffer=self.req_candidate_features_shm.buf,
        )
        self.req_candidate_mask_shm = _create_shm(
            _shm_name("req_candidate_mask", self.worker_id),
            self.max_batch * MAX_CANDIDATES,
        )
        self.req_candidate_mask = np.ndarray(
            (self.max_batch, MAX_CANDIDATES),
            dtype=np.uint8,
            buffer=self.req_candidate_mask_shm.buf,
        )
        self.res_sparse_logits_shm = _create_shm(
            _shm_name("res_sparse_logits", self.worker_id),
            self.max_batch * MAX_CANDIDATES * 4,
        )
        self.res_sparse_logits = np.ndarray(
            (self.max_batch, MAX_CANDIDATES),
            dtype=np.float32,
            buffer=self.res_sparse_logits_shm.buf,
        )
        self.req_pair_count_shm = _create_shm(
            _shm_name("req_pair_count", self.worker_id), self.max_batch * 2
        )
        self.req_pair_count = np.ndarray(
            (self.max_batch,), dtype=np.uint16, buffer=self.req_pair_count_shm.buf
        )
        self.req_pair_count.fill(0)
        self.req_pair_indices_shm = _create_shm(
            _shm_name("req_pair_indices", self.worker_id),
            self.max_batch * MAX_PAIR_CANDIDATES * 2 * 8,
        )
        self.req_pair_indices = np.ndarray(
            (self.max_batch, MAX_PAIR_CANDIDATES, 2),
            dtype=np.int64,
            buffer=self.req_pair_indices_shm.buf,
        )
        self.req_pair_mask_shm = _create_shm(
            _shm_name("req_pair_mask", self.worker_id),
            self.max_batch * MAX_PAIR_CANDIDATES,
        )
        self.req_pair_mask = np.ndarray(
            (self.max_batch, MAX_PAIR_CANDIDATES),
            dtype=np.uint8,
            buffer=self.req_pair_mask_shm.buf,
        )
        self.res_pair_logits_shm = _create_shm(
            _shm_name("res_pair_logits", self.worker_id),
            self.max_batch * MAX_PAIR_CANDIDATES * 4,
        )
        self.res_pair_logits = np.ndarray(
            (self.max_batch, MAX_PAIR_CANDIDATES),
            dtype=np.float32,
            buffer=self.res_pair_logits_shm.buf,
        )
        self._allocate_graph_slots()

        self.req_ready = SharedEvent(_shm_name("req_ready", self.worker_id), create=True)
        self.res_ready = SharedEvent(_shm_name("res_ready", self.worker_id), create=True)

    def _allocate_graph_slots(self):
        """Create one padded graph request/response slot for this worker."""
        self.req_graph_meta_shm = _create_shm(_shm_name("req_graph_meta", self.worker_id), 8 * 2)
        self.req_graph_meta = np.ndarray((8,), dtype=np.uint16, buffer=self.req_graph_meta_shm.buf)
        self.req_graph_meta[:] = (
            GRAPH_SCHEMA_VERSION,
            RELATION_SCHEMA_VERSION,
            0,
            0,
            0,
            0,
            MAX_GRAPH_TOKENS,
            MAX_GRAPH_ACTIONS,
        )
        self.req_graph_token_features_shm = _create_shm(
            _shm_name("req_graph_token_features", self.worker_id),
            MAX_GRAPH_TOKENS * GRAPH_FEATURE_DIM * 4,
        )
        self.req_graph_token_features = np.ndarray(
            (MAX_GRAPH_TOKENS, GRAPH_FEATURE_DIM), dtype=np.float32, buffer=self.req_graph_token_features_shm.buf
        )
        self.req_graph_token_type_shm = _create_shm(
            _shm_name("req_graph_token_type", self.worker_id), MAX_GRAPH_TOKENS * 2
        )
        self.req_graph_token_type = np.ndarray(
            (MAX_GRAPH_TOKENS,), dtype=np.int16, buffer=self.req_graph_token_type_shm.buf
        )
        self.req_graph_token_qr_shm = _create_shm(
            _shm_name("req_graph_token_qr", self.worker_id), MAX_GRAPH_TOKENS * 2 * 4
        )
        self.req_graph_token_qr = np.ndarray(
            (MAX_GRAPH_TOKENS, 2), dtype=np.int32, buffer=self.req_graph_token_qr_shm.buf
        )
        self.req_graph_token_mask_shm = _create_shm(
            _shm_name("req_graph_token_mask", self.worker_id), MAX_GRAPH_TOKENS
        )
        self.req_graph_token_mask = np.ndarray(
            (MAX_GRAPH_TOKENS,), dtype=np.uint8, buffer=self.req_graph_token_mask_shm.buf
        )
        self.req_graph_legal_token_indices_shm = _create_shm(
            _shm_name("req_graph_legal_token_indices", self.worker_id), MAX_GRAPH_ACTIONS * 8
        )
        self.req_graph_legal_token_indices = np.ndarray(
            (MAX_GRAPH_ACTIONS,), dtype=np.int64, buffer=self.req_graph_legal_token_indices_shm.buf
        )
        self.req_graph_legal_qr_shm = _create_shm(
            _shm_name("req_graph_legal_qr", self.worker_id), MAX_GRAPH_ACTIONS * 2 * 4
        )
        self.req_graph_legal_qr = np.ndarray(
            (MAX_GRAPH_ACTIONS, 2), dtype=np.int32, buffer=self.req_graph_legal_qr_shm.buf
        )
        self.req_graph_legal_mask_shm = _create_shm(
            _shm_name("req_graph_legal_mask", self.worker_id), MAX_GRAPH_ACTIONS
        )
        self.req_graph_legal_mask = np.ndarray(
            (MAX_GRAPH_ACTIONS,), dtype=np.uint8, buffer=self.req_graph_legal_mask_shm.buf
        )
        self.req_graph_opp_legal_qr_shm = _create_shm(
            _shm_name("req_graph_opp_legal_qr", self.worker_id), MAX_GRAPH_ACTIONS * 2 * 4
        )
        self.req_graph_opp_legal_qr = np.ndarray(
            (MAX_GRAPH_ACTIONS, 2), dtype=np.int32, buffer=self.req_graph_opp_legal_qr_shm.buf
        )
        self.req_graph_opp_legal_mask_shm = _create_shm(
            _shm_name("req_graph_opp_legal_mask", self.worker_id), MAX_GRAPH_ACTIONS
        )
        self.req_graph_opp_legal_mask = np.ndarray(
            (MAX_GRAPH_ACTIONS,), dtype=np.uint8, buffer=self.req_graph_opp_legal_mask_shm.buf
        )
        self.req_graph_pair_token_indices_shm = _create_shm(
            _shm_name("req_graph_pair_token_indices", self.worker_id), MAX_GRAPH_PAIRS * 8
        )
        self.req_graph_pair_token_indices = np.ndarray(
            (MAX_GRAPH_PAIRS,), dtype=np.int64, buffer=self.req_graph_pair_token_indices_shm.buf
        )
        self.req_graph_pair_first_indices_shm = _create_shm(
            _shm_name("req_graph_pair_first_indices", self.worker_id), MAX_GRAPH_PAIRS * 8
        )
        self.req_graph_pair_first_indices = np.ndarray(
            (MAX_GRAPH_PAIRS,), dtype=np.int64, buffer=self.req_graph_pair_first_indices_shm.buf
        )
        self.req_graph_pair_second_indices_shm = _create_shm(
            _shm_name("req_graph_pair_second_indices", self.worker_id), MAX_GRAPH_PAIRS * 8
        )
        self.req_graph_pair_second_indices = np.ndarray(
            (MAX_GRAPH_PAIRS,), dtype=np.int64, buffer=self.req_graph_pair_second_indices_shm.buf
        )
        self.req_graph_relation_type_shm = _create_shm(
            _shm_name("req_graph_relation_type", self.worker_id),
            MAX_GRAPH_TOKENS * MAX_GRAPH_TOKENS * 2,
        )
        self.req_graph_relation_type = np.ndarray(
            (MAX_GRAPH_TOKENS, MAX_GRAPH_TOKENS), dtype=np.int16, buffer=self.req_graph_relation_type_shm.buf
        )
        self.req_graph_relation_bias_shm = _create_shm(
            _shm_name("req_graph_relation_bias", self.worker_id),
            MAX_GRAPH_TOKENS * MAX_GRAPH_TOKENS * 4,
        )
        self.req_graph_relation_bias = np.ndarray(
            (1, MAX_GRAPH_TOKENS, MAX_GRAPH_TOKENS), dtype=np.float32, buffer=self.req_graph_relation_bias_shm.buf
        )
        self.res_graph_meta_shm = _create_shm(_shm_name("res_graph_meta", self.worker_id), 8 * 2)
        self.res_graph_meta = np.ndarray((8,), dtype=np.uint16, buffer=self.res_graph_meta_shm.buf)
        self.res_graph_meta[:] = 0
        self.res_graph_place_logits_shm = _create_shm(
            _shm_name("res_graph_place_logits", self.worker_id), MAX_GRAPH_ACTIONS * 4
        )
        self.res_graph_place_logits = np.ndarray(
            (MAX_GRAPH_ACTIONS,), dtype=np.float32, buffer=self.res_graph_place_logits_shm.buf
        )
        self.res_graph_opp_logits_shm = _create_shm(
            _shm_name("res_graph_opp_logits", self.worker_id), MAX_GRAPH_ACTIONS * 4
        )
        self.res_graph_opp_logits = np.ndarray(
            (MAX_GRAPH_ACTIONS,), dtype=np.float32, buffer=self.res_graph_opp_logits_shm.buf
        )
        self.res_graph_pair_first_logits_shm = _create_shm(
            _shm_name("res_graph_pair_first_logits", self.worker_id), MAX_GRAPH_ACTIONS * 4
        )
        self.res_graph_pair_first_logits = np.ndarray(
            (MAX_GRAPH_ACTIONS,), dtype=np.float32, buffer=self.res_graph_pair_first_logits_shm.buf
        )
        self.res_graph_pair_logits_shm = _create_shm(
            _shm_name("res_graph_pair_logits", self.worker_id), MAX_GRAPH_PAIRS * 4
        )
        self.res_graph_pair_logits = np.ndarray(
            (MAX_GRAPH_PAIRS,), dtype=np.float32, buffer=self.res_graph_pair_logits_shm.buf
        )
        self.res_graph_pair_second_logits_shm = _create_shm(
            _shm_name("res_graph_pair_second_logits", self.worker_id), MAX_GRAPH_PAIRS * 4
        )
        self.res_graph_pair_second_logits = np.ndarray(
            (MAX_GRAPH_PAIRS,), dtype=np.float32, buffer=self.res_graph_pair_second_logits_shm.buf
        )
        self.res_graph_regret_rank_shm = _create_shm(
            _shm_name("res_graph_regret_rank", self.worker_id), 4
        )
        self.res_graph_regret_rank = np.ndarray(
            (1,), dtype=np.float32, buffer=self.res_graph_regret_rank_shm.buf
        )

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
        self.req_mode_shm = SharedMemory(
            name=_shm_name("req_mode", self.worker_id), create=False
        )
        self.req_mode = np.ndarray((1,), dtype=np.uint8, buffer=self.req_mode_shm.buf)

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
        self.res_regret_rank_shm = SharedMemory(
            name=_shm_name("res_regret_rank", self.worker_id), create=False
        )
        self.res_regret_rank = np.ndarray(
            (self.max_batch,), dtype=np.float32, buffer=self.res_regret_rank_shm.buf
        )

        self.req_candidate_count_shm = SharedMemory(
            name=_shm_name("req_candidate_count", self.worker_id), create=False
        )
        self.req_candidate_count = np.ndarray(
            (self.max_batch,), dtype=np.uint16, buffer=self.req_candidate_count_shm.buf
        )
        self.req_candidate_indices_shm = SharedMemory(
            name=_shm_name("req_candidate_indices", self.worker_id), create=False
        )
        self.req_candidate_indices = np.ndarray(
            (self.max_batch, MAX_CANDIDATES),
            dtype=np.int64,
            buffer=self.req_candidate_indices_shm.buf,
        )
        self.req_candidate_features_shm = SharedMemory(
            name=_shm_name("req_candidate_features", self.worker_id), create=False
        )
        self.req_candidate_features = np.ndarray(
            (self.max_batch, MAX_CANDIDATES, CANDIDATE_FEATURES),
            dtype=np.float32,
            buffer=self.req_candidate_features_shm.buf,
        )
        self.req_candidate_mask_shm = SharedMemory(
            name=_shm_name("req_candidate_mask", self.worker_id), create=False
        )
        self.req_candidate_mask = np.ndarray(
            (self.max_batch, MAX_CANDIDATES),
            dtype=np.uint8,
            buffer=self.req_candidate_mask_shm.buf,
        )
        self.res_sparse_logits_shm = SharedMemory(
            name=_shm_name("res_sparse_logits", self.worker_id), create=False
        )
        self.res_sparse_logits = np.ndarray(
            (self.max_batch, MAX_CANDIDATES),
            dtype=np.float32,
            buffer=self.res_sparse_logits_shm.buf,
        )
        self.req_pair_count_shm = SharedMemory(
            name=_shm_name("req_pair_count", self.worker_id), create=False
        )
        self.req_pair_count = np.ndarray(
            (self.max_batch,), dtype=np.uint16, buffer=self.req_pair_count_shm.buf
        )
        self.req_pair_indices_shm = SharedMemory(
            name=_shm_name("req_pair_indices", self.worker_id), create=False
        )
        self.req_pair_indices = np.ndarray(
            (self.max_batch, MAX_PAIR_CANDIDATES, 2),
            dtype=np.int64,
            buffer=self.req_pair_indices_shm.buf,
        )
        self.req_pair_mask_shm = SharedMemory(
            name=_shm_name("req_pair_mask", self.worker_id), create=False
        )
        self.req_pair_mask = np.ndarray(
            (self.max_batch, MAX_PAIR_CANDIDATES),
            dtype=np.uint8,
            buffer=self.req_pair_mask_shm.buf,
        )
        self.res_pair_logits_shm = SharedMemory(
            name=_shm_name("res_pair_logits", self.worker_id), create=False
        )
        self.res_pair_logits = np.ndarray(
            (self.max_batch, MAX_PAIR_CANDIDATES),
            dtype=np.float32,
            buffer=self.res_pair_logits_shm.buf,
        )
        self._connect_graph_slots()

        self.req_ready = SharedEvent(_shm_name("req_ready", self.worker_id), create=False)
        self.res_ready = SharedEvent(_shm_name("res_ready", self.worker_id), create=False)

    def _connect_graph_slots(self):
        self.req_graph_meta_shm = SharedMemory(name=_shm_name("req_graph_meta", self.worker_id), create=False)
        self.req_graph_meta = np.ndarray((8,), dtype=np.uint16, buffer=self.req_graph_meta_shm.buf)
        self.req_graph_token_features_shm = SharedMemory(name=_shm_name("req_graph_token_features", self.worker_id), create=False)
        self.req_graph_token_features = np.ndarray((MAX_GRAPH_TOKENS, GRAPH_FEATURE_DIM), dtype=np.float32, buffer=self.req_graph_token_features_shm.buf)
        self.req_graph_token_type_shm = SharedMemory(name=_shm_name("req_graph_token_type", self.worker_id), create=False)
        self.req_graph_token_type = np.ndarray((MAX_GRAPH_TOKENS,), dtype=np.int16, buffer=self.req_graph_token_type_shm.buf)
        self.req_graph_token_qr_shm = SharedMemory(name=_shm_name("req_graph_token_qr", self.worker_id), create=False)
        self.req_graph_token_qr = np.ndarray((MAX_GRAPH_TOKENS, 2), dtype=np.int32, buffer=self.req_graph_token_qr_shm.buf)
        self.req_graph_token_mask_shm = SharedMemory(name=_shm_name("req_graph_token_mask", self.worker_id), create=False)
        self.req_graph_token_mask = np.ndarray((MAX_GRAPH_TOKENS,), dtype=np.uint8, buffer=self.req_graph_token_mask_shm.buf)
        self.req_graph_legal_token_indices_shm = SharedMemory(name=_shm_name("req_graph_legal_token_indices", self.worker_id), create=False)
        self.req_graph_legal_token_indices = np.ndarray((MAX_GRAPH_ACTIONS,), dtype=np.int64, buffer=self.req_graph_legal_token_indices_shm.buf)
        self.req_graph_legal_qr_shm = SharedMemory(name=_shm_name("req_graph_legal_qr", self.worker_id), create=False)
        self.req_graph_legal_qr = np.ndarray((MAX_GRAPH_ACTIONS, 2), dtype=np.int32, buffer=self.req_graph_legal_qr_shm.buf)
        self.req_graph_legal_mask_shm = SharedMemory(name=_shm_name("req_graph_legal_mask", self.worker_id), create=False)
        self.req_graph_legal_mask = np.ndarray((MAX_GRAPH_ACTIONS,), dtype=np.uint8, buffer=self.req_graph_legal_mask_shm.buf)
        self.req_graph_opp_legal_qr_shm = SharedMemory(name=_shm_name("req_graph_opp_legal_qr", self.worker_id), create=False)
        self.req_graph_opp_legal_qr = np.ndarray((MAX_GRAPH_ACTIONS, 2), dtype=np.int32, buffer=self.req_graph_opp_legal_qr_shm.buf)
        self.req_graph_opp_legal_mask_shm = SharedMemory(name=_shm_name("req_graph_opp_legal_mask", self.worker_id), create=False)
        self.req_graph_opp_legal_mask = np.ndarray((MAX_GRAPH_ACTIONS,), dtype=np.uint8, buffer=self.req_graph_opp_legal_mask_shm.buf)
        self.req_graph_pair_token_indices_shm = SharedMemory(name=_shm_name("req_graph_pair_token_indices", self.worker_id), create=False)
        self.req_graph_pair_token_indices = np.ndarray((MAX_GRAPH_PAIRS,), dtype=np.int64, buffer=self.req_graph_pair_token_indices_shm.buf)
        self.req_graph_pair_first_indices_shm = SharedMemory(name=_shm_name("req_graph_pair_first_indices", self.worker_id), create=False)
        self.req_graph_pair_first_indices = np.ndarray((MAX_GRAPH_PAIRS,), dtype=np.int64, buffer=self.req_graph_pair_first_indices_shm.buf)
        self.req_graph_pair_second_indices_shm = SharedMemory(name=_shm_name("req_graph_pair_second_indices", self.worker_id), create=False)
        self.req_graph_pair_second_indices = np.ndarray((MAX_GRAPH_PAIRS,), dtype=np.int64, buffer=self.req_graph_pair_second_indices_shm.buf)
        self.req_graph_relation_type_shm = SharedMemory(name=_shm_name("req_graph_relation_type", self.worker_id), create=False)
        self.req_graph_relation_type = np.ndarray((MAX_GRAPH_TOKENS, MAX_GRAPH_TOKENS), dtype=np.int16, buffer=self.req_graph_relation_type_shm.buf)
        self.req_graph_relation_bias_shm = SharedMemory(name=_shm_name("req_graph_relation_bias", self.worker_id), create=False)
        self.req_graph_relation_bias = np.ndarray((1, MAX_GRAPH_TOKENS, MAX_GRAPH_TOKENS), dtype=np.float32, buffer=self.req_graph_relation_bias_shm.buf)
        self.res_graph_meta_shm = SharedMemory(name=_shm_name("res_graph_meta", self.worker_id), create=False)
        self.res_graph_meta = np.ndarray((8,), dtype=np.uint16, buffer=self.res_graph_meta_shm.buf)
        self.res_graph_place_logits_shm = SharedMemory(name=_shm_name("res_graph_place_logits", self.worker_id), create=False)
        self.res_graph_place_logits = np.ndarray((MAX_GRAPH_ACTIONS,), dtype=np.float32, buffer=self.res_graph_place_logits_shm.buf)
        self.res_graph_opp_logits_shm = SharedMemory(name=_shm_name("res_graph_opp_logits", self.worker_id), create=False)
        self.res_graph_opp_logits = np.ndarray((MAX_GRAPH_ACTIONS,), dtype=np.float32, buffer=self.res_graph_opp_logits_shm.buf)
        self.res_graph_pair_first_logits_shm = SharedMemory(name=_shm_name("res_graph_pair_first_logits", self.worker_id), create=False)
        self.res_graph_pair_first_logits = np.ndarray((MAX_GRAPH_ACTIONS,), dtype=np.float32, buffer=self.res_graph_pair_first_logits_shm.buf)
        self.res_graph_pair_logits_shm = SharedMemory(name=_shm_name("res_graph_pair_logits", self.worker_id), create=False)
        self.res_graph_pair_logits = np.ndarray((MAX_GRAPH_PAIRS,), dtype=np.float32, buffer=self.res_graph_pair_logits_shm.buf)
        self.res_graph_pair_second_logits_shm = SharedMemory(name=_shm_name("res_graph_pair_second_logits", self.worker_id), create=False)
        self.res_graph_pair_second_logits = np.ndarray((MAX_GRAPH_PAIRS,), dtype=np.float32, buffer=self.res_graph_pair_second_logits_shm.buf)
        self.res_graph_regret_rank_shm = SharedMemory(name=_shm_name("res_graph_regret_rank", self.worker_id), create=False)
        self.res_graph_regret_rank = np.ndarray((1,), dtype=np.float32, buffer=self.res_graph_regret_rank_shm.buf)

    def close(self):
        """Close and unlink all shared memory segments."""
        for attr in (
            "req_tensor_shm",
            "req_count_shm",
            "req_mode_shm",
            "res_policy_shm",
            "res_value_shm",
            "res_regret_rank_shm",
            "req_candidate_count_shm",
            "req_candidate_indices_shm",
            "req_candidate_features_shm",
            "req_candidate_mask_shm",
            "res_sparse_logits_shm",
            "req_pair_count_shm",
            "req_pair_indices_shm",
            "req_pair_mask_shm",
            "res_pair_logits_shm",
            "req_graph_meta_shm",
            "req_graph_token_features_shm",
            "req_graph_token_type_shm",
            "req_graph_token_qr_shm",
            "req_graph_token_mask_shm",
            "req_graph_legal_token_indices_shm",
            "req_graph_legal_qr_shm",
            "req_graph_legal_mask_shm",
            "req_graph_opp_legal_qr_shm",
            "req_graph_opp_legal_mask_shm",
            "req_graph_pair_token_indices_shm",
            "req_graph_pair_first_indices_shm",
            "req_graph_pair_second_indices_shm",
            "req_graph_relation_type_shm",
            "req_graph_relation_bias_shm",
            "res_graph_meta_shm",
            "res_graph_place_logits_shm",
            "res_graph_opp_logits_shm",
            "res_graph_pair_first_logits_shm",
            "res_graph_pair_logits_shm",
            "res_graph_pair_second_logits_shm",
            "res_graph_regret_rank_shm",
        ):
            shm = getattr(self, attr, None)
            if shm is not None:
                shm.close()
                if self._create:
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        logger.debug("SharedMemory %s already unlinked", shm.name)
        for evt in (self.req_ready, self.res_ready):
            if evt is not None:
                evt.close()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()


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
        with contextlib.suppress(Exception):
            self.close()


def create_inference_queue(num_workers: int, max_batch_size: int) -> InferenceQueue:
    """Create shared-memory slots for all workers (called by server)."""
    return InferenceQueue(num_workers, max_batch_size, create=True)


def connect_inference_queue(num_workers: int, max_batch_size: int) -> InferenceQueue:
    """Connect to existing shared-memory slots (called by workers)."""
    return InferenceQueue(num_workers, max_batch_size, create=False)
