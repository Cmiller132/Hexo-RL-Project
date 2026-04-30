import numpy as np

from hexorl.inference.shm_queue import (
    BOARD_AREA,
    CANDIDATE_FEATURES,
    MAX_CANDIDATES,
    MAX_PAIR_CANDIDATES,
)


class FakeEvent:
    def __init__(self, ready=False, wait_result=None):
        self.ready = ready
        self.wait_result = wait_result

    def set(self):
        self.ready = True

    def clear(self):
        self.ready = False

    def wait(self, timeout=None):
        if self.wait_result is not None:
            return self.wait_result
        return self.ready


class FakeSlot:
    def __init__(self, max_batch=4):
        self.req_ready = FakeEvent(False)
        self.res_ready = FakeEvent(True, wait_result=True)
        self.req_kind = np.zeros(1, dtype=np.uint8)
        self.req_count = np.zeros(1, dtype=np.uint32)
        self.req_tensor = np.zeros((max_batch, 13, 33, 33), dtype=np.float32)
        self.req_candidate_count = np.zeros(max_batch, dtype=np.uint16)
        self.req_candidate_indices = np.full((max_batch, MAX_CANDIDATES), -1, dtype=np.int64)
        self.req_candidate_features = np.zeros((max_batch, MAX_CANDIDATES, CANDIDATE_FEATURES), dtype=np.float32)
        self.req_candidate_mask = np.zeros((max_batch, MAX_CANDIDATES), dtype=np.uint8)
        self.req_pair_count = np.zeros(max_batch, dtype=np.uint16)
        self.req_pair_indices = np.full((max_batch, MAX_PAIR_CANDIDATES, 2), -1, dtype=np.int64)
        self.req_pair_mask = np.zeros((max_batch, MAX_PAIR_CANDIDATES), dtype=np.uint8)
        self.res_policy = np.zeros((max_batch, BOARD_AREA), dtype=np.float32)
        self.res_value = np.zeros(max_batch, dtype=np.float32)
        self.res_sparse_logits = np.zeros((max_batch, MAX_CANDIDATES), dtype=np.float32)
        self.res_pair_logits = np.zeros((max_batch, MAX_PAIR_CANDIDATES), dtype=np.float32)
        self.res_regret_rank = np.zeros(max_batch, dtype=np.float32)
