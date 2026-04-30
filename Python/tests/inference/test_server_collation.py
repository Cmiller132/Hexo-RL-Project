import numpy as np
import pytest
import torch

from hexorl.config import load_config
from hexorl.inference.protocol import InferenceRequestKind, REQUEST_KIND_TO_CODE
from hexorl.inference.server.collation import ServerCollator
from hexorl.inference.shm_queue import (
    CANDIDATE_FEATURES,
    GRAPH_FEATURE_DIM,
    GRAPH_SCHEMA_VERSION,
    MAX_GRAPH_ACTIONS,
    MAX_GRAPH_PAIRS,
    MAX_GRAPH_TOKENS,
    RELATION_SCHEMA_VERSION,
)
from .fakes import FakeSlot


class _Queue:
    def __init__(self, slots):
        self.slots = slots

    def get_slot(self, worker_id):
        return self.slots[worker_id]


class _GraphSlot:
    def __init__(self, *, schema_version=GRAPH_SCHEMA_VERSION, relation_schema_version=RELATION_SCHEMA_VERSION):
        self.req_count = np.array([1], dtype=np.uint32)
        self.req_graph_meta = np.array(
            [schema_version, relation_schema_version, 2, 1, 0, 0, MAX_GRAPH_TOKENS, MAX_GRAPH_ACTIONS],
            dtype=np.uint16,
        )
        self.req_graph_token_features = np.ones((MAX_GRAPH_TOKENS, GRAPH_FEATURE_DIM), dtype=np.float32)
        self.req_graph_token_type = np.zeros(MAX_GRAPH_TOKENS, dtype=np.int16)
        self.req_graph_token_qr = np.zeros((MAX_GRAPH_TOKENS, 2), dtype=np.int32)
        self.req_graph_token_mask = np.zeros(MAX_GRAPH_TOKENS, dtype=np.uint8)
        self.req_graph_legal_token_indices = np.zeros(MAX_GRAPH_ACTIONS, dtype=np.int64)
        self.req_graph_legal_mask = np.ones(MAX_GRAPH_ACTIONS, dtype=np.uint8)
        self.req_graph_opp_legal_qr = np.zeros((MAX_GRAPH_ACTIONS, 2), dtype=np.int32)
        self.req_graph_opp_legal_mask = np.zeros(MAX_GRAPH_ACTIONS, dtype=np.uint8)
        self.req_graph_pair_token_indices = np.full(MAX_GRAPH_PAIRS, -1, dtype=np.int64)
        self.req_graph_pair_first_indices = np.full(MAX_GRAPH_PAIRS, -1, dtype=np.int64)
        self.req_graph_pair_second_indices = np.full(MAX_GRAPH_PAIRS, -1, dtype=np.int64)
        self.req_graph_relation_type = np.zeros((MAX_GRAPH_TOKENS, MAX_GRAPH_TOKENS), dtype=np.int16)
        self.req_graph_relation_bias = np.zeros((1, MAX_GRAPH_TOKENS, MAX_GRAPH_TOKENS), dtype=np.float32)


def _collator(queue, *, global_graph_kind=False):
    cfg = load_config()
    cfg.inference.max_batch_size = 4
    return ServerCollator(
        cfg=cfg,
        queue=queue,
        device=torch.device("cpu"),
        max_batch=4,
        global_graph_kind=global_graph_kind,
    )


def test_dense_sparse_pair_collation_from_slots():
    slot = FakeSlot(max_batch=4)
    slot.req_kind[0] = REQUEST_KIND_TO_CODE[InferenceRequestKind.PAIR_SCORING]
    slot.req_count[0] = 2
    slot.req_tensor[:2] = 3.0
    slot.req_candidate_count[:2] = [2, 2]
    slot.req_candidate_indices[:2, :2] = [[0, 1], [2, 3]]
    slot.req_candidate_features[:2, :2] = 0.5
    slot.req_candidate_mask[:2, :2] = 1
    slot.req_pair_count[:2] = [1, 1]
    slot.req_pair_indices[:2, :1] = [[[0, 1]], [[0, 1]]]
    slot.req_pair_mask[:2, :1] = 1

    batch = _collator(_Queue([slot])).collate_dense([0])

    assert batch.total_count == 2
    assert batch.per_worker_counts == [2]
    assert batch.dense_tensor.shape == (2, 13, 33, 33)
    assert batch.sparse_inputs["candidate_features"].shape == (2, 2, CANDIDATE_FEATURES)
    assert batch.sparse_inputs["pair_candidate_indices"].shape == (2, 1, 2)


def test_graph_collation_rejects_schema_mismatch():
    slot = _GraphSlot(schema_version=GRAPH_SCHEMA_VERSION + 1)

    with pytest.raises(ValueError, match="graph IPC schema mismatch"):
        _collator(_Queue([slot]), global_graph_kind=True).collate_graph([0])
