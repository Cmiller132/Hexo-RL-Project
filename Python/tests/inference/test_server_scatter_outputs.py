import numpy as np
import pytest
import torch

from hexorl.inference.server.outputs import DenseForwardOutputs, GraphForwardOutputs, bounded_policy_logits
from hexorl.inference.server.scatter import ServerScatterer
from hexorl.inference.shm_queue import MAX_GRAPH_ACTIONS, MAX_GRAPH_PAIRS, MAX_GRAPH_TOKENS
from .fakes import FakeSlot


class _Queue:
    def __init__(self, slots):
        self.slots = slots

    def get_slot(self, worker_id):
        return self.slots[worker_id]


class _GraphSlot:
    def __init__(self, *, legal_count=2, pair_count=0):
        self.req_kind = np.array([6], dtype=np.uint8)
        self.req_graph_meta = np.array([2, 2, 3, legal_count, 0, pair_count, MAX_GRAPH_TOKENS, MAX_GRAPH_ACTIONS], dtype=np.uint16)
        self.res_value = np.zeros(1, dtype=np.float32)
        self.res_graph_meta = np.zeros(8, dtype=np.uint16)
        self.res_graph_place_logits = np.zeros(MAX_GRAPH_ACTIONS, dtype=np.float32)
        self.res_graph_opp_logits = np.zeros(MAX_GRAPH_ACTIONS, dtype=np.float32)
        self.res_graph_pair_first_logits = np.zeros(MAX_GRAPH_ACTIONS, dtype=np.float32)
        self.res_graph_pair_logits = np.zeros(MAX_GRAPH_PAIRS, dtype=np.float32)
        self.res_graph_pair_second_logits = np.zeros(MAX_GRAPH_PAIRS, dtype=np.float32)
        self.res_graph_regret_rank = np.zeros(1, dtype=np.float32)
        self.res_graph_regret_value = np.zeros(1, dtype=np.float32)


def test_dense_scatter_writes_active_rows_and_clears_request_kind():
    slot = FakeSlot(max_batch=3)
    slot.req_kind[0] = 5
    outputs = DenseForwardOutputs(
        policies=np.ones((2, 1089), dtype=np.float32),
        values=np.array([0.25, -0.5], dtype=np.float32),
        sparse_logits=np.ones((2, 2), dtype=np.float32) * 2.0,
        pair_logits=np.ones((2, 1), dtype=np.float32) * 3.0,
    )

    ServerScatterer(queue=_Queue([slot])).scatter_dense(
        ready_workers=[0],
        per_worker_counts=[2],
        outputs=outputs,
    )

    assert slot.req_kind[0] == 0
    assert np.all(slot.res_policy[:2] == 1.0)
    assert np.all(slot.res_value[:2] == [0.25, -0.5])
    assert np.all(slot.res_sparse_logits[:2, :2] == 2.0)
    assert np.all(slot.res_pair_logits[:2, :1] == 3.0)


def test_graph_scatter_requires_pair_heads_only_for_pair_requests():
    no_pair_slot = _GraphSlot(legal_count=2, pair_count=0)
    outputs = GraphForwardOutputs(
        place_logits=np.ones((1, 2), dtype=np.float32),
        values=np.array([0.1], dtype=np.float32),
    )
    ServerScatterer(queue=_Queue([no_pair_slot])).scatter_graph(ready_workers=[0], outputs=outputs)
    assert no_pair_slot.req_kind[0] == 0
    assert np.all(no_pair_slot.res_graph_place_logits[:2] == 1.0)

    pair_slot = _GraphSlot(legal_count=2, pair_count=1)
    with pytest.raises(ValueError, match="joint/second pair logits"):
        ServerScatterer(queue=_Queue([pair_slot])).scatter_graph(ready_workers=[0], outputs=outputs)


def test_output_validation_rejects_non_finite_before_scatter():
    with pytest.raises(RuntimeError, match="non-finite"):
        bounded_policy_logits(torch.tensor([[float("nan")]]), head_name="policy")
