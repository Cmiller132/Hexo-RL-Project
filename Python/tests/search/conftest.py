import numpy as np
import pytest

from hexorl.contracts.legal import LegalActionTable
from hexorl.models.specs import ModelSpec


@pytest.fixture
def legal_table():
    return LegalActionTable.from_rows(
        [(0, 0), (1, 0), (0, 1)],
        source="fixture",
        allow_fixture=True,
        history_hash="fixture-history",
    )


@pytest.fixture
def dense_spec():
    return ModelSpec(kind="dense_cnn", source_name="fixture")


class FakeInferenceClient:
    def __init__(self):
        self.manifest = type("Manifest", (), {"transport": "test-protocol"})()
        self.dense_calls = 0
        self.sparse_calls = 0
        self.graph_calls = 0
        self.pair_calls = 0

    def evaluate_dense(self, tensor, count):
        self.dense_calls += 1
        policy = np.zeros(1089, dtype=np.float32)
        policy[544] = 3.0
        policy[577] = 2.0
        policy[545] = 1.0
        return policy, np.asarray([0.25], dtype=np.float32)

    def evaluate_sparse(self, tensor, count, candidate_indices, candidate_features, candidate_mask):
        self.sparse_calls += 1
        return (
            np.zeros(1089, dtype=np.float32),
            np.asarray([0.5], dtype=np.float32),
            np.asarray([[3.0, 2.0, 1.0]], dtype=np.float32),
        )

    def evaluate_global_graph(self, graph_batch):
        self.graph_calls += 1
        return {
            "policy_place": np.asarray([3.0, 2.0, 1.0], dtype=np.float32),
            "value": np.asarray([0.75], dtype=np.float32),
            "metadata": {"legal_qr": np.asarray([(0, 0), (1, 0), (0, 1)], dtype=np.int32)},
            "policy_pair_first": np.asarray([9.0, 8.0, 7.0], dtype=np.float32),
        }


@pytest.fixture
def fake_client():
    return FakeInferenceClient()
