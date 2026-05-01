import numpy as np
import pytest

from hexorl.contracts.legal import LegalActionTable
from hexorl.models.inference_contracts import OP_GRAPH_PLACE_VALUE, OP_PLACE_VALUE, OP_SPARSE_PLACE_VALUE
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

    def evaluate(self, operation_name, payload):
        if operation_name == OP_PLACE_VALUE:
            return self._response(**self._dense(payload["tensor"]))
        if operation_name == OP_SPARSE_PLACE_VALUE:
            dense, value, sparse = self._sparse(payload)
            return self._response(policy=dense, value=value, sparse_policy=sparse)
        if operation_name == OP_GRAPH_PLACE_VALUE:
            return self._response(**self._graph(payload))
        raise ValueError(operation_name)

    @staticmethod
    def _response(**heads):
        return type("Response", (), {"head_outputs": heads, "telemetry": {"wait_ms": 0.0}})()

    def _dense(self, tensor):
        self.dense_calls += 1
        count = int(np.asarray(tensor).shape[0])
        policy = np.zeros((count, 1089), dtype=np.float32)
        policy[:, 544] = 3.0
        policy[:, 577] = 2.0
        policy[:, 545] = 1.0
        return {"policy": policy, "value": np.full((count,), 0.25, dtype=np.float32)}

    def _sparse(self, payload):
        self.sparse_calls += 1
        return (
            np.zeros(1089, dtype=np.float32),
            np.asarray([0.5], dtype=np.float32),
            np.asarray([[3.0, 2.0, 1.0]], dtype=np.float32),
        )

    def _graph(self, payload):
        self.graph_calls += 1
        if "graph_batch" in payload:
            legal_count = int(np.asarray(payload["graph_batch"].legal_mask).reshape(-1).shape[0])
        else:
            legal_count = int(np.asarray(payload["legal_mask"]).reshape(-1).shape[0])
        return {
            "policy_place": np.asarray([3.0, 2.0, 1.0], dtype=np.float32)[:legal_count],
            "value": np.asarray([0.75], dtype=np.float32),
            "policy_pair_first": np.asarray([9.0, 8.0, 7.0], dtype=np.float32)[:legal_count],
        }


@pytest.fixture
def fake_client():
    return FakeInferenceClient()
