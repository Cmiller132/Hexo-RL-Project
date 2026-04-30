import numpy as np

from hexorl.inference.adapters.sparse import SparsePolicyValueAdapter
from hexorl.inference.protocol import InferenceRequestKind, default_protocol_manifest, make_request
from hexorl.inference.shm_queue import CANDIDATE_FEATURES


def test_sparse_adapter_accepts_candidate_contract_request():
    manifest = default_protocol_manifest(max_batch_size=2, timeout_ms=100.0)
    request = make_request(
        kind=InferenceRequestKind.SPARSE_POLICY_VALUE,
        manifest=manifest,
        payload={
            "tensor": np.zeros((2, 13, 33, 33), dtype=np.float32),
            "count": 2,
            "candidate_indices": np.zeros((2, 3), dtype=np.int64),
            "candidate_features": np.zeros((2, 3, CANDIDATE_FEATURES), dtype=np.float32),
            "candidate_mask": np.ones((2, 3), dtype=np.bool_),
        },
        deadline_monotonic_s=999.0,
        slot_generation=0,
    )
    SparsePolicyValueAdapter(manifest).validate_request(request)
