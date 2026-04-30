import numpy as np

from hexorl.inference.adapters.pair_scoring import PairScoringAdapter
from hexorl.inference.protocol import InferenceRequestKind, default_protocol_manifest, make_request
from hexorl.inference.shm_queue import CANDIDATE_FEATURES


def test_pair_scoring_adapter_validates_supplied_pair_rows_only():
    manifest = default_protocol_manifest(max_batch_size=1, timeout_ms=100.0)
    request = make_request(
        kind=InferenceRequestKind.PAIR_SCORING,
        manifest=manifest,
        payload={
            "tensor": np.zeros((1, 13, 33, 33), dtype=np.float32),
            "count": 1,
            "candidate_indices": np.array([[0, 1, 2]], dtype=np.int64),
            "candidate_features": np.zeros((1, 3, CANDIDATE_FEATURES), dtype=np.float32),
            "candidate_mask": np.ones((1, 3), dtype=np.bool_),
            "pair_candidate_indices": np.array([[[0, 1], [1, 2]]], dtype=np.int64),
            "pair_candidate_mask": np.ones((1, 2), dtype=np.bool_),
        },
        deadline_monotonic_s=999.0,
        slot_generation=0,
    )
    PairScoringAdapter(manifest).validate_request(request)
