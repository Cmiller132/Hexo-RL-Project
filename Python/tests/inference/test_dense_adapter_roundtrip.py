import numpy as np

from hexorl.inference.adapters.dense import DensePolicyValueAdapter
from hexorl.inference.protocol import InferenceRequestKind, default_protocol_manifest, make_request


def test_dense_adapter_accepts_finite_contract_request():
    manifest = default_protocol_manifest(max_batch_size=2, timeout_ms=100.0)
    request = make_request(
        kind=InferenceRequestKind.DENSE_POLICY_VALUE,
        manifest=manifest,
        payload={"tensor": np.zeros((2, 13, 33, 33), dtype=np.float32), "count": 2},
        deadline_monotonic_s=999.0,
        slot_generation=0,
    )
    DensePolicyValueAdapter(manifest).validate_request(request)
