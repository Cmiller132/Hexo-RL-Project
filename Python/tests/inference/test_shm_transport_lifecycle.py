import numpy as np

from hexorl.inference.adapters.dense import DensePolicyValueAdapter
from hexorl.inference.protocol import InferenceRequestKind, default_protocol_manifest, make_request
from hexorl.inference.shm_transport import ShmTransport, TransportState
from .fakes import FakeSlot


def test_shm_transport_lifecycle_roundtrip_returns_to_ready():
    manifest = default_protocol_manifest(max_batch_size=4, timeout_ms=100.0)
    slot = FakeSlot(max_batch=4)
    transport = ShmTransport(worker_id=0, slot=slot, timeout_ms=100.0)
    transport.mark_ready()
    request = make_request(
        kind=InferenceRequestKind.DENSE_POLICY_VALUE,
        manifest=manifest,
        payload={"tensor": np.zeros((1, 13, 33, 33), dtype=np.float32), "count": 1},
        deadline_monotonic_s=999.0,
        slot_generation=transport.generation,
    )
    response = transport.round_trip(request, DensePolicyValueAdapter(manifest))
    assert response.status == "ok"
    assert transport.state == TransportState.READY
    assert transport.generation == 1
