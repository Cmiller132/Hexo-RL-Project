import numpy as np

from hexorl.inference.adapters.dense import DensePolicyValueAdapter
from hexorl.inference.protocol import InferenceRequestKind, default_protocol_manifest, make_request
from hexorl.inference.client.transport import ShmTransport
from .fakes import FakeSlot


def test_response_telemetry_contains_request_identity_and_transport_state():
    manifest = default_protocol_manifest(max_batch_size=1, timeout_ms=100.0)
    transport = ShmTransport(worker_id=0, slot=FakeSlot(max_batch=1), timeout_ms=100.0)
    transport.mark_ready()
    request = make_request(
        kind=InferenceRequestKind.DENSE_POLICY_VALUE,
        manifest=manifest,
        payload={"tensor": np.zeros((1, 13, 33, 33), dtype=np.float32), "count": 1},
        deadline_monotonic_s=999.0,
        slot_generation=0,
    )
    response = transport.round_trip(request, DensePolicyValueAdapter(manifest))
    assert response.telemetry["request_id"] == request.request_id
    assert response.telemetry["trace_id"] == request.trace_id
    assert response.telemetry["request_kind"] == "dense_policy_value"
    assert response.telemetry["transport_state"] == "draining"
