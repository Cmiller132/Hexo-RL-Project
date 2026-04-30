import numpy as np
import pytest

from hexorl.inference.adapters.dense import DensePolicyValueAdapter
from hexorl.inference.protocol import InferenceRequestKind, default_protocol_manifest, make_request
from hexorl.inference.client.transport import ShmTransport, TransportState
from .fakes import FakeSlot


def test_shm_transport_timeout_has_structured_context():
    manifest = default_protocol_manifest(max_batch_size=4, timeout_ms=1.0)
    slot = FakeSlot(max_batch=4)
    slot.res_ready.ready = False
    slot.res_ready.wait_result = False
    transport = ShmTransport(worker_id=0, slot=slot, timeout_ms=1.0)
    transport.mark_ready()
    request = make_request(
        kind=InferenceRequestKind.DENSE_POLICY_VALUE,
        manifest=manifest,
        payload={"tensor": np.zeros((1, 13, 33, 33), dtype=np.float32), "count": 1},
        deadline_monotonic_s=999.0,
        slot_generation=transport.generation,
    )
    with pytest.raises(TimeoutError) as exc:
        transport.round_trip(request, DensePolicyValueAdapter(manifest))
    message = str(exc.value)
    assert request.request_id in message
    assert "trace_id=" in message
    assert "kind=dense_policy_value" in message
    assert "queue_depth=1" in message
    assert transport.state == TransportState.FAILED
