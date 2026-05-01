import numpy as np
import pytest

from hexorl.config import load_config
from hexorl.inference.arena import WorkerSlots, arena_layout_from_manifest
from hexorl.inference.client.transport import ShmTransport, TransportState
from hexorl.inference.control import CTL_OPCODE, CTL_STATUS, STATUS_OK, read_all_dyn_dims, write_dyn_dims
from hexorl.inference.protocol import make_request, protocol_manifest_from_contract
from hexorl.models.factory import REGISTRY
from hexorl.models.inference_contracts import OP_PAIR_POLICY, OP_PLACE_VALUE
from hexorl.models.specs import ModelSpec


def _manifest():
    cfg = load_config()
    cfg.inference.max_batch_size = 2
    cfg.model.architecture = "graph_hybrid_0"
    cfg.model.sparse_policy = True
    cfg.model.heads = ["policy", "value", "sparse_policy", "pair_policy"]
    contract = REGISTRY.resolve("graph_hybrid").inference_contract_factory(ModelSpec(kind="graph_hybrid"), cfg)
    return protocol_manifest_from_contract(contract, timeout_ms=100.0)


def test_arena_layout_has_manifest_declared_tensors_only():
    layout = arena_layout_from_manifest(_manifest())
    assert {"tensor", "candidate_indices", "candidate_features", "pair_candidate_indices"} <= set(layout.request_tensors)
    assert {"policy", "value", "sparse_policy", "pair_policy"} <= set(layout.response_tensors)


def test_dynamic_dim_table_carries_arbitrary_named_dims():
    manifest = _manifest()
    slot = WorkerSlots(worker_id=97, max_batch_size=2, manifest=manifest, create=True)
    try:
        write_dyn_dims(slot.control, {"B": 2, "surprise": 17})
        assert read_all_dyn_dims(slot.control, ("B", "surprise")) == {"B": 2, "surprise": 17}
    finally:
        slot.close()


def test_transport_roundtrip_uses_operation_code_and_dynamic_head_buffers():
    manifest = _manifest()
    slot = WorkerSlots(worker_id=98, max_batch_size=2, manifest=manifest, create=True)
    try:
        transport = ShmTransport(worker_id=98, slot=slot, timeout_ms=100.0, manifest=manifest)
        transport.mark_ready()
        request = make_request(operation_name=OP_PLACE_VALUE, manifest=manifest, payload={"tensor": np.ones((1, 13, 33, 33), dtype=np.float32)})

        def _server_wait(timeout=None):
            slot.response_tensor("policy")[:1] = 2.0
            slot.response_tensor("value")[:1] = 0.25
            write_dyn_dims(slot.control, {"B": 1})
            slot.control[CTL_STATUS] = STATUS_OK
            return True

        slot.res_ready.wait = _server_wait
        response = transport.round_trip(request)
        assert slot.control[CTL_OPCODE] == manifest.operation_code(OP_PLACE_VALUE)
        assert response.head_outputs["policy"].shape == (1, 1089)
        assert response.head_outputs["value"].shape == (1,)
        assert transport.state == TransportState.READY
    finally:
        slot.close()


def test_transport_rejects_missing_required_operation_payload():
    manifest = _manifest()
    slot = WorkerSlots(worker_id=99, max_batch_size=2, manifest=manifest, create=True)
    try:
        transport = ShmTransport(worker_id=99, slot=slot, timeout_ms=100.0, manifest=manifest)
        transport.mark_ready()
        request = make_request(operation_name=OP_PAIR_POLICY, manifest=manifest, payload={"tensor": np.zeros((1, 13, 33, 33), dtype=np.float32)})
        with pytest.raises(ValueError):
            transport.round_trip(request)
    finally:
        slot.close()
