from pathlib import Path

import numpy as np
import torch

from hexorl.inference.arena import create_inference_queue
from hexorl.inference.client import RemoteEvaluator
from hexorl.inference.control import CTL_STATUS, STATUS_OK, write_dyn_dims
from hexorl.inference.local import LocalEvaluator
from hexorl.inference.protocol import protocol_manifest_from_contract
from hexorl.models.inference_contracts import (
    CapacitySpec,
    HeadDecoderSpec,
    InferenceOperationSpec,
    ModelInferenceContract,
    OutputHeadSpec,
    TensorSpec,
    TransportLayoutSpec,
)


def _fake_manifest():
    input_spec = TensorSpec("alien_signal", "float32", ("B", 2), "test_only", "stack_over_b")
    output_spec = TensorSpec("new_head", "float32", ("B",), "test_only", "pad_and_stack")
    head = OutputHeadSpec("new_head", output_spec, HeadDecoderSpec("new_head", "scalar"), "B", required=True)
    op = InferenceOperationSpec(
        "never_seen_operation",
        "TEST_ONLY",
        ("alien_signal",),
        ("new_head",),
        ("new_head",),
        TransportLayoutSpec("never_seen_layout", (input_spec,), (output_spec,)),
    )
    contract = ModelInferenceContract(
        "fake_family",
        1,
        99,
        (op,),
        (head,),
        CapacitySpec(max_batch_size=4),
        "fake_input",
        "fake_action",
    )
    return protocol_manifest_from_contract(contract, timeout_ms=100.0)


class _FakeModel(torch.nn.Module):
    def forward(self, **inputs):
        return {"new_head": inputs["alien_signal"].sum(dim=1)}


def test_no_semantic_strings_in_inference_package():
    root = Path(__file__).resolve().parents[2] / "src" / "hexorl" / "inference"
    banned = ("crop_batch", "graph_batch", "policy_place", "candidate_indices", "graph_token")
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        offenders.extend(f"{path}:{word}" for word in banned if word in text)
        offenders.extend(f"{path}:REQ_" for word in ("REQ_CANDIDATE", "REQ_PAIR", "REQ_TOKEN", "REQ_LEGAL", "REQ_OPP", "REQ_GRAPH") if word in text)
    assert offenders == []


def test_register_new_operation_runs_end_to_end_local_and_remote():
    manifest = _fake_manifest()
    payload = {"alien_signal": np.asarray([[2.0, 5.0]], dtype=np.float32)}
    local = LocalEvaluator(_FakeModel(), manifest=manifest, device=torch.device("cpu"))
    local_response = local.evaluate("never_seen_operation", payload)
    assert np.allclose(local_response.head_outputs["new_head"], [7.0])

    queue = create_inference_queue(1, 4, manifest)
    remote = RemoteEvaluator(0, 1, 4, timeout_ms=100.0, manifest=manifest, server_manifest=manifest)
    try:
        remote.connect()

        def _server_wait(timeout=None):
            remote._slot.response_tensor("new_head")[:1] = 7.0
            write_dyn_dims(remote._slot.control, {"B": 1})
            remote._slot.control[CTL_STATUS] = STATUS_OK
            return True

        remote._slot.res_ready.wait = _server_wait
        remote_response = remote.evaluate("never_seen_operation", payload)
        assert np.allclose(remote_response.head_outputs["new_head"], local_response.head_outputs["new_head"])
    finally:
        remote.close()
        queue.close()

