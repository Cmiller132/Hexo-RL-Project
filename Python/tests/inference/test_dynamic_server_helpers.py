import numpy as np
import pytest
import torch

from hexorl.config import load_config
from hexorl.inference.arena import InferenceQueue
from hexorl.inference.control import CTL_STATUS, STATUS_OK, write_dyn_dims
from hexorl.inference.protocol import protocol_manifest_from_contract
from hexorl.inference.server.collation import ServerCollator
from hexorl.inference.server.outputs import bounded_policy_logits
from hexorl.inference.server.scatter import ServerScatterer
from hexorl.models.factory import REGISTRY
from hexorl.models.inference_contracts import OP_PAIR_POLICY
from hexorl.models.specs import ModelSpec


def _manifest():
    cfg = load_config()
    cfg.inference.max_batch_size = 2
    cfg.model.architecture = "graph_hybrid_0"
    cfg.model.sparse_policy = True
    cfg.model.heads = ["policy", "value", "sparse_policy", "pair_policy"]
    contract = REGISTRY.resolve("graph_hybrid").inference_contract_factory(ModelSpec(kind="graph_hybrid"), cfg)
    return protocol_manifest_from_contract(contract, timeout_ms=100.0)


def test_collation_and_scatter_use_dynamic_tensors_only():
    manifest = _manifest()
    queue = InferenceQueue(1, 2, manifest, create=True)
    try:
        slot = queue.get_slot(0)
        write_dyn_dims(slot.control, {"B": 2, "K": 2, "P": 1})
        slot.request_tensor("tensor")[:2] = 3.0
        slot.request_tensor("candidate_indices")[:2, :2] = [[0, 1], [2, 3]]
        slot.request_tensor("candidate_features")[:2, :2] = 0.5
        slot.request_tensor("candidate_mask")[:2, :2] = 1
        slot.request_tensor("pair_candidate_indices")[:2, :1] = [[[0, 1]], [[0, 1]]]
        slot.request_tensor("pair_candidate_mask")[:2, :1] = 1
        collator = ServerCollator(cfg=load_config(), queue=queue, device=torch.device("cpu"), max_batch=2, manifest=manifest)
        collated = collator.collate([0], OP_PAIR_POLICY)
        assert collated.total_count == 2
        assert collated.model_inputs["candidate_indices"].shape == (2, 2)
        outputs = {
            "policy": np.ones((2, 1089), dtype=np.float32),
            "value": np.asarray([0.2, -0.1], dtype=np.float32),
            "sparse_policy": np.full((2, 2), 2.0, dtype=np.float32),
            "pair_policy": np.full((2, 1), 3.0, dtype=np.float32),
        }
        ServerScatterer(queue=queue, manifest=manifest).scatter(collated=collated, outputs=outputs)
        assert slot.control[CTL_STATUS] == STATUS_OK
        assert np.all(slot.response_tensor("policy")[:2] == 1.0)
        assert np.all(slot.response_tensor("pair_policy")[:2, :1] == 3.0)
    finally:
        queue.close()


def test_output_decode_rejects_non_finite_before_scatter():
    with pytest.raises(RuntimeError):
        bounded_policy_logits(torch.tensor([[float("nan")]]), head_name="policy")
