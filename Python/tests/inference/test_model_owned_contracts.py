from dataclasses import replace

import pytest

from hexorl.config import load_config
from hexorl.inference.protocol import InferenceProtocolMismatch, negotiate_protocol, protocol_manifest_from_contract
from hexorl.models.factory import REGISTRY
from hexorl.models.inference_contracts import OP_GRAPH_PLACE_VALUE, OP_PLACE_VALUE
from hexorl.models.registry import FamilyComponents, ModelFamilyDescriptor, ModelFamilyRegistry
from hexorl.models.capabilities import CROP_INPUT, DENSE_PLACE_POLICY, CapabilitySet
from hexorl.models.specs import ModelSpec


def _manifest_for(family: str):
    cfg = load_config()
    cfg.inference.max_batch_size = 4
    if family.startswith("global_"):
        cfg.model.architecture = f"{family}_0" if family != "global_relation_graph" else "global_graph_option1"
        cfg.model.heads = ["value", "policy_place", "regret_rank", "regret_value"]
    elif family == "graph_hybrid":
        cfg.model.architecture = "graph_hybrid_0"
        cfg.model.sparse_policy = True
        cfg.model.heads = ["policy", "value", "sparse_policy", "pair_policy"]
    elif family == "restnet":
        cfg.model.architecture = "restnet"
        cfg.model.heads = ["policy", "value", "regret_rank", "regret_value"]
    else:
        cfg.model.architecture = "cnn"
        cfg.model.heads = ["policy", "value", "regret_rank", "regret_value"]
    descriptor = REGISTRY.resolve(family)
    contract = descriptor.inference_contract_factory(ModelSpec(kind=descriptor.name), cfg)
    return protocol_manifest_from_contract(contract, timeout_ms=100.0)


def test_every_registered_family_declares_model_owned_inference_contract():
    for descriptor in REGISTRY.descriptors():
        manifest = _manifest_for(descriptor.name)
        assert manifest.model_family == descriptor.name
        assert manifest.model_contract_hash == manifest.model_contract.hash()
        assert manifest.operations
        assert set(manifest.operations) == {op.name for op in manifest.model_contract.operations}
        assert all(manifest.layout_hashes[name] for name in manifest.operations)


def test_contract_hash_changes_when_head_semantics_change():
    manifest = _manifest_for("dense_cnn")
    changed_heads = tuple(head for head in manifest.model_contract.heads if head.name != "regret_rank")
    changed = replace(manifest.model_contract, heads=changed_heads)
    assert changed.hash() != manifest.model_contract_hash


def test_fake_family_registers_operation_contract_without_inference_runtime_edits():
    registry = ModelFamilyRegistry()
    cfg = load_config()

    def _builder(spec, cfg, *, device=None, inference=False):
        del spec, cfg, device, inference
        return None

    def _train(spec, cfg, model, *, device):
        return None

    def _infer(spec, cfg, model):
        contract = REGISTRY.resolve("dense_cnn").inference_contract_factory(ModelSpec(kind="dense_cnn"), cfg)
        return type("FakeInference", (), {"manifest": None, "contract": replace(contract, model_family=spec.kind)})()

    def _contract(spec, cfg):
        contract = REGISTRY.resolve("dense_cnn").inference_contract_factory(ModelSpec(kind="dense_cnn"), cfg)
        return replace(contract, model_family=spec.kind)

    descriptor = ModelFamilyDescriptor(
        name="fake_contract_family",
        aliases=frozenset(),
        capabilities=CapabilitySet.of([CROP_INPUT, DENSE_PLACE_POLICY]),
        spec_schema=ModelSpec,
        components=FamilyComponents(trunk="fake", heads=("policy", "value")),
        model_builder=_builder,
        train_adapter_factory=_train,
        inference_adapter_factory=_infer,
        inference_contract_factory=_contract,
        policy_provider_factory=lambda spec, cfg, model: model,
        loss_plan_provider=lambda spec, cfg: {},
        recipe_provider=lambda host: {},
        tune_space_provider=lambda host: {},
        checkpoint_manifest_provider=lambda spec, cfg: {},
    )
    registry.register(descriptor)
    contract = registry.resolve("fake_contract_family").inference_contract_factory(ModelSpec(kind="fake_contract_family"), cfg)
    assert contract.operation(OP_PLACE_VALUE).name == OP_PLACE_VALUE


def test_handshake_rejects_operation_head_and_layout_mismatches():
    manifest = _manifest_for("dense_cnn")
    negotiate_protocol(client_manifest=manifest, server_manifest=manifest, operation_name=OP_PLACE_VALUE)
    with pytest.raises(InferenceProtocolMismatch):
        negotiate_protocol(client_manifest=manifest, server_manifest=manifest, operation_name=OP_GRAPH_PLACE_VALUE)
    with pytest.raises(InferenceProtocolMismatch):
        negotiate_protocol(
            client_manifest=manifest,
            server_manifest=manifest,
            operation_name=OP_PLACE_VALUE,
            required_heads=("missing_head",),
        )
    changed_contract = replace(manifest.model_contract, contract_version=manifest.model_contract.contract_version + 1)
    changed = protocol_manifest_from_contract(changed_contract, timeout_ms=100.0)
    with pytest.raises(InferenceProtocolMismatch):
        negotiate_protocol(client_manifest=manifest, server_manifest=changed, operation_name=OP_PLACE_VALUE)
