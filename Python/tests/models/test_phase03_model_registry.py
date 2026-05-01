import ast
import re
from pathlib import Path

import pytest
import torch

from hexorl.config import Config
from hexorl.models.capabilities import CROP_INPUT, DENSE_PLACE_POLICY, CapabilitySet
from hexorl.models.families import builtin_descriptors
from hexorl.models.factory import REGISTRY, build_model
from hexorl.models.inputs import CropInputs
from hexorl.models.registry import FamilyComponents, ModelFamilyDescriptor, ModelFamilyRegistry
from hexorl.models.specs import REQUIRED_MODEL_KINDS, ModelSpec
from hexorl.models.trunks import TRUNK_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_ROOT = PROJECT_ROOT / "Python" / "src" / "hexorl" / "models"


def _cfg(architecture: str) -> Config:
    model = {"architecture": architecture, "channels": 4, "blocks": 1}
    if architecture in {"restnet", "graph_hybrid_0", "global_xattn_0", "global_line_window_0", "global_graph_option1"}:
        model["attention_heads"] = 1
    if architecture in {"graph_hybrid_0", "global_xattn_0", "global_line_window_0", "global_graph_option1"}:
        model["graph_layers"] = 1
    return Config.model_validate(
        {
            "model": model,
            "train": {"batches_per_epoch": 1},
            "inference": {"fp16": False},
        }
    )


def test_registry_lists_all_required_families():
    assert set(REQUIRED_MODEL_KINDS) <= set(REGISTRY.names())


def test_builtin_family_modules_own_descriptor_construction():
    descriptors = builtin_descriptors()
    assert {descriptor.name for descriptor in descriptors} == set(REQUIRED_MODEL_KINDS)
    for descriptor in descriptors:
        assert descriptor.components.trunk
        assert descriptor.components.heads
        assert descriptor.model_builder.__module__.startswith("hexorl.models.trunks.")
        assert descriptor.checkpoint_manifest_provider(ModelSpec(kind=descriptor.name), _cfg("cnn"))["model_family"] == descriptor.name


def test_head_and_trunk_modules_expose_used_components():
    from hexorl.models.heads import (
        GLOBAL_GRAPH_OUTPUT_HEADS,
        GRAPH_HYBRID_POLICY_HEADS,
        AuxPolicyHead,
        AxisHead,
        AxisMapHead,
        MovesLeftHead,
        PairPolicyHead,
        PolicyHead,
        RegretRankHead,
        SparsePolicyHead,
        ValueBinnedHead,
    )
    from hexorl.models.trunks import (
        GatedResBlock,
        HexConv2d,
        SparseHexGraphHybrid0Encoder,
        SpatialTransformerBlock,
        build_dense_cnn_model,
        build_global_relation_graph_model,
        build_graph_hybrid_model,
        build_restnet_model,
    )

    assert GRAPH_HYBRID_POLICY_HEADS == ("policy", "sparse_policy")
    assert GLOBAL_GRAPH_OUTPUT_HEADS == (
        "policy_place",
        "policy_pair_first",
        "policy_pair_second",
        "policy_pair_joint",
        "value",
    )
    assert build_dense_cnn_model.__module__ == "hexorl.models.trunks.crop_cnn"
    assert build_restnet_model.__module__ == "hexorl.models.trunks.crop_xformer"
    assert build_graph_hybrid_model.__module__ == "hexorl.models.trunks.graph_hybrid"
    assert build_global_relation_graph_model.__module__ == "hexorl.models.trunks.global_relation_graph"
    assert PolicyHead.__module__ == "hexorl.models.heads.policy"
    assert ValueBinnedHead.__module__ == "hexorl.models.heads.value"
    assert RegretRankHead.__module__ == "hexorl.models.heads.regret"
    assert SparsePolicyHead.__module__ == "hexorl.models.heads.sparse_policy"
    assert PairPolicyHead.__module__ == "hexorl.models.heads.pair_policy"
    assert AuxPolicyHead.__module__ == "hexorl.models.heads.tactical"
    assert AxisHead.__module__ == "hexorl.models.heads.tactical"
    assert AxisMapHead.__module__ == "hexorl.models.heads.tactical"
    assert MovesLeftHead.__module__ == "hexorl.models.heads.tactical"
    assert HexConv2d.__module__ == "hexorl.models.trunks.crop_cnn"
    assert GatedResBlock.__module__ == "hexorl.models.trunks.crop_cnn"
    assert SpatialTransformerBlock.__module__ == "hexorl.models.trunks.crop_xformer"
    assert SparseHexGraphHybrid0Encoder.__module__ == "hexorl.models.trunks.graph_hybrid"


def test_crop_components_are_not_owned_by_legacy_network_module():
    assert not (MODELS_ROOT / "network.py").exists()

    head_sources = {
        path.name: ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted((MODELS_ROOT / "heads").glob("*.py"))
        if path.name != "__init__.py"
    }
    forbidden_network_imports: dict[str, list[str]] = {}
    for filename, tree in head_sources.items():
        imported = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "hexorl.models.network"
            for alias in node.names
        ]
        if imported:
            forbidden_network_imports[filename] = imported
    assert forbidden_network_imports == {}

    trunk_sources = {
        path.name: ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted((MODELS_ROOT / "trunks").glob("*.py"))
        if path.name != "__init__.py"
    }
    forbidden_trunk_network_imports: dict[str, list[str]] = {}
    for filename, tree in trunk_sources.items():
        imported = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "hexorl.models.network"
            for alias in node.names
        ]
        if imported:
            forbidden_trunk_network_imports[filename] = imported
    assert forbidden_trunk_network_imports == {}

    runtime_imports: dict[str, list[int]] = {}
    for path in sorted((PROJECT_ROOT / "Python" / "src" / "hexorl").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "hexorl.models.network"
        ]
        if lines:
            runtime_imports[str(path.relative_to(PROJECT_ROOT))] = lines
    assert runtime_imports == {}


def test_every_registered_family_validates_default_recipe():
    for descriptor in REGISTRY.descriptors():
        recipe = descriptor.recipe_provider("ci-cpu")
        assert recipe["valid"] is True
        assert recipe["model_kind"] == descriptor.name


def test_every_registered_family_builds_model():
    aliases = {
        "dense_cnn": "cnn",
        "restnet": "restnet",
        "graph_hybrid": "graph_hybrid_0",
        "global_xattn": "global_xattn_0",
        "global_line_window": "global_line_window_0",
        "global_relation_graph": "global_graph_option1",
    }
    for family, architecture in aliases.items():
        model = build_model(_cfg(architecture), device=torch.device("cpu"), inference=False)
        assert isinstance(model, torch.nn.Module), family


def test_every_registered_family_builds_train_adapter():
    cfg = _cfg("cnn")
    for descriptor in REGISTRY.descriptors():
        spec = ModelSpec(kind=descriptor.name)
        adapter = descriptor.train_adapter_factory(spec, cfg, torch.nn.Linear(1, 1), device=torch.device("cpu"))
        assert adapter.spec.kind == descriptor.name


def test_every_registered_family_builds_inference_adapter_contract():
    cfg = _cfg("cnn")
    for descriptor in REGISTRY.descriptors():
        spec = ModelSpec(kind=descriptor.name)
        contract = descriptor.inference_adapter_factory(spec, cfg, torch.nn.Identity()).contract
        assert contract.model_family == descriptor.name
        assert contract.contract_version >= 1


def test_every_registered_family_builds_policy_provider():
    cfg = _cfg("cnn")
    for descriptor in REGISTRY.descriptors():
        provider = descriptor.policy_provider_factory(ModelSpec(kind=descriptor.name), cfg, torch.nn.Identity())
        assert provider.model is not None


def test_every_registered_family_declares_loss_plan():
    cfg = _cfg("cnn")
    for descriptor in REGISTRY.descriptors():
        plan = descriptor.loss_plan_provider(ModelSpec(kind=descriptor.name), cfg)
        assert isinstance(plan.weights, dict)
        assert plan.finite_required is True


def test_every_registered_family_declares_tune_space():
    for descriptor in REGISTRY.descriptors():
        tune = descriptor.tune_space_provider("ci-cpu")
        assert tune["model_kind"] == descriptor.name
        assert tune["mutations"]


def test_fake_family_registers_without_runtime_internal_edits():
    registry = REGISTRY.clone()

    def _builder(spec, cfg, *, device=None, inference=False):
        return torch.nn.Linear(1, 1)

    def _train(spec, cfg, model, *, device):
        return {"spec": spec.kind, "device": str(device)}

    def _infer(spec, cfg, model):
        return type("FakeInference", (), {"manifest": type("FakeManifest", (), {"model_family": spec.kind})(), "contract": None})()

    def _contract(spec, cfg):
        return REGISTRY.resolve("dense_cnn").inference_contract_factory(ModelSpec(kind="dense_cnn"), cfg)

    descriptor = ModelFamilyDescriptor(
        name="fake_extension",
        aliases=frozenset({"fake_alias"}),
        capabilities=CapabilitySet.of([CROP_INPUT, DENSE_PLACE_POLICY]),
        spec_schema=ModelSpec,
        components=FamilyComponents(trunk="fake", heads=("policy",)),
        model_builder=_builder,
        train_adapter_factory=_train,
        inference_adapter_factory=_infer,
        inference_contract_factory=_contract,
        policy_provider_factory=lambda spec, cfg, model: model,
        loss_plan_provider=lambda spec, cfg: {"loss": "policy"},
        recipe_provider=lambda host: {"valid": True},
        tune_space_provider=lambda host: {"mutations": {"channels": [1]}},
        checkpoint_manifest_provider=lambda spec, cfg: {"model_family": spec.kind},
    )

    registry.register(descriptor)

    assert registry.resolve("fake_alias").name == "fake_extension"


def test_no_family_kind_string_dispatch():
    offenders = {}
    for root_name in ("trunks", "composers"):
        for path in (MODELS_ROOT / root_name).glob("*.py"):
            text = path.read_text(encoding="utf-8")
            matches = re.findall(r"family_kind\s*[=!]=|family_kind\s+in\s", text)
            if matches:
                offenders[str(path.relative_to(MODELS_ROOT))] = matches
    for path in MODELS_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if 'name.startswith("lookahead_")' in text:
            offenders[str(path.relative_to(MODELS_ROOT))] = ['name.startswith("lookahead_")']
    assert offenders == {}


def test_each_family_only_allocates_declared_heads():
    aliases = {
        "dense_cnn": "cnn",
        "restnet": "restnet",
        "graph_hybrid": "graph_hybrid_0",
        "global_xattn": "global_xattn_0",
        "global_line_window": "global_line_window_0",
        "global_relation_graph": "global_graph_option1",
    }
    for descriptor in REGISTRY.descriptors():
        model = build_model(_cfg(aliases[descriptor.name]), device=torch.device("cpu"))
        assert set(model.heads.keys()) == set(descriptor.components.heads)


def test_inference_contract_matches_built_modules():
    for descriptor in REGISTRY.descriptors():
        cfg = _cfg({"global_xattn": "global_xattn_0", "global_line_window": "global_line_window_0", "global_relation_graph": "global_graph_option1"}.get(descriptor.name, "cnn"))
        if descriptor.name == "restnet":
            cfg = _cfg("restnet")
        if descriptor.name == "graph_hybrid":
            cfg = _cfg("graph_hybrid_0")
        model = build_model(cfg, device=torch.device("cpu"))
        contract = descriptor.inference_contract_factory(ModelSpec(kind=descriptor.name), cfg)
        assert {head.name for head in contract.heads} == set(model.heads.keys())


def test_register_new_trunk_without_editing_existing_trunks():
    class FakeTrunk(torch.nn.Module):
        feature_channels = 4
        input_tensors = ()

        def forward(self, inputs: CropInputs) -> torch.Tensor:
            return torch.zeros(inputs.tensor.shape[0], 4, 33, 33)

    def _builder(spec, cfg, *, device=None, inference=False):
        from hexorl.models.composers import CropModel
        from hexorl.models.heads import build_heads_for_family

        trunk = FakeTrunk()
        return CropModel(trunk, build_heads_for_family(spec, cfg, trunk))

    registry = REGISTRY.clone()
    descriptor = ModelFamilyDescriptor(
        name="fake_trunk_family",
        aliases=frozenset(),
        capabilities=CapabilitySet.of([CROP_INPUT, DENSE_PLACE_POLICY]),
        spec_schema=ModelSpec,
        components=FamilyComponents(trunk="fake_trunk", heads=("policy", "value")),
        model_builder=_builder,
        train_adapter_factory=lambda spec, cfg, model, *, device: None,
        inference_adapter_factory=lambda spec, cfg, model: None,
        inference_contract_factory=lambda spec, cfg: None,
        policy_provider_factory=lambda spec, cfg, model: model,
        loss_plan_provider=lambda spec, cfg: {},
        recipe_provider=lambda host: {},
        tune_space_provider=lambda host: {},
        checkpoint_manifest_provider=lambda spec, cfg: {},
    )
    registry.register(descriptor)
    TRUNK_REGISTRY["fake_trunk"] = _builder
    model = registry.resolve("fake_trunk_family").model_builder(ModelSpec(kind="fake_trunk_family"), _cfg("cnn"))
    assert model(CropInputs(torch.zeros(1, 13, 33, 33)))["policy"].shape == (1, 1089)
    del TRUNK_REGISTRY["fake_trunk"]


def test_per_family_params_reject_unknown_keys():
    cfg = Config.model_validate({"model": {"architecture": "cnn", "graph_token_set": "graph256_cells"}})
    with pytest.raises(ValueError, match="does not accept model fields"):
        build_model(cfg, device=torch.device("cpu"))


def test_no_dead_variants():
    reachable = {descriptor.components.trunk for descriptor in REGISTRY.descriptors()}
    assert reachable <= set(TRUNK_REGISTRY)
