import ast
from pathlib import Path

import torch

from hexorl.config import Config
from hexorl.models.capabilities import CROP_INPUT, DENSE_PLACE_POLICY, CapabilitySet
from hexorl.models.families import builtin_descriptors
from hexorl.models.factory import REGISTRY, build_model
from hexorl.models.registry import FamilyComponents, ModelFamilyDescriptor, ModelFamilyRegistry
from hexorl.models.specs import REQUIRED_MODEL_KINDS, ModelSpec

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_ROOT = PROJECT_ROOT / "Python" / "src" / "hexorl" / "models"


def _cfg(architecture: str) -> Config:
    return Config.model_validate(
        {
            "model": {
                "architecture": architecture,
                "channels": 4,
                "blocks": 1,
                "attention_heads": 1,
                "graph_layers": 1,
            },
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
    assert build_dense_cnn_model.__module__ == "hexorl.models.trunks.dense_cnn"
    assert build_restnet_model.__module__ == "hexorl.models.trunks.restnet"
    assert build_graph_hybrid_model.__module__ == "hexorl.models.trunks.graph_hybrid"
    assert build_global_relation_graph_model.__module__ == "hexorl.models.trunks.global_graph"
    assert PolicyHead.__module__ == "hexorl.models.heads.policy"
    assert ValueBinnedHead.__module__ == "hexorl.models.heads.value"
    assert RegretRankHead.__module__ == "hexorl.models.heads.regret"
    assert SparsePolicyHead.__module__ == "hexorl.models.heads.sparse_policy"
    assert PairPolicyHead.__module__ == "hexorl.models.heads.pair_policy"
    assert AuxPolicyHead.__module__ == "hexorl.models.heads.tactical"
    assert AxisHead.__module__ == "hexorl.models.heads.tactical"
    assert AxisMapHead.__module__ == "hexorl.models.heads.tactical"
    assert MovesLeftHead.__module__ == "hexorl.models.heads.tactical"
    assert HexConv2d.__module__ == "hexorl.models.trunks.dense_cnn"
    assert GatedResBlock.__module__ == "hexorl.models.trunks.dense_cnn"
    assert SpatialTransformerBlock.__module__ == "hexorl.models.trunks.restnet"
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


def test_every_registered_family_builds_inference_adapter_manifest():
    cfg = _cfg("cnn")
    for descriptor in REGISTRY.descriptors():
        spec = ModelSpec(kind=descriptor.name)
        manifest = descriptor.inference_adapter_factory(spec, cfg, torch.nn.Identity()).manifest
        assert manifest.model_family == descriptor.name
        assert manifest.protocol_version == 1


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
        return type("FakeInference", (), {"manifest": type("FakeManifest", (), {"model_family": spec.kind})()})()

    descriptor = ModelFamilyDescriptor(
        name="fake_extension",
        aliases=frozenset({"fake_alias"}),
        capabilities=CapabilitySet.of([CROP_INPUT, DENSE_PLACE_POLICY]),
        spec_schema=ModelSpec,
        components=FamilyComponents(trunk="fake", heads=("policy",)),
        model_builder=_builder,
        train_adapter_factory=_train,
        inference_adapter_factory=_infer,
        policy_provider_factory=lambda spec, cfg, model: model,
        loss_plan_provider=lambda spec, cfg: {"loss": "policy"},
        recipe_provider=lambda host: {"valid": True},
        tune_space_provider=lambda host: {"mutations": {"channels": [1]}},
        checkpoint_manifest_provider=lambda spec, cfg: {"model_family": spec.kind},
    )

    registry.register(descriptor)

    assert registry.resolve("fake_alias").name == "fake_extension"
