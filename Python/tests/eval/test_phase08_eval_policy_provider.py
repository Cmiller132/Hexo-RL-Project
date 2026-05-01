import numpy as np
import pytest

from hexorl.config import Config
from hexorl.eval.position_services import build_search_context
from hexorl.eval.players import PolicyPlayer
from hexorl.models.factory import get_model_registry
from hexorl.models.specs import ModelSpec
from hexorl.search.policy_provider import create_policy_provider
from hexorl.search.priors import SearchEvaluation


class _Provider:
    name = "UnitPolicyProvider"

    def __init__(self, family: str):
        self.family = family
        self.seen = []

    def evaluate_root(self, context):
        self.seen.append(context)
        priors = np.ones(context.legal_table.rows.shape[0], dtype=np.float32)
        priors /= priors.sum()
        return SearchEvaluation(
            context=context,
            value=0.0,
            legal_row_ids=np.arange(priors.shape[0], dtype=np.int64),
            legal_dense_indices=context.legal_table.dense_indices,
            row_priors=priors,
            prior_source=np.full(priors.shape[0], 2, dtype=np.uint8),
            policy_provider=self.name,
            model_family=self.family,
            model_spec_version="1",
            inference_protocol="unit",
        )

    def evaluate_leaves(self, contexts):
        return [self.evaluate_root(ctx) for ctx in contexts]


class _Manifest:
    transport = "unit-registry"


class _InferenceClient:
    manifest = _Manifest()

    def evaluate(self, operation_name, payload):
        if operation_name == "place_value":
            count = int(np.asarray(payload["tensor"]).shape[0])
            heads = {"policy": np.ones((count, 1089), dtype=np.float32), "value": np.zeros((count,), dtype=np.float32)}
        elif operation_name == "sparse_place_value":
            count = int(np.asarray(payload["tensor"]).shape[0])
            candidate_indices = payload["candidate_indices"]
            heads = {
                "policy": np.ones((count, 1089), dtype=np.float32),
                "value": np.zeros((count,), dtype=np.float32),
                "sparse_policy": np.ones(np.asarray(candidate_indices).shape, dtype=np.float32),
            }
        elif operation_name == "graph_place_value":
            legal_count = int(np.asarray(payload["legal_mask"]).shape[0])
            heads = {
                "policy_place": np.ones((legal_count,), dtype=np.float32),
                "value": np.zeros((1,), dtype=np.float32),
                "policy_pair_first": np.zeros((0,), dtype=np.float32),
            }
        else:
            raise ValueError(operation_name)
        return type("Response", (), {"head_outputs": heads, "telemetry": {"wait_ms": 0.0}})()


def test_arena_policy_player_covers_every_registered_family_through_provider():
    pytest.importorskip("_engine")
    for family in get_model_registry().names():
        provider = _Provider(family)
        player = PolicyPlayer(provider, model_spec=ModelSpec(kind=family), recipe_id=f"test:{family}")
        move = player([], 100, 0)
        assert move is not None
        assert provider.seen[0].model_family == family
        assert player.telemetry[-1].provider_type == "UnitPolicyProvider"
        assert player.telemetry[-1].pair_rows_scored == 0


def test_eval_uses_real_policy_provider_registry_for_every_registered_family():
    pytest.importorskip("_engine")
    client = _InferenceClient()
    for family in get_model_registry().names():
        spec = ModelSpec(kind=family, source_name="test")
        provider = create_policy_provider(model_spec=spec, client=client)
        context = build_search_context(
            b"",
            model_spec=spec,
            recipe_id=f"registry:{family}",
            candidate_budget=32,
        )
        evaluation = provider.evaluate_root(context)
        assert evaluation.model_family == family
        assert evaluation.policy_provider.endswith("PolicyProvider")
        assert evaluation.legal_row_ids.shape[0] == context.legal_table.rows.shape[0]
        assert np.isclose(float(evaluation.row_priors.sum()), 1.0)


def test_eval_import_boundary_has_no_model_class_or_architecture_dispatch():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "hexorl" / "eval"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert "from hexorl.models.network import HexNet" not in text
    assert "architecture ==" not in text
    assert "architecture.startswith" not in text
