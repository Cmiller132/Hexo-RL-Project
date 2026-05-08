import numpy as np
import pytest
import torch

from hexorl.autotune import candidate_recipes_from_config
from hexorl.config import Config
from hexorl.graph.batch import (
    GRAPH_FEATURE_DIM,
    GRAPH_FEATURE_PLACEMENTS_REMAINING,
    GRAPH_SCHEMA_VERSION,
    RELATION_SCHEMA_VERSION,
    GraphBatch,
    GraphTokenType,
    V1_PAIR_FEATURE_DIM,
    collate_graph_batches,
    graph_batch_with_admitted_pair_rows,
)
from hexorl.inference.adapters import decode_global_graph_outputs
from hexorl.models.assembly import bins_to_value
from hexorl.models.families.global_graph import GlobalHexGraphNet
from hexorl.models.registry import architecture_ids, resolve_model_spec
from hexorl.search.pair_strategy import build_pair_strategy


V1_ARCHITECTURE_ID = "global_pair_biaffine_0"
V1_PAIR_STRATEGY_ID = "sampled_joint_pair_v1"
V1_OUTPUTS = [
    "cell_marginal_logits",
    "pair_completion_logits",
    "pair_proposal_score",
    "pair_joint_logits",
    "value",
    "terminal_tactical_v1",
]


def _manual_graph(legal_mask: tuple[bool, bool, bool] = (True, True, False)) -> GraphBatch:
    token_features = np.zeros((5, GRAPH_FEATURE_DIM), dtype=np.float32)
    token_features[:, GRAPH_FEATURE_PLACEMENTS_REMAINING] = 1.0
    token_type = np.asarray(
        [
            int(GraphTokenType.STATE),
            int(GraphTokenType.TURN),
            int(GraphTokenType.LEGAL),
            int(GraphTokenType.LEGAL),
            int(GraphTokenType.LEGAL),
        ],
        dtype=np.int64,
    )
    token_qr = np.asarray(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1],
        ],
        dtype=np.int32,
    )
    legal_qr = token_qr[2:5].copy()
    return GraphBatch(
        token_features=token_features,
        token_type=token_type,
        token_qr=token_qr,
        token_mask=np.ones(5, dtype=np.bool_),
        legal_token_indices=np.asarray([2, 3, 4], dtype=np.int64),
        legal_qr=legal_qr,
        legal_mask=np.asarray(legal_mask, dtype=np.bool_),
        pair_token_indices=np.zeros(0, dtype=np.int64),
        pair_first_indices=np.zeros(0, dtype=np.int64),
        pair_second_indices=np.zeros(0, dtype=np.int64),
        relation_bias=np.zeros((1, 5, 5), dtype=np.float32),
        relation_type=np.zeros((5, 5), dtype=np.int16),
        policy_target=np.zeros(3, dtype=np.float32),
        opp_legal_qr=np.zeros((0, 2), dtype=np.int32),
        opp_legal_mask=np.zeros(0, dtype=np.bool_),
        opp_policy_target=np.zeros(0, dtype=np.float32),
        pair_first_policy_target=np.zeros(3, dtype=np.float32),
        pair_policy_target=np.zeros(0, dtype=np.float32),
        pair_second_policy_target=np.zeros(0, dtype=np.float32),
        tactical_target=np.zeros(4, dtype=np.float32),
        placements_remaining=2,
        current_player=0,
        schema_version=GRAPH_SCHEMA_VERSION,
        relation_schema_version=RELATION_SCHEMA_VERSION,
    )


def _tensor_inputs(batch: GraphBatch) -> dict[str, torch.Tensor]:
    unbatched = np.asarray(batch.token_features).ndim == 2
    names = (
        "token_features",
        "token_type",
        "token_qr",
        "token_mask",
        "legal_token_indices",
        "legal_mask",
        "pair_token_indices",
        "pair_first_indices",
        "pair_second_indices",
        "relation_type",
        "relation_bias",
    )
    tensors = {}
    for name in names:
        tensor = torch.from_numpy(getattr(batch, name))
        if unbatched and name in {"token_features", "token_type", "token_qr", "token_mask", "legal_token_indices", "legal_mask", "pair_token_indices", "pair_first_indices", "pair_second_indices"} and tensor.ndim in {1, 2}:
            tensor = tensor.unsqueeze(0)
        elif unbatched and name in {"relation_type", "relation_bias"} and tensor.ndim in {2, 3}:
            tensor = tensor.unsqueeze(0)
        tensors[name] = tensor
    if batch.pair_features is not None:
        pair_features = torch.from_numpy(batch.pair_features)
        if unbatched and pair_features.ndim == 2:
            pair_features = pair_features.unsqueeze(0)
        tensors["pair_features"] = pair_features
    return tensors


def test_v1_config_and_autotune_materialize_side_by_side():
    cfg = Config.model_validate(
        {
            "autotune": {
                "scout": {
                    "candidate_plan": [
                        "global_xattn_0:none",
                        "global_graph768_champion:none",
                        f"{V1_ARCHITECTURE_ID}:{V1_PAIR_STRATEGY_ID}",
                    ]
                }
            }
        }
    )

    recipes = {recipe.candidate_id: recipe for recipe in candidate_recipes_from_config(cfg)}

    assert V1_ARCHITECTURE_ID in set(architecture_ids())
    assert set(recipes) == {
        "global_xattn_0__none__v1",
        "global_graph768_champion__none__v1",
        "global_pair_biaffine_0__sampled_joint_pair_v1__v1",
    }

    xattn_cfg = recipes["global_xattn_0__none__v1"].materialize_config(Config())
    champion_cfg = recipes["global_graph768_champion__none__v1"].materialize_config(Config())
    v1_cfg = recipes["global_pair_biaffine_0__sampled_joint_pair_v1__v1"].materialize_config(Config())

    assert xattn_cfg.model.architecture == "global_xattn_0"
    assert xattn_cfg.model.heads == ["policy_place", "value"]
    assert xattn_cfg.model.pair_strategy == "none"
    assert champion_cfg.model.architecture == "global_graph768_champion"
    assert champion_cfg.model.heads == ["policy_place", "value"]
    assert champion_cfg.model.pair_strategy == "none"

    assert v1_cfg.model.architecture == V1_ARCHITECTURE_ID
    assert v1_cfg.model.heads == V1_OUTPUTS
    assert v1_cfg.model.pair_strategy == V1_PAIR_STRATEGY_ID
    assert v1_cfg.model.pair_strategy_max_pairs == 256
    assert v1_cfg.selfplay.legal_row_mode == "full_rust_legal"
    assert v1_cfg.selfplay.tactical_mode == "proposal_and_label"
    assert v1_cfg.selfplay.constrain_threats is False
    assert resolve_model_spec(v1_cfg).outputs == tuple(V1_OUTPUTS)


def test_v1_requires_explicit_strategy_and_full_rust_legal_rows():
    no_strategy = Config.model_validate(
        {
            "model": {
                "architecture": V1_ARCHITECTURE_ID,
                "heads": V1_OUTPUTS,
            }
        }
    )
    assert no_strategy.model.pair_strategy == "none"

    with pytest.raises(ValueError, match="selfplay.constrain_threats must be false"):
        Config.model_validate(
            {
                "model": {
                    "architecture": V1_ARCHITECTURE_ID,
                    "heads": V1_OUTPUTS,
                    "pair_strategy": V1_PAIR_STRATEGY_ID,
                    "pair_strategy_max_pairs": 16,
                },
                "selfplay": {
                    "legal_row_mode": "full_rust_legal",
                    "tactical_mode": "proposal_and_label",
                },
            }
        )

    with pytest.raises(ValueError, match="requires model.architecture"):
        Config.model_validate(
            {
                "model": {
                    "architecture": "global_xattn_0",
                    "heads": [
                        "policy_place",
                        "value",
                        "policy_pair_first",
                        "policy_pair_joint",
                        "policy_pair_second",
                    ],
                    "pair_strategy": V1_PAIR_STRATEGY_ID,
                    "pair_strategy_max_pairs": 16,
                },
                "selfplay": {
                    "legal_row_mode": "full_rust_legal",
                    "tactical_mode": "proposal_and_label",
                    "constrain_threats": False,
                },
            }
        )

    strategy = build_pair_strategy(V1_PAIR_STRATEGY_ID, max_pairs=16, prior_mix=0.35)
    assert strategy.enabled
    assert strategy.leaf_pair_scoring_enabled
    assert "pair_joint_logits" in strategy.required_output_contracts


def test_graph_admitted_pair_rows_reference_legal_rows_without_pair_tokens():
    base = _manual_graph((True, True, True))
    admitted = graph_batch_with_admitted_pair_rows(
        base,
        np.asarray(
            [
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.int32,
        ),
        pair_features=np.zeros((3, V1_PAIR_FEATURE_DIM), dtype=np.float32),
    )

    assert admitted.token_features.shape == base.token_features.shape
    assert admitted.pair_first_indices.tolist() == [2, 2]
    assert admitted.pair_second_indices.tolist() == [3, 4]
    assert np.all(admitted.pair_token_indices == -1)
    assert admitted.pair_features.shape == (2, V1_PAIR_FEATURE_DIM)

    batch = collate_graph_batches([admitted])
    assert batch.pair_features.shape == (1, 2, V1_PAIR_FEATURE_DIM)


def test_v1_pair_biaffine_outputs_are_finite_masked_and_unordered_symmetric():
    base = _manual_graph()
    batch = GraphBatch(
        **{
            **base.__dict__,
            "pair_token_indices": np.full(3, -1, dtype=np.int64),
            "pair_first_indices": np.asarray([2, 3, 2], dtype=np.int64),
            "pair_second_indices": np.asarray([3, 2, 2], dtype=np.int64),
            "pair_policy_target": np.zeros(3, dtype=np.float32),
            "pair_second_policy_target": np.zeros(3, dtype=np.float32),
        }
    )
    model = GlobalHexGraphNet(
        channels=16,
        heads=4,
        layers=1,
        architecture=V1_ARCHITECTURE_ID,
        output_heads=V1_OUTPUTS,
    )
    model.eval()

    with torch.no_grad():
        out = model(**_tensor_inputs(batch))

    assert set(V1_OUTPUTS) <= set(out)
    assert "policy_place" not in out
    assert "policy_pair_joint" not in out
    assert out["cell_marginal_logits"].shape == (1, 3)
    assert out["pair_completion_logits"].shape == (1, 3)
    assert out["pair_proposal_score"].shape == (1, 3)
    assert out["pair_joint_logits"].shape == (1, 3)
    assert out["value"].shape == (1, 65)
    assert out["terminal_tactical_v1"].shape == (1, 8)
    assert out["cell_marginal_logits"][0, 2].item() == pytest.approx(-80.0)
    assert out["pair_joint_logits"][0, 2].item() == pytest.approx(-80.0)
    assert torch.isfinite(out["pair_joint_logits"][0, :2]).all()
    torch.testing.assert_close(out["pair_joint_logits"][0, 0], out["pair_joint_logits"][0, 1])
    torch.testing.assert_close(out["pair_proposal_score"][0, 0], out["pair_proposal_score"][0, 1])


def test_v1_inference_decode_carries_output_names_and_row_metadata():
    base = _manual_graph()
    admitted = graph_batch_with_admitted_pair_rows(
        base,
        np.asarray([[0, 0, 1, 0]], dtype=np.int32),
    )
    batch = collate_graph_batches([admitted])
    outputs = {
        "cell_marginal_logits": torch.zeros(1, 3),
        "pair_completion_logits": torch.ones(1, 1),
        "pair_proposal_score": torch.full((1, 1), 0.25),
        "pair_joint_logits": torch.full((1, 1), 0.5),
        "value": torch.zeros(1, 65),
        "terminal_tactical_v1": torch.zeros(1, 8),
    }
    decoded = decode_global_graph_outputs(
        outputs,
        _tensor_inputs(batch),
        value_decoder=bins_to_value,
    )

    assert decoded.cell_marginal_logits.shape == (1, 3)
    assert decoded.pair_completion_logits.shape == (1, 1)
    assert decoded.pair_proposal_score.shape == (1, 1)
    assert decoded.pair_joint_logits.shape == (1, 1)
    assert decoded.terminal_tactical_v1.shape == (1, 8)
    metadata = decoded.metadata["outputs"]
    assert metadata["cell_marginal_logits"]["row_table"]["family"] == "legal"
    assert metadata["pair_joint_logits"]["row_table"]["family"] == "pair_joint"
    assert metadata["pair_completion_logits"]["row_table"]["identity_hash"] == metadata["pair_joint_logits"]["row_table"]["identity_hash"]
    assert metadata["pair_proposal_score"]["row_table"]["identity_hash"] == metadata["pair_joint_logits"]["row_table"]["identity_hash"]
    assert metadata["terminal_tactical_v1"]["kind"] == "auxiliary"
