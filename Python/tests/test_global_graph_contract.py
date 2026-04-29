import struct

import numpy as np
import pytest
import torch

from hexorl.config import Config
from hexorl.graph import (
    GRAPH_SCHEMA_VERSION,
    GraphTokenType,
    RelationType,
    build_graph_batch_from_history as _build_graph_batch_from_history,
    collate_graph_batches,
    graph_batch_with_reference_pair_rows,
    transform_history,
    transform_pair_policy_target,
    transform_policy_target,
    transform_qr,
)
from hexorl.model.global_graph import GlobalHexGraphNet
from hexorl.model.network import build_model_from_config
from hexorl.selfplay.worker import _graph_batch_with_pair_rows
from hexorl.train.trainer import Trainer


def _hist(*moves):
    data = bytearray()
    for player, q, r in moves:
        data.extend(struct.pack("<iii", player, q, r))
    return bytes(data)


def build_graph_batch_from_history(history, **kwargs):
    """Test helper: graph contract always uses the Rust radius-8 legal table."""
    kwargs.pop("radius", None)
    kwargs.setdefault("radius", 8)
    kwargs.setdefault(
        "include_pair_rows",
        bool(kwargs.get("pair_policy_target"))
        or "max_pair_rows" in kwargs
        or bool(kwargs.get("allow_pair_truncation", False)),
    )
    return _build_graph_batch_from_history(history, **kwargs)


def test_global_graph_builder_preserves_all_legal_rows():
    history = _hist((0, 0, 0), (1, 1, 0), (1, 0, 1))
    graph = build_graph_batch_from_history(history, radius=2)

    assert graph.schema_version == GRAPH_SCHEMA_VERSION
    assert graph.legal_qr.shape[0] == graph.legal_token_indices.shape[0]
    assert graph.legal_mask.all()
    assert len({tuple(qr) for qr in graph.legal_qr.tolist()}) == graph.legal_qr.shape[0]
    assert (0, 0) not in {tuple(qr) for qr in graph.legal_qr.tolist()}
    legal_token_types = graph.token_type[graph.legal_token_indices]
    assert np.all(legal_token_types == int(GraphTokenType.LEGAL))


def test_global_graph_rejects_sub_rust_legal_radius():
    with pytest.raises(ValueError, match="radius must be 8"):
        _build_graph_batch_from_history(_hist((0, 0, 0)), radius=2)


def test_global_graph_opponent_policy_uses_independent_legal_rows():
    history = _hist((0, 0, 0), (1, 1, 0))
    opp_legal = [(-3, 1), (2, -1), (2, 0)]
    graph = build_graph_batch_from_history(
        history,
        radius=2,
        opp_legal_moves=opp_legal,
        opp_policy_target=[(2, -1, 3.0), (-3, 1, 1.0)],
    )

    assert [tuple(qr) for qr in graph.opp_legal_qr.tolist()] == opp_legal
    assert graph.opp_legal_mask.all()
    assert graph.opp_policy_target.sum() == pytest.approx(1.0)
    target_by_qr = {
        tuple(qr): float(prob)
        for qr, prob in zip(graph.opp_legal_qr.tolist(), graph.opp_policy_target.tolist())
    }
    assert target_by_qr[(2, -1)] == pytest.approx(0.75)
    assert target_by_qr[(-3, 1)] == pytest.approx(0.25)

    with pytest.raises(ValueError, match="opp_legal_moves"):
        build_graph_batch_from_history(
            history,
            radius=2,
            opp_policy_target=[(2, -1, 1.0)],
        )


def test_global_graph_targets_must_match_their_own_legal_tables():
    history = _hist((0, 0, 0), (1, 1, 0))

    with pytest.raises(ValueError, match="policy_target"):
        build_graph_batch_from_history(
            history,
            radius=2,
            policy_target=[(99, 99, 1.0)],
        )
    with pytest.raises(ValueError, match="opp_policy_target"):
        build_graph_batch_from_history(
            history,
            radius=2,
            opp_legal_moves=[(2, 0), (2, -1)],
            opp_policy_target=[(99, 99, 1.0)],
        )
    with pytest.raises(ValueError, match="opp_legal_moves contains occupied"):
        build_graph_batch_from_history(
            history,
            radius=2,
            opp_legal_moves=[(0, 0), (2, 0)],
        )


def test_global_graph_builder_includes_required_token_families_and_relations():
    history = _hist((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, -1, 0), (0, -2, 0))
    graph = build_graph_batch_from_history(history, radius=2)
    token_types = set(int(x) for x in graph.token_type.tolist())

    for token_type in [
        GraphTokenType.STATE,
        GraphTokenType.TURN,
        GraphTokenType.PLAYER,
        GraphTokenType.STONE,
        GraphTokenType.LEGAL,
        GraphTokenType.WINDOW6,
        GraphTokenType.LINE,
        GraphTokenType.COMPONENT,
    ]:
        assert int(token_type) in token_types
    assert graph.relation_type.shape == (graph.token_type.shape[0], graph.token_type.shape[0])
    assert graph.relation_bias.shape[1:] == graph.relation_type.shape
    assert int(RelationType.SAME_LINE) in set(int(x) for x in graph.relation_type.reshape(-1).tolist())


def test_global_graph_relation_bias_contract_includes_cover_pair_and_component_edges():
    history = _hist(
        (0, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
        (0, 1, 0),
        (0, 2, 0),
        (1, 0, 3),
        (1, 0, 4),
        (0, 3, 0),
    )
    graph = build_graph_batch_from_history(
        history,
        radius=2,
        pair_policy_target=[((3, 0), (4, 0), 1.0)],
    )
    relation_ids = set(int(x) for x in graph.relation_type.reshape(-1).tolist())

    assert int(GraphTokenType.COVER_SET) in set(int(x) for x in graph.token_type.tolist())
    assert int(GraphTokenType.PAIR_ACTION) in set(int(x) for x in graph.token_type.tolist())
    for relation in [
        RelationType.DISTANCE_BUCKET,
        RelationType.DIRECTION_BUCKET,
        RelationType.SAME_AXIS,
        RelationType.SAME_LINE,
        RelationType.SAME_WINDOW6,
        RelationType.STONE_IN_WINDOW6,
        RelationType.LEGAL_IN_WINDOW6,
        RelationType.LEGAL_IN_COVER_SET,
        RelationType.WINDOW6_TO_COVER_SET,
        RelationType.LINE_TO_WINDOW6,
        RelationType.LEGAL_TO_PAIR_ACTION,
        RelationType.SAME_COMPONENT,
        RelationType.AGE_ORDER_BUCKET,
        RelationType.RECENT_MOVE_RELATION,
        RelationType.FIRST_SECOND_PAIR_RELATION,
        RelationType.D6_ORBIT_RELATION,
    ]:
        assert int(relation) in relation_ids
    assert np.isfinite(graph.relation_bias).all()


def test_global_graph_pair_rows_mask_opening_and_exist_on_two_placement_turns():
    opening = build_graph_batch_from_history(b"", radius=2)
    normal_turn = build_graph_batch_from_history(
        _hist((0, 0, 0)),
        radius=2,
        max_pair_rows=4,
        allow_pair_truncation=True,
    )
    second_placement = build_graph_batch_from_history(
        _hist((0, 0, 0), (1, 1, 0)),
        radius=2,
        include_pair_rows=True,
    )

    assert opening.pair_token_indices.shape[0] == 0
    assert normal_turn.pair_token_indices.shape[0] > 0
    assert np.all(normal_turn.pair_first_indices != normal_turn.pair_second_indices)
    assert second_placement.pair_token_indices.shape[0] == second_placement.legal_qr.shape[0]
    assert np.all(second_placement.token_type[second_placement.pair_first_indices] == int(GraphTokenType.STONE))
    assert np.all(second_placement.token_type[second_placement.pair_second_indices] == int(GraphTokenType.LEGAL))


def test_global_graph_second_placement_pair_targets_are_ordered_and_conditional():
    history = _hist((0, 0, 0), (1, 1, 0))
    probe = build_graph_batch_from_history(history, radius=2)
    first = (1, 0)
    second = tuple(probe.legal_qr[0].tolist())

    graph = build_graph_batch_from_history(
        history,
        radius=2,
        pair_policy_target=[(first, second, 1.0)],
    )

    assert graph.pair_policy_target.sum() == pytest.approx(1.0)
    second_tokens = graph.pair_second_indices[graph.pair_policy_target > 0.0]
    assert second_tokens.shape[0] == 1
    token_to_legal = {int(tok): i for i, tok in enumerate(graph.legal_token_indices.tolist())}
    assert tuple(graph.legal_qr[token_to_legal[int(second_tokens[0])]].tolist()) == second

    with pytest.raises(ValueError, match="first action"):
        build_graph_batch_from_history(
            history,
            radius=2,
            pair_policy_target=[((0, 0), second, 1.0)],
        )

    with pytest.raises(ValueError, match="pair-action table is empty"):
        build_graph_batch_from_history(
            b"",
            radius=2,
            pair_policy_target=[((0, 0), (1, 0), 1.0)],
        )


def test_global_graph_pair_targets_reject_duplicate_and_illegal_pairs():
    history = _hist((0, 0, 0), (1, 1, 0))

    with pytest.raises(ValueError, match="duplicate coordinates"):
        build_graph_batch_from_history(
            history,
            pair_policy_target=[((1, 0), (1, 0), 1.0)],
            radius=2,
        )

    with pytest.raises(ValueError, match="illegal"):
        build_graph_batch_from_history(
            history,
            pair_policy_target=[((1, 0), (99, 99), 1.0)],
            radius=2,
        )


def test_global_graph_pair_rows_fail_instead_of_silent_truncation():
    with pytest.raises(ValueError, match="pair rows would be truncated"):
        build_graph_batch_from_history(_hist((0, 0, 0)), radius=3, max_pair_rows=4)

    graph = build_graph_batch_from_history(
        _hist((0, 0, 0)),
        radius=3,
        max_pair_rows=4,
        allow_pair_truncation=True,
    )
    assert graph.pair_token_indices.shape[0] == 4


def test_global_graph_pair_chunks_remove_ipc_pair_cap_as_semantic_limit():
    graph = build_graph_batch_from_history(
        _hist((0, 0, 0)),
        include_pair_rows=False,
    )
    assert graph.legal_qr.shape[0] > 91

    pair_first = []
    pair_second = []
    for a_idx in range(graph.legal_qr.shape[0]):
        for b_idx in range(a_idx + 1, graph.legal_qr.shape[0]):
            pair_first.append(int(graph.legal_token_indices[a_idx]))
            pair_second.append(int(graph.legal_token_indices[b_idx]))
            if len(pair_first) == 4096:
                break
        if len(pair_first) == 4096:
            break
    chunk = _graph_batch_with_pair_rows(
        graph,
        np.asarray(pair_first, dtype=np.int64),
        np.asarray(pair_second, dtype=np.int64),
    )
    model = GlobalHexGraphNet(channels=16, heads=4, layers=1, architecture="global_pair_twostage_0")

    out = model(
        token_features=torch.from_numpy(chunk.token_features).unsqueeze(0),
        token_type=torch.from_numpy(chunk.token_type).unsqueeze(0),
        token_qr=torch.from_numpy(chunk.token_qr).unsqueeze(0),
        token_mask=torch.from_numpy(chunk.token_mask).unsqueeze(0),
        legal_token_indices=torch.from_numpy(chunk.legal_token_indices).unsqueeze(0),
        legal_mask=torch.from_numpy(chunk.legal_mask).unsqueeze(0),
        pair_first_indices=torch.from_numpy(chunk.pair_first_indices).unsqueeze(0),
        pair_second_indices=torch.from_numpy(chunk.pair_second_indices).unsqueeze(0),
        pair_token_indices=torch.from_numpy(chunk.pair_token_indices).unsqueeze(0),
    )

    assert out["policy_pair_joint"].shape == (1, 4096)
    assert out["policy_pair_second"].shape == (1, 4096)


def test_global_graph_reference_pair_rows_cover_full_first_placement_table():
    graph = build_graph_batch_from_history(
        _hist((0, 0, 0)),
        include_pair_rows=False,
    )
    first = tuple(graph.legal_qr[0].tolist())
    second = tuple(graph.legal_qr[1].tolist())

    graph = graph_batch_with_reference_pair_rows(
        graph,
        [(first, second, 1.0)],
    )

    legal_count = int(graph.legal_qr.shape[0])
    assert graph.token_type.shape[0] < graph.pair_token_indices.shape[0]
    assert graph.pair_token_indices.shape[0] == legal_count * (legal_count - 1) // 2
    assert graph.pair_policy_target.sum() == pytest.approx(1.0)
    assert graph.pair_first_policy_target.sum() == pytest.approx(1.0)
    first_row = {
        tuple(qr.tolist()): row
        for row, qr in enumerate(graph.legal_qr)
    }[first]
    second_row = {
        tuple(qr.tolist()): row
        for row, qr in enumerate(graph.legal_qr)
    }[second]
    assert graph.pair_first_policy_target[first_row] == pytest.approx(1.0)
    assert graph.pair_first_policy_target[second_row] == pytest.approx(0.0)


def test_global_graph_model_forward_with_padded_batch():
    graphs = [
        build_graph_batch_from_history(_hist((0, 0, 0)), radius=2),
        build_graph_batch_from_history(_hist((0, 0, 0), (1, 1, 0), (1, 0, 1)), radius=2),
    ]
    batch = collate_graph_batches(graphs)
    cfg = Config.model_validate(
        {
            "model": {
                "architecture": "global_graph_option1",
                "channels": 16,
                "attention_heads": 4,
                "graph_layers": 1,
            },
            "inference": {"fp16": False},
        }
    )
    model = build_model_from_config(cfg, device=torch.device("cpu"))
    tensors = {
        "token_features": torch.from_numpy(batch.token_features),
        "token_type": torch.from_numpy(batch.token_type),
        "token_qr": torch.from_numpy(batch.token_qr),
        "token_mask": torch.from_numpy(batch.token_mask),
        "legal_token_indices": torch.from_numpy(batch.legal_token_indices),
        "legal_mask": torch.from_numpy(batch.legal_mask),
        "pair_first_indices": torch.from_numpy(batch.pair_first_indices),
        "pair_second_indices": torch.from_numpy(batch.pair_second_indices),
        "pair_token_indices": torch.from_numpy(batch.pair_token_indices),
        "relation_type": torch.from_numpy(batch.relation_type),
        "relation_bias": torch.from_numpy(batch.relation_bias),
    }

    out = model(**tensors)

    assert out["policy_place"].shape == batch.legal_mask.shape
    assert out["policy_pair_first"].shape == batch.legal_mask.shape
    assert out["value"].shape == (2, 65)
    assert "policy_pair_joint" in out


def test_global_graph_trainer_runs_graph_native_step_without_dense_policy():
    graph = collate_graph_batches([
        build_graph_batch_from_history(
            _hist((0, 0, 0), (1, 1, 0), (1, 0, 1)),
            policy_target=[(2, 0, 1.0)],
            include_pair_rows=False,
        )
    ])
    cfg = Config.model_validate(
        {
            "model": {
                "architecture": "global_xattn_0",
                "channels": 16,
                "attention_heads": 4,
                "graph_layers": 1,
            },
            "train": {
                "batches_per_epoch": 1,
                "loss_weights": {
                    "policy": 1.0,
                    "policy_place": 1.0,
                    "value": 1.0,
                    "opp_policy": 0.1,
                    "tactical": 0.1,
                },
            },
            "inference": {"fp16": False},
        }
    )
    model = build_model_from_config(cfg, device=torch.device("cpu"))
    aux = {
        "token_features": torch.from_numpy(graph.token_features),
        "token_type": torch.from_numpy(graph.token_type),
        "token_qr": torch.from_numpy(graph.token_qr),
        "token_mask": torch.from_numpy(graph.token_mask),
        "legal_token_indices": torch.from_numpy(graph.legal_token_indices),
        "legal_qr": torch.from_numpy(graph.legal_qr),
        "legal_mask": torch.from_numpy(graph.legal_mask),
        "policy_target": torch.from_numpy(graph.policy_target),
        "opp_legal_qr": torch.from_numpy(graph.opp_legal_qr),
        "opp_legal_mask": torch.from_numpy(graph.opp_legal_mask),
        "opp_policy_target": torch.from_numpy(graph.opp_policy_target),
        "pair_token_indices": torch.from_numpy(graph.pair_token_indices),
        "pair_first_indices": torch.from_numpy(graph.pair_first_indices),
        "pair_second_indices": torch.from_numpy(graph.pair_second_indices),
        "pair_policy_target": torch.from_numpy(graph.pair_policy_target),
        "relation_type": torch.from_numpy(graph.relation_type),
        "relation_bias": torch.from_numpy(graph.relation_bias),
        "tactical_target": torch.from_numpy(graph.tactical_target),
        "policy_weight": torch.ones(1),
        "opp_policy_weight": torch.ones(1),
    }
    batch = (
        torch.zeros(1, 13, 33, 33),
        torch.zeros(1, 1089),
        torch.zeros(1),
        [],
        aux,
    )
    trainer = Trainer(model, cfg, dataloader=[], device=torch.device("cpu"))

    losses = trainer._train_step(batch, 0)

    assert np.isfinite(losses["total"])


def test_global_graph_pair_logits_mask_invalid_rows_even_without_pair_token_indices():
    graphs = [
        build_graph_batch_from_history(b"", radius=2),
        build_graph_batch_from_history(_hist((0, 0, 0)), radius=2),
    ]
    batch = collate_graph_batches(graphs)
    model = GlobalHexGraphNet(channels=16, heads=4, layers=1, architecture="global_xattn_0")
    tensors = {
        "token_features": torch.from_numpy(batch.token_features),
        "token_type": torch.from_numpy(batch.token_type),
        "token_qr": torch.from_numpy(batch.token_qr),
        "token_mask": torch.from_numpy(batch.token_mask),
        "legal_token_indices": torch.from_numpy(batch.legal_token_indices),
        "legal_mask": torch.from_numpy(batch.legal_mask),
        "pair_first_indices": torch.from_numpy(batch.pair_first_indices),
        "pair_second_indices": torch.from_numpy(batch.pair_second_indices),
    }

    out = model(**tensors)

    assert torch.all(out["policy_pair_joint"][0] == -80.0)
    assert torch.isfinite(out["policy_pair_joint"][1][batch.pair_token_indices[1] >= 0]).all()


def test_global_graph_full_requires_relation_bias_contract():
    graph = build_graph_batch_from_history(_hist((0, 0, 0)), radius=2)
    model = GlobalHexGraphNet(channels=16, heads=4, layers=1, architecture="global_graph_full_0")

    with pytest.raises(ValueError, match="relation_type and relation_bias"):
        model(
            token_features=torch.from_numpy(graph.token_features).unsqueeze(0),
            token_type=torch.from_numpy(graph.token_type).unsqueeze(0),
            token_qr=torch.from_numpy(graph.token_qr).unsqueeze(0),
            token_mask=torch.from_numpy(graph.token_mask).unsqueeze(0),
            legal_token_indices=torch.from_numpy(graph.legal_token_indices).unsqueeze(0),
            legal_mask=torch.from_numpy(graph.legal_mask).unsqueeze(0),
        )


def test_global_graph_relation_tensor_shapes_are_validated():
    graph = build_graph_batch_from_history(_hist((0, 0, 0)), radius=2)
    model = GlobalHexGraphNet(channels=16, heads=4, layers=1, architecture="global_xattn_0")
    tensors = {
        "token_features": torch.from_numpy(graph.token_features).unsqueeze(0),
        "token_type": torch.from_numpy(graph.token_type).unsqueeze(0),
        "token_qr": torch.from_numpy(graph.token_qr).unsqueeze(0),
        "token_mask": torch.from_numpy(graph.token_mask).unsqueeze(0),
        "legal_token_indices": torch.from_numpy(graph.legal_token_indices).unsqueeze(0),
        "legal_mask": torch.from_numpy(graph.legal_mask).unsqueeze(0),
        "relation_type": torch.from_numpy(graph.relation_type[:-1, :-1]).unsqueeze(0),
        "relation_bias": torch.from_numpy(graph.relation_bias).unsqueeze(0),
    }

    with pytest.raises(ValueError, match="relation_type"):
        model(**tensors)

    tensors["relation_type"] = torch.from_numpy(graph.relation_type).unsqueeze(0)
    tensors["relation_bias"] = torch.zeros(
        1,
        2,
        graph.relation_type.shape[0],
        graph.relation_type.shape[1],
    )
    with pytest.raises(ValueError, match="head dimension"):
        model(**tensors)


@pytest.mark.parametrize("architecture", sorted(GlobalHexGraphNet.ARCHITECTURES))
def test_global_graph_alternatives_share_targets_and_masks(architecture):
    graph = build_graph_batch_from_history(_hist((0, 0, 0)), radius=1)
    cfg = Config.model_validate(
        {
            "model": {
                "architecture": architecture,
                "channels": 16,
                "attention_heads": 4,
                "graph_layers": 1,
            },
            "inference": {"fp16": False},
        }
    )
    model = build_model_from_config(cfg, device=torch.device("cpu"))
    tensors = {
        "token_features": torch.from_numpy(graph.token_features).unsqueeze(0),
        "token_type": torch.from_numpy(graph.token_type).unsqueeze(0),
        "token_qr": torch.from_numpy(graph.token_qr).unsqueeze(0),
        "token_mask": torch.from_numpy(graph.token_mask).unsqueeze(0),
        "legal_token_indices": torch.from_numpy(graph.legal_token_indices).unsqueeze(0),
        "legal_mask": torch.from_numpy(graph.legal_mask).unsqueeze(0),
        "pair_first_indices": torch.from_numpy(graph.pair_first_indices).unsqueeze(0),
        "pair_second_indices": torch.from_numpy(graph.pair_second_indices).unsqueeze(0),
        "pair_token_indices": torch.from_numpy(graph.pair_token_indices).unsqueeze(0),
        "relation_type": torch.from_numpy(graph.relation_type).unsqueeze(0),
        "relation_bias": torch.from_numpy(graph.relation_bias).unsqueeze(0),
    }

    out = model(**tensors)

    assert "policy" not in out
    assert out["policy_place"].shape == (1, graph.legal_mask.shape[0])
    assert out["policy_pair_first"].shape == (1, graph.legal_mask.shape[0])
    assert torch.isfinite(out["policy_place"][0, graph.legal_mask]).all()


def test_global_graph_pair_heads_are_distinct_first_second_and_joint_contracts():
    graph = build_graph_batch_from_history(
        _hist((0, 0, 0)),
        max_pair_rows=8,
        allow_pair_truncation=True,
    )
    model = GlobalHexGraphNet(channels=16, heads=4, layers=1, architecture="global_pair_twostage_0")

    out = model(
        token_features=torch.from_numpy(graph.token_features).unsqueeze(0),
        token_type=torch.from_numpy(graph.token_type).unsqueeze(0),
        token_qr=torch.from_numpy(graph.token_qr).unsqueeze(0),
        token_mask=torch.from_numpy(graph.token_mask).unsqueeze(0),
        legal_token_indices=torch.from_numpy(graph.legal_token_indices).unsqueeze(0),
        legal_mask=torch.from_numpy(graph.legal_mask).unsqueeze(0),
        pair_first_indices=torch.from_numpy(graph.pair_first_indices).unsqueeze(0),
        pair_second_indices=torch.from_numpy(graph.pair_second_indices).unsqueeze(0),
        pair_token_indices=torch.from_numpy(graph.pair_token_indices).unsqueeze(0),
    )

    assert model.pair_second is not model.pair_joint
    assert out["policy_pair_first"].shape == (1, graph.legal_qr.shape[0])
    assert out["policy_pair_second"].shape == out["policy_pair_joint"].shape
    assert not torch.allclose(out["policy_pair_second"], out["policy_pair_joint"])


def test_global_graph_alternatives_have_distinct_model_families():
    families = {
        arch: GlobalHexGraphNet(channels=16, heads=4, layers=1, architecture=arch).architecture_family
        for arch in GlobalHexGraphNet.ARCHITECTURES
    }

    assert families["global_xattn_0"] == "context_cross_attention"
    assert families["global_line_window_0"] == "line_window_cover"
    assert families["global_pair_twostage_0"] == "pair_two_stage"
    assert families["global_graph_full_0"] == "full_relation_graph"
    assert len(set(families.values())) == len(families)


def _policy_by_qr(graph):
    return {
        tuple(qr): float(prob)
        for qr, prob in zip(graph.legal_qr.tolist(), graph.policy_target.tolist())
        if float(prob) > 0.0
    }


def _pair_policy_by_cells(graph):
    token_to_legal = {int(tok): i for i, tok in enumerate(graph.legal_token_indices.tolist())}
    out = {}
    for first_tok, second_tok, prob in zip(
        graph.pair_first_indices.tolist(),
        graph.pair_second_indices.tolist(),
        graph.pair_policy_target.tolist(),
    ):
        if float(prob) <= 0.0:
            continue
        if int(first_tok) in token_to_legal:
            first = tuple(graph.legal_qr[token_to_legal[int(first_tok)]].tolist())
            unordered = True
        else:
            first = tuple(graph.token_qr[int(first_tok)].tolist())
            unordered = False
        second = tuple(graph.legal_qr[token_to_legal[int(second_tok)]].tolist())
        key = frozenset({first, second}) if unordered else (first, second)
        out[key] = float(prob)
    return out


def _transform_pair_key(key, sym):
    if isinstance(key, frozenset):
        return frozenset({transform_qr(qr, sym) for qr in key})
    return (transform_qr(key[0], sym), transform_qr(key[1], sym))


def test_d6_graph_token_relation_pair_equivariance():
    history = _hist((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, -1, 0))
    policy = [(2, 0, 0.6), (-2, 0, 0.4)]
    pair_policy = [((-1, 0), (2, 0), 1.0)]
    opp_legal = [(2, 0), (-2, 0), (1, -2)]
    opp_policy = [(1, -2, 1.0)]
    base = build_graph_batch_from_history(
        history,
        radius=2,
        policy_target=policy,
        pair_policy_target=pair_policy,
        opp_legal_moves=opp_legal,
        opp_policy_target=opp_policy,
    )
    base_legal_set = {tuple(qr) for qr in base.legal_qr.tolist()}
    base_type_counts = np.bincount(base.token_type, minlength=max(int(t) for t in GraphTokenType) + 1)

    for sym in range(12):
        graph = build_graph_batch_from_history(
            transform_history(history, sym),
            radius=2,
            policy_target=transform_policy_target(policy, sym),
            pair_policy_target=transform_pair_policy_target(pair_policy, sym),
            opp_legal_moves=[transform_qr(qr, sym) for qr in opp_legal],
            opp_policy_target=transform_policy_target(opp_policy, sym),
        )
        expected_legal = {transform_qr(qr, sym) for qr in base_legal_set}
        assert {tuple(qr) for qr in graph.legal_qr.tolist()} == expected_legal
        assert np.array_equal(
            np.bincount(graph.token_type, minlength=base_type_counts.shape[0]),
            base_type_counts,
        )
        assert _policy_by_qr(graph) == pytest.approx(
            {transform_qr(qr, sym): prob for qr, prob in _policy_by_qr(base).items()}
        )
        assert _pair_policy_by_cells(graph) == pytest.approx(
            {
                _transform_pair_key(pair, sym): prob
                for pair, prob in _pair_policy_by_cells(base).items()
            }
        )
        assert graph.opp_policy_target.sum() == pytest.approx(1.0)

        base_legal_tokens = {
            tuple(base.legal_qr[row].tolist()): int(tok)
            for row, tok in enumerate(base.legal_token_indices.tolist())
        }
        graph_legal_tokens = {
            tuple(graph.legal_qr[row].tolist()): int(tok)
            for row, tok in enumerate(graph.legal_token_indices.tolist())
        }
        sample = sorted(base_legal_set)[:5]
        for a in sample:
            for b in sample:
                base_i = base_legal_tokens[a]
                base_j = base_legal_tokens[b]
                graph_i = graph_legal_tokens[transform_qr(a, sym)]
                graph_j = graph_legal_tokens[transform_qr(b, sym)]
                assert graph.relation_bias[0, graph_i, graph_j] == pytest.approx(
                    base.relation_bias[0, base_i, base_j]
                )
