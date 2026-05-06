import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hexorl.graph.batch import build_graph_batch_from_history
from hexorl.inference.adapters import (
    decode_dense_outputs,
    decode_global_graph_outputs,
    decode_graph_slot_response,
)
from hexorl.inference.protocol import GRAPH_HEAD_PAIR_FIRST, GRAPH_HEAD_PAIR_JOINT
from hexorl.models.assembly import bins_to_value
from hexorl.search.engine_adapter import EngineAdapter
from hexorl.search.pair_strategy import build_pair_strategy


def _hist(*moves):
    data = bytearray()
    for player, q, r in moves:
        data.extend(struct.pack("<iii", player, q, r))
    return bytes(data)


def test_dense_inference_adapter_emits_value_decoder_metadata():
    decoded = decode_dense_outputs(
        {
            "policy": torch.zeros(2, 1089),
            "value": torch.zeros(2, 65),
        },
        value_decoder=bins_to_value,
        sparse_requested=False,
    )

    assert decoded.policy.shape == (2, 1089)
    assert decoded.value.shape == (2,)
    metadata = decoded.metadata["outputs"]
    assert metadata["policy"]["row_table"]["family"] == "dense_board"
    assert metadata["value"]["value_decoder"]["name"] == "binned_expected_value_65"


def test_global_graph_adapter_rejects_unmapped_policy_output_shape():
    graph = build_graph_batch_from_history(_hist((0, 0, 0)), include_pair_rows=False)
    inputs = {
        "legal_mask": torch.from_numpy(graph.legal_mask).unsqueeze(0),
        "opp_legal_qr": torch.from_numpy(graph.opp_legal_qr).unsqueeze(0),
        "pair_first_indices": torch.from_numpy(graph.pair_first_indices).unsqueeze(0),
    }

    with pytest.raises(ValueError, match="policy_place output has width"):
        decode_global_graph_outputs(
            {
                "policy_place": torch.zeros(1, max(0, graph.legal_qr.shape[0] - 1)),
                "value": torch.zeros(1, 65),
            },
            inputs,
            value_decoder=bins_to_value,
        )


def test_graph_response_rejects_same_count_reordered_rows():
    graph = build_graph_batch_from_history(_hist((0, 0, 0)), include_pair_rows=False)
    legal = np.asarray(graph.legal_qr, dtype=np.int32)
    mask = np.asarray(graph.legal_mask, dtype=np.uint8)
    slot = SimpleNamespace(
        res_graph_meta=np.array([graph.schema_version, graph.relation_schema_version, legal.shape[0], 0, 0, legal.shape[0], 0, 0], dtype=np.int64),
        req_graph_legal_qr=legal[::-1].copy(),
        req_graph_legal_mask=mask[::-1].copy(),
        res_graph_place_logits=np.zeros(legal.shape[0], dtype=np.float32),
        res_value=np.zeros(1, dtype=np.float32),
        res_graph_pair_first_logits=np.zeros(legal.shape[0], dtype=np.float32),
        res_graph_pair_logits=np.zeros(0, dtype=np.float32),
        res_graph_pair_second_logits=np.zeros(0, dtype=np.float32),
        res_graph_opp_logits=np.zeros(0, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="row-table identity mismatch"):
        decode_graph_slot_response(
            slot,
            [graph],
            [(legal.shape[0], legal.shape[0], 0, 0)],
            [(0, 0, 0, 0)],
            head_flags=0,
        )


def test_graph_response_metadata_declares_pair_contracts_when_present():
    graph = build_graph_batch_from_history(
        _hist((0, 0, 0)),
        max_pair_rows=3,
        allow_pair_truncation=True,
    )
    legal = np.asarray(graph.legal_qr, dtype=np.int32)
    mask = np.asarray(graph.legal_mask, dtype=np.uint8)
    pair_count = int(graph.pair_first_indices.shape[0])
    slot = SimpleNamespace(
        res_graph_meta=np.array([graph.schema_version, graph.relation_schema_version, legal.shape[0], 0, pair_count, graph.token_features.shape[0], 0, GRAPH_HEAD_PAIR_FIRST | GRAPH_HEAD_PAIR_JOINT], dtype=np.int64),
        req_graph_legal_qr=legal.copy(),
        req_graph_legal_mask=mask.copy(),
        res_graph_place_logits=np.zeros(legal.shape[0], dtype=np.float32),
        res_value=np.zeros(1, dtype=np.float32),
        res_graph_pair_first_logits=np.ones(legal.shape[0], dtype=np.float32),
        res_graph_pair_logits=np.ones(pair_count, dtype=np.float32),
        res_graph_pair_second_logits=np.zeros(pair_count, dtype=np.float32),
        res_graph_opp_logits=np.zeros(0, dtype=np.float32),
    )

    result = decode_graph_slot_response(
        slot,
        [graph],
        [(graph.token_features.shape[0], legal.shape[0], 0, pair_count)],
        [(0, 0, 0, 0)],
        head_flags=GRAPH_HEAD_PAIR_FIRST | GRAPH_HEAD_PAIR_JOINT,
    )[0]

    assert result["policy_pair_first"].shape == (legal.shape[0],)
    assert result["policy_pair_joint"].shape == (pair_count,)
    assert result["metadata"]["row_tables"]["policy_place"]["identity_hash"].startswith("sha256:")


def test_pair_strategy_is_required_for_pair_behavior():
    none = build_pair_strategy("none", max_pairs=0, prior_mix=0.0)
    assert not none.enabled
    with pytest.raises(ValueError, match="pair behavior requires an explicit pair strategy"):
        none.require_enabled(context="unit")

    strategy = build_pair_strategy("diagnostic_full_pair", max_pairs=16, prior_mix=0.25)
    assert strategy.enabled
    assert strategy.pair_rows_owned
    assert "policy_pair_joint" in strategy.required_output_contracts
    with pytest.raises(ValueError, match="known first"):
        strategy.require_pair_phase(second_placement=True, known_first=False, context="unit")


def test_engine_adapter_validates_legal_order_and_value_range():
    graph_legal = np.asarray([[0, 0], [1, 0]], dtype=np.int32)
    rust_legal = np.asarray([[1, 0], [0, 0]], dtype=np.int32)
    logits = np.asarray([10.0, 20.0], dtype=np.float32)

    legal, aligned = EngineAdapter.align_global_logits_to_rust_legal(
        graph_legal,
        rust_legal,
        logits,
        context="unit",
    )

    assert np.array_equal(legal, rust_legal)
    assert aligned.tolist() == pytest.approx([20.0, 10.0])
    with pytest.raises(ValueError, match="outside"):
        EngineAdapter().validate_value(1.5, context="unit")

