import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hexorl.config import Config
from hexorl.eval import model_provider
from hexorl.graph.batch import (
    build_graph_batch_from_history,
    collate_graph_batches,
    sparse_relation_edge_count,
)
from hexorl.inference.adapters import (
    decode_dense_outputs,
    decode_global_graph_outputs,
    decode_graph_slot_response,
)
from hexorl.inference.protocol import (
    GRAPH_HEAD_PAIR_FIRST,
    GRAPH_HEAD_PAIR_JOINT,
    GRAPH_HEAD_PAIR_SECOND,
)
from hexorl.models.assembly import bins_to_value
from hexorl.contracts import ValueDecoderContract
from hexorl.models.loading import build_runtime_model
from hexorl.models.registry import resolve_model_spec
from hexorl.search.engine_adapter import EngineAdapter
from hexorl.search.pair_strategy import build_legacy_pair_baseline_strategy, build_pair_strategy
from hexorl.search.legacy_pair_projection import pair_logits_to_action_logits
from hexorl.selfplay.worker import _expand_crop_root_with_optional_sparse


def _hist(*moves):
    data = bytearray()
    for player, q, r in moves:
        data.extend(struct.pack("<iii", player, q, r))
    return bytes(data)


def _graph_count_tuple(graph, legal_count: int | None = None, opp_count: int = 0, pair_count: int | None = None):
    return (
        int(graph.token_features.shape[0]),
        int(graph.legal_qr.shape[0] if legal_count is None else legal_count),
        int(opp_count),
        int(graph.pair_first_indices.shape[0] if pair_count is None else pair_count),
        int(sparse_relation_edge_count(graph)),
    )


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


def test_dense_inference_adapter_supports_scalar_value_contract():
    decoded = decode_dense_outputs(
        {
            "policy": torch.zeros(2, 4),
            "value": torch.tensor([[2.0], [-2.0]]),
        },
        value_decoder=lambda _x: (_ for _ in ()).throw(AssertionError("binned decoder should not run")),
        sparse_requested=False,
        value_contract=ValueDecoderContract(
            name="scalar_tanh",
            logits_key="value",
            n_bins=1,
            output_range=(-1.0, 1.0),
            perspective="current_player",
        ),
    )

    assert decoded.value.tolist() == pytest.approx([1.0, -1.0])
    meta = decoded.metadata["outputs"]["value"]["value_decoder"]
    assert meta["name"] == "scalar_tanh"
    assert meta["n_bins"] == 1


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


def test_runtime_global_graph_honors_explicit_output_heads_without_lookahead():
    cfg = Config.model_validate(
        {
            "model": {
                "architecture": "global_xattn_0",
                "channels": 16,
                "attention_heads": 4,
                "graph_layers": 1,
                "heads": ["policy_place", "value"],
            },
            "inference": {"fp16": False},
        }
    )
    resolved = resolve_model_spec(cfg)
    assert resolved.outputs == ("policy_place", "value")

    model = build_runtime_model(cfg, device=torch.device("cpu"), inference=True)
    graph = collate_graph_batches([
        build_graph_batch_from_history(
            _hist((0, 0, 0)),
            include_pair_rows=True,
            max_pair_rows=4,
            allow_pair_truncation=True,
        )
    ])
    inputs = {
        name: torch.from_numpy(getattr(graph, name))
        for name in (
            "token_features",
            "token_type",
            "token_qr",
            "token_mask",
            "legal_token_indices",
            "legal_mask",
            "opp_legal_qr",
            "opp_legal_mask",
            "pair_token_indices",
            "pair_first_indices",
            "pair_second_indices",
            "relation_type",
            "relation_bias",
        )
    }

    with torch.inference_mode():
        out = model(**inputs)

    assert set(out) == {"policy_place", "value"}


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
            [_graph_count_tuple(graph, legal_count=legal.shape[0], pair_count=0)],
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
        [_graph_count_tuple(graph, legal_count=legal.shape[0], pair_count=pair_count)],
        [(0, 0, 0, 0)],
        head_flags=GRAPH_HEAD_PAIR_FIRST | GRAPH_HEAD_PAIR_JOINT,
    )[0]

    assert result["policy_pair_first"].shape == (legal.shape[0],)
    assert result["policy_pair_joint"].shape == (pair_count,)
    row_tables = result["metadata"]["row_tables"]
    outputs = result["metadata"]["outputs"]
    assert row_tables["policy_place"]["identity_hash"].startswith("sha256:")
    assert row_tables["policy_pair_first"]["family"] == "legal"
    assert row_tables["policy_pair_joint"]["family"] == "pair_joint"
    assert outputs["policy_pair_first"]["row_table"]["identity_hash"] == row_tables["policy_pair_first"]["identity_hash"]
    assert outputs["policy_pair_joint"]["row_table"]["identity_hash"] == row_tables["policy_pair_joint"]["identity_hash"]


def test_graph_response_metadata_declares_known_first_pair_contract():
    graph = build_graph_batch_from_history(
        _hist((0, 0, 0), (1, 1, 0)),
        max_pair_rows=3,
        allow_pair_truncation=True,
    )
    legal = np.asarray(graph.legal_qr, dtype=np.int32)
    mask = np.asarray(graph.legal_mask, dtype=np.uint8)
    pair_count = int(graph.pair_first_indices.shape[0])
    slot = SimpleNamespace(
        res_graph_meta=np.array([graph.schema_version, graph.relation_schema_version, legal.shape[0], 0, pair_count, graph.token_features.shape[0], 0, GRAPH_HEAD_PAIR_SECOND], dtype=np.int64),
        req_graph_legal_qr=legal.copy(),
        req_graph_legal_mask=mask.copy(),
        res_graph_place_logits=np.zeros(legal.shape[0], dtype=np.float32),
        res_value=np.zeros(1, dtype=np.float32),
        res_graph_pair_first_logits=np.zeros(legal.shape[0], dtype=np.float32),
        res_graph_pair_logits=np.zeros(pair_count, dtype=np.float32),
        res_graph_pair_second_logits=np.ones(pair_count, dtype=np.float32),
        res_graph_opp_logits=np.zeros(0, dtype=np.float32),
    )

    result = decode_graph_slot_response(
        slot,
        [graph],
        [_graph_count_tuple(graph, legal_count=legal.shape[0], pair_count=pair_count)],
        [(0, 0, 0, 0)],
        head_flags=GRAPH_HEAD_PAIR_SECOND,
    )[0]

    assert result["policy_pair_second"].shape == (pair_count,)
    row_table = result["metadata"]["row_tables"]["policy_pair_second"]
    assert row_table["family"] == "known_first_pair"
    assert row_table["phase"] == "second_placement_known_first"


def test_pair_strategy_is_required_for_pair_behavior():
    none = build_pair_strategy("none", max_pairs=0, prior_mix=0.0)
    assert not none.enabled
    with pytest.raises(ValueError, match="pair behavior requires an explicit pair strategy"):
        none.require_enabled(context="unit")

    with pytest.raises(ValueError, match="quarantined"):
        build_pair_strategy("root_pair_mcts", max_pairs=16, prior_mix=0.25)

    root = build_legacy_pair_baseline_strategy("root_pair_mcts", max_pairs=16, prior_mix=0.25)
    assert root.enabled
    assert root.pair_rows_owned
    assert not root.leaf_pair_scoring_enabled
    assert "policy_pair_joint" in root.required_output_contracts
    with pytest.raises(ValueError, match="known first"):
        root.require_pair_phase(second_placement=True, known_first=False, context="unit")

    full = build_legacy_pair_baseline_strategy("full_pair_mcts", max_pairs=16, prior_mix=0.25)
    assert full.leaf_pair_scoring_enabled


def test_pair_strategy_rejects_non_finite_pair_logits():
    strategy = build_legacy_pair_baseline_strategy("root_pair_mcts", max_pairs=16, prior_mix=0.25)
    pair_qr = np.asarray([[0, 0, 1, 0]], dtype=np.int32)
    legal = np.asarray([[0, 0], [1, 0]], dtype=np.int32)

    with pytest.raises(ValueError, match="finite"):
        pair_logits_to_action_logits(
            pair_qr,
            np.asarray([np.nan], dtype=np.float32),
            legal,
        )


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


def test_engine_adapter_validates_runtime_contracts_before_mcts():
    adapter = EngineAdapter()

    assert adapter.validate_search_phase(1, root=True, context="unit") == 1
    with pytest.raises(ValueError, match="placements_remaining"):
        adapter.validate_search_phase(0, root=False, context="unit")

    assert adapter.validate_batch_generation(3, 3, context="unit") == 3
    with pytest.raises(ValueError, match="stale MCTS batch generation"):
        adapter.validate_batch_generation(2, 3, context="unit")

    legal = np.asarray([[3, 4], [5, 6]], dtype=np.int32)
    assert np.array_equal(
        adapter.validate_legal_bytes_alignment(legal, legal.tobytes(), context="unit"),
        legal,
    )
    with pytest.raises(ValueError, match="Rust legal rows changed"):
        adapter.validate_legal_bytes_alignment(legal[::-1], legal.tobytes(), context="unit")

    assert adapter.validate_dense_offset_mapping(legal, 0, 0, board_size=33, context="unit").tolist() == [103, 171]
    with pytest.raises(ValueError, match="dense policy offset"):
        adapter.validate_dense_offset_mapping(legal, 4, 0, board_size=3, context="unit")

    adapter.validate_value_perspective(
        {"outputs": {"value": {"value_decoder": {"perspective": "current_player"}}}},
        context="unit",
    )
    with pytest.raises(ValueError, match="value perspective"):
        adapter.validate_value_perspective(
            {"outputs": {"value": {"value_decoder": {"perspective": "absolute_player0"}}}},
            context="unit",
        )

    rows = np.asarray([[1, 0, 2, 0]], dtype=np.int32)
    assert np.array_equal(
        adapter.validate_pair_phase(rows, placements_remaining=1, first_qr=(1, 0), context="unit"),
        rows,
    )
    with pytest.raises(ValueError, match="known-first pair rows"):
        adapter.validate_pair_phase(rows, placements_remaining=1, first_qr=(0, 1), context="unit")
    with pytest.raises(ValueError, match="duplicate"):
        adapter.validate_pair_phase(np.asarray([[1, 0, 1, 0]], dtype=np.int32), placements_remaining=2, context="unit")


def test_sparse_root_expansion_allows_legal_rows_outside_dense_crop():
    class FakeEngine:
        def __init__(self):
            self.dense_calls = 0
            self.sparse_calls = 0
            self.sparse_qr = None
            self.sparse_logits = None

        def expand_root(self, *_args):
            self.dense_calls += 1

        def expand_root_with_sparse_priors(
            self,
            _policy,
            _value,
            _offset_q,
            _offset_r,
            _legal_bytes,
            _root_generation,
            sparse_qr,
            sparse_logits,
            _stage,
            _sparse_mix,
        ):
            self.sparse_calls += 1
            self.sparse_qr = np.asarray(sparse_qr)
            self.sparse_logits = np.asarray(sparse_logits)

    legal = np.asarray([[50, 50], [0, 0]], dtype=np.int32)
    engine = FakeEngine()

    _expand_crop_root_with_optional_sparse(
        engine,
        EngineAdapter(),
        np.zeros(33 * 33, dtype=np.float32),
        0.25,
        -16,
        -16,
        legal.tobytes(),
        1,
        legal,
        np.asarray([[50, 50]], dtype=np.int32),
        np.asarray([2.0], dtype=np.float32),
        sparse_root_active=True,
        sparse_prior_stage=1,
        sparse_prior_mix=0.6,
    )

    assert engine.sparse_calls == 1
    assert engine.dense_calls == 0
    assert engine.sparse_qr.tolist() == [[50, 50]]
    assert engine.sparse_logits.tolist() == pytest.approx([2.0])


def test_dense_root_expansion_still_rejects_legal_rows_outside_dense_crop():
    class FakeEngine:
        def expand_root(self, *_args):
            raise AssertionError("dense expansion should not run after validation failure")

        def expand_root_with_sparse_priors(self, *_args):
            raise AssertionError("sparse expansion should not run for dense root")

    legal = np.asarray([[50, 50], [0, 0]], dtype=np.int32)

    with pytest.raises(ValueError, match="dense policy offset"):
        _expand_crop_root_with_optional_sparse(
            FakeEngine(),
            EngineAdapter(),
            np.zeros(33 * 33, dtype=np.float32),
            0.0,
            -16,
            -16,
            legal.tobytes(),
            1,
            legal,
            np.zeros((0, 2), dtype=np.int32),
            np.zeros(0, dtype=np.float32),
            sparse_root_active=False,
            sparse_prior_stage=0,
            sparse_prior_mix=0.0,
        )


def test_eval_checkpoint_loading_uses_provider_boundary(monkeypatch, tmp_path):
    cfg = Config()
    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {
            "cfg_json": cfg.model_dump(mode="json"),
            "model_state_dict": {"weight": torch.tensor([1.0])},
        },
        ckpt_path,
    )
    calls: list[tuple[str, object]] = []

    class DummyModel(torch.nn.Module):
        pass

    dummy = DummyModel()

    def fake_build(model_cfg, *, device, inference):
        calls.append(("build", (model_cfg, device, inference)))
        return dummy

    def fake_restore(model, state, *, allow_partial):
        calls.append(("restore", (model, state, allow_partial)))

    monkeypatch.setattr(model_provider, "build_runtime_model", fake_build)
    monkeypatch.setattr(model_provider, "restore_model_weights", fake_restore)

    model = model_provider.load_eval_model(
        ckpt_path,
        cfg,
        device=torch.device("cpu"),
        allow_partial=True,
    )

    assert model is dummy
    assert calls[0][0] == "build"
    assert calls[0][1][2] is True
    assert calls[1][0] == "restore"
    assert calls[1][1][2] is True
