import numpy as np
import pytest

from hexorl.contracts.candidates import CandidateContractBuilder
from hexorl.contracts.pairs import PairActionTable, PairActionTableBuilder, PairStrategy
from hexorl.contracts.validation import ContractValidationError
from hexorl.graph.batch import build_graph_batch_from_history
from hexorl.graph.tensorize import graph_batch_with_pair_table
from hexorl.models.specs import ModelSpec
from hexorl.search.context import SearchContext
from hexorl.search.engine_adapter import create_engine_adapter
from hexorl.search.pair_strategy import PairEvaluation, PairStrategySpec, create_pair_strategy
from hexorl.search.policy_provider import GlobalGraphPolicyProvider
from hexorl.search.priors import SearchEvaluation


def _candidate_table():
    return CandidateContractBuilder().build(
        [(0, 0), (1, 0), (0, 1)],
        [],
        offset_q=0,
        offset_r=0,
        budget=3,
        storage_width=3,
        source="fixture",
        allow_fixture=True,
    )


def _pair_table(*, known_first=None, mode="capped_fill", max_pairs=3):
    return PairActionTableBuilder().build(
        _candidate_table(),
        [((0, 0), (1, 0), 1.0)],
        strategy=PairStrategy(mode=mode, max_pairs=max_pairs, allow_full=mode == "full_capped"),
        legal_moves=[(0, 0), (1, 0), (0, 1)],
        known_first=known_first,
        source="fixture",
        allow_fixture=True,
    )


def _base_eval(ctx):
    return SearchEvaluation(
        context=ctx,
        value=0.0,
        legal_row_ids=np.arange(ctx.legal_table.rows.shape[0]),
        legal_dense_indices=ctx.legal_table.dense_indices,
        row_priors=np.ones(ctx.legal_table.rows.shape[0], dtype=np.float32),
        prior_source=np.ones(ctx.legal_table.rows.shape[0], dtype=np.uint8),
        policy_provider="test",
        model_family=ctx.model_family,
        model_spec_version="v2",
        inference_protocol="test",
    )


def test_policy_place_one_logit_per_legal_row(legal_table, fake_client):
    graph = build_graph_batch_from_history(b"", include_pair_rows=False)
    graph = graph.__class__(**{**graph.__dict__, "legal_qr": legal_table.rows, "legal_mask": np.ones(3, dtype=np.bool_)})
    ctx = SearchContext.create(phase="root", legal_table=legal_table, model_family="global_xattn", graph_batch=graph)
    ev = GlobalGraphPolicyProvider(client=fake_client, model_spec=ModelSpec(kind="global_xattn", source_name="fixture")).evaluate_root(ctx)
    assert ev.row_priors.shape == (legal_table.rows.shape[0],)
    assert ev.raw_metadata["policy_pair_first_rows"] == legal_table.rows.shape[0]


def test_policy_pair_first_one_logit_per_legal_first_row(legal_table, fake_client):
    graph = build_graph_batch_from_history(b"", include_pair_rows=False)
    graph = graph.__class__(**{**graph.__dict__, "legal_qr": legal_table.rows, "legal_mask": np.ones(3, dtype=np.bool_)})
    response = fake_client.evaluate_global_graph(graph)
    assert response["policy_pair_first"].shape == (legal_table.rows.shape[0],)


def test_policy_pair_second_requires_known_first():
    with pytest.raises(ContractValidationError, match="known_first"):
        PairActionTable(
            rows=np.zeros((1, 4), dtype=np.int32),
            first_candidate_rows=np.zeros(1, dtype=np.int64),
            second_candidate_rows=np.ones(1, dtype=np.int64),
            mask=np.ones(1, dtype=np.bool_),
            target=np.ones(1, dtype=np.float32),
            phase="second_placement_known_first",
            source="fixture",
            allow_fixture=True,
        )


def test_policy_pair_second_uses_post_first_legal_table():
    table = _pair_table(known_first=(0, 0))
    assert table.phase == "second_placement_known_first"
    assert table.known_first == (0, 0)
    assert all(tuple(row[:2]) == (0, 0) for row in table.rows[table.mask].tolist())


def test_policy_pair_joint_one_logit_per_pair_action_row():
    table = _pair_table(mode="full_capped", max_pairs=3)
    base = build_graph_batch_from_history(b"", include_pair_rows=False)
    base = base.__class__(**{**base.__dict__, "legal_token_indices": np.arange(3, dtype=np.int64)})
    graph = graph_batch_with_pair_table(base, table)
    assert graph.pair_policy_target.shape == (table.selected_pair_count,)
    assert graph.pair_first_indices.shape == graph.pair_second_indices.shape == (table.selected_pair_count,)


def test_pair_action_rows_from_canonical_pair_action_table():
    table = _pair_table(mode="full_capped", max_pairs=3)
    active = table.rows[table.mask]
    assert active.shape[1] == 4
    assert all(tuple(row[:2]) <= tuple(row[2:]) for row in active.tolist())


def test_opening_position_has_no_pair_prior_or_pair_loss(legal_table):
    ctx = SearchContext.create(phase="root", legal_table=legal_table, model_family="global_xattn")
    pair_eval = create_pair_strategy(PairStrategySpec()).score_root(ctx, _base_eval(ctx))
    assert pair_eval.scored_pair_rows == 0
    assert pair_eval.pair_priors.shape == (0,)


def test_pair_prior_telemetry_reports_pair_head_influence():
    engine = create_engine_adapter(num_simulations=1, c_puct=1.5, near_radius=2, seed=1, force_mock=True)
    pair = PairEvaluation(
        strategy_name="two_stage_root_only",
        phase="root",
        root_scope=True,
        pair_table_identity="fixture-pair-table",
        pair_rows=np.asarray([[(0, 0, 1, 0)]], dtype=np.int32),
        pair_priors=np.asarray([1.0], dtype=np.float32),
        pair_prior_source=np.asarray([1], dtype=np.uint8),
        known_first=None,
        total_possible_pairs=1,
        selected_pair_rows=1,
        scored_pair_rows=1,
        influence="pair_joint",
    )
    engine.apply_root_pair_priors(pair)
    assert engine.prior_source_summary()["pair_influence"] == "pair_joint"
