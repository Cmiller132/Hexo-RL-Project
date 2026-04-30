import numpy as np
import pytest

from hexorl.contracts.candidates import CandidateContractBuilder
from hexorl.contracts.pairs import PairActionTableBuilder, PairStrategy
from hexorl.contracts.validation import ContractValidationError
from hexorl.search.context import SearchContext
from hexorl.search.pair_strategy import PairStrategySpec, create_pair_strategy
from hexorl.search.priors import PRIOR_SOURCE_PAIR, SearchEvaluation


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
        model_spec_version="1",
        inference_protocol="test",
    )


def _pair_context(legal_table, *, strategy):
    cand = CandidateContractBuilder().build(
        [(0, 0), (1, 0), (0, 1)],
        [],
        offset_q=0,
        offset_r=0,
        budget=3,
        storage_width=3,
        source="fixture",
        allow_fixture=True,
    )
    pair_table = PairActionTableBuilder().build(
        cand,
        [((0, 0), (1, 0), 1.0)],
        strategy=strategy,
        legal_moves=[(0, 0), (1, 0), (0, 1)],
        source="fixture",
        allow_fixture=True,
    )
    return SearchContext.create(
        phase="root",
        legal_table=legal_table,
        model_family="dense_cnn",
        candidate_table=cand,
        pair_table=pair_table,
        tensor=np.zeros((1, 13, 33, 33), dtype=np.float32),
    )


class FakePairScorer:
    name = "fake_pair_head"

    def __init__(self):
        self.calls = []

    def score_pairs(self, context, table, active_rows):
        self.calls.append((context, table, np.asarray(active_rows, dtype=np.int64).copy()))
        return np.asarray([4.0, 1.0, -2.0], dtype=np.float32)[: len(active_rows)]


def test_pair_strategy_none_generates_zero_pair_rows(legal_table):
    ctx = _pair_context(legal_table, strategy=PairStrategy(mode="none", max_pairs=0))
    ev = create_pair_strategy(PairStrategySpec()).score_root(ctx, _base_eval(ctx))
    assert ev.selected_pair_rows == 0
    assert ev.scored_pair_rows == 0


def test_pair_strategy_none_scores_zero_pairs(legal_table):
    ctx = _pair_context(legal_table, strategy=PairStrategy(mode="none", max_pairs=0))
    strategy = create_pair_strategy(PairStrategySpec())
    assert strategy.score_root(ctx, _base_eval(ctx)).scored_pair_rows == 0
    assert strategy.score_leaves([ctx], [_base_eval(ctx)])[0].scored_pair_rows == 0


def test_global_xattn_default_pair_strategy_none_zero_rows(legal_table):
    ctx = SearchContext.create(phase="root", legal_table=legal_table, model_family="global_xattn")
    ev = create_pair_strategy(PairStrategySpec()).score_root(ctx, _base_eval(ctx))
    assert ev.strategy_name == "none"
    assert ev.scored_pair_rows == 0


def test_global_graph_default_pair_strategy_none_zero_rows(legal_table):
    ctx = SearchContext.create(phase="root", legal_table=legal_table, model_family="global_relation_graph")
    ev = create_pair_strategy(PairStrategySpec()).score_root(ctx, _base_eval(ctx))
    assert ev.scored_pair_rows == 0


def test_pair_head_presence_does_not_enable_pair_scoring(legal_table):
    ctx = SearchContext.create(phase="root", legal_table=legal_table, model_family="global_xattn", extra={"heads": ["policy_pair_joint"]})
    ev = create_pair_strategy(PairStrategySpec()).score_root(ctx, _base_eval(ctx))
    assert ev.scored_pair_rows == 0


def test_pair_prior_mix_does_not_enable_pair_scoring(legal_table):
    ctx = SearchContext.create(phase="root", legal_table=legal_table, model_family="dense_cnn", extra={"pair_prior_mix": 1.0})
    ev = create_pair_strategy(PairStrategySpec()).score_root(ctx, _base_eval(ctx))
    assert ev.scored_pair_rows == 0


def test_architecture_prefix_does_not_enable_pair_scoring(legal_table):
    ctx = SearchContext.create(phase="root", legal_table=legal_table, model_family="global_xattn", extra={"architecture": "global_xattn_0"})
    ev = create_pair_strategy(PairStrategySpec()).score_root(ctx, _base_eval(ctx))
    assert ev.scored_pair_rows == 0


def test_leaf_pair_scoring_requires_explicit_enable_and_cap():
    with pytest.raises(ContractValidationError):
        PairStrategySpec(name="tactical_only", root_enabled=False, leaf_enabled=True, max_leaf_pair_rows=0)


def test_full_pair_strategy_requires_diagnostic_root_only_and_cap():
    with pytest.raises(ContractValidationError):
        PairStrategySpec(name="diagnostic_full_root", diagnostic=False, root_enabled=True, max_full_pair_rows=10)


def test_capped_pair_strategy_enforces_root_cap(legal_table):
    ctx = _pair_context(legal_table, strategy=PairStrategy(mode="capped_fill", max_pairs=3))
    spec = PairStrategySpec(name="two_stage_root_only", root_enabled=True, max_root_pair_rows=1)
    ev = create_pair_strategy(spec, pair_scorer=FakePairScorer()).score_root(ctx, _base_eval(ctx))
    assert ev.scored_pair_rows <= 1


def test_capped_pair_strategy_enforces_leaf_cap(legal_table):
    ctx = _pair_context(legal_table, strategy=PairStrategy(mode="capped_fill", max_pairs=3))
    spec = PairStrategySpec(name="tactical_only", root_enabled=False, leaf_enabled=True, max_leaf_pair_rows=1)
    ev = create_pair_strategy(spec, pair_scorer=FakePairScorer()).score_leaves([ctx], [_base_eval(ctx)])[0]
    assert ev.scored_pair_rows <= 1


def test_diagnostic_full_root_strategy_never_scores_leaves(legal_table):
    ctx = _pair_context(legal_table, strategy=PairStrategy(mode="full_capped", max_pairs=3, allow_full=True))
    spec = PairStrategySpec(name="diagnostic_full_root", diagnostic=True, root_enabled=True, leaf_enabled=False, max_full_pair_rows=3)
    ev = create_pair_strategy(spec).score_leaves([ctx], [_base_eval(ctx)])[0]
    assert ev.scored_pair_rows == 0


def test_explicit_pair_strategy_consumes_pair_scorer_not_pair_targets(legal_table):
    ctx = _pair_context(legal_table, strategy=PairStrategy(mode="capped_fill", max_pairs=3))
    scorer = FakePairScorer()
    spec = PairStrategySpec(name="two_stage_root_only", root_enabled=True, max_root_pair_rows=2)
    ev = create_pair_strategy(spec, pair_scorer=scorer).score_root(ctx, _base_eval(ctx))

    assert len(scorer.calls) == 1
    assert ev.scored_pair_rows == 2
    assert np.all(ev.pair_prior_source == PRIOR_SOURCE_PAIR)
    assert "fake_pair_head" in ev.influence


def test_explicit_pair_strategy_fails_without_pair_scorer(legal_table):
    ctx = _pair_context(legal_table, strategy=PairStrategy(mode="capped_fill", max_pairs=3))
    spec = PairStrategySpec(name="two_stage_root_only", root_enabled=True, max_root_pair_rows=1)
    with pytest.raises(ContractValidationError, match="pair-scoring provider"):
        create_pair_strategy(spec).score_root(ctx, _base_eval(ctx))
