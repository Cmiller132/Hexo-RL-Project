import numpy as np

from hexorl.contracts.candidates import CandidateContractBuilder
from hexorl.contracts.pairs import PairActionTableBuilder, PairStrategy
from hexorl.search.context import SearchContext
from hexorl.search.pair_strategy import PairStrategySpec, create_pair_strategy
from hexorl.search.priors import PRIOR_SOURCE_DENSE, SearchEvaluation


class FakePairScorer:
    name = "fake_pair_adapter"

    def __init__(self):
        self.calls = 0

    def score_pairs(self, context, table, active_rows):
        self.calls += 1
        return np.linspace(1.0, 2.0, num=len(active_rows), dtype=np.float32)


def _evaluation(legal_table, pair_table=None):
    context = SearchContext.create(
        phase="root",
        legal_table=legal_table,
        model_family="dense_cnn",
        pair_table=pair_table,
        pair_strategy_id="diagnostic_full_root" if pair_table is not None else "none",
    )
    width = int(context.legal_table.rows.shape[0])
    return context, SearchEvaluation(
        context=context,
        value=0.0,
        legal_row_ids=np.arange(width, dtype=np.int64),
        legal_dense_indices=context.legal_table.dense_indices,
        row_priors=np.ones(width, dtype=np.float32),
        prior_source=np.full(width, PRIOR_SOURCE_DENSE, dtype=np.uint8),
        policy_provider="fake",
        model_family="dense_cnn",
        model_spec_version="v2",
        inference_protocol="fake",
    )


def test_default_pair_strategy_scores_zero_rows(legal_table):
    context, evaluation = _evaluation(legal_table)
    pair_eval = create_pair_strategy(PairStrategySpec()).score_root(context, evaluation)

    assert pair_eval.strategy_name == "none"
    assert pair_eval.scored_pair_rows == 0


def test_pair_scoring_occurs_only_through_explicit_strategy(legal_table):
    candidate_table = CandidateContractBuilder().build(
        legal_table.rows.tolist(),
        [(0, 0, 1.0)],
        offset_q=-16,
        offset_r=-16,
        budget=2,
        source="fixture",
        allow_fixture=True,
    )
    table = PairActionTableBuilder().build(
        candidate_table,
        [((0, 0), (1, 0), 1.0)],
        strategy=PairStrategy(mode="full_capped", max_pairs=1, allow_full=True),
        legal_moves=legal_table.rows.tolist(),
        source="fixture",
        allow_fixture=True,
    )
    context, evaluation = _evaluation(legal_table, pair_table=table)
    scorer = FakePairScorer()
    pair_eval = create_pair_strategy(
        PairStrategySpec(
            name="diagnostic_full_root",
            diagnostic=True,
            root_enabled=True,
            max_full_pair_rows=1,
        ),
        pair_scorer=scorer,
    ).score_root(context, evaluation)

    assert pair_eval.scored_pair_rows == 1
    assert pair_eval.caps_applied["cap"] == 1
    assert scorer.calls == 1
