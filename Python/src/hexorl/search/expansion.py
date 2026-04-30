"""Search expansion helpers."""

from __future__ import annotations

from hexorl.search.engine_adapter import EngineAdapter
from hexorl.search.pair_strategy import PairEvaluation
from hexorl.search.priors import SearchEvaluation


def expand_root_with_evaluation(engine: EngineAdapter, evaluation: SearchEvaluation, pair_eval: PairEvaluation | None = None) -> None:
    if evaluation.policy_provider == "GlobalGraphPolicyProvider":
        engine.expand_root_with_global_priors(evaluation)
    elif evaluation.policy_provider == "GraphHybridPolicyProvider":
        engine.expand_root_with_sparse_priors(evaluation)
    else:
        engine.expand_root(evaluation)
    if pair_eval is not None and pair_eval.scored_pair_rows > 0:
        if pair_eval.known_first is not None:
            engine.apply_root_pair_second_priors(pair_eval)
        else:
            engine.apply_root_pair_priors(pair_eval)
