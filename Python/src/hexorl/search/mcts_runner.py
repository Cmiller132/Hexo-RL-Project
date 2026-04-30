"""MCTS runner trace utilities."""

from __future__ import annotations

from dataclasses import dataclass

from hexorl.search.engine_adapter import EngineAdapter
from hexorl.search.pair_strategy import PairEvaluation
from hexorl.search.priors import SearchEvaluation


@dataclass(frozen=True)
class SearchAction:
    q: int
    r: int


@dataclass(frozen=True)
class LeafBatch:
    count: int
    batch_generation: int
    trace_id: str


def start_root(engine: EngineAdapter):
    return engine.init_root()


def choose_leaf_batch(engine: EngineAdapter, batch_size: int):
    return engine.select_leaves(batch_size)


def commit_root(engine: EngineAdapter, evaluation: SearchEvaluation, pair_eval: PairEvaluation) -> None:
    if evaluation.policy_provider == "GlobalGraphPolicyProvider":
        engine.expand_root_with_global_priors(evaluation)
    elif evaluation.policy_provider == "GraphHybridPolicyProvider":
        engine.expand_root_with_sparse_priors(evaluation)
    else:
        engine.expand_root(evaluation)
    if pair_eval.scored_pair_rows > 0:
        if pair_eval.known_first is not None:
            engine.apply_root_pair_second_priors(pair_eval)
        else:
            engine.apply_root_pair_priors(pair_eval)


def commit_leaf_batch(engine: EngineAdapter, evaluations: list[SearchEvaluation], *, source_mode: str) -> None:
    if source_mode == "global":
        engine.expand_and_backprop_sparse_sources(evaluations)
    elif source_mode == "sparse":
        engine.expand_and_backprop_with_sparse(evaluations)
    else:
        engine.expand_and_backprop(evaluations)
