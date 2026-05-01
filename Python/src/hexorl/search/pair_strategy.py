"""Explicit pair strategy ownership for search."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np

from hexorl.contracts.pair_strategy import PAIR_STRATEGY_REGISTRY, PairStrategyRegistry, PairStrategySpec
from hexorl.contracts.pairs import PairActionTable
from hexorl.contracts.validation import ContractValidationError
from hexorl.inference.evaluator import Evaluator
from hexorl.models.inference_contracts import OP_PAIR_POLICY
from hexorl.search.context import SearchContext
from hexorl.search.priors import PRIOR_SOURCE_PAIR, SearchEvaluation, priors_from_logits


@dataclass(frozen=True)
class PairEvaluation:
    strategy_name: str
    phase: str
    root_scope: bool
    pair_table_identity: str
    pair_rows: np.ndarray
    pair_priors: np.ndarray
    pair_prior_source: np.ndarray
    known_first: tuple[int, int] | None
    total_possible_pairs: int
    selected_pair_rows: int
    scored_pair_rows: int
    caps_applied: Mapping[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    timings: Mapping[str, float] = field(default_factory=dict)
    influence: str = "none"

    def __post_init__(self) -> None:
        rows = np.array(self.pair_rows, dtype=np.int32, copy=True).reshape(-1, 4)
        priors = np.array(self.pair_priors, dtype=np.float32, copy=True).reshape(-1)
        sources = np.array(self.pair_prior_source, dtype=np.uint8, copy=True).reshape(-1)
        if rows.shape[0] != priors.shape[0] or rows.shape[0] != sources.shape[0]:
            raise ContractValidationError("PairEvaluation rows/priors/source length mismatch", owner="PairEvaluation")
        if not np.isfinite(priors).all() or np.any(priors < -1e-7):
            raise ContractValidationError("PairEvaluation priors must be finite non-negative", owner="PairEvaluation")
        rows.setflags(write=False)
        priors.setflags(write=False)
        sources.setflags(write=False)
        object.__setattr__(self, "pair_rows", rows)
        object.__setattr__(self, "pair_priors", priors)
        object.__setattr__(self, "pair_prior_source", sources)
        object.__setattr__(self, "total_possible_pairs", int(self.total_possible_pairs))
        object.__setattr__(self, "selected_pair_rows", int(self.selected_pair_rows))
        object.__setattr__(self, "scored_pair_rows", int(self.scored_pair_rows))
        object.__setattr__(self, "caps_applied", MappingProxyType(dict(self.caps_applied)))
        object.__setattr__(self, "timings", MappingProxyType(dict(self.timings)))

    @classmethod
    def empty(cls, *, strategy_name: str, context: SearchContext, total_possible_pairs: int = 0) -> "PairEvaluation":
        return cls(
            strategy_name=strategy_name,
            phase=context.phase,
            root_scope=context.phase == "root",
            pair_table_identity="" if context.pair_table is None else context.pair_table.table_hash,
            pair_rows=np.zeros((0, 4), dtype=np.int32),
            pair_priors=np.zeros(0, dtype=np.float32),
            pair_prior_source=np.zeros(0, dtype=np.uint8),
            known_first=None if context.pair_table is None else context.pair_table.known_first,
            total_possible_pairs=total_possible_pairs,
            selected_pair_rows=0,
            scored_pair_rows=0,
            caps_applied={},
            influence="none",
        )


class PairStrategy(Protocol):
    name: str

    def score_root(self, context: SearchContext, base_eval: SearchEvaluation) -> PairEvaluation: ...

    def score_leaves(self, contexts: list[SearchContext], base_evals: list[SearchEvaluation]) -> list[PairEvaluation]: ...


class PairScoringProvider(Protocol):
    name: str

    def score_pairs(self, context: SearchContext, table: PairActionTable, active_rows: np.ndarray) -> np.ndarray: ...


class InferencePairScoringProvider:
    name = "inference_pair_scoring"

    def __init__(self, client: Evaluator):
        self.client = client

    def score_pairs(self, context: SearchContext, table: PairActionTable, active_rows: np.ndarray) -> np.ndarray:
        if context.tensor is None:
            raise ContractValidationError("pair scoring requires tensor in SearchContext", owner=self.name)
        if context.candidate_table is None:
            raise ContractValidationError("pair scoring requires canonical candidate_table in SearchContext", owner=self.name)
        active = np.asarray(active_rows, dtype=np.int64).reshape(-1)
        if active.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        candidate_table = context.candidate_table
        response = self.client.evaluate(
            OP_PAIR_POLICY,
            {
                "tensor": context.tensor.reshape(1, 13, 33, 33),
                "candidate_indices": candidate_table.dense_indices.reshape(1, -1),
                "candidate_features": candidate_table.features.reshape(1, candidate_table.features.shape[0], candidate_table.features.shape[1]),
                "candidate_mask": candidate_table.mask.reshape(1, -1),
                "pair_candidate_indices": table.pair_indices[active].reshape(1, active.shape[0], 2),
                "pair_candidate_mask": table.mask[active].reshape(1, -1),
            },
        )
        pair_logits = response.head_outputs["pair_policy"]
        logits = np.asarray(pair_logits, dtype=np.float32).reshape(1, -1)[0, : active.shape[0]]
        if logits.shape[0] != active.shape[0]:
            raise ContractValidationError("pair-scoring output length does not match selected pair rows", owner=self.name)
        if not np.isfinite(logits).all():
            raise ContractValidationError("pair-scoring output contains non-finite logits", owner=self.name)
        return logits


class NoPairStrategy:
    name = "none"

    def score_root(self, context: SearchContext, base_eval: SearchEvaluation) -> PairEvaluation:
        return PairEvaluation.empty(strategy_name=self.name, context=context, total_possible_pairs=_possible_pairs(context.pair_table))

    def score_leaves(self, contexts: list[SearchContext], base_evals: list[SearchEvaluation]) -> list[PairEvaluation]:
        return [PairEvaluation.empty(strategy_name=self.name, context=ctx, total_possible_pairs=_possible_pairs(ctx.pair_table)) for ctx in contexts]


class ExplicitPairStrategy:
    def __init__(self, spec: PairStrategySpec, *, pair_scorer: PairScoringProvider | None = None):
        self.spec = spec
        self.name = spec.name
        self.pair_scorer = pair_scorer

    def score_root(self, context: SearchContext, base_eval: SearchEvaluation) -> PairEvaluation:
        if not self.spec.root_enabled:
            return PairEvaluation.empty(strategy_name=self.name, context=context, total_possible_pairs=_possible_pairs(context.pair_table))
        return self._score(context, base_eval, cap=self._root_cap())

    def score_leaves(self, contexts: list[SearchContext], base_evals: list[SearchEvaluation]) -> list[PairEvaluation]:
        if not self.spec.leaf_enabled:
            return [PairEvaluation.empty(strategy_name=self.name, context=ctx, total_possible_pairs=_possible_pairs(ctx.pair_table)) for ctx in contexts]
        return [self._score(ctx, ev, cap=self.spec.max_leaf_pair_rows) for ctx, ev in zip(contexts, base_evals)]

    def _root_cap(self) -> int:
        return self.spec.root_pair_row_cap

    def _score(self, context: SearchContext, base_eval: SearchEvaluation, *, cap: int) -> PairEvaluation:
        t0 = time.monotonic()
        table = context.pair_table
        if table is None or table.rows.shape[0] == 0:
            return PairEvaluation.empty(strategy_name=self.name, context=context, total_possible_pairs=0)
        active = np.flatnonzero(table.mask)[: int(cap)]
        rows = table.rows[active]
        if rows.shape[0] == 0:
            return PairEvaluation.empty(strategy_name=self.name, context=context, total_possible_pairs=table.possible_pair_count)
        if self.pair_scorer is None:
            raise ContractValidationError(
                f"pair strategy {self.name!r} requires an inference pair-scoring provider",
                owner="ExplicitPairStrategy",
            )
        logits = self.pair_scorer.score_pairs(context, table, active)
        priors, _fallback = priors_from_logits(logits)
        return PairEvaluation(
            strategy_name=self.name,
            phase=context.phase,
            root_scope=context.phase == "root",
            pair_table_identity=table.table_hash,
            pair_rows=rows,
            pair_priors=priors,
            pair_prior_source=np.full(rows.shape[0], PRIOR_SOURCE_PAIR, dtype=np.uint8),
            known_first=table.known_first,
            total_possible_pairs=table.possible_pair_count,
            selected_pair_rows=int(rows.shape[0]),
            scored_pair_rows=int(rows.shape[0]),
            caps_applied={"cap": int(cap)},
            timings={
                "pair_strategy_ms": (time.monotonic() - t0) * 1000.0,
                "pair_chunk_count": 1,
                "pair_chunk_forward_ms": (time.monotonic() - t0) * 1000.0,
            },
            influence=f"{self.name}:{self.pair_scorer.name}",
        )


def create_pair_strategy(
    spec: PairStrategySpec | None = None,
    *,
    pair_scorer: PairScoringProvider | None = None,
    registry: PairStrategyRegistry = PAIR_STRATEGY_REGISTRY,
) -> PairStrategy:
    spec = registry.build_spec("none", max_pairs=0) if spec is None else registry.validate_spec(spec)
    descriptor = registry.resolve(spec)
    if not descriptor.scoring_enabled:
        return NoPairStrategy()
    return ExplicitPairStrategy(spec, pair_scorer=pair_scorer)


def _possible_pairs(table: PairActionTable | None) -> int:
    return 0 if table is None else int(table.possible_pair_count)
