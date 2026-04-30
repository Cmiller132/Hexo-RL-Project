"""Explicit pair strategy ownership for search."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping, Protocol

import numpy as np

from hexorl.contracts.pairs import PairActionTable
from hexorl.contracts.validation import ContractValidationError
from hexorl.search.context import SearchContext
from hexorl.search.priors import SearchEvaluation, priors_from_logits

PairStrategyName = Literal["none", "two_stage_root_only", "tactical_only", "diagnostic_full_root"]


@dataclass(frozen=True)
class PairStrategySpec:
    name: PairStrategyName = "none"
    enabled_sources: tuple[str, ...] = ()
    root_enabled: bool = False
    leaf_enabled: bool = False
    max_root_pair_rows: int = 0
    max_leaf_pair_rows: int = 0
    max_full_pair_rows: int = 0
    chunk_size: int = 0
    phase_eligibility: tuple[str, ...] = ("root",)
    known_first_required: bool = False
    diagnostic: bool = False
    telemetry_level: str = "summary"

    def __post_init__(self) -> None:
        name = str(self.name)
        object.__setattr__(self, "name", name)
        for attr in ("max_root_pair_rows", "max_leaf_pair_rows", "max_full_pair_rows", "chunk_size"):
            object.__setattr__(self, attr, int(getattr(self, attr)))
        object.__setattr__(self, "root_enabled", bool(self.root_enabled))
        object.__setattr__(self, "leaf_enabled", bool(self.leaf_enabled))
        object.__setattr__(self, "diagnostic", bool(self.diagnostic))
        if name == "none":
            if self.root_enabled or self.leaf_enabled or self.max_root_pair_rows or self.max_leaf_pair_rows or self.max_full_pair_rows:
                raise ContractValidationError("none pair strategy must not enable or cap scoring", owner="PairStrategySpec")
        elif name == "two_stage_root_only":
            if not self.root_enabled or self.leaf_enabled or self.max_root_pair_rows <= 0:
                raise ContractValidationError("two_stage_root_only requires root cap and no leaf scoring", owner="PairStrategySpec")
        elif name == "tactical_only":
            if not (self.root_enabled or self.leaf_enabled):
                raise ContractValidationError("tactical_only must declare an enabled scope", owner="PairStrategySpec")
            if self.root_enabled and self.max_root_pair_rows <= 0:
                raise ContractValidationError("tactical_only root scoring requires root cap", owner="PairStrategySpec")
            if self.leaf_enabled and self.max_leaf_pair_rows <= 0:
                raise ContractValidationError("tactical_only leaf scoring requires leaf cap", owner="PairStrategySpec")
        elif name == "diagnostic_full_root":
            if not self.diagnostic or not self.root_enabled or self.leaf_enabled or self.max_full_pair_rows <= 0:
                raise ContractValidationError("diagnostic_full_root must be diagnostic root-only with max_full_pair_rows", owner="PairStrategySpec")
        else:
            raise ContractValidationError(f"unknown pair strategy {name!r}", owner="PairStrategySpec")


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


class NoPairStrategy:
    name = "none"

    def score_root(self, context: SearchContext, base_eval: SearchEvaluation) -> PairEvaluation:
        return PairEvaluation.empty(strategy_name=self.name, context=context, total_possible_pairs=_possible_pairs(context.pair_table))

    def score_leaves(self, contexts: list[SearchContext], base_evals: list[SearchEvaluation]) -> list[PairEvaluation]:
        return [PairEvaluation.empty(strategy_name=self.name, context=ctx, total_possible_pairs=_possible_pairs(ctx.pair_table)) for ctx in contexts]


class ExplicitPairStrategy:
    def __init__(self, spec: PairStrategySpec):
        self.spec = spec
        self.name = spec.name

    def score_root(self, context: SearchContext, base_eval: SearchEvaluation) -> PairEvaluation:
        if not self.spec.root_enabled:
            return PairEvaluation.empty(strategy_name=self.name, context=context, total_possible_pairs=_possible_pairs(context.pair_table))
        return self._score(context, base_eval, cap=self._root_cap())

    def score_leaves(self, contexts: list[SearchContext], base_evals: list[SearchEvaluation]) -> list[PairEvaluation]:
        if not self.spec.leaf_enabled:
            return [PairEvaluation.empty(strategy_name=self.name, context=ctx, total_possible_pairs=_possible_pairs(ctx.pair_table)) for ctx in contexts]
        return [self._score(ctx, ev, cap=self.spec.max_leaf_pair_rows) for ctx, ev in zip(contexts, base_evals)]

    def _root_cap(self) -> int:
        if self.spec.name == "diagnostic_full_root":
            return self.spec.max_full_pair_rows
        return self.spec.max_root_pair_rows

    def _score(self, context: SearchContext, base_eval: SearchEvaluation, *, cap: int) -> PairEvaluation:
        t0 = time.monotonic()
        table = context.pair_table
        if table is None or table.rows.shape[0] == 0:
            return PairEvaluation.empty(strategy_name=self.name, context=context, total_possible_pairs=0)
        active = np.flatnonzero(table.mask)[: int(cap)]
        rows = table.rows[active]
        if rows.shape[0] == 0:
            return PairEvaluation.empty(strategy_name=self.name, context=context, total_possible_pairs=table.possible_pair_count)
        logits = table.target[active]
        if float(np.sum(logits)) <= 0.0:
            logits = np.ones(rows.shape[0], dtype=np.float32)
        priors, _fallback = priors_from_logits(logits)
        return PairEvaluation(
            strategy_name=self.name,
            phase=context.phase,
            root_scope=context.phase == "root",
            pair_table_identity=table.table_hash,
            pair_rows=rows,
            pair_priors=priors,
            pair_prior_source=np.ones(rows.shape[0], dtype=np.uint8),
            known_first=table.known_first,
            total_possible_pairs=table.possible_pair_count,
            selected_pair_rows=int(rows.shape[0]),
            scored_pair_rows=int(rows.shape[0]),
            caps_applied={"cap": int(cap)},
            timings={"pair_strategy_ms": (time.monotonic() - t0) * 1000.0},
            influence=str(self.name),
        )


def create_pair_strategy(spec: PairStrategySpec | None = None) -> PairStrategy:
    spec = spec or PairStrategySpec()
    if spec.name == "none":
        return NoPairStrategy()
    return ExplicitPairStrategy(spec)


def _possible_pairs(table: PairActionTable | None) -> int:
    return 0 if table is None else int(table.possible_pair_count)
