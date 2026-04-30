"""Single Python boundary for Rust MCTS calls."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from hexorl.contracts.validation import ContractValidationError
from hexorl.engine.legal import decode_legal_bytes
from hexorl.engine.rust import engine_available, mcts_engine_class
from hexorl.search.pair_strategy import PairEvaluation
from hexorl.search.priors import PRIOR_SOURCE_DENSE, SearchEvaluation


@dataclass(frozen=True)
class MCTSTrace:
    operation: str
    trace_id: str
    root_token: int | None = None
    batch_token: int | None = None
    legal_table_hash: str = ""
    pair_table_hash: str = ""
    prior_source: str = ""
    elapsed_ms: float = 0.0
    extra: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))


class EngineAdapterError(RuntimeError):
    def __init__(self, message: str, *, trace: MCTSTrace):
        super().__init__(message)
        self.trace = trace


class EngineAdapter:
    """Only Python owner of MCTS lifecycle calls."""

    def __init__(
        self,
        backend: Any,
        *,
        is_mock: bool = False,
        subtree_reuse: bool = False,
        game: Any = None,
        engine_factory_args: dict[str, Any] | None = None,
    ):
        self._backend = backend
        self._is_mock = bool(is_mock)
        self._subtree_reuse = bool(subtree_reuse)
        self._game = game
        self._engine_factory_args = dict(engine_factory_args or {})
        self._root_offset: tuple[int, int] | None = None
        self._root_legal_bytes: bytes | None = None
        self._root_generation: int | None = None
        self._batch_generation: int | None = None
        self.timings: dict[str, float] = {}
        self.last_trace: MCTSTrace | None = None
        self.pair_influence = "none"

    def init_root(self):
        t0 = time.monotonic()
        try:
            init = self._backend.init_root()
            if init is None:
                return None
            if len(init) != 5:
                raise EngineAdapterError(
                    "MCTS init_root must return tensor, offsets, legal bytes, and root token",
                    trace=MCTSTrace("init_root", trace_id="unknown"),
                )
            tensor_3d, oq, or_, legal_bytes, root_generation = init
            self._root_offset = (int(oq), int(or_))
            self._root_legal_bytes = bytes(legal_bytes)
            self._root_generation = int(root_generation)
            return np.asarray(tensor_3d, dtype=np.float32), int(oq), int(or_), bytes(legal_bytes), int(root_generation)
        finally:
            self._record_timing("init_root", t0)

    def expand_root(self, evaluation: SearchEvaluation) -> None:
        self._require_evaluation(evaluation)
        self._validate_root_identity(evaluation)
        t0 = time.monotonic()
        try:
            self._call_backend(
                "expand_root",
                evaluation.dense_policy(),
                float(evaluation.value),
                int(evaluation.context.extra.get("offset_q", 0) if evaluation.context.extra else 0),
                int(evaluation.context.extra.get("offset_r", 0) if evaluation.context.extra else 0),
                bytes(evaluation.context.extra.get("legal_bytes", b"") if evaluation.context.extra else b""),
                int(evaluation.context.root_generation or 0),
                trace=self._trace("expand_root", evaluation),
            )
        finally:
            self._record_timing("expand_root", t0)

    def expand_root_with_sparse_priors(self, evaluation: SearchEvaluation) -> None:
        self._require_evaluation(evaluation)
        self._validate_root_identity(evaluation)
        sparse_qr, sparse_logits, sparse_counts = evaluation.sparse_payload()
        t0 = time.monotonic()
        try:
            self._call_backend(
                "expand_root_with_sparse_priors",
                evaluation.dense_policy(),
                float(evaluation.value),
                int(evaluation.context.extra.get("offset_q", 0) if evaluation.context.extra else 0),
                int(evaluation.context.extra.get("offset_r", 0) if evaluation.context.extra else 0),
                bytes(evaluation.context.extra.get("legal_bytes", b"") if evaluation.context.extra else b""),
                int(evaluation.context.root_generation or 0),
                sparse_qr.reshape(-1, 2),
                sparse_logits.reshape(-1),
                int(evaluation.context.extra.get("sparse_stage", 1) if evaluation.context.extra else 1),
                float(evaluation.context.extra.get("sparse_mix", 1.0) if evaluation.context.extra else 1.0),
                trace=self._trace("expand_root_with_sparse_priors", evaluation),
            )
        finally:
            self._record_timing("expand_root_with_sparse_priors", t0)

    def expand_root_with_global_priors(self, evaluation: SearchEvaluation) -> None:
        self._require_evaluation(evaluation)
        self._validate_root_identity(evaluation)
        t0 = time.monotonic()
        try:
            self._call_backend(
                "expand_root_with_global_priors",
                bytes(evaluation.context.extra.get("legal_bytes", b"") if evaluation.context.extra else b""),
                int(evaluation.context.root_generation or 0),
                evaluation.context.legal_table.rows,
                evaluation.row_priors,
                float(evaluation.value),
                trace=self._trace("expand_root_with_global_priors", evaluation),
            )
        finally:
            self._record_timing("expand_root_with_global_priors", t0)

    def apply_root_pair_first_priors(self, pair_eval: PairEvaluation) -> None:
        self._apply_pair(pair_eval, "apply_root_pair_first_priors")

    def apply_root_pair_priors(self, pair_eval: PairEvaluation) -> None:
        self._apply_pair(pair_eval, "apply_root_pair_priors")

    def apply_root_pair_second_priors(self, pair_eval: PairEvaluation) -> None:
        self._apply_pair(pair_eval, "apply_root_pair_second_priors")

    def select_leaves(self, context_or_batch_size, batch_size: int | None = None):
        size = int(context_or_batch_size if batch_size is None else batch_size)
        t0 = time.monotonic()
        try:
            selected = self._backend.select_leaves(size)
            if len(selected) != 3:
                raise EngineAdapterError(
                    "MCTS select_leaves must return tensor batch, count, and batch token",
                    trace=MCTSTrace("select_leaves", trace_id="unknown"),
                )
            tensor_4d, count, batch_generation = selected
            self._batch_generation = int(batch_generation)
            return np.asarray(tensor_4d, dtype=np.float32), int(count), int(batch_generation)
        except Exception as exc:
            raise EngineAdapterError("MCTS select_leaves failed", trace=MCTSTrace("select_leaves", trace_id="unknown")) from exc
        finally:
            self._record_timing("select_leaves", t0)

    def expand_and_backprop(self, evaluations: list[SearchEvaluation]) -> None:
        self._expand_backprop_dense(evaluations)

    def expand_and_backprop_with_sparse(self, evaluations: list[SearchEvaluation]) -> None:
        self._expand_backprop_sparse(evaluations, with_sources=False)

    def expand_and_backprop_sparse_sources(self, evaluations: list[SearchEvaluation]) -> None:
        self._expand_backprop_sparse(evaluations, with_sources=True)

    def sample_action(self, temperature: float, rng_state: int | None = None):
        t0 = time.monotonic()
        try:
            return self._backend.sample_action(temperature, rng_state)
        finally:
            self._record_timing("sample_action", t0)

    def re_root(self, action, new_sims: int):
        q, r = action
        t0 = time.monotonic()
        try:
            if self._subtree_reuse:
                self._backend.re_root(q, r, new_sims)
                if self._game is not None:
                    self._game.place(q, r)
                return True
            if self._game is not None:
                self._game.place(q, r)
            if self._engine_factory_args:
                engine_cls = mcts_engine_class(required=True)
                args = dict(self._engine_factory_args)
                args["game"] = self._game
                args["num_simulations"] = int(new_sims)
                self._backend = engine_cls(**args)
            else:
                return self._backend.re_root(q, r, new_sims)
            return True
        finally:
            self._record_timing("re_root", t0)

    def add_dirichlet_noise(self, noise, fraction):
        return self._backend.add_dirichlet_noise(noise, fraction)

    def done(self):
        return self._backend.done()

    def pending_leaf_metadata(self):
        return self._backend.pending_leaf_metadata()

    def get_results(self):
        return self._backend.get_results()

    def root_child_priors(self):
        return self._backend.root_child_priors()

    def root_child_prior_sources(self):
        if hasattr(self._backend, "root_child_prior_sources"):
            return self._backend.root_child_prior_sources()
        return [PRIOR_SOURCE_DENSE] * len(self.root_child_priors())

    def prior_source_summary(self):
        if hasattr(self._backend, "prior_source_summary"):
            summary = dict(self._backend.prior_source_summary())
        else:
            n = len(self.root_child_prior_sources())
            summary = {
                "root_total_count": n,
                "root_sparse_count": 0,
                "root_dense_count": n,
                "root_default_count": 0,
                "root_pair_count": 0,
                "leaf_pair_count": 0,
                "leaf_total_count": 0,
                "leaf_sparse_count": 0,
                "leaf_dense_count": 0,
                "leaf_default_count": 0,
                "root_sparse_candidate_count": 0,
                "leaf_sparse_candidate_count": 0,
                "root_pair_candidate_count": 0,
                "leaf_expansion_count": 0,
            }
        summary["pair_influence"] = self.pair_influence
        return summary

    def root_child_q_values(self):
        return self._backend.root_child_q_values()

    def root_pair_visit_targets(self):
        return self._backend.root_pair_visit_targets() if hasattr(self._backend, "root_pair_visit_targets") else []

    def move_history_bytes(self):
        return self._backend.move_history_bytes()

    def extract_tree_node_histories(self, min_visits: int = 2):
        return self._backend.extract_tree_node_histories(min_visits) if hasattr(self._backend, "extract_tree_node_histories") else []

    @property
    def winner(self):
        return getattr(self._backend, "winner", None)

    @property
    def is_over(self):
        return getattr(self._backend, "is_over", False)

    @property
    def _game(self):
        return self.__dict__.get("game")

    @_game.setter
    def _game(self, value):
        self.__dict__["game"] = value

    def _expand_backprop_dense(self, evaluations: list[SearchEvaluation]) -> None:
        if not evaluations:
            return
        values = np.asarray([ev.value for ev in evaluations], dtype=np.float32)
        policies = np.concatenate([ev.dense_policy() for ev in evaluations]).astype(np.float32)
        batch_generation = int(evaluations[0].context.batch_generation or 0)
        self._validate_leaf_batch(evaluations, batch_generation)
        t0 = time.monotonic()
        try:
            self._call_backend("expand_and_backprop", policies, values, batch_generation, trace=self._trace("expand_and_backprop", evaluations[0]))
        finally:
            self._record_timing("expand_and_backprop", t0)

    def _expand_backprop_sparse(self, evaluations: list[SearchEvaluation], *, with_sources: bool) -> None:
        if not evaluations:
            return
        max_width = max(int(ev.row_priors.shape[0]) for ev in evaluations)
        count = len(evaluations)
        sparse_qr = np.zeros((count, max_width, 2), dtype=np.int32)
        sparse_logits = np.zeros((count, max_width), dtype=np.float32)
        sparse_sources = np.zeros((count, max_width), dtype=np.uint8)
        sparse_counts = np.zeros(count, dtype=np.uint16)
        values = np.zeros(count, dtype=np.float32)
        policies = np.concatenate([ev.dense_policy() for ev in evaluations]).astype(np.float32)
        for row, ev in enumerate(evaluations):
            width = int(ev.row_priors.shape[0])
            sparse_qr[row, :width] = ev.context.legal_table.rows
            sparse_logits[row, :width] = ev.row_priors
            sparse_sources[row, :width] = ev.prior_source
            sparse_counts[row] = width
            values[row] = ev.value
        batch_generation = int(evaluations[0].context.batch_generation or 0)
        self._validate_leaf_batch(evaluations, batch_generation)
        t0 = time.monotonic()
        try:
            if with_sources:
                self._call_backend(
                    "expand_and_backprop_with_sparse_sources",
                    policies,
                    values,
                    batch_generation,
                    sparse_qr,
                    sparse_logits,
                    sparse_counts,
                    sparse_sources,
                    2,
                    1.0,
                    trace=self._trace("expand_and_backprop_with_sparse_sources", evaluations[0]),
                )
            else:
                self._call_backend(
                    "expand_and_backprop_with_sparse",
                    policies,
                    values,
                    batch_generation,
                    sparse_qr,
                    sparse_logits,
                    sparse_counts,
                    2,
                    1.0,
                    trace=self._trace("expand_and_backprop_with_sparse", evaluations[0]),
                )
        finally:
            self._record_timing("expand_and_backprop_sparse", t0)

    def _apply_pair(self, pair_eval: PairEvaluation, operation: str) -> None:
        if not isinstance(pair_eval, PairEvaluation):
            raise TypeError("EngineAdapter pair operations accept only PairEvaluation")
        if pair_eval.scored_pair_rows == 0:
            self.pair_influence = "none"
            return
        t0 = time.monotonic()
        try:
            if operation == "apply_root_pair_first_priors":
                self._call_backend(operation, pair_eval.pair_priors, 1.0, trace=MCTSTrace(operation, trace_id="pair", pair_table_hash=pair_eval.pair_table_identity))
            else:
                self._call_backend(operation, pair_eval.pair_rows, pair_eval.pair_priors, 1.0, trace=MCTSTrace(operation, trace_id="pair", pair_table_hash=pair_eval.pair_table_identity))
            self.pair_influence = pair_eval.influence
        finally:
            self._record_timing(operation, t0)

    def _require_evaluation(self, evaluation: SearchEvaluation) -> None:
        if not isinstance(evaluation, SearchEvaluation):
            raise TypeError("EngineAdapter accepts only SearchEvaluation")

    def _validate_root_identity(self, evaluation: SearchEvaluation) -> None:
        extra = evaluation.context.extra or {}
        legal_bytes = bytes(extra.get("legal_bytes", b""))
        if not legal_bytes:
            raise ContractValidationError("root SearchEvaluation is missing legal_bytes", owner="EngineAdapter")
        expected = decode_legal_bytes(self._root_legal_bytes or legal_bytes)
        received = decode_legal_bytes(legal_bytes)
        if expected.shape != received.shape or not np.array_equal(expected, received):
            raise ContractValidationError("root legal row mismatch", owner="EngineAdapter")
        table_rows = evaluation.context.legal_table.rows
        if table_rows.shape != received.shape or not np.array_equal(table_rows, received):
            raise ContractValidationError("SearchEvaluation legal table does not match root legal bytes", owner="EngineAdapter")
        oq = int(extra.get("offset_q", 0))
        or_ = int(extra.get("offset_r", 0))
        if self._root_offset is not None and (oq, or_) != self._root_offset:
            raise ContractValidationError("root offset mismatch", owner="EngineAdapter")
        generation = evaluation.context.root_generation
        if self._root_generation is not None and generation != self._root_generation:
            raise ContractValidationError("stale root token", owner="EngineAdapter")

    def _validate_leaf_batch(self, evaluations: list[SearchEvaluation], batch_generation: int) -> None:
        if self._batch_generation is not None and batch_generation != self._batch_generation:
            raise ContractValidationError("stale batch token", owner="EngineAdapter")
        for ev in evaluations:
            if ev.context.batch_generation != batch_generation:
                raise ContractValidationError("mixed batch token in leaf evaluations", owner="EngineAdapter")

    def _call_backend(self, name: str, *args, trace: MCTSTrace):
        try:
            return getattr(self._backend, name)(*args)
        except Exception as exc:
            raise EngineAdapterError(f"MCTS operation failed: {name}: {exc}", trace=trace) from exc

    def _trace(self, operation: str, evaluation: SearchEvaluation) -> MCTSTrace:
        trace = MCTSTrace(
            operation=operation,
            trace_id=evaluation.context.trace_id,
            root_token=evaluation.context.root_generation,
            batch_token=evaluation.context.batch_generation,
            legal_table_hash=evaluation.context.legal_table.table_hash,
            pair_table_hash=evaluation.context.pair_hash,
            prior_source=evaluation.policy_provider,
        )
        self.last_trace = trace
        return trace

    def _record_timing(self, name: str, start: float) -> None:
        self.timings[name] = self.timings.get(name, 0.0) + (time.monotonic() - start) * 1000.0


class _MockBackend:
    NUM_CHANNELS = 13
    BOARD_SIZE = 33
    BOARD_AREA = 33 * 33

    def __init__(self, num_simulations: int = 100, c_puct: float = 1.5, near_radius: int = 8, seed: int = 0):
        self.num_simulations = int(num_simulations)
        self.near_radius = int(near_radius)
        self._rng = np.random.RandomState(seed)
        self._move_count = 0
        self._max_moves = self._rng.randint(20, 80)
        self._is_over = False
        self._winner = None
        self._num_children = self._rng.randint(5, 25)
        self._priors = None
        self._visits = None
        self._q_values = None
        self._root_value = 0.0
        self._sims_done = 0
        self._batch_size = 4
        self._root_generation = 0
        self._batch_generation = 0

    def init_root(self):
        if self._is_over:
            return None
        tensor = self._rng.randn(self.NUM_CHANNELS, self.BOARD_SIZE, self.BOARD_SIZE).astype(np.float32)
        legal_bytes = bytearray()
        rows: set[tuple[int, int]] = set()
        target = int(self._rng.randint(5, 25))
        while len(rows) < target:
            rows.add((int(self._rng.randint(-8, 9)), int(self._rng.randint(-8, 9))))
        for q, r in sorted(rows):
            legal_bytes.extend(q.to_bytes(4, "little", signed=True))
            legal_bytes.extend(r.to_bytes(4, "little", signed=True))
        self._root_generation += 1
        return tensor, 16, 16, bytes(legal_bytes), self._root_generation

    def expand_root(self, policy, value, oq, or_, legal_bytes, root_generation):
        legal = decode_legal_bytes(legal_bytes)
        self._num_children = int(len(legal))
        priors = np.asarray(policy, dtype=np.float32).reshape(-1)
        dense_idx = np.clip((legal[:, 0] + 16) * 33 + (legal[:, 1] + 16), 0, 1088)
        row = priors[dense_idx]
        mass = float(row.sum())
        self._priors = (row / mass if mass > 0 else np.full(self._num_children, 1.0 / max(self._num_children, 1))).astype(np.float32)
        self._root_value = float(value)
        self._visits = np.zeros(self._num_children, dtype=np.uint32)

    def expand_root_with_sparse_priors(self, policy, value, oq, or_, legal_bytes, root_generation, sparse_qr, sparse_logits, stage, sparse_mix):
        self.expand_root(policy, value, oq, or_, legal_bytes, root_generation)

    def expand_root_with_global_priors(self, legal_bytes, root_generation, global_qr, global_logits, value):
        legal = decode_legal_bytes(legal_bytes)
        if legal.shape != np.asarray(global_qr).shape or not np.array_equal(legal, np.asarray(global_qr, dtype=np.int32)):
            raise ValueError("global priors legal rows do not match legal_bytes")
        self._num_children = int(legal.shape[0])
        logits = np.asarray(global_logits, dtype=np.float32)[: self._num_children]
        mass = float(logits.sum())
        self._priors = (logits / mass if mass > 0 else np.full(self._num_children, 1.0 / max(self._num_children, 1))).astype(np.float32)
        self._root_value = float(value)
        self._visits = np.zeros(self._num_children, dtype=np.uint32)

    def apply_root_pair_first_priors(self, pair_first_logits, pair_mix):
        return None

    def apply_root_pair_priors(self, pair_qr, pair_logits, pair_mix):
        return None

    def apply_root_pair_second_priors(self, pair_qr, pair_logits, pair_mix):
        return None

    def add_dirichlet_noise(self, noise, fraction):
        return None

    def done(self):
        return self._sims_done >= self.num_simulations

    def select_leaves(self, batch_size):
        count = min(int(batch_size), self.num_simulations - self._sims_done, self._batch_size)
        self._batch_generation += 1
        if count <= 0:
            self._sims_done = self.num_simulations
            return np.zeros((0, 13, 33, 33), dtype=np.float32), 0, self._batch_generation
        self._sims_done += count
        return self._rng.randn(count, 13, 33, 33).astype(np.float32), count, self._batch_generation

    def pending_leaf_metadata(self):
        return []

    def expand_and_backprop(self, policies, values, batch_generation=0):
        return None

    def expand_and_backprop_with_sparse(self, policies, values, batch_generation, sparse_qr, sparse_logits, sparse_counts, stage, sparse_mix):
        return None

    def expand_and_backprop_with_sparse_sources(self, policies, values, batch_generation, sparse_qr, sparse_logits, sparse_counts, sparse_sources, stage, sparse_mix):
        return None

    def get_results(self):
        n = max(1, int(self._num_children))
        self._visits = self._rng.randint(1, 100, size=n).astype(np.uint32)
        moves_q = [int(self._rng.randint(-8, 9)) for _ in range(n)]
        moves_r = [int(self._rng.randint(-8, 9)) for _ in range(n)]
        return moves_q, moves_r, [int(v) for v in self._visits.tolist()], float(self._rng.uniform(-1, 1))

    def sample_action(self, temperature, rng_state=None):
        return int(self._rng.randint(-8, 9)), int(self._rng.randint(-8, 9))

    def re_root(self, q, r, new_sims):
        self._move_count += 1
        self._sims_done = 0
        if self._move_count >= self._max_moves:
            self._is_over = True
            self._winner = int(self._rng.randint(0, 2))
        return True

    def root_child_priors(self):
        if self._priors is None:
            self._priors = np.full(max(1, self._num_children), 1.0 / max(1, self._num_children), dtype=np.float32)
        return np.array(self._priors, copy=True)

    def root_child_prior_sources(self):
        return [PRIOR_SOURCE_DENSE] * max(0, int(self._num_children))

    def prior_source_summary(self):
        n = max(0, int(self._num_children))
        return {"root_total_count": n, "root_sparse_count": 0, "root_dense_count": n, "root_default_count": 0, "root_pair_count": 0, "leaf_pair_count": 0, "leaf_total_count": 0, "leaf_sparse_count": 0, "leaf_dense_count": 0, "leaf_default_count": 0, "root_sparse_candidate_count": 0, "leaf_sparse_candidate_count": 0, "root_pair_candidate_count": 0, "leaf_expansion_count": 0}

    def root_child_q_values(self):
        return self._rng.uniform(-1, 1, size=max(1, self._num_children)).astype(np.float32).tolist()

    def root_pair_visit_targets(self):
        return []

    def move_history_bytes(self):
        return self._rng.bytes(self._move_count * 12)

    def extract_tree_node_histories(self, min_visits: int = 2):
        return []

    @property
    def winner(self):
        return self._winner

    @property
    def is_over(self):
        return self._is_over


def create_engine_adapter(
    *,
    game: Any = None,
    num_simulations: int,
    c_puct: float,
    near_radius: int,
    seed: int,
    c_puct_init: float = 19652.0,
    constrain_threats: bool = True,
    subtree_reuse: bool = False,
    force_mock: bool | None = None,
) -> EngineAdapter:
    use_mock = (not engine_available()) if force_mock is None else bool(force_mock)
    if use_mock:
        return EngineAdapter(_MockBackend(num_simulations=num_simulations, c_puct=c_puct, near_radius=near_radius, seed=seed), is_mock=True)
    engine_cls = mcts_engine_class(required=True)
    args = {
        "game": game,
        "num_simulations": int(num_simulations),
        "c_puct": float(c_puct),
        "near_radius": int(near_radius),
        "c_puct_init": float(c_puct_init),
        "constrain_threats": bool(constrain_threats),
        "seed": int(seed),
    }
    backend = engine_cls(**args)
    return EngineAdapter(backend, game=game, subtree_reuse=subtree_reuse, engine_factory_args=args)
