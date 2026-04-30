"""Per-game self-play runner.

GameRunner owns one game execution at a time through explicit V2 dependencies.
Process lifecycle, IPC connection management, and heartbeat scheduling stay in
selfplay.worker.
"""

import time
import logging
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Protocol

from hexorl.config import Config
from hexorl.contracts.candidates import CandidateContractBuilder
from hexorl.contracts.identity import stable_digest
from hexorl.contracts.legal import LegalActionTable
from hexorl.action_contract.tactical_oracle import (
    scan_tactical_oracle_from_game,
    scan_tactical_oracle_from_history,
)
from hexorl.engine.legal import decode_legal_bytes
from hexorl.engine.rust import engine_available, hex_game_class
from hexorl.inference.shm_queue import MAX_GRAPH_PAIRS, MAX_PAIR_CANDIDATES
from hexorl.models.specs import model_spec_from_config
from hexorl.graph.tensorize import GraphBatch, build_graph_batch_from_history
from hexorl.search.context import SearchContext
from hexorl.search.engine_adapter import create_engine_adapter
from hexorl.search.pair_strategy import PairEvaluation, PairStrategySpec, create_pair_strategy
from hexorl.search.policy_provider import PolicyProvider, create_policy_provider
from hexorl.search.priors import PRIOR_SOURCE_DEFAULT, SearchEvaluation
from hexorl.search.mcts_runner import choose_leaf_batch, commit_leaf_batch, commit_root, start_root
from hexorl.selfplay.rgsc import RGSCRestartService, encode_move_history
from hexorl.selfplay.records import (
    GameRecord,
    PositionRecord,
    action_to_board_index,
    dense_policy_from_v2,
    policy_v2_from_visits,
)
from hexorl.selfplay.record_writer import QueueSelfPlayRecordWriter, RecordWriteResult, SelfPlayRecordWriter
from hexorl.selfplay.telemetry import (
    ContractTrace,
    SelfPlayDebugBundle,
    LoggingSelfPlayTelemetrySink,
    SelfPlayTelemetrySink,
    heartbeat_payload,
    no_progress_payload,
)

logger = logging.getLogger(__name__)

ENGINE_AVAILABLE = engine_available()

PRIOR_SOURCE_SPARSE = 1
PRIOR_SOURCE_DENSE = 2
PRIOR_SOURCE_PAIR = 4
PAIR_STRATEGY_NONE = "none"
PAIR_STRATEGY_DIAGNOSTIC_FULL_PAIR = "diagnostic_full_pair"


@dataclass(frozen=True)
class RuntimeResourceSpec:
    worker_processes: int
    inference_queue_capacity: int
    record_queue_capacity: int
    leaf_batch_size: int
    max_in_flight_requests_per_worker: int
    rust_threads: int
    torch_threads: int
    shutdown_timeout_s: float


@dataclass(frozen=True)
class GameRunnerConfig:
    worker_id: int
    run_seed: int
    num_simulations: int
    max_game_moves: int
    batch_size: int
    c_puct: float
    c_puct_init: float
    near_radius: int
    constrain_threats: bool
    temperature_schedule: tuple[tuple[float, float], ...]
    pcr_low_sim_prob: float
    pcr_low_sims: int
    policy_target_top_k: int
    dirichlet_alpha: float
    dirichlet_fraction: float
    sparse_prior_stage: int
    sparse_prior_mix: float
    sparse_policy_enabled: bool
    candidate_budget: int
    subtree_reuse: bool
    pair_strategy_name: str
    pair_strategy_max_pairs: int
    configured_heads: tuple[str, ...] = ()


@dataclass(frozen=True)
class GameRunRequest:
    run_id: str
    game_id: int
    game_index: int
    seed: int
    debug_positions: tuple[int, ...] = ()
    cancellation_deadline_s: float | None = None


@dataclass(frozen=True)
class GameRunResult:
    ok: bool
    request: GameRunRequest
    game_record: GameRecord | None
    positions_written: int
    terminal_reason: str
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    failure_status: str = ""
    record_write: RecordWriteResult | None = None


@dataclass(frozen=True)
class SelfPlayContractBuilders:
    candidate_builder_factory: Callable[[], CandidateContractBuilder] = CandidateContractBuilder
    graph_batch_builder: Callable[..., GraphBatch] = build_graph_batch_from_history
    tactical_from_game: Callable[..., object] = scan_tactical_oracle_from_game
    tactical_from_history: Callable[..., object] = scan_tactical_oracle_from_history


class EngineAdapterFactory(Protocol):
    def __call__(
        self,
        *,
        game: object | None,
        num_simulations: int,
        c_puct: float,
        near_radius: int,
        seed: int,
        c_puct_init: float = 19652.0,
        constrain_threats: bool = True,
        subtree_reuse: bool = False,
        force_mock: bool | None = None,
    ):
        ...


def get_temperature(
    move_index: int,
    temperature_schedule: List[List[float]],
) -> float:
    """Interpolate temperature from a piecewise-constant schedule.

    Args:
        move_index: Current move number (0-indexed).
        temperature_schedule: List of [move_threshold, temperature] pairs,
            sorted by move_threshold ascending. Example: [[0, 1.0], [30, 0.0]].

    Returns:
        Temperature at the given move index.
    """
    temp = 0.0
    for threshold, t in temperature_schedule:
        if move_index >= threshold:
            temp = t
    return max(temp, 0.0)


def _critical_actions_from_root_tensor(
    tensor: np.ndarray,
    legal: np.ndarray,
    offset_q: int,
    offset_r: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    winning: list[tuple[int, int]] = []
    forced: list[tuple[int, int]] = []
    cover: list[tuple[int, int]] = []
    if tensor.shape[0] <= 10:
        return winning, forced, cover
    for q_raw, r_raw in legal:
        q, r = int(q_raw), int(r_raw)
        flat = action_to_board_index(q, r, offset_q, offset_r)
        if flat < 0:
            continue
        gi, gj = divmod(flat, 33)
        if tensor[10, gi, gj] > 0.0:
            winning.append((q, r))
        if tensor[9, gi, gj] > 0.0:
            forced.append((q, r))
            cover.append((q, r))
    return winning, forced, cover


def _prior_source_fraction(summary: dict, source_key: str, prefix: str) -> float:
    total = float(summary.get(f"{prefix}_total_count", 0.0))
    if total <= 0.0:
        return 0.0
    return float(summary.get(f"{prefix}_{source_key}_count", 0.0)) / total


def _fallback_use_on_topk(
    visits: List[int],
    sources: List[int],
    k: int,
) -> float:
    return _source_fraction_on_topk(visits, sources, k, PRIOR_SOURCE_DEFAULT)


def _source_fraction_on_topk(
    visits: List[int],
    sources: List[int],
    k: int,
    source: int,
) -> float:
    if not visits or not sources:
        return 0.0
    n = min(len(visits), len(sources))
    order = sorted(range(n), key=lambda i: (-int(visits[i]), i))
    top = order[: min(k, n)]
    if not top:
        return 0.0
    hits = sum(1 for i in top if int(sources[i]) == int(source))
    return hits / float(len(top))


def _last_move_qr(move_history: bytes | bytearray) -> tuple[int, int] | None:
    """Return the most recent placement `(q, r)` from packed move history."""
    if len(move_history) < 12:
        return None
    tail = bytes(move_history[-12:])
    q = int.from_bytes(tail[4:8], "little", signed=True)
    r = int.from_bytes(tail[8:12], "little", signed=True)
    return q, r


def _pair_qr_from_graph_batch(graph_batch) -> np.ndarray:
    """Return `(q1, r1, q2, r2)` rows for graph pair logits."""
    pair_count = int(np.asarray(graph_batch.pair_first_indices).shape[0])
    if pair_count <= 0:
        return np.zeros((0, 4), dtype=np.int32)
    token_qr = np.asarray(graph_batch.token_qr, dtype=np.int32)
    first = np.asarray(graph_batch.pair_first_indices, dtype=np.int64)
    second = np.asarray(graph_batch.pair_second_indices, dtype=np.int64)
    rows = np.zeros((pair_count, 4), dtype=np.int32)
    for idx, (a, b) in enumerate(zip(first, second)):
        if int(a) < 0 or int(b) < 0:
            continue
        rows[idx, :2] = token_qr[int(a)]
        rows[idx, 2:] = token_qr[int(b)]
    return rows


def _legal_bytes_from_qr(qr_rows: np.ndarray) -> bytes:
    """Encode global `(q,r)` rows for the Rust MCTS legal-row byte contract."""
    return np.asarray(qr_rows, dtype=np.int32).reshape(-1, 2).tobytes(order="C")


def _blend_action_logits(base_logits: np.ndarray, aux_logits: np.ndarray, mix: float) -> np.ndarray:
    """Blend two keyed action-logit vectors in probability space."""
    base = np.asarray(base_logits, dtype=np.float32)
    aux = np.asarray(aux_logits, dtype=np.float32)
    if base.shape != aux.shape:
        raise ValueError(f"logit blend shape mismatch: {base.shape} vs {aux.shape}")
    if base.size == 0:
        return base
    base_exp = np.exp(base - np.max(base))
    aux_exp = np.exp(aux - np.max(aux))
    base_prob = base_exp / max(float(base_exp.sum()), 1e-6)
    aux_prob = aux_exp / max(float(aux_exp.sum()), 1e-6)
    alpha = float(np.clip(mix, 0.0, 1.0))
    blended = (1.0 - alpha) * base_prob + alpha * aux_prob
    return np.log(np.maximum(blended, 1e-12)).astype(np.float32)


def _normalize_pair_visit_targets(rows) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    parsed: list[tuple[tuple[int, int], tuple[int, int], float]] = []
    total = 0.0
    for row in rows or []:
        q1, r1, q2, r2, weight = row
        weight = float(weight)
        if weight <= 0.0:
            continue
        first = (int(q1), int(r1))
        second = (int(q2), int(r2))
        if first == second:
            continue
        parsed.append((first, second, weight))
        total += weight
    if total <= 0.0:
        return []
    return [(first, second, float(weight / total)) for first, second, weight in parsed]


def _pair_policy_target_is_complete(
    pair_policy_v2: list[tuple[tuple[int, int], tuple[int, int], float]],
    legal_rows: np.ndarray,
    placements_remaining: int,
) -> bool:
    """Return whether a first-placement joint target covers every legal pair row."""
    if int(placements_remaining) < 2:
        return True
    legal = np.asarray(legal_rows, dtype=np.int32).reshape(-1, 2)
    legal_count = int(legal.shape[0])
    if legal_count < 2:
        return True
    expected = legal_count * (legal_count - 1) // 2
    if len(pair_policy_v2) != expected:
        return False
    legal_set = {(int(q), int(r)) for q, r in legal.tolist()}
    seen: set[frozenset[tuple[int, int]]] = set()
    for first, second, prob in pair_policy_v2:
        if float(prob) < 0.0:
            return False
        a = (int(first[0]), int(first[1]))
        b = (int(second[0]), int(second[1]))
        if a == b or a not in legal_set or b not in legal_set:
            return False
        seen.add(frozenset({a, b}))
    return len(seen) == expected


# â”€â”€ Mock MCTS Engine â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class GameRunner:
    """Execute one self-play game through explicit V2 runtime dependencies."""

    def __init__(
        self,
        *,
        policy_provider: PolicyProvider,
        pair_strategy,
        engine_adapter_factory: EngineAdapterFactory,
        record_writer: SelfPlayRecordWriter,
        telemetry_sink: SelfPlayTelemetrySink,
        contract_builders: SelfPlayContractBuilders,
        runtime_spec: RuntimeResourceSpec,
        runner_config: GameRunnerConfig,
        model_spec,
        rgsc_service: RGSCRestartService | None = None,
    ):
        self.policy_provider = policy_provider
        self.pair_strategy_impl = pair_strategy
        self.engine_adapter_factory = engine_adapter_factory
        self.record_writer = record_writer
        self.telemetry_sink = telemetry_sink
        self.contract_builders = contract_builders
        self.runtime_spec = runtime_spec
        self.runner_config = runner_config
        self.model_spec = model_spec
        self.rgsc = rgsc_service or RGSCRestartService(enabled=False)
        self.worker_id = runner_config.worker_id
        self.num_simulations = runner_config.num_simulations
        self.max_game_moves = runner_config.max_game_moves
        self.batch_size = runner_config.batch_size
        self.c_puct = runner_config.c_puct
        self.c_puct_init = runner_config.c_puct_init
        self.near_radius = runner_config.near_radius
        self.constrain_threats = runner_config.constrain_threats
        self.temperature_schedule = [list(item) for item in runner_config.temperature_schedule]
        self.pcr_low_sim_prob = runner_config.pcr_low_sim_prob
        self.pcr_low_sims = runner_config.pcr_low_sims
        self.policy_target_top_k = runner_config.policy_target_top_k
        self.dirichlet_alpha = runner_config.dirichlet_alpha
        self.dirichlet_fraction = runner_config.dirichlet_fraction
        self.sparse_prior_stage = runner_config.sparse_prior_stage
        self.sparse_prior_mix = runner_config.sparse_prior_mix
        self.sparse_policy_enabled = runner_config.sparse_policy_enabled
        self.uses_global_policy = self.model_spec.is_global_graph
        if self.uses_global_policy:
            self.near_radius = 8
            self.constrain_threats = False
        self.pair_strategy = str(runner_config.pair_strategy_name).lower()
        self.pair_strategy_max_pairs = int(runner_config.pair_strategy_max_pairs)
        self.pair_policy_enabled = self.pair_strategy != PAIR_STRATEGY_NONE
        self.candidate_budget = max(int(runner_config.candidate_budget), int(runner_config.policy_target_top_k))

    def pair_strategy_summary(
        self,
        *,
        pair_rows_possible: int = 0,
        pair_rows_scored: int = 0,
    ) -> dict[str, int | float | str]:
        return {
            "event": "pair_strategy_summary",
            "worker_id": int(self.worker_id),
            "model_family": self.model_spec.kind,
            "pair_strategy": self.pair_strategy,
            "pair_rows_possible": int(pair_rows_possible),
            "pair_rows_scored": int(pair_rows_scored),
        }

    def emit_worker_heartbeat(
        self,
        *,
        process_id: int,
        run_id: str,
        game_id: int,
        phase: str,
        move_index: int,
        positions_completed: int,
        warning_count: int = 0,
        last_warning: str = "",
    ) -> None:
        self.telemetry_sink.emit(
            "selfplay_worker_heartbeat",
            heartbeat_payload(
                worker_id=self.worker_id,
                process_id=int(process_id),
                run_id=run_id,
                game_id=int(game_id),
                phase=phase,
                move_index=int(move_index),
                positions_completed=int(positions_completed),
                model_family=self.model_spec.kind,
                recipe_id="default",
                policy_provider=self.policy_provider.name,
                pair_strategy=self.pair_strategy,
                warning_count=warning_count,
                last_warning=last_warning,
                recent_timing_summary={},
            ),
        )

    def _pair_strategy_spec(self) -> PairStrategySpec:
        if self.pair_strategy == PAIR_STRATEGY_NONE:
            return PairStrategySpec()
        if self.pair_strategy in {"two_stage_root_only", "two_stage_root"}:
            return PairStrategySpec(
                name="two_stage_root_only",
                root_enabled=True,
                leaf_enabled=False,
                max_root_pair_rows=self.pair_strategy_max_pairs,
                chunk_size=0 if self.pair_strategy_max_pairs <= 0 else min(self.pair_strategy_max_pairs, MAX_GRAPH_PAIRS),
            )
        if self.pair_strategy in {"tactical_only", "tactical"}:
            return PairStrategySpec(
                name="tactical_only",
                root_enabled=True,
                leaf_enabled=False,
                max_root_pair_rows=self.pair_strategy_max_pairs,
                chunk_size=0 if self.pair_strategy_max_pairs <= 0 else min(self.pair_strategy_max_pairs, MAX_PAIR_CANDIDATES),
            )
        if self.pair_strategy in {PAIR_STRATEGY_DIAGNOSTIC_FULL_PAIR, "diagnostic_full_root"}:
            return PairStrategySpec(
                name="diagnostic_full_root",
                diagnostic=True,
                root_enabled=True,
                leaf_enabled=False,
                max_full_pair_rows=self.pair_strategy_max_pairs,
                chunk_size=0 if self.pair_strategy_max_pairs <= 0 else min(self.pair_strategy_max_pairs, MAX_GRAPH_PAIRS),
            )
        raise ValueError(f"unsupported explicit pair strategy {self.pair_strategy!r}")

    def _legal_table_from_bytes(
        self,
        legal_bytes: bytes,
        *,
        history_bytes: bytes,
        placements_remaining: int = 1,
    ) -> LegalActionTable:
        rows = decode_legal_bytes(bytes(legal_bytes))
        return LegalActionTable.from_rows(
            [(int(q), int(r)) for q, r in rows.tolist()],
            source="rust:legal",
            history_hash=stable_digest(("search-history", bytes(history_bytes))),
            placements_remaining=int(placements_remaining),
        )

    def _candidate_table_for_search(
        self,
        *,
        tensor_3d: np.ndarray,
        legal: np.ndarray,
        offset_q: int,
        offset_r: int,
        history_bytes: bytes,
        engine,
    ):
        winning_moves, forced_blocks, cover_cells = _critical_actions_from_root_tensor(
            tensor_3d,
            legal,
            int(offset_q),
            int(offset_r),
        )
        root_game = getattr(engine, "_game", None)
        if root_game is not None:
            oracle = self.contract_builders.tactical_from_game(
                root_game,
                [(int(q), int(r)) for q, r in legal],
                offset_q=int(offset_q),
                offset_r=int(offset_r),
            )
        else:
            oracle = self.contract_builders.tactical_from_history(
                bytes(history_bytes),
                [(int(q), int(r)) for q, r in legal],
                offset_q=int(offset_q),
                offset_r=int(offset_r),
            )
        return self.contract_builders.candidate_builder_factory().build(
            [(int(q), int(r)) for q, r in legal],
            [],
            offset_q=int(offset_q),
            offset_r=int(offset_r),
            budget=self.candidate_budget,
            storage_width=self.candidate_budget,
            winning_moves=list(winning_moves) + list(oracle.win_now_cells),
            forced_block_moves=list(forced_blocks) + list(oracle.forced_block_cells),
            cover_cells=list(cover_cells) + list(oracle.cover_cells),
            open_four_cells=oracle.open_four_cells,
            open_five_cells=oracle.open_five_cells,
        )

    def _root_context(
        self,
        *,
        tensor_3d: np.ndarray,
        offset_q: int,
        offset_r: int,
        legal_bytes: bytes,
        root_generation: int,
        move_history: bytes,
        engine,
    ) -> tuple[SearchContext, float]:
        legal = decode_legal_bytes(legal_bytes)
        legal_table = self._legal_table_from_bytes(
            legal_bytes,
            history_bytes=move_history,
            placements_remaining=1,
        )
        candidate_table = None
        graph_batch = None
        candidate_ms = 0.0
        if self.uses_global_policy:
            graph_batch = self.contract_builders.graph_batch_builder(
                bytes(move_history),
                radius=8,
                max_pair_rows=0,
                include_pair_rows=False,
            )
        elif self.sparse_policy_enabled and self.sparse_prior_stage > 0:
            t0 = time.monotonic()
            candidate_table = self._candidate_table_for_search(
                tensor_3d=tensor_3d,
                legal=legal,
                offset_q=int(offset_q),
                offset_r=int(offset_r),
                history_bytes=bytes(move_history),
                engine=engine,
            )
            candidate_ms = (time.monotonic() - t0) * 1000.0
        context = SearchContext.create(
            phase="root",
            legal_table=legal_table,
            model_family=self.model_spec.kind,
            tensor=tensor_3d.reshape(1, 13, 33, 33).astype(np.float32, copy=False),
            history_bytes=bytes(move_history),
            root_generation=int(root_generation),
            candidate_table=candidate_table,
            graph_batch=graph_batch,
            pair_strategy_id=self.pair_strategy,
            extra={
                "offset_q": int(offset_q),
                "offset_r": int(offset_r),
                "legal_bytes": bytes(legal_bytes),
                "sparse_stage": int(self.sparse_prior_stage),
                "sparse_mix": float(self.sparse_prior_mix),
            },
        )
        return context, candidate_ms

    def _evaluate_root_with_search(
        self,
        *,
        engine,
        tensor_3d: np.ndarray,
        offset_q: int,
        offset_r: int,
        legal_bytes: bytes,
        root_generation: int,
        move_history: bytes,
    ) -> tuple[SearchEvaluation, PairEvaluation, float, float]:
        context, candidate_ms = self._root_context(
            tensor_3d=tensor_3d,
            offset_q=offset_q,
            offset_r=offset_r,
            legal_bytes=legal_bytes,
            root_generation=root_generation,
            move_history=move_history,
            engine=engine,
        )
        t0 = time.monotonic()
        evaluation = self.policy_provider.evaluate_root(context)
        forward_ms = (time.monotonic() - t0) * 1000.0
        pair_eval = self.pair_strategy_impl.score_root(context, evaluation)
        trace = ContractTrace.from_context(
            context,
            timings_ms={
                "candidate_build_ms": float(candidate_ms),
                "model_forward_ms": float(forward_ms),
                **dict(evaluation.timings),
                **dict(pair_eval.timings),
            },
            pair_rows_scored=pair_eval.scored_pair_rows,
            warnings=tuple(evaluation.warnings) + tuple(pair_eval.warnings),
        )
        self.telemetry_sink.emit(
            "policy_eval_timing",
            {
                "trace_id": context.trace_id,
                "history_hash": context.history_hash,
                "move_index": int(len(context.history_bytes) // 12),
                "phase": context.phase,
                "request_kind": "policy",
                "provider_name": self.policy_provider.name,
                "model_family": self.model_spec.kind,
                "protocol_version": evaluation.inference_protocol,
                "ipc_pack_ms": trace.timings_ms["ipc_pack_ms"],
                "ipc_wait_ms": trace.timings_ms["ipc_wait_ms"],
                "queue_wait_ms": trace.timings_ms["queue_wait_ms"],
                "collate_ms": trace.timings_ms["collate_ms"],
                "model_forward_ms": trace.timings_ms["model_forward_ms"],
                "scatter_ms": trace.timings_ms["scatter_ms"],
                "decode_ms": trace.timings_ms["decode_ms"],
                "prior_source": evaluation.policy_provider,
                "masked_count": 0,
                "non_finite_count": 0,
                "legal_row_coverage": int(evaluation.row_priors.shape[0]),
                "contract_trace": trace.to_event_payload(),
            },
        )
        self.telemetry_sink.emit(
            "pair_strategy_summary",
            {
                "pair_strategy_name": pair_eval.strategy_name,
                "enabled_disabled_reason": "enabled" if pair_eval.scored_pair_rows else "no_pair_rows_scored",
                "phase": context.phase,
                "total_possible_pairs": int(pair_eval.total_possible_pairs),
                "selected_rows": int(pair_eval.selected_pair_rows),
                "scored_rows": int(pair_eval.scored_pair_rows),
                "cap_values": dict(pair_eval.caps_applied),
                "cap_hit": pair_eval.selected_pair_rows < pair_eval.total_possible_pairs,
                "chunk_count": int(pair_eval.timings.get("pair_chunk_count", 0)),
                "chunk_forward_time_ms": float(pair_eval.timings.get("pair_chunk_forward_ms", 0.0)),
                "head_usage": pair_eval.influence,
            },
        )
        return evaluation, pair_eval, candidate_ms, forward_ms

    def _commit_root_search_evaluation(
        self,
        engine,
        evaluation: SearchEvaluation,
        pair_eval: PairEvaluation,
    ) -> None:
        commit_root(engine, evaluation, pair_eval)

    def _leaf_contexts(
        self,
        *,
        engine,
        batch_4d: np.ndarray,
        count: int,
        batch_generation: int,
    ) -> list[SearchContext]:
        meta = engine.pending_leaf_metadata()
        contexts: list[SearchContext] = []
        if len(meta) != int(count):
            return contexts
        for row, (leaf_oq, leaf_or, leaf_legal_bytes, leaf_history_bytes) in enumerate(meta):
            legal = decode_legal_bytes(bytes(leaf_legal_bytes))
            legal_table = self._legal_table_from_bytes(
                bytes(leaf_legal_bytes),
                history_bytes=bytes(leaf_history_bytes),
                placements_remaining=1,
            )
            candidate_table = None
            graph_batch = None
            if self.uses_global_policy:
                graph_batch = self.contract_builders.graph_batch_builder(
                    bytes(leaf_history_bytes),
                    opp_legal_moves=[(int(q), int(r)) for q, r in legal],
                    radius=8,
                    max_pair_rows=0,
                    include_pair_rows=False,
                )
            elif self.sparse_policy_enabled and self.sparse_prior_stage >= 2:
                candidate_table = self._candidate_table_for_search(
                    tensor_3d=batch_4d[row],
                    legal=legal,
                    offset_q=int(leaf_oq),
                    offset_r=int(leaf_or),
                    history_bytes=bytes(leaf_history_bytes),
                    engine=engine,
                )
            contexts.append(
                SearchContext.create(
                    phase="leaf",
                    legal_table=legal_table,
                    model_family=self.model_spec.kind,
                    tensor=batch_4d[row].reshape(1, 13, 33, 33).astype(np.float32, copy=False),
                    history_bytes=bytes(leaf_history_bytes),
                    batch_generation=int(batch_generation),
                    candidate_table=candidate_table,
                    graph_batch=graph_batch,
                    pair_strategy_id=self.pair_strategy,
                    extra={
                        "offset_q": int(leaf_oq),
                        "offset_r": int(leaf_or),
                        "legal_bytes": bytes(leaf_legal_bytes),
                        "sparse_stage": int(self.sparse_prior_stage),
                        "sparse_mix": float(self.sparse_prior_mix),
                    },
                )
            )
        return contexts

    def _expand_leaf_batch_with_search(
        self,
        *,
        engine,
        batch_4d: np.ndarray,
        count: int,
        batch_generation: int,
    ) -> float:
        contexts = self._leaf_contexts(
            engine=engine,
            batch_4d=batch_4d,
            count=count,
            batch_generation=batch_generation,
        )
        if len(contexts) != int(count):
            raise ValueError("MCTS leaf expansion requires pending leaf metadata for SearchEvaluation")
        t0 = time.monotonic()
        evaluations = self.policy_provider.evaluate_leaves(contexts)
        elapsed = (time.monotonic() - t0) * 1000.0
        if self.uses_global_policy:
            commit_leaf_batch(engine, evaluations, source_mode="global")
        elif self.policy_provider.name == "GraphHybridPolicyProvider":
            commit_leaf_batch(engine, evaluations, source_mode="sparse")
        else:
            commit_leaf_batch(engine, evaluations, source_mode="dense")
        return elapsed

    def run_game(self, request: GameRunRequest) -> GameRunResult:
        t0 = time.monotonic()
        self.telemetry_sink.emit(
            "selfplay_phase_transition",
            {
                "worker_id": int(self.worker_id),
                "run_id": request.run_id,
                "game_id": int(request.game_id),
                "phase": "game_start",
            },
        )
        try:
            game_record = self._play_one_game(request)
            if game_record is None:
                return GameRunResult(
                    ok=False,
                    request=request,
                    game_record=None,
                    positions_written=0,
                    terminal_reason="no_record",
                    timings_ms={"game_runner_ms": (time.monotonic() - t0) * 1000.0},
                    failure_status="game_execution_failed",
                )
            write_result = self.record_writer.write(game_record, run_request=request)
            self.telemetry_sink.emit(
                "selfplay_game_summary",
                {
                    "game_id": int(game_record.game_id),
                    "seed": int(request.seed),
                    "winner_result": float(game_record.outcome),
                    "move_count": int(game_record.game_length),
                    "terminal_reason": str(game_record.terminal_reason),
                    "positions_written": int(write_result.positions_written),
                    "records_skipped": 0 if write_result.ok else len(game_record.positions),
                    "total_engine_ms": 0.0,
                    "total_search_ms": 0.0,
                    "total_inference_ms": 0.0,
                    "total_contract_ms": 0.0,
                    "total_record_writer_ms": float(write_result.elapsed_ms),
                    "average_legal_count": 0.0,
                    "average_candidate_count": 0.0,
                    "average_pair_count": 0.0,
                    "average_token_count": 0.0,
                    "average_relation_count": 0.0,
                    "pair_strategy_used": self.pair_strategy,
                    "total_pair_rows_scored": sum(int(pos.pair_prior_candidate_count) for pos in game_record.positions),
                    "validation_failures": [] if write_result.ok else [write_result.error],
                    "warning_summary": [],
                },
            )
            return GameRunResult(
                ok=write_result.ok,
                request=request,
                game_record=game_record,
                positions_written=write_result.positions_written,
                terminal_reason=game_record.terminal_reason,
                timings_ms={"game_runner_ms": (time.monotonic() - t0) * 1000.0},
                failure_status="" if write_result.ok else write_result.error,
                record_write=write_result,
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            self.telemetry_sink.emit(
                "selfplay_no_progress",
                no_progress_payload(
                    phase="game_runner_exception",
                    elapsed_ms=elapsed_ms,
                    last_completed_position=0,
                    suggested_owner="EngineAdapter/MCTS",
                ),
            )
            return GameRunResult(
                ok=False,
                request=request,
                game_record=None,
                positions_written=0,
                terminal_reason="exception",
                timings_ms={"game_runner_ms": elapsed_ms},
                failure_status=str(exc),
            )

    def _play_one_game(
        self, request: GameRunRequest
    ) -> Optional[GameRecord]:
        """Play one complete self-play game.

        Returns a GameRecord with all position data, or None on failure.
        """
        use_pcr = np.random.random() < self.pcr_low_sim_prob
        sims = self.pcr_low_sims if use_pcr else self.num_simulations
        game_seed = int(request.seed)
        game_id = int(request.game_id)
        rgsc_restart = None
        if ENGINE_AVAILABLE:
            game_cls = hex_game_class(required=True)
            rgsc_restart = self.rgsc.maybe_restart(
                game_cls,
                max_game_moves=self.max_game_moves,
            )
            game = rgsc_restart.game if rgsc_restart.used else game_cls()
            engine = self.engine_adapter_factory(
                game=game,
                num_simulations=sims,
                c_puct=self.c_puct,
                near_radius=self.near_radius,
                seed=game_seed,
                c_puct_init=self.c_puct_init,
                constrain_threats=self.constrain_threats,
                subtree_reuse=self.runner_config.subtree_reuse,
            )
        else:
            engine = self.engine_adapter_factory(
                game=None,
                num_simulations=sims,
                c_puct=self.c_puct,
                near_radius=self.near_radius,
                seed=game_seed,
                force_mock=True,
            )

        positions: List[PositionRecord] = []
        move_history = bytearray(rgsc_restart.move_history if rgsc_restart and rgsc_restart.used else b"")
        move_idx = int(rgsc_restart.move_count) if rgsc_restart and rgsc_restart.used else 0
        terminal_reason = "unknown"

        while True:
            if move_idx >= self.max_game_moves:
                terminal_reason = "max_game_moves"
                break
            init = start_root(engine)
            if init is None:
                terminal_reason = "no_root"
                break

            tensor, offset_q, offset_r, legal_bytes, root_generation = init
            if isinstance(tensor, np.ndarray):
                tensor_3d = tensor
            else:
                tensor_3d = np.array(tensor)

            sparse_prior_forward_ms = 0.0
            sparse_prior_candidate_build_ms = 0.0
            sparse_vs_dense_disagreement = 0.0

            try:
                root_eval, pair_eval, candidate_ms, forward_ms = self._evaluate_root_with_search(
                    engine=engine,
                    tensor_3d=tensor_3d,
                    offset_q=int(offset_q),
                    offset_r=int(offset_r),
                    legal_bytes=bytes(legal_bytes),
                    root_generation=int(root_generation),
                    move_history=bytes(move_history),
                )
                sparse_prior_candidate_build_ms += candidate_ms
                sparse_prior_forward_ms += forward_ms
                self._commit_root_search_evaluation(engine, root_eval, pair_eval)
            except Exception as exc:
                logger.warning(
                    "Worker %s: root search evaluation failed at move %s: %s",
                    self.worker_id,
                    move_idx,
                    exc,
                )
                terminal_reason = "invalid_root_inference"
                return None

            if self.dirichlet_alpha > 0:
                try:
                    child_priors = engine.root_child_priors()
                    n_children = (
                        child_priors.shape[0]
                        if hasattr(child_priors, "shape")
                        else len(child_priors)
                    )
                except Exception as exc:
                    logger.debug("Worker %s: root noise skipped: %s", self.worker_id, exc)
                    n_children = 20
                noise = np.random.dirichlet(
                    [self.dirichlet_alpha] * max(n_children, 1)
                )
                engine.add_dirichlet_noise(
                    noise.astype(np.float32), self.dirichlet_fraction
                )

            while not engine.done():
                try:
                    batch_tensor, count, batch_generation = choose_leaf_batch(engine, self.batch_size)
                    if count == 0:
                        commit_leaf_batch(engine, [], source_mode="dense")
                        break
                    if isinstance(batch_tensor, np.ndarray):
                        batch_4d = batch_tensor
                    else:
                        batch_4d = np.array(batch_tensor)

                    leaf_forward_ms = self._expand_leaf_batch_with_search(
                        engine=engine,
                        batch_4d=batch_4d.astype(np.float32, copy=False),
                        count=int(count),
                        batch_generation=int(batch_generation),
                    )
                    sparse_prior_forward_ms += leaf_forward_ms
                except Exception as exc:
                    logger.warning(
                        "Worker %s: leaf expansion failed at move %s: %s",
                        self.worker_id,
                        move_idx,
                        exc,
                    )
                    if self.uses_global_policy:
                        terminal_reason = "invalid_graph_leaf_inference"
                        return None
                    break

            moves_q, moves_r, visits, root_value = engine.get_results()
            pair_policy_v2 = _normalize_pair_visit_targets(
                engine.root_pair_visit_targets()
                if hasattr(engine, "root_pair_visit_targets")
                else []
            )
            legal_root = decode_legal_bytes(legal_bytes)
            root_placements_remaining_for_targets = (
                int(getattr(engine._game, "placements_remaining", 1))
                if ENGINE_AVAILABLE and hasattr(engine, "_game")
                else (1 if move_idx == 0 else 2 if move_idx % 2 == 1 else 1)
            )
            pair_policy_complete = _pair_policy_target_is_complete(
                pair_policy_v2,
                legal_root,
                root_placements_remaining_for_targets,
            )
            prior_summary = (
                engine.prior_source_summary()
                if hasattr(engine, "prior_source_summary")
                else {}
            )
            root_prior_sources = (
                list(engine.root_child_prior_sources())
                if hasattr(engine, "root_child_prior_sources")
                else []
            )
            total_prior_count = float(prior_summary.get("root_total_count", 0.0)) + float(
                prior_summary.get("leaf_total_count", 0.0)
            )
            fallback_prior_use = 0.0
            if total_prior_count > 0.0:
                fallback_prior_use = (
                    float(prior_summary.get("root_default_count", 0.0))
                    + float(prior_summary.get("leaf_default_count", 0.0))
                ) / total_prior_count
            leaf_expansions = float(prior_summary.get("leaf_expansion_count", 0.0))
            sparse_prior_leaf_candidate_count = (
                float(prior_summary.get("leaf_sparse_candidate_count", 0.0)) / leaf_expansions
                if leaf_expansions > 0.0
                else 0.0
            )
            rgsc_tree_node_insertions = 0
            if self.rgsc.enabled and hasattr(engine, "extract_tree_node_histories"):
                try:
                    min_tree_visits = max(2, int(sims) // 16)
                    histories = engine.extract_tree_node_histories(min_tree_visits)
                    scored = []
                    if histories:
                        self.telemetry_sink.emit(
                            "selfplay_phase_transition",
                            {
                                "worker_id": int(self.worker_id),
                                "game_id": int(game_id),
                                "phase": "rgsc_tree_candidates_unscored",
                                "candidate_count": len(histories),
                            },
                        )
                    if scored:
                        rgsc_tree_node_insertions = self.rgsc.observe_tree_node_candidates(
                            scored,
                            game_id=game_id,
                            score_source="explicit_regret_scorer",
                        )
                except Exception as exc:
                    logger.debug("Worker %s: RGSC tree extraction failed: %s", self.worker_id, exc)

            temp = get_temperature(move_idx, self.temperature_schedule)
            q, r = engine.sample_action(temp)
            if q is None:
                q, r = 0, 0
            q, r = int(q), int(r)
            selected_action_value = None
            try:
                q_values = list(engine.root_child_q_values())
                for child_q, child_r, child_value in zip(moves_q, moves_r, q_values):
                    if int(child_q) == q and int(child_r) == r:
                        selected_action_value = float(child_value)
                        break
            except Exception:
                selected_action_value = None

            if ENGINE_AVAILABLE:
                player = engine._game.current_player
            else:
                player = move_idx % 2

            record_history = bytes(move_history)

            # Preserve global action identity first, then project to the legacy
            # 33x33 crop target. This keeps outside-window MCTS mass measurable.
            policy_v2 = policy_v2_from_visits(
                moves_q,
                moves_r,
                visits,
            )
            policy, outside_mass = dense_policy_from_v2(
                policy_v2,
                offset_q,
                offset_r,
                top_k=self.policy_target_top_k,
            )
            target_mass = sum(prob for _q, _r, prob in policy_v2)
            missing_mass = max(0.0, 1.0 - target_mass) if policy_v2 else 1.0
            winning_moves, forced_blocks, cover_cells = _critical_actions_from_root_tensor(
                tensor_3d,
                legal_root,
                int(offset_q),
                int(offset_r),
            )
            oracle = self.contract_builders.tactical_from_history(
                record_history,
                [(int(q), int(r)) for q, r in legal_root],
                offset_q=int(offset_q),
                offset_r=int(offset_r),
            )
            candidate_probe = self.contract_builders.candidate_builder_factory().build(
                legal_root.tolist(),
                policy_v2,
                offset_q=int(offset_q),
                offset_r=int(offset_r),
                budget=self.candidate_budget,
                winning_moves=list(winning_moves) + list(oracle.win_now_cells),
                forced_block_moves=list(forced_blocks) + list(oracle.forced_block_cells),
                cover_cells=list(cover_cells) + list(oracle.cover_cells),
                open_four_cells=oracle.open_four_cells,
                open_five_cells=oracle.open_five_cells,
            )
            pair_prior_applicable = move_idx > 0
            pair_prior_hit_frac = _prior_source_fraction(prior_summary, "pair", "root")
            pair_fallback_prior_use = (
                max(0.0, 1.0 - pair_prior_hit_frac)
                if pair_prior_applicable
                else 0.0
            )
            pair_fallback_prior_use_on_mcts_top1 = (
                max(
                    0.0,
                    1.0
                    - _source_fraction_on_topk(
                        visits, root_prior_sources, 1, PRIOR_SOURCE_PAIR
                    ),
                )
                if pair_prior_applicable
                else 0.0
            )
            pair_fallback_prior_use_on_mcts_top4 = (
                max(
                    0.0,
                    1.0
                    - _source_fraction_on_topk(
                        visits, root_prior_sources, 4, PRIOR_SOURCE_PAIR
                    ),
                )
                if pair_prior_applicable
                else 0.0
            )
            pair_fallback_prior_use_on_mcts_top8 = (
                max(
                    0.0,
                    1.0
                    - _source_fraction_on_topk(
                        visits, root_prior_sources, 8, PRIOR_SOURCE_PAIR
                    ),
                )
                if pair_prior_applicable
                else 0.0
            )

            positions.append(
                PositionRecord(
                    move_history=record_history,
                    policy_target=policy,
                    policy_target_v2=policy_v2,
                    pair_policy_target_v2=pair_policy_v2,
                    pair_policy_complete=pair_policy_complete,
                    target_policy_mass_outside_window=outside_mass,
                    missing_target_policy_mass=missing_mass,
                    candidate_recall_mcts_top1=candidate_probe.recall_top1,
                    candidate_recall_mcts_top4=candidate_probe.recall_top4,
                    candidate_recall_mcts_top8=candidate_probe.recall_top8,
                    candidate_recall_winning_move=candidate_probe.recall_winning_move,
                    candidate_recall_forced_block=candidate_probe.recall_forced_block,
                    candidate_recall_two_placement_cover=candidate_probe.recall_two_placement_cover,
                    candidate_discovery_top1=candidate_probe.discovery_top1,
                    candidate_discovery_top4=candidate_probe.discovery_top4,
                    candidate_discovery_top8=candidate_probe.discovery_top8,
                    candidate_discovery_winning_move=candidate_probe.discovery_winning_move,
                    candidate_discovery_forced_block=candidate_probe.discovery_forced_block,
                    candidate_discovery_two_placement_cover=candidate_probe.discovery_two_placement_cover,
                    candidate_discovery_open_four=candidate_probe.discovery_open_four,
                    candidate_discovery_open_five=candidate_probe.discovery_open_five,
                    candidate_critical_count=candidate_probe.critical_count,
                    candidate_critical_overflow_count=candidate_probe.critical_overflow_count,
                    candidate_critical_overflow_examples=candidate_probe.critical_overflow_examples,
                    sparse_prior_stage=self.sparse_prior_stage,
                    sparse_prior_root_candidate_count=int(
                        prior_summary.get("root_sparse_candidate_count", 0)
                    ),
                    sparse_prior_leaf_candidate_count=sparse_prior_leaf_candidate_count,
                    sparse_prior_root_hit_frac=_prior_source_fraction(
                        prior_summary, "sparse", "root"
                    ),
                    sparse_prior_leaf_hit_frac=_prior_source_fraction(
                        prior_summary, "sparse", "leaf"
                    ),
                    fallback_prior_use=fallback_prior_use,
                    fallback_prior_use_on_mcts_top1=_fallback_use_on_topk(
                        visits, root_prior_sources, 1
                    ),
                    fallback_prior_use_on_mcts_top4=_fallback_use_on_topk(
                        visits, root_prior_sources, 4
                    ),
                    fallback_prior_use_on_mcts_top8=_fallback_use_on_topk(
                        visits, root_prior_sources, 8
                    ),
                    sparse_vs_dense_disagreement=sparse_vs_dense_disagreement,
                    sparse_prior_forward_ms=sparse_prior_forward_ms,
                    sparse_prior_candidate_build_ms=sparse_prior_candidate_build_ms,
                    pair_prior_candidate_count=int(
                        prior_summary.get("root_pair_candidate_count", 0)
                    ),
                    pair_prior_hit_frac=pair_prior_hit_frac,
                    pair_fallback_prior_use=pair_fallback_prior_use,
                    pair_fallback_prior_use_on_mcts_top1=pair_fallback_prior_use_on_mcts_top1,
                    pair_fallback_prior_use_on_mcts_top4=pair_fallback_prior_use_on_mcts_top4,
                    pair_fallback_prior_use_on_mcts_top8=pair_fallback_prior_use_on_mcts_top8,
                    root_value=root_value,
                    selected_action_value=selected_action_value,
                    player=player,
                    game_id=game_id,
                    is_full_search=not use_pcr,
                    turn_index=move_idx,
                )
            )

            move_history.extend(player.to_bytes(4, "little", signed=True))
            move_history.extend(q.to_bytes(4, "little", signed=True))
            move_history.extend(r.to_bytes(4, "little", signed=True))

            move_idx += 1

            engine.re_root(q, r, sims)

            if engine.is_over:
                terminal_reason = "win"
                break

        if ENGINE_AVAILABLE:
            outcome = (
                1.0
                if engine.winner == 0
                else -1.0
                if engine.winner == 1
                else 0.0
            )
        else:
            winner = engine.winner
            if winner is None:
                outcome = 0.0
            elif winner == 0:
                outcome = 1.0
            else:
                outcome = -1.0

        full_history = bytes(move_history)
        truncated = terminal_reason != "win"
        record = GameRecord(
            positions=positions,
            outcome=outcome,
            game_id=game_id,
            game_length=len(positions),
        )
        if rgsc_restart is not None:
            record.rgsc_restart_attempted = rgsc_restart.attempted
            record.rgsc_restart_used = rgsc_restart.used
            record.rgsc_restart_reason = rgsc_restart.reason
            record.rgsc_restart_entry_index = rgsc_restart.entry_index
            record.rgsc_restart_entry_id = rgsc_restart.entry_id
            record.rgsc_restart_move_count = rgsc_restart.move_count
        record.rgsc_tree_node_insertions = int(self.rgsc.tree_node_insertions)
        record.final_move_history = full_history
        record.truncated = truncated
        record.terminal_reason = terminal_reason

        record.assign_outcomes()
        return record


def create_default_game_runner(
    *,
    cfg: Config,
    worker_id: int,
    output_queue,
    client,
    num_workers: int,
    max_batch_size: int,
    telemetry_sink: SelfPlayTelemetrySink | None = None,
) -> GameRunner:
    sp = cfg.selfplay
    model_spec = model_spec_from_config(cfg)
    runner_config = GameRunnerConfig(
        worker_id=int(worker_id),
        run_seed=int(cfg.run.seed),
        num_simulations=int(sp.mcts_simulations),
        max_game_moves=int(sp.max_game_moves),
        batch_size=int(sp.batch_size_per_worker),
        c_puct=float(sp.c_puct),
        c_puct_init=float(sp.c_puct_init),
        near_radius=int(sp.near_radius),
        constrain_threats=bool(sp.constrain_threats),
        temperature_schedule=tuple((float(a), float(b)) for a, b in sp.temperature_schedule),
        pcr_low_sim_prob=float(sp.pcr_low_sim_prob),
        pcr_low_sims=int(sp.pcr_low_sims),
        policy_target_top_k=int(sp.policy_target_top_k),
        dirichlet_alpha=float(sp.dirichlet_alpha),
        dirichlet_fraction=float(sp.dirichlet_fraction),
        sparse_prior_stage=int(getattr(cfg.model, "sparse_prior_stage", 0)),
        sparse_prior_mix=float(getattr(cfg.model, "sparse_prior_mix", 0.25)),
        sparse_policy_enabled=bool(getattr(cfg.model, "sparse_policy", False)),
        candidate_budget=max(int(getattr(cfg.model, "candidate_budget", 256)), int(sp.policy_target_top_k)),
        subtree_reuse=bool(getattr(sp, "subtree_reuse", False)),
        pair_strategy_name=str(getattr(cfg.model, "pair_strategy", PAIR_STRATEGY_NONE)).lower(),
        pair_strategy_max_pairs=int(getattr(cfg.model, "pair_strategy_max_pairs", 0)),
        configured_heads=tuple(str(item) for item in getattr(cfg.model, "heads", ())),
    )
    runtime_spec = RuntimeResourceSpec(
        worker_processes=int(num_workers),
        inference_queue_capacity=int(max_batch_size),
        record_queue_capacity=5000,
        leaf_batch_size=int(sp.batch_size_per_worker),
        max_in_flight_requests_per_worker=1,
        rust_threads=max(1, (os.cpu_count() or 1) // max(1, int(num_workers))),
        torch_threads=max(1, (os.cpu_count() or 1) // max(1, int(num_workers))),
        shutdown_timeout_s=2.0,
    )
    telemetry = telemetry_sink or LoggingSelfPlayTelemetrySink(logger)
    rgsc = RGSCRestartService(
        beta=float(getattr(sp, "rgsc_beta", 0.0)),
        capacity=int(getattr(sp, "rgsc_prb_capacity", 100)),
        ema_alpha=float(getattr(sp, "rgsc_prb_ema_alpha", 0.5)),
        sampling_temperature=float(getattr(sp, "rgsc_prb_temperature", 0.1)),
        seed=int(cfg.run.seed + worker_id * 10000),
        enabled=ENGINE_AVAILABLE,
    )
    writer = QueueSelfPlayRecordWriter(
        output_queue,
        lookahead_horizons=list(cfg.buffer.lookahead_horizons),
        lookahead_lambdas=list(cfg.buffer.lookahead_lambdas),
        telemetry_sink=telemetry,
        rgsc_service=rgsc,
    )
    pair_strategy = create_pair_strategy(_pair_strategy_spec_from_config(runner_config))
    policy_provider = create_policy_provider(model_spec=model_spec, client=client)
    return GameRunner(
        policy_provider=policy_provider,
        pair_strategy=pair_strategy,
        engine_adapter_factory=create_engine_adapter,
        record_writer=writer,
        telemetry_sink=telemetry,
        contract_builders=SelfPlayContractBuilders(),
        runtime_spec=runtime_spec,
        runner_config=runner_config,
        model_spec=model_spec,
        rgsc_service=rgsc,
    )


def _pair_strategy_spec_from_config(config: GameRunnerConfig) -> PairStrategySpec:
    pair_strategy = str(config.pair_strategy_name).lower()
    max_pairs = int(config.pair_strategy_max_pairs)
    if pair_strategy == PAIR_STRATEGY_NONE:
        return PairStrategySpec()
    if pair_strategy in {"two_stage_root_only", "two_stage_root"}:
        return PairStrategySpec(
            name="two_stage_root_only",
            root_enabled=True,
            leaf_enabled=False,
            max_root_pair_rows=max_pairs,
            chunk_size=0 if max_pairs <= 0 else min(max_pairs, MAX_GRAPH_PAIRS),
        )
    if pair_strategy in {"tactical_only", "tactical"}:
        return PairStrategySpec(
            name="tactical_only",
            root_enabled=True,
            leaf_enabled=False,
            max_root_pair_rows=max_pairs,
            chunk_size=0 if max_pairs <= 0 else min(max_pairs, MAX_PAIR_CANDIDATES),
        )
    if pair_strategy in {PAIR_STRATEGY_DIAGNOSTIC_FULL_PAIR, "diagnostic_full_root"}:
        return PairStrategySpec(
            name="diagnostic_full_root",
            diagnostic=True,
            root_enabled=True,
            leaf_enabled=False,
            max_full_pair_rows=max_pairs,
            chunk_size=0 if max_pairs <= 0 else min(max_pairs, MAX_GRAPH_PAIRS),
        )
    raise ValueError(f"unsupported explicit pair strategy {pair_strategy!r}")
