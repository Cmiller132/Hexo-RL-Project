"""Self-play worker — plays games using MCTSEngine + InferenceClient.

Each worker is a separate multiprocessing.Process. It:
  1. Connects to the inference server via SharedMemory.
  2. Plays games in a loop using the MCTS engine.
  3. Pushes completed game records to the buffer queue.
  4. Handles temperature schedule and terminal detection.
  5. Falls back to MockMCTSEngine when the Rust extension is unavailable.
"""

import time
import queue
import logging
import multiprocessing as mp
import signal
import numpy as np
from dataclasses import replace
from typing import Optional, List, Tuple

try:
    import _engine

    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

from hexorl.config import Config
from hexorl.action_contract.candidates import build_candidate_batch, build_pair_candidate_batch
from hexorl.action_contract.tactical_oracle import (
    scan_tactical_oracle_from_game,
    scan_tactical_oracle_from_history,
)
from hexorl.inference.client import InferenceClient
from hexorl.inference.shm_queue import MAX_GRAPH_PAIRS
from hexorl.graph.batch import GraphBatch, GraphTokenType, build_graph_batch_from_history
from hexorl.selfplay.rgsc import RGSCRestartService, encode_move_history
from hexorl.selfplay.records import (
    GameRecord,
    PositionRecord,
    action_to_board_index,
    dense_policy_from_v2,
    policy_v2_from_visits,
)
from hexorl.buffer.targets import process_game_record

logger = logging.getLogger(__name__)

PRIOR_SOURCE_SPARSE = 1
PRIOR_SOURCE_DENSE = 2
PRIOR_SOURCE_DEFAULT = 3
PRIOR_SOURCE_PAIR = 4


def _graph_batch_with_pair_rows(
    graph_batch: GraphBatch,
    pair_first_indices: np.ndarray,
    pair_second_indices: np.ndarray,
) -> GraphBatch:
    pair_count = int(pair_first_indices.shape[0])
    return replace(
        graph_batch,
        pair_token_indices=np.zeros(pair_count, dtype=np.int64),
        pair_first_indices=np.asarray(pair_first_indices, dtype=np.int64),
        pair_second_indices=np.asarray(pair_second_indices, dtype=np.int64),
        pair_policy_target=np.zeros(pair_count, dtype=np.float32),
    )


def _graph_stone_token_for_qr(graph_batch: GraphBatch, qr: tuple[int, int]) -> int:
    q, r = int(qr[0]), int(qr[1])
    for idx, (token_q, token_r) in enumerate(np.asarray(graph_batch.token_qr, dtype=np.int32).tolist()):
        if (
            int(token_q) == q
            and int(token_r) == r
            and int(graph_batch.token_type[idx]) == int(GraphTokenType.STONE)
        ):
            return idx
    raise ValueError(f"first placement {qr} is not present as a graph STONE token")


def _score_graph_pair_chunks(
    client: InferenceClient,
    graph_batch: GraphBatch,
    *,
    second_placement: bool,
    first_qr: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every legal pair row while using the IPC pair table as a chunk."""
    legal = np.asarray(graph_batch.legal_qr, dtype=np.int32)
    legal_tokens = np.asarray(graph_batch.legal_token_indices, dtype=np.int64)
    if legal.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.int32), np.zeros(0, dtype=np.float32)

    pair_qr_chunks: list[np.ndarray] = []
    logit_chunks: list[np.ndarray] = []

    if second_placement:
        if first_qr is None:
            raise ValueError("second-placement pair scoring requires first_qr")
        first_token = _graph_stone_token_for_qr(graph_batch, first_qr)
        first = np.asarray(first_qr, dtype=np.int32)
        for start in range(0, legal.shape[0], MAX_GRAPH_PAIRS):
            stop = min(start + MAX_GRAPH_PAIRS, legal.shape[0])
            width = stop - start
            pair_first = np.full(width, first_token, dtype=np.int64)
            pair_second = legal_tokens[start:stop]
            chunk = _graph_batch_with_pair_rows(graph_batch, pair_first, pair_second)
            out = client.submit_graph(chunk)
            logits = np.asarray(out.get("policy_pair_joint", []), dtype=np.float32)[:width]
            pair_qr = np.column_stack([
                np.full(width, int(first[0]), dtype=np.int32),
                np.full(width, int(first[1]), dtype=np.int32),
                legal[start:stop, 0],
                legal[start:stop, 1],
            ])
            pair_qr_chunks.append(pair_qr)
            logit_chunks.append(logits)
    else:
        first_rows: list[int] = []
        second_rows: list[int] = []
        qr_rows: list[tuple[int, int, int, int]] = []
        for a_idx in range(legal.shape[0]):
            for b_idx in range(a_idx + 1, legal.shape[0]):
                first_rows.append(int(legal_tokens[a_idx]))
                second_rows.append(int(legal_tokens[b_idx]))
                qr_rows.append((
                    int(legal[a_idx, 0]),
                    int(legal[a_idx, 1]),
                    int(legal[b_idx, 0]),
                    int(legal[b_idx, 1]),
                ))
                if len(first_rows) == MAX_GRAPH_PAIRS:
                    chunk = _graph_batch_with_pair_rows(
                        graph_batch,
                        np.asarray(first_rows, dtype=np.int64),
                        np.asarray(second_rows, dtype=np.int64),
                    )
                    out = client.submit_graph(chunk)
                    logit_chunks.append(np.asarray(out.get("policy_pair_joint", []), dtype=np.float32)[: len(first_rows)])
                    pair_qr_chunks.append(np.asarray(qr_rows, dtype=np.int32))
                    first_rows.clear()
                    second_rows.clear()
                    qr_rows.clear()
        if first_rows:
            chunk = _graph_batch_with_pair_rows(
                graph_batch,
                np.asarray(first_rows, dtype=np.int64),
                np.asarray(second_rows, dtype=np.int64),
            )
            out = client.submit_graph(chunk)
            logit_chunks.append(np.asarray(out.get("policy_pair_joint", []), dtype=np.float32)[: len(first_rows)])
            pair_qr_chunks.append(np.asarray(qr_rows, dtype=np.int32))

    if not pair_qr_chunks:
        return np.zeros((0, 4), dtype=np.int32), np.zeros(0, dtype=np.float32)
    return np.concatenate(pair_qr_chunks, axis=0), np.concatenate(logit_chunks, axis=0)


# ── Temperature Schedule ─────────────────────────────────────────────────────

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


def _sparse_dense_disagreement(
    policy_logits: np.ndarray,
    sparse_qr: np.ndarray,
    sparse_logits: np.ndarray,
    legal: np.ndarray,
    offset_q: int,
    offset_r: int,
) -> float:
    if legal.size == 0 or sparse_qr.size == 0 or sparse_logits.size == 0:
        return 0.0
    legal_set = {(int(q), int(r)) for q, r in legal}
    dense_best: tuple[int, int] | None = None
    dense_score = float("-inf")
    flat_logits = np.asarray(policy_logits, dtype=np.float32).reshape(-1)
    for q_raw, r_raw in legal:
        q, r = int(q_raw), int(r_raw)
        idx = action_to_board_index(q, r, offset_q, offset_r)
        score = float(flat_logits[idx]) if idx >= 0 else -10.0
        if score > dense_score:
            dense_score = score
            dense_best = (q, r)

    sparse_best: tuple[int, int] | None = None
    sparse_score = float("-inf")
    for row, logit in zip(np.asarray(sparse_qr), np.asarray(sparse_logits).reshape(-1)):
        q, r = int(row[0]), int(row[1])
        if (q, r) not in legal_set:
            continue
        score = float(logit)
        if score > sparse_score:
            sparse_score = score
            sparse_best = (q, r)
    if dense_best is None or sparse_best is None:
        return 0.0
    return 0.0 if dense_best == sparse_best else 1.0


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


# ── Mock MCTS Engine ─────────────────────────────────────────────────────────

class MockMCTSEngine:
    """Plausible mock of the Rust MCTSEngine for pipeline testing.

    Generates random visit distributions, legal moves, and terminal signals.
    Simulates realistic game lengths (20-80 moves) and branching factors.
    """

    NUM_CHANNELS = 13
    BOARD_SIZE = 33
    BOARD_AREA = 33 * 33  # 1089

    def __init__(
        self,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        near_radius: int = 8,
        seed: int = 0,
    ):
        self.num_simulations = num_simulations
        self.near_radius = near_radius
        self._rng = np.random.RandomState(seed)
        self._move_count = 0
        self._max_moves = self._rng.randint(20, 80)
        self._is_over = False
        self._winner: Optional[int] = None
        self._num_children = self._rng.randint(5, 25)
        self._visits: Optional[np.ndarray] = None
        self._priors: Optional[np.ndarray] = None
        self._q_values: Optional[np.ndarray] = None
        self._root_value = 0.0
        self._sims_done = 0
        self._batch_size = 4

    def init_root(self) -> Optional[Tuple[np.ndarray, int, int, bytes]]:
        """Return a mock (13,33,33) tensor, offsets, and legal moves bytes."""
        if self._is_over:
            return None

        tensor = self._rng.randn(
            self.NUM_CHANNELS, self.BOARD_SIZE, self.BOARD_SIZE
        ).astype(np.float32)

        offset_q, offset_r = 16, 16

        num_legal = self._rng.randint(5, 25)
        legal_bytes = bytearray()
        for _ in range(num_legal):
            q = self._rng.randint(-8, 9)
            r = self._rng.randint(-8, 9)
            legal_bytes.extend(q.to_bytes(4, "little", signed=True))
            legal_bytes.extend(r.to_bytes(4, "little", signed=True))

        return tensor, offset_q, offset_r, bytes(legal_bytes)

    def expand_root(
        self,
        policy: np.ndarray,
        value: float,
        offset_q: int,
        offset_r: int,
        legal_bytes: bytes,
    ):
        """Mock root expansion with random priors."""
        legal = np.frombuffer(legal_bytes, dtype=np.int32).reshape(-1, 2)
        n = len(legal)
        self._num_children = n
        self._priors = np.random.dirichlet([0.3] * n).astype(np.float32)
        self._root_value = value
        self._visits = np.zeros(n, dtype=np.uint32)

    def expand_root_with_sparse_priors(
        self,
        policy: np.ndarray,
        value: float,
        offset_q: int,
        offset_r: int,
        legal_bytes: bytes,
        sparse_qr: np.ndarray,
        sparse_logits: np.ndarray,
        stage: int,
        sparse_mix: float,
    ):
        """Mock sparse root expansion by preserving dense mock semantics."""
        self.expand_root(policy, value, offset_q, offset_r, legal_bytes)

    def expand_root_with_global_priors(self, legal_bytes: bytes, global_qr: np.ndarray, global_logits: np.ndarray, value: float):
        """Mock graph root expansion from keyed global priors."""
        legal = np.frombuffer(legal_bytes, dtype=np.int32).reshape(-1, 2)
        if legal.shape != np.asarray(global_qr).shape or not np.array_equal(legal, np.asarray(global_qr, dtype=np.int32)):
            raise ValueError("mock global priors legal_qr does not match legal_bytes")
        self._num_children = int(legal.shape[0])
        logits = np.asarray(global_logits, dtype=np.float32)[: self._num_children]
        logits = logits - np.max(logits) if logits.size else logits
        exp = np.exp(logits)
        self._priors = (exp / max(float(exp.sum()), 1e-6)).astype(np.float32)
        self._root_value = float(value)
        self._visits = np.zeros(self._num_children, dtype=np.uint32)

    def apply_root_pair_priors(
        self,
        pair_qr: np.ndarray,
        pair_logits: np.ndarray,
        pair_mix: float,
    ):
        """Mock pair-prior hook; records no active pair source."""
        return None

    def apply_root_pair_second_priors(
        self,
        pair_qr: np.ndarray,
        pair_logits: np.ndarray,
        pair_mix: float,
    ):
        """Mock conditional pair-prior hook; records no active pair source."""
        return None

    def add_dirichlet_noise(self, noise: np.ndarray, fraction: float):
        """Mock Dirichlet noise (no-op in mock)."""
        pass

    def done(self) -> bool:
        """Check if MCTS simulations are complete."""
        return self._sims_done >= self.num_simulations

    def select_leaves(self, batch_size: int) -> Tuple[np.ndarray, int]:
        """Return a mock batch of (count, 13, 33, 33) tensors."""
        count = min(batch_size, self.num_simulations - self._sims_done)
        count = min(count, self._batch_size)
        if count == 0:
            self._sims_done = self.num_simulations
            return np.zeros((0, 13, 33, 33), dtype=np.float32), 0

        self._sims_done += count
        tensors = self._rng.randn(
            count, self.NUM_CHANNELS, self.BOARD_SIZE, self.BOARD_SIZE
        ).astype(np.float32)
        return tensors, count

    def expand_and_backprop(self, policies: np.ndarray, values: np.ndarray):
        """Mock backpropagation."""
        pass

    def pending_leaf_metadata(self) -> List[Tuple[int, int, bytes]]:
        """Mock leaf metadata; dense fallback remains valid for mock tests."""
        return []

    def expand_and_backprop_with_sparse(
        self,
        policies: np.ndarray,
        values: np.ndarray,
        sparse_qr: np.ndarray,
        sparse_logits: np.ndarray,
        sparse_counts: np.ndarray,
        stage: int,
        sparse_mix: float,
    ):
        """Mock sparse backpropagation by preserving dense mock semantics."""
        self.expand_and_backprop(policies, values)

    def get_results(self) -> Tuple[List[int], List[int], List[int], float]:
        """Return mock (moves_q, moves_r, visits, root_value)."""
        n = self._num_children
        self._visits = self._rng.randint(0, 100, size=n).astype(np.uint32)
        self._q_values = self._rng.uniform(-1, 1, size=n).astype(np.float32)

        moves_q = []
        moves_r = []
        for _ in range(n):
            moves_q.append(int(self._rng.randint(-8, 9)))
            moves_r.append(int(self._rng.randint(-8, 9)))

        visits = self._visits.tolist()
        root_value = float(self._rng.uniform(-1, 1))
        return moves_q, moves_r, [int(v) for v in visits], root_value

    def sample_action(
        self, temperature: float, rng_state: Optional[int] = None
    ) -> Tuple[int, int]:
        """Sample a mock action."""
        return int(self._rng.randint(-8, 9)), int(self._rng.randint(-8, 9))

    def re_root(self, q: int, r: int, new_sims: int) -> bool:
        """Mock tree advancement."""
        self._move_count += 1
        self._sims_done = 0
        self._num_children = self._rng.randint(5, 25)
        self._visits = None
        self._priors = None
        if self._move_count >= self._max_moves:
            self._is_over = True
            self._winner = self._rng.randint(0, 2)
        return True

    def root_child_priors(self) -> np.ndarray:
        """Return mock priors."""
        if self._priors is None:
            self._priors = np.random.dirichlet(
                [0.3] * max(1, self._num_children)
            ).astype(np.float32)
        return np.array(self._priors)

    def root_child_prior_sources(self) -> List[int]:
        """Return mock prior sources; mock policy is dense/default only."""
        return [PRIOR_SOURCE_DENSE] * max(0, int(self._num_children))

    def prior_source_summary(self) -> dict:
        """Return mock prior-source counters compatible with the Rust engine."""
        n = max(0, int(self._num_children))
        return {
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

    def root_child_q_values(self) -> List[float]:
        """Return mock Q-values."""
        if self._q_values is None:
            self._q_values = self._rng.uniform(
                -1, 1, size=max(1, self._num_children)
            ).astype(np.float32)
        return self._q_values.tolist()

    def root_pair_visit_targets(self):
        return []

    def move_history_bytes(self) -> bytes:
        """Return mock packed move history."""
        n_bytes = self._move_count * 12
        return self._rng.bytes(n_bytes)

    def extract_tree_node_histories(self, min_visits: int = 2) -> list[list[tuple[int, int, int]]]:
        """Mock engine has no real tree-node states."""
        return []

    @property
    def winner(self) -> Optional[int]:
        return self._winner

    @property
    def is_over(self) -> bool:
        return self._is_over


# ── Real MCTS Engine Wrapper ─────────────────────────────────────────────────

class RealMCTSEngine:
    """Wrapper around the Rust PyMCTSEngine for type-compatible API."""

    NUM_CHANNELS = 13
    BOARD_SIZE = 33
    BOARD_AREA = 33 * 33

    def __init__(
        self,
        game,
        num_simulations: int,
        c_puct: float,
        near_radius: int,
        seed: int,
        c_puct_init: float = 19652.0,
        constrain_threats: bool = True,
        subtree_reuse: bool = False,
    ):
        self._num_simulations = num_simulations
        self._c_puct = c_puct
        self._near_radius = near_radius
        self._c_puct_init = c_puct_init
        self._constrain_threats = constrain_threats
        self._seed = seed
        self._subtree_reuse = subtree_reuse
        self._engine = _engine.MCTSEngine(
            game=game,
            num_simulations=num_simulations,
            c_puct=c_puct,
            near_radius=near_radius,
            c_puct_init=c_puct_init,
            constrain_threats=constrain_threats,
            seed=seed,
        )
        self._game = game

    def init_root(self):
        init = self._engine.init_root()
        if init is None:
            return None
        tensor_3d, oq, or_, legal_bytes = init
        return (
            np.asarray(tensor_3d, dtype=np.float32),
            oq,
            or_,
            legal_bytes,
        )

    def expand_root(self, policy, value, oq, or_, legal_bytes):
        self._engine.expand_root(policy, value, oq, or_, legal_bytes)

    def expand_root_with_sparse_priors(
        self,
        policy,
        value,
        oq,
        or_,
        legal_bytes,
        sparse_qr,
        sparse_logits,
        stage,
        sparse_mix,
    ):
        if hasattr(self._engine, "expand_root_with_sparse_priors"):
            self._engine.expand_root_with_sparse_priors(
                policy,
                value,
                oq,
                or_,
                legal_bytes,
                np.asarray(sparse_qr, dtype=np.int32),
                np.asarray(sparse_logits, dtype=np.float32),
                int(stage),
                float(sparse_mix),
            )
        else:
            self._engine.expand_root(policy, value, oq, or_, legal_bytes)

    def expand_root_with_global_priors(self, legal_bytes, global_qr, global_logits, value):
        if not hasattr(self._engine, "expand_root_with_global_priors"):
            raise RuntimeError("Rust engine does not expose expand_root_with_global_priors")
        self._engine.expand_root_with_global_priors(
            legal_bytes,
            np.asarray(global_qr, dtype=np.int32),
            np.asarray(global_logits, dtype=np.float32),
            float(value),
        )

    def apply_root_pair_priors(self, pair_qr, pair_logits, pair_mix):
        if hasattr(self._engine, "apply_root_pair_priors"):
            self._engine.apply_root_pair_priors(
                np.asarray(pair_qr, dtype=np.int32),
                np.asarray(pair_logits, dtype=np.float32),
                float(pair_mix),
            )

    def apply_root_pair_second_priors(self, pair_qr, pair_logits, pair_mix):
        if hasattr(self._engine, "apply_root_pair_second_priors"):
            self._engine.apply_root_pair_second_priors(
                np.asarray(pair_qr, dtype=np.int32),
                np.asarray(pair_logits, dtype=np.float32),
                float(pair_mix),
            )

    def add_dirichlet_noise(self, noise, fraction):
        self._engine.add_dirichlet_noise(noise, fraction)

    def done(self):
        return self._engine.done()

    def select_leaves(self, batch_size):
        tensor_4d, count = self._engine.select_leaves(batch_size)
        return np.asarray(tensor_4d, dtype=np.float32), count

    def pending_leaf_metadata(self):
        if hasattr(self._engine, "pending_leaf_metadata"):
            return self._engine.pending_leaf_metadata()
        return []

    def expand_and_backprop(self, policies, values):
        self._engine.expand_and_backprop(policies, values)

    def expand_and_backprop_with_sparse(
        self,
        policies,
        values,
        sparse_qr,
        sparse_logits,
        sparse_counts,
        stage,
        sparse_mix,
    ):
        if hasattr(self._engine, "expand_and_backprop_with_sparse"):
            self._engine.expand_and_backprop_with_sparse(
                policies,
                values,
                np.asarray(sparse_qr, dtype=np.int32),
                np.asarray(sparse_logits, dtype=np.float32),
                np.asarray(sparse_counts, dtype=np.uint16),
                int(stage),
                float(sparse_mix),
            )
        else:
            self._engine.expand_and_backprop(policies, values)

    def get_results(self):
        return self._engine.get_results()

    def sample_action(self, temperature, rng_state=None):
        return self._engine.sample_action(temperature, rng_state)

    def re_root(self, q, r, new_sims):
        if self._subtree_reuse:
            self._engine.re_root(q, r, new_sims)
            self._game.place(q, r)
            return True
        self._game.place(q, r)
        self._num_simulations = new_sims
        self._engine = _engine.MCTSEngine(
            game=self._game,
            num_simulations=new_sims,
            c_puct=self._c_puct,
            near_radius=self._near_radius,
            c_puct_init=self._c_puct_init,
            constrain_threats=self._constrain_threats,
            seed=self._seed,
        )
        return True

    def root_child_priors(self):
        return self._engine.root_child_priors()

    def root_child_prior_sources(self):
        if hasattr(self._engine, "root_child_prior_sources"):
            return self._engine.root_child_prior_sources()
        priors = self.root_child_priors()
        return [PRIOR_SOURCE_DENSE] * len(priors)

    def prior_source_summary(self):
        if hasattr(self._engine, "prior_source_summary"):
            return dict(self._engine.prior_source_summary())
        n = len(self.root_child_prior_sources())
        return {
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

    def root_child_q_values(self):
        return self._engine.root_child_q_values()

    def root_pair_visit_targets(self):
        if hasattr(self._engine, "root_pair_visit_targets"):
            return self._engine.root_pair_visit_targets()
        return []

    def extract_tree_node_histories(self, min_visits: int = 2):
        if hasattr(self._engine, "extract_tree_node_states"):
            _tensors, histories, _count = self._engine.extract_tree_node_states(int(min_visits))
            return histories
        return []

    def move_history_bytes(self):
        return self._game.move_history_bytes()

    @property
    def winner(self):
        return self._game.winner

    @property
    def is_over(self):
        return self._game.is_over


# ── Worker ───────────────────────────────────────────────────────────────────

class SelfPlayWorker:
    """A single self-play worker that plays games and pushes records."""

    def __init__(
        self,
        worker_id: int,
        cfg: Config,
        record_queue: mp.Queue,
        num_workers: int = 24,
        max_batch_size: int = 128,
        stop_event: Optional[mp.Event] = None,
    ):
        self.worker_id = worker_id
        self.cfg = cfg
        self.record_queue = record_queue
        self.stop_event = stop_event
        self.num_workers = num_workers
        self.max_batch = max_batch_size

        sp = cfg.selfplay
        self.num_simulations = sp.mcts_simulations
        self.max_game_moves = sp.max_game_moves
        self.batch_size = sp.batch_size_per_worker
        self.c_puct = sp.c_puct
        self.c_puct_init = sp.c_puct_init
        self.near_radius = sp.near_radius
        self.constrain_threats = sp.constrain_threats
        self.temperature_schedule = sp.temperature_schedule
        self.pcr_low_sim_prob = sp.pcr_low_sim_prob
        self.pcr_low_sims = sp.pcr_low_sims
        self.policy_target_top_k = sp.policy_target_top_k
        self.dirichlet_alpha = sp.dirichlet_alpha
        self.dirichlet_fraction = sp.dirichlet_fraction
        self.sparse_prior_stage = int(getattr(cfg.model, "sparse_prior_stage", 0))
        self.sparse_prior_mix = float(getattr(cfg.model, "sparse_prior_mix", 0.25))
        self.sparse_policy_enabled = bool(getattr(cfg.model, "sparse_policy", False))
        self.global_graph_enabled = str(getattr(cfg.model, "architecture", "")).lower().startswith("global_")
        if self.global_graph_enabled:
            self.near_radius = 8
            self.constrain_threats = False
        self.pair_prior_mix = float(getattr(cfg.model, "pair_prior_mix", 0.35))
        configured_heads = set(getattr(cfg.model, "heads", []))
        self.pair_policy_enabled = bool(
            (self.global_graph_enabled and self.pair_prior_mix > 0.0)
            or ({"pair_policy", "policy_pair_first", "policy_pair_second", "policy_pair_joint"} & configured_heads)
        )
        self.candidate_budget = max(
            int(getattr(cfg.model, "candidate_budget", 256)),
            int(sp.policy_target_top_k),
        )
        self.rgsc = RGSCRestartService(
            beta=float(getattr(sp, "rgsc_beta", 0.0)),
            capacity=int(getattr(sp, "rgsc_prb_capacity", 100)),
            ema_alpha=float(getattr(sp, "rgsc_prb_ema_alpha", 0.5)),
            sampling_temperature=float(getattr(sp, "rgsc_prb_temperature", 0.1)),
            seed=int(cfg.run.seed + worker_id * 10000),
            enabled=HAS_ENGINE,
        )

        self._engine_factory = MockMCTSEngine if not HAS_ENGINE else RealMCTSEngine
        self._game_counter = 0
        self._crash_count = 0

    def run(self):
        """Main worker loop — runs in a separate multiprocessing.Process."""
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        logger.info(
            f"Worker {self.worker_id} starting (engine={'rust' if HAS_ENGINE else 'mock'})"
        )

        client = InferenceClient(
            worker_id=self.worker_id,
            num_workers=self.num_workers,
            max_batch_size=self.max_batch,
            timeout_ms=30000,
        )

        try:
            try:
                client.connect()
            except Exception as exc:
                logger.warning(
                    f"Worker {self.worker_id}: inference server not available, using mock evaluations: {exc}"
                )
                client = None

            while self.stop_event is None or not self.stop_event.is_set():
                try:
                    game_record = self._play_one_game(client)
                    if game_record is not None and len(game_record.positions) > 0:
                        process_game_record(
                            game_record,
                            lookahead_horizons=self.cfg.buffer.lookahead_horizons,
                            lookahead_lambdas=self.cfg.buffer.lookahead_lambdas,
                        )
                        restart_idx = getattr(game_record, "rgsc_restart_entry_index", None)
                        refreshes_before = self.rgsc.refreshes
                        inserted = self.rgsc.observe_game(
                            game_record,
                            restart_entry_index=restart_idx,
                        )
                        game_record.rgsc_prb_inserted = bool(inserted)
                        game_record.rgsc_metrics = {
                            "rgsc_prb_size": float(len(self.rgsc.prb)),
                            "rgsc_restart_attempts": 1.0
                            if getattr(game_record, "rgsc_restart_attempted", False)
                            else 0.0,
                            "rgsc_restart_successes": 1.0
                            if getattr(game_record, "rgsc_restart_used", False)
                            else 0.0,
                            "rgsc_restart_rejections": 1.0
                            if (
                                getattr(game_record, "rgsc_restart_attempted", False)
                                and not getattr(game_record, "rgsc_restart_used", False)
                            )
                            else 0.0,
                            "rgsc_prb_insertions": 1.0 if inserted else 0.0,
                            "rgsc_prb_refreshes": float(self.rgsc.refreshes - refreshes_before),
                            "rgsc_last_ema_delta": float(self.rgsc.last_ema_delta),
                            "rgsc_last_staleness": float(self.rgsc.last_staleness),
                            "rgsc_tree_node_insertions": float(
                                getattr(game_record, "rgsc_tree_node_insertions", 0)
                            ),
                        }
                        game_record.rgsc_prb_snapshot = self.rgsc.snapshot_entries()
                        while self.stop_event is None or not self.stop_event.is_set():
                            try:
                                self.record_queue.put(game_record, timeout=0.5)
                                break
                            except queue.Full:
                                logger.warning(
                                    f"Worker {self.worker_id}: record queue full, retrying..."
                                )
                        self._game_counter += 1
                except queue.Full:
                    logger.warning(
                        f"Worker {self.worker_id}: record queue full, retrying..."
                    )
                    time.sleep(0.1)
                except Exception as e:
                    self._crash_count += 1
                    logger.error(
                        f"Worker {self.worker_id} crash #{self._crash_count}: {e}"
                    )
                    time.sleep(0.5)
                    if self._crash_count > 10:
                        logger.critical(
                            f"Worker {self.worker_id} exceeded max crashes, stopping"
                        )
                        break
        finally:
            if client is not None:
                client.disconnect()

    def _play_one_game(
        self, client: Optional[InferenceClient]
    ) -> Optional[GameRecord]:
        """Play one complete self-play game.

        Returns a GameRecord with all position data, or None on failure.
        """
        use_pcr = np.random.random() < self.pcr_low_sim_prob
        sims = self.pcr_low_sims if use_pcr else self.num_simulations
        game_seed = (
            self.cfg.run.seed + self.worker_id * 10000 + self._game_counter
        )

        game_id = self._game_id()
        rgsc_restart = None
        if HAS_ENGINE:
            rgsc_restart = self.rgsc.maybe_restart(
                _engine.HexGame,
                max_game_moves=self.max_game_moves,
            )
            game = rgsc_restart.game if rgsc_restart.used else _engine.HexGame()
            engine = RealMCTSEngine(
                game,
                sims,
                self.c_puct,
                self.near_radius,
                game_seed,
                c_puct_init=self.c_puct_init,
                constrain_threats=self.constrain_threats,
                subtree_reuse=getattr(self.cfg.selfplay, "subtree_reuse", False),
            )
        else:
            engine = MockMCTSEngine(
                sims, self.c_puct, self.near_radius, game_seed
            )

        positions: List[PositionRecord] = []
        move_history = bytearray(rgsc_restart.move_history if rgsc_restart and rgsc_restart.used else b"")
        move_idx = int(rgsc_restart.move_count) if rgsc_restart and rgsc_restart.used else 0
        terminal_reason = "unknown"

        while True:
            if move_idx >= self.max_game_moves:
                terminal_reason = "max_game_moves"
                break
            init = engine.init_root()
            if init is None:
                terminal_reason = "no_root"
                break

            tensor, offset_q, offset_r, legal_bytes = init
            if isinstance(tensor, np.ndarray):
                tensor_3d = tensor
            else:
                tensor_3d = np.array(tensor)

            sparse_prior_forward_ms = 0.0
            sparse_prior_candidate_build_ms = 0.0
            sparse_vs_dense_disagreement = 0.0

            if client is not None:
                try:
                    root_tensor = tensor_3d.reshape(1, 13, 33, 33).astype(np.float32, copy=False)
                    if self.global_graph_enabled:
                        graph_batch = build_graph_batch_from_history(
                            bytes(move_history),
                            radius=8,
                            max_pair_rows=0,
                            include_pair_rows=False,
                        )
                        graph_out = client.submit_graph(graph_batch)
                        graph_legal = np.asarray(graph_out["metadata"]["legal_qr"], dtype=np.int32)
                        graph_legal_bytes = _legal_bytes_from_qr(graph_legal)
                        graph_value = float(np.asarray(graph_out["value"], dtype=np.float32)[0])
                        engine.expand_root_with_global_priors(
                            graph_legal_bytes,
                            graph_legal,
                            np.asarray(graph_out["policy_place"], dtype=np.float32),
                            graph_value,
                        )
                        root_placements_remaining = (
                            int(getattr(engine._game, "placements_remaining", 1))
                            if HAS_ENGINE and hasattr(engine, "_game")
                            else int(graph_batch.placements_remaining)
                        )
                        if self.pair_policy_enabled:
                            first_qr = _last_move_qr(move_history) if root_placements_remaining == 1 else None
                            pair_qr, pair_logits = _score_graph_pair_chunks(
                                client,
                                graph_batch,
                                second_placement=root_placements_remaining == 1,
                                first_qr=first_qr,
                            )
                        else:
                            pair_qr = np.zeros((0, 4), dtype=np.int32)
                            pair_logits = np.zeros(0, dtype=np.float32)
                        if self.pair_policy_enabled and pair_qr.shape[0] > 0 and pair_logits.shape[0] >= pair_qr.shape[0]:
                            if root_placements_remaining == 1 and hasattr(engine, "apply_root_pair_second_priors"):
                                engine.apply_root_pair_second_priors(
                                    pair_qr,
                                    pair_logits[: pair_qr.shape[0]],
                                    self.pair_prior_mix,
                                )
                            elif root_placements_remaining >= 2:
                                engine.apply_root_pair_priors(
                                    pair_qr,
                                    pair_logits[: pair_qr.shape[0]],
                                    self.pair_prior_mix,
                                )
                    else:
                        use_action_keyed_root = (
                            (self.sparse_policy_enabled and self.sparse_prior_stage > 0)
                            or self.pair_policy_enabled
                        )
                        if use_action_keyed_root:
                            legal = np.frombuffer(legal_bytes, dtype=np.int32).reshape(-1, 2)
                            winning_moves, forced_blocks, cover_cells = _critical_actions_from_root_tensor(
                                tensor_3d,
                                legal,
                                int(offset_q),
                                int(offset_r),
                            )
                            root_game = getattr(engine, "_game", None)
                            if root_game is not None:
                                oracle = scan_tactical_oracle_from_game(
                                    root_game,
                                    [(int(q), int(r)) for q, r in legal],
                                    offset_q=int(offset_q),
                                    offset_r=int(offset_r),
                                )
                            else:
                                oracle = scan_tactical_oracle_from_history(
                                    bytes(move_history),
                                    [(int(q), int(r)) for q, r in legal],
                                    offset_q=int(offset_q),
                                    offset_r=int(offset_r),
                                )
                            root_placements_remaining = (
                                int(getattr(engine._game, "placements_remaining", 1))
                                if HAS_ENGINE and hasattr(engine, "_game")
                                else (2 if move_idx > 0 and move_idx % 2 == 1 else 1)
                            )
                            candidate_budget = self.candidate_budget
                            if self.pair_policy_enabled:
                                if legal.shape[0] > 512:
                                    raise ValueError(
                                        "pair_policy root prior requires all legal rows in the current IPC path; "
                                        f"legal_count={legal.shape[0]} exceeds 512"
                                    )
                                candidate_budget = int(legal.shape[0])
                            t_build = time.monotonic()
                            cand = build_candidate_batch(
                                [(int(q), int(r)) for q, r in legal],
                                [],
                                offset_q=int(offset_q),
                                offset_r=int(offset_r),
                                budget=candidate_budget,
                                storage_width=candidate_budget,
                                winning_moves=list(winning_moves) + list(oracle.win_now_cells),
                                forced_block_moves=list(forced_blocks) + list(oracle.forced_block_cells),
                                cover_cells=list(cover_cells) + list(oracle.cover_cells),
                                open_four_cells=oracle.open_four_cells,
                                open_five_cells=oracle.open_five_cells,
                            )
                            active_rows = np.flatnonzero(cand.mask)
                            active_width = int(active_rows.shape[0])
                            sparse_prior_candidate_build_ms += (time.monotonic() - t_build) * 1000.0
                            if active_width <= 0:
                                p, v = client.submit(root_tensor, 1)
                                engine.expand_root(
                                    p, v[0], offset_q, offset_r, legal_bytes
                                )
                            else:
                                root_candidate_qr = cand.qr[active_rows]
                                forward_indices = cand.indices[active_rows]
                                forward_features = cand.features[active_rows]
                                forward_mask = cand.mask[active_rows]
                                pair_indices_for_forward = None
                                pair_mask_for_forward = None
                                pair_qr = np.zeros((0, 4), dtype=np.int32)
                                pair_width = 0
                                sparse_logit_start = 0
                                use_second_pair_priors = False
                                if (
                                    self.pair_policy_enabled
                                    and root_placements_remaining >= 2
                                    and active_width >= 2
                                ):
                                    total_pair_rows = active_width * (active_width - 1) // 2
                                    if total_pair_rows > 512:
                                        raise ValueError(
                                            "pair_policy root prior requires all legal pair rows in the current IPC path; "
                                            f"pair_count={total_pair_rows} exceeds 512"
                                        )
                                    pair_batch = build_pair_candidate_batch(
                                        root_candidate_qr.tolist(),
                                        [],
                                        budget=total_pair_rows,
                                        candidate_mask=np.ones(active_width, dtype=np.bool_),
                                        legal_moves=legal.tolist(),
                                    )
                                    pair_rows = np.flatnonzero(pair_batch.mask)
                                    pair_width = int(pair_rows.shape[0])
                                    if pair_width > 0:
                                        pair_indices = pair_batch.pair_indices[pair_rows[:pair_width]]
                                        pair_qr = np.asarray(
                                            [
                                                [
                                                    int(root_candidate_qr[a, 0]),
                                                    int(root_candidate_qr[a, 1]),
                                                    int(root_candidate_qr[b, 0]),
                                                    int(root_candidate_qr[b, 1]),
                                                ]
                                                for a, b in pair_indices
                                            ],
                                            dtype=np.int32,
                                        )
                                        pair_indices_for_forward = pair_indices
                                        pair_mask_for_forward = pair_batch.mask[pair_rows[:pair_width]]
                                elif self.pair_policy_enabled and root_placements_remaining == 1 and active_width >= 1:
                                    first_qr = _last_move_qr(move_history)
                                    legal_set = {(int(q), int(r)) for q, r in legal.tolist()}
                                    if first_qr is not None and first_qr not in legal_set:
                                        first_cand = build_candidate_batch(
                                            [first_qr],
                                            [],
                                            offset_q=int(offset_q),
                                            offset_r=int(offset_r),
                                            budget=1,
                                            critical_actions=[first_qr],
                                        )
                                        forward_indices = np.concatenate(
                                            [first_cand.indices[:1], cand.indices[active_rows]],
                                            axis=0,
                                        )
                                        forward_features = np.concatenate(
                                            [first_cand.features[:1], cand.features[active_rows]],
                                            axis=0,
                                        )
                                        forward_mask = np.concatenate(
                                            [first_cand.mask[:1], cand.mask[active_rows]],
                                            axis=0,
                                        )
                                        sparse_logit_start = 1
                                        pair_width = active_width
                                        pair_indices_for_forward = np.asarray(
                                            [[0, i] for i in range(1, active_width + 1)],
                                            dtype=np.int64,
                                        )
                                        pair_mask_for_forward = np.ones(pair_width, dtype=np.bool_)
                                        first_q, first_r = first_qr
                                        pair_qr = np.asarray(
                                            [
                                                [
                                                    int(first_q),
                                                    int(first_r),
                                                    int(root_candidate_qr[i, 0]),
                                                    int(root_candidate_qr[i, 1]),
                                                ]
                                                for i in range(active_width)
                                            ],
                                            dtype=np.int32,
                                        )
                                        use_second_pair_priors = True
                                t_forward = time.monotonic()
                                if pair_indices_for_forward is not None and pair_width > 0:
                                    p, v, sparse, pair_logits = client.submit_sparse_pair(
                                        root_tensor,
                                        1,
                                        forward_indices.reshape(1, -1),
                                        forward_features.reshape(1, forward_features.shape[0], forward_features.shape[1]),
                                        forward_mask.reshape(1, -1),
                                        pair_indices_for_forward.reshape(1, pair_width, 2),
                                        pair_mask_for_forward.reshape(1, pair_width),
                                    )
                                else:
                                    p, v, sparse = client.submit_sparse(
                                        root_tensor,
                                        1,
                                        forward_indices.reshape(1, -1),
                                        forward_features.reshape(1, forward_features.shape[0], forward_features.shape[1]),
                                        forward_mask.reshape(1, -1),
                                    )
                                    pair_logits = None
                                sparse_prior_forward_ms += (time.monotonic() - t_forward) * 1000.0
                                root_sparse_logits = sparse[0, sparse_logit_start:sparse_logit_start + active_width]
                                sparse_vs_dense_disagreement = _sparse_dense_disagreement(
                                    p,
                                    root_candidate_qr,
                                    root_sparse_logits,
                                    legal,
                                    int(offset_q),
                                    int(offset_r),
                                )
                                if self.sparse_policy_enabled and self.sparse_prior_stage > 0:
                                    engine.expand_root_with_sparse_priors(
                                        p,
                                        v[0],
                                        offset_q,
                                        offset_r,
                                        legal_bytes,
                                        root_candidate_qr,
                                        root_sparse_logits,
                                        self.sparse_prior_stage,
                                        self.sparse_prior_mix,
                                    )
                                else:
                                    engine.expand_root(p, v[0], offset_q, offset_r, legal_bytes)
                                if pair_logits is not None and pair_width > 0 and pair_qr.size > 0:
                                    if use_second_pair_priors and hasattr(engine, "apply_root_pair_second_priors"):
                                        engine.apply_root_pair_second_priors(
                                            pair_qr,
                                            pair_logits[0, :pair_width],
                                            self.pair_prior_mix,
                                        )
                                    else:
                                        engine.apply_root_pair_priors(
                                            pair_qr,
                                            pair_logits[0, :pair_width],
                                            self.pair_prior_mix,
                                        )
                        else:
                            p, v = client.submit(root_tensor, 1)
                            engine.expand_root(
                                p, v[0], offset_q, offset_r, legal_bytes
                            )
                except Exception as exc:
                    logger.warning(
                        "Worker %s: root inference failed at move %s: %s",
                        self.worker_id,
                        move_idx,
                        exc,
                    )
                    if self.global_graph_enabled:
                        terminal_reason = "invalid_graph_root_inference"
                        return None
                    engine.expand_root(
                        np.ones(1089, dtype=np.float32) / 1089,
                        0.0,
                        offset_q,
                        offset_r,
                        legal_bytes,
                    )
            else:
                if self.global_graph_enabled:
                    logger.warning(
                        "Worker %s: global graph model has no inference client; marking run invalid",
                        self.worker_id,
                    )
                    terminal_reason = "invalid_graph_no_inference_client"
                    return None
                engine.expand_root(
                    np.ones(1089, dtype=np.float32) / 1089,
                    0.0,
                    offset_q,
                    offset_r,
                    legal_bytes,
                )

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
                    batch_tensor, count = engine.select_leaves(
                        self.batch_size
                    )
                    if count == 0:
                        engine.expand_and_backprop(
                            np.zeros(0, dtype=np.float32),
                            np.zeros(0, dtype=np.float32),
                        )
                        break
                    if isinstance(batch_tensor, np.ndarray):
                        batch_4d = batch_tensor
                    else:
                        batch_4d = np.array(batch_tensor)

                    if client is not None:
                        if self.global_graph_enabled:
                            meta = engine.pending_leaf_metadata() if hasattr(engine, "pending_leaf_metadata") else []
                            if len(meta) != count:
                                raise ValueError("global graph leaf expansion requires pending leaf metadata")
                            graph_values = np.zeros(count, dtype=np.float32)
                            legal_rows: list[np.ndarray] = []
                            legal_logits: list[np.ndarray] = []
                            max_width = 0
                            t_forward = time.monotonic()
                            for row, (_leaf_oq, _leaf_or, leaf_legal_bytes, leaf_history_bytes) in enumerate(meta):
                                legal = np.frombuffer(bytes(leaf_legal_bytes), dtype=np.int32).reshape(-1, 2)
                                graph_batch = build_graph_batch_from_history(
                                    bytes(leaf_history_bytes),
                                    opp_legal_moves=[(int(q), int(r)) for q, r in legal],
                                    radius=8,
                                    max_pair_rows=0,
                                    include_pair_rows=False,
                                )
                                graph_out = client.submit_graph(graph_batch)
                                graph_legal = np.asarray(graph_out["metadata"]["legal_qr"], dtype=np.int32)
                                if graph_legal.shape != legal.shape or not np.array_equal(graph_legal, legal):
                                    raise ValueError(
                                        "graph inference legal_qr does not match Rust legal moves at leaf"
                                    )
                                logits = np.asarray(graph_out["policy_place"], dtype=np.float32)
                                graph_values[row] = float(np.asarray(graph_out["value"], dtype=np.float32)[0])
                                legal_rows.append(graph_legal)
                                legal_logits.append(logits)
                                max_width = max(max_width, int(graph_legal.shape[0]))
                            sparse_qr = np.zeros((count, max_width, 2), dtype=np.int32)
                            sparse_logits = np.zeros((count, max_width), dtype=np.float32)
                            sparse_counts = np.zeros(count, dtype=np.uint16)
                            for row, (qr_rows, logits) in enumerate(zip(legal_rows, legal_logits)):
                                width = int(qr_rows.shape[0])
                                sparse_qr[row, :width] = qr_rows
                                sparse_logits[row, :width] = logits[:width]
                                sparse_counts[row] = width
                            sparse_prior_forward_ms += (time.monotonic() - t_forward) * 1000.0
                            engine.expand_and_backprop_with_sparse(
                                np.zeros(count * 1089, dtype=np.float32),
                                graph_values,
                                sparse_qr,
                                sparse_logits,
                                sparse_counts,
                                2,
                                1.0,
                            )
                        elif self.sparse_policy_enabled and self.sparse_prior_stage >= 2:
                            meta = engine.pending_leaf_metadata() if hasattr(engine, "pending_leaf_metadata") else []
                            if len(meta) == count:
                                cand_qr = np.zeros((count, self.candidate_budget, 2), dtype=np.int32)
                                cand_indices = np.full((count, self.candidate_budget), -1, dtype=np.int64)
                                cand_features = np.zeros((count, self.candidate_budget, 12), dtype=np.float32)
                                cand_mask = np.zeros((count, self.candidate_budget), dtype=np.bool_)
                                cand_counts = np.zeros(count, dtype=np.uint16)
                                t_build = time.monotonic()
                                for row, (leaf_oq, leaf_or, leaf_legal_bytes, leaf_history_bytes) in enumerate(meta):
                                    legal = np.frombuffer(bytes(leaf_legal_bytes), dtype=np.int32).reshape(-1, 2)
                                    leaf_winning, leaf_forced, leaf_cover = _critical_actions_from_root_tensor(
                                        batch_4d[row],
                                        legal,
                                        int(leaf_oq),
                                        int(leaf_or),
                                    )
                                    leaf_oracle = scan_tactical_oracle_from_history(
                                        bytes(leaf_history_bytes),
                                        [(int(q), int(r)) for q, r in legal],
                                        offset_q=int(leaf_oq),
                                        offset_r=int(leaf_or),
                                    )
                                    cand = build_candidate_batch(
                                        [(int(q), int(r)) for q, r in legal],
                                        [],
                                        offset_q=int(leaf_oq),
                                        offset_r=int(leaf_or),
                                        budget=self.candidate_budget,
                                        winning_moves=list(leaf_winning) + list(leaf_oracle.win_now_cells),
                                        forced_block_moves=list(leaf_forced) + list(leaf_oracle.forced_block_cells),
                                        cover_cells=list(leaf_cover) + list(leaf_oracle.cover_cells),
                                        open_four_cells=leaf_oracle.open_four_cells,
                                        open_five_cells=leaf_oracle.open_five_cells,
                                    )
                                    active_rows = np.flatnonzero(cand.mask)
                                    width = min(self.candidate_budget, int(active_rows.shape[0]))
                                    rows = active_rows[:width]
                                    cand_qr[row, :width] = cand.qr[rows]
                                    cand_indices[row, :width] = cand.indices[rows]
                                    cand_features[row, :width] = cand.features[rows]
                                    cand_mask[row, :width] = cand.mask[rows]
                                    cand_counts[row] = width
                                sparse_prior_candidate_build_ms += (time.monotonic() - t_build) * 1000.0
                                t_forward = time.monotonic()
                                p, v, sparse = client.submit_sparse(
                                    batch_4d.astype(np.float32, copy=False),
                                    count,
                                    cand_indices,
                                    cand_features,
                                    cand_mask,
                                )
                                sparse_prior_forward_ms += (time.monotonic() - t_forward) * 1000.0
                                engine.expand_and_backprop_with_sparse(
                                    p,
                                    v,
                                    cand_qr,
                                    sparse,
                                    cand_counts,
                                    self.sparse_prior_stage,
                                    self.sparse_prior_mix,
                                )
                            else:
                                p, v = client.submit(
                                    batch_4d.astype(np.float32, copy=False), count
                                )
                                engine.expand_and_backprop(p, v)
                        else:
                            p, v = client.submit(
                                batch_4d.astype(np.float32, copy=False), count
                            )
                            engine.expand_and_backprop(p, v)
                    else:
                        uniform_policy = np.full(
                            count * 1089,
                            1.0 / 1089.0,
                            dtype=np.float32,
                        )
                        engine.expand_and_backprop(
                            uniform_policy,
                            np.zeros(count, dtype=np.float32),
                        )
                except Exception as exc:
                    logger.warning(
                        "Worker %s: leaf expansion failed at move %s: %s",
                        self.worker_id,
                        move_idx,
                        exc,
                    )
                    if self.global_graph_enabled:
                        terminal_reason = "invalid_graph_leaf_inference"
                        return None
                    break

            moves_q, moves_r, visits, root_value = engine.get_results()
            pair_policy_v2 = _normalize_pair_visit_targets(
                engine.root_pair_visit_targets()
                if hasattr(engine, "root_pair_visit_targets")
                else []
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
                    if histories and (not self.global_graph_enabled or client is None):
                        logger.debug(
                            "Worker %s: RGSC tree extraction skipped because no regret-network scorer is available",
                            self.worker_id,
                        )
                    for history in histories if self.global_graph_enabled and client is not None else []:
                        if not history:
                            continue
                        move_bytes = encode_move_history(history)
                        graph = build_graph_batch_from_history(
                            move_bytes,
                            radius=8,
                            max_pair_rows=0,
                            include_pair_rows=False,
                        )
                        out = client.submit_graph(graph)
                        regret_rank = float(np.asarray(out.get("regret_rank", [0.0]), dtype=np.float32)[0])
                        if regret_rank > 0.0 and np.isfinite(regret_rank):
                            scored.append((move_bytes, regret_rank, regret_rank))
                    if scored:
                        rgsc_tree_node_insertions = self.rgsc.observe_tree_node_candidates(
                            scored,
                            game_id=game_id,
                            score_source="graph_regret_rank",
                        )
                except Exception as exc:
                    logger.debug("Worker %s: RGSC tree extraction skipped: %s", self.worker_id, exc)

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

            if HAS_ENGINE:
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
            legal_root = np.frombuffer(legal_bytes, dtype=np.int32).reshape(-1, 2)
            winning_moves, forced_blocks, cover_cells = _critical_actions_from_root_tensor(
                tensor_3d,
                legal_root,
                int(offset_q),
                int(offset_r),
            )
            oracle = scan_tactical_oracle_from_history(
                record_history,
                [(int(q), int(r)) for q, r in legal_root],
                offset_q=int(offset_q),
                offset_r=int(offset_r),
            )
            candidate_probe = build_candidate_batch(
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

        if HAS_ENGINE:
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

    def _game_id(self) -> int:
        worker_part = (int(self.worker_id) & 0xFF) << 24
        game_part = int(self._game_counter) & 0xFF_FFFF
        return worker_part | game_part
