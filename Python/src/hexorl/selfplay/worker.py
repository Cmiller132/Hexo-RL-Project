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
from typing import Optional, List, Tuple

try:
    import _engine

    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

from hexorl.config import Config
from hexorl.inference.client import InferenceClient
from hexorl.selfplay.records import (
    GameRecord,
    PositionRecord,
    sparsify_policy,
    action_to_board_index,
    BOARD_AREA,
)
from hexorl.buffer.targets import process_game_record

logger = logging.getLogger(__name__)


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

    def root_child_q_values(self) -> List[float]:
        """Return mock Q-values."""
        if self._q_values is None:
            self._q_values = self._rng.uniform(
                -1, 1, size=max(1, self._num_children)
            ).astype(np.float32)
        return self._q_values.tolist()

    def move_history_bytes(self) -> bytes:
        """Return mock packed move history."""
        n_bytes = self._move_count * 12
        return self._rng.bytes(n_bytes)

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
            np.array(tensor_3d).astype(np.float32),
            oq,
            or_,
            legal_bytes,
        )

    def expand_root(self, policy, value, oq, or_, legal_bytes):
        self._engine.expand_root(policy, value, oq, or_, legal_bytes)

    def add_dirichlet_noise(self, noise, fraction):
        self._engine.add_dirichlet_noise(noise, fraction)

    def done(self):
        return self._engine.done()

    def select_leaves(self, batch_size):
        tensor_4d, count = self._engine.select_leaves(batch_size)
        return np.array(tensor_4d).astype(np.float32), count

    def expand_and_backprop(self, policies, values):
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

    def root_child_q_values(self):
        return self._engine.root_child_q_values()

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
        if HAS_ENGINE:
            game = _engine.HexGame()
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
        move_history = bytearray()
        move_idx = 0
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

            if client is not None:
                try:
                    p, v = client.submit(
                        tensor_3d.reshape(1, 13, 33, 33).astype(np.float32),
                        1,
                    )
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
                    engine.expand_root(
                        np.ones(1089, dtype=np.float32) / 1089,
                        0.0,
                        offset_q,
                        offset_r,
                        legal_bytes,
                    )
            else:
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
                        p, v = client.submit(
                            batch_4d.astype(np.float32), count
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
                    break

            moves_q, moves_r, visits, root_value = engine.get_results()

            temp = get_temperature(move_idx, self.temperature_schedule)
            q, r = engine.sample_action(temp)
            if q is None:
                q, r = 0, 0

            if HAS_ENGINE:
                player = engine._game.current_player
            else:
                player = move_idx % 2

            record_history = bytes(move_history)

            # Build visit distribution over the full board (MCTS-improved policy).
            # offset_q/offset_r come from init_root() above — still in scope.
            visit_arr = np.zeros(BOARD_AREA, dtype=np.float32)
            for q_coord, r_coord, v in zip(moves_q, moves_r, visits):
                flat_idx = action_to_board_index(q_coord, r_coord, offset_q, offset_r)
                if flat_idx >= 0:
                    visit_arr[flat_idx] = float(v)
            policy = sparsify_policy(visit_arr, top_k=self.policy_target_top_k)

            positions.append(
                PositionRecord(
                    move_history=record_history,
                    policy_target=policy,
                    root_value=root_value,
                    player=player,
                    game_id=game_id,
                    is_full_search=not use_pcr,
                    turn_index=move_idx,
                )
            )

            q, r = int(q), int(r)
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
        record = GameRecord.from_game_data(
            move_history_bytes=full_history,
            policy_targets=[p.policy_target for p in positions],
            root_values=[p.root_value for p in positions],
            players=[p.player for p in positions],
            outcome=outcome,
            game_id=game_id,
            is_full_search=not use_pcr,
        )
        record.final_move_history = full_history
        record.truncated = truncated
        record.terminal_reason = terminal_reason

        record.assign_outcomes()
        return record

    def _game_id(self) -> int:
        worker_part = (int(self.worker_id) & 0xFF) << 24
        game_part = int(self._game_counter) & 0xFF_FFFF
        return worker_part | game_part
