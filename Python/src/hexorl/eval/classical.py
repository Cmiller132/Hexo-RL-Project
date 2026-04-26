"""Classical alpha-beta opponent for ELO anchoring.

Uses the Rust iterative deepening alpha-beta search exposed via PyO3 FFI.
Serves as a stable baseline for evaluating neural network strength.
"""

import logging
import random
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

try:
    import _engine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False


class ClassicalOpponent:
    """Alpha-beta search opponent using the Rust classical engine."""

    def __init__(
        self,
        time_ms: int = 100,
        max_depth: int = 6,
        near_radius: int = 2,
        noise_level: float = 0.0,
    ):
        self.time_ms = time_ms
        self.max_depth = max_depth
        self.near_radius = near_radius
        self.noise_level = noise_level
        self._game = None
        self._moves_played = 0

        if not HAS_ENGINE:
            logger.warning("Rust engine not available — ClassicalOpponent using mock")

    def reset(self):
        if HAS_ENGINE:
            self._game = _engine.HexGame()
        else:
            self._game = None
        self._moves_played = 0

    def select_move(
        self,
        move_history: List[Tuple[int, int, int]],
        time_ms: Optional[int] = None,
        player: int = 0,
    ) -> Tuple[int, int]:
        if not HAS_ENGINE:
            return self._mock_move()

        if self._game is None:
            self.reset()

        if self._moves_played == 0 and move_history:
            for p, q, r in move_history:
                try:
                    self._game.place(q, r)
                except Exception as exc:
                    logger.debug("Ignoring invalid replayed move (%s,%s,%s): %s", p, q, r, exc)
            self._moves_played = len(move_history)

        t = time_ms if time_ms is not None else self.time_ms

        try:
            q, r, score, depth, nodes = self._game.classical_search(
                time_ms=t,
                max_depth=self.max_depth,
                near_radius=self.near_radius,
                noise_level=self.noise_level,
            )
            logger.debug(f"Classical: ({q},{r}) score={score} depth={depth} nodes={nodes}")
            self._game.place(q, r)
            self._moves_played += 1
            return q, r
        except Exception as e:
            logger.warning(f"Classical search failed: {e}, using fallback")
            return self._mock_move()

    def _mock_move(self) -> Tuple[int, int]:
        return (random.randint(-2, 2), random.randint(-2, 2))


def classical_opponent_fn(
    time_ms: int = 100,
    max_depth: int = 6,
) -> callable:
    opponent = ClassicalOpponent(
        time_ms=time_ms, max_depth=max_depth, near_radius=2, noise_level=0.05,
    )

    def _fn(move_history, time_ms_override, player):
        opponent.reset()
        for p, q, r in move_history:
            if opponent._game:
                try:
                    opponent._game.place(q, r)
                except Exception as exc:
                    logger.debug("Ignoring invalid replayed move (%s,%s,%s): %s", p, q, r, exc)
        return opponent.select_move(move_history, time_ms, player)

    return _fn
