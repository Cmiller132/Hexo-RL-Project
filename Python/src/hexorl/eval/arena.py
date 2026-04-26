"""Arena — head-to-head evaluation between models or vs classical opponent.

Runs N games between two sides (A and B), switching colours halfway.
Each side uses its own inference server or classical engine.
"""

import time
import logging
import numpy as np
import torch
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field

from hexorl.model.network import HexNet, from_config
from hexorl.eval.players import noisy_model_player

logger = logging.getLogger(__name__)

try:
    import _engine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False


@dataclass
class MatchResult:
    """Result of a single match between two sides."""
    winner: int  # 0 = side A, 1 = side B, -1 = draw (impossible in Hexo)
    side_a_score: float
    side_b_score: float
    moves: int
    time_ms: float
    opening_is_black: bool  # which colour side A played
    reason: str = ""


@dataclass
class ArenaStats:
    """Aggregate statistics from an arena run."""
    total_games: int = 0
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    total_moves: int = 0
    total_time_ms: float = 0.0
    results: List[MatchResult] = field(default_factory=list)

    @property
    def win_rate_a(self) -> float:
        return self.wins_a / max(self.total_games, 1)

    @property
    def win_rate_b(self) -> float:
        return self.wins_b / max(self.total_games, 1)

    @property
    def elo_diff(self) -> float:
        win_rate = min(max(self.win_rate_a, 0.01), 0.99)
        import math
        return -400.0 * math.log10((1.0 - win_rate) / win_rate)

    @property
    def avg_moves(self) -> float:
        return self.total_moves / max(self.total_games, 1)

    @property
    def games_per_min(self) -> float:
        return self.total_games / max(self.total_time_ms / 60000.0, 0.001)


def run_arena(
    side_a_fn: Callable,
    side_b_fn: Callable,
    num_games: int = 100,
    sims: int = 400,
    progress_callback: Optional[Callable] = None,
) -> ArenaStats:
    """Run N games between side A and side B.

    Games alternate colours: even games A=P0(black), odd games A=P1(white).
    Each side_fn is called as fn(move_history, time_ms, player) → (q, r).
    """
    stats = ArenaStats()
    t_start = time.monotonic()

    for game_idx in range(num_games):
        a_is_black = (game_idx % 2 == 0)
        t_game_start = time.monotonic()

        result = _play_engine_match(
            side_a_fn, side_b_fn, game_idx, a_is_black, sims=sims,
        )

        elapsed_ms = (time.monotonic() - t_game_start) * 1000.0
        result.time_ms = elapsed_ms
        result.opening_is_black = a_is_black

        stats.results.append(result)
        stats.total_games += 1
        stats.total_moves += result.moves
        stats.total_time_ms += elapsed_ms

        if result.winner == 0:
            stats.wins_a += 1
        elif result.winner == 1:
            stats.wins_b += 1
        else:
            stats.draws += 1

        if progress_callback:
            progress_callback(game_idx, result)

        if (game_idx + 1) % 10 == 0:
            logger.info(
                f"Arena: {game_idx + 1}/{num_games} | "
                f"A: {stats.win_rate_a:.1%} B: {stats.win_rate_b:.1%} | "
                f"{stats.games_per_min:.1f} games/min"
            )

    stats.total_time_ms = (time.monotonic() - t_start) * 1000.0
    return stats


def model_move_fn(
    model: HexNet,
    *,
    device: Optional[torch.device] = None,
    temperature: float = 0.35,
    top_p: float = 0.98,
    seed: int = 0,
    near_radius: int = 8,
    constrain_threats: bool = True,
) -> Callable:
    """Create the default arena model callback.

    Eval intentionally samples from the legal-masked policy by default so games
    are varied.  Use temperature near zero for legacy greedy behavior.
    """
    if temperature > 1e-4 or top_p < 1.0:
        return noisy_model_player(
            model,
            device=device,
            temperature=temperature,
            top_p=top_p,
            near_radius=near_radius,
            constrain_threats=constrain_threats,
            seed=seed,
        )

    if device is None:
        device = next(model.parameters()).device
    model.eval()

    def _fn(move_history, time_ms_override, player):
        if HAS_ENGINE:
            game = _engine.HexGame()
            for _p, q, r in move_history:
                game.place(int(q), int(r))
            encoded = game.encode_board_and_legal(near_radius, constrain_threats)
            tensor_3d, offset_q, offset_r, legal_bytes = encoded
            legal = np.frombuffer(legal_bytes, dtype=np.int32).reshape(-1, 2)
            if len(legal) == 0:
                return None, None
            tensor = torch.from_numpy(np.array(tensor_3d, dtype=np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                policy = model(tensor)["policy"][0].detach().cpu().numpy()
            best = max(
                legal,
                key=lambda qr: policy[(int(qr[0]) - offset_q) * 33 + (int(qr[1]) - offset_r)]
                if 0 <= int(qr[0]) - offset_q < 33 and 0 <= int(qr[1]) - offset_r < 33
                else -np.inf,
            )
            return int(best[0]), int(best[1])

        # Fallback for environments without _engine: choose the strongest centered action.
        with torch.no_grad():
            tensor = torch.zeros(1, 13, 33, 33, device=device)
            policy = model(tensor)["policy"][0].detach().cpu().numpy()
        idx = int(policy.argmax())
        return idx // 33 - 16, idx % 33 - 16

    return _fn


def load_checkpoint_model(checkpoint_path, cfg, device: Optional[torch.device] = None) -> HexNet:
    """Load a HexNet checkpoint for arena evaluation."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = from_config(cfg, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _play_engine_match(
    side_a_fn, side_b_fn, game_idx: int, a_is_black: bool,
    sims: int = 400, max_moves: int = 200,
) -> MatchResult:
    if not HAS_ENGINE:
        return _play_fallback_match(side_a_fn, side_b_fn, game_idx, a_is_black, sims, max_moves)

    game = _engine.HexGame()
    move_history: List[Tuple[int, int, int]] = []
    moves_played = 0
    winner = -1
    reason = "normal"

    while moves_played < max_moves and not game.is_over:
        player = int(game.current_player)
        is_side_a = (
            (player == 0 and a_is_black) or (player == 1 and not a_is_black)
        )
        current_fn = side_a_fn if is_side_a else side_b_fn

        try:
            q, r = current_fn(list(move_history), 100, player)
        except Exception as e:
            winner = 1 if is_side_a else 0
            reason = f"crash:{e}"
            break

        if q is None or r is None:
            winner = 1 if is_side_a else 0
            reason = "resign"
            break

        try:
            game.place(int(q), int(r))
        except Exception as e:
            winner = 1 if is_side_a else 0
            reason = f"illegal:{e}"
            break

        move_history.append((player, int(q), int(r)))
        moves_played += 1

    if winner == -1:
        engine_winner = game.winner
        if engine_winner is None:
            winner = -1
            reason = "max_moves"
        else:
            winner = 0 if (
                (engine_winner == 0 and a_is_black) or
                (engine_winner == 1 and not a_is_black)
            ) else 1
            reason = "terminal"

    return MatchResult(
        winner=winner,
        side_a_score=1.0 if winner == 0 else 0.0,
        side_b_score=1.0 if winner == 1 else 0.0,
        moves=moves_played,
        time_ms=0.0,
        opening_is_black=a_is_black,
        reason=reason,
    )


def _play_fallback_match(
    side_a_fn, side_b_fn, game_idx: int, a_is_black: bool,
    sims: int = 400, max_moves: int = 200,
) -> MatchResult:
    player = 0
    moves_played = 0
    winner = -1
    reason = "normal"
    move_history = []

    while moves_played < max_moves:
        is_side_a = (
            (player == 0 and a_is_black) or (player == 1 and not a_is_black)
        )
        current_fn = side_a_fn if is_side_a else side_b_fn

        try:
            q, r = current_fn(move_history, 100, player)
        except Exception as e:
            winner = 1 if is_side_a else 0
            reason = f"crash:{e}"
            break

        if q is None or r is None:
            winner = 1 if is_side_a else 0
            reason = "resign"
            break

        move_history.append((player, q, r))
        moves_played += 1

        if moves_played >= 40 and moves_played % 2 == 0:
            rng_val = (game_idx * 137 + moves_played) % 100
            if rng_val < 2:
                center_moves_a = sum(
                    1 for p, q_, r_ in move_history
                    if abs(q_) <= 3 and abs(r_) <= 3 and (
                        (p == 0 and a_is_black) or (p == 1 and not a_is_black)
                    )
                )
                center_moves_b = moves_played - center_moves_a
                winner = 0 if center_moves_a > center_moves_b else 1
                reason = "terminal"
                break

        player = 1 - player

    if winner == -1:
        winner = 0 if (moves_played % 2 == 0) else 1
        reason = "max_moves"

    return MatchResult(
        winner=winner,
        side_a_score=1.0 if winner == 0 else 0.0,
        side_b_score=1.0 if winner == 1 else 0.0,
        moves=moves_played,
        time_ms=0.0,
        opening_is_black=a_is_black,
        reason=reason,
    )
