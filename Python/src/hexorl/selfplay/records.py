"""Game record format — compact serialization for the ring buffer.

Each record represents one position from a self-play game:
  - move_history: compact bytes (i32 LE triples: player, q, r) 
  - policy_target: sparse dict {action_idx: probability} 
  - value_target: f32 (outcome or EMA lookahead)
  - game_id: u32 (for recency weighting)
  - player: u8 (which player generated this record — used for perspective flip)
"""

import struct
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# Constants matching Rust encoder (must stay in sync)
NUM_CHANNELS = 13
BOARD_SIZE = 33
BOARD_AREA = 33 * 33  # 1089


@dataclass
class PositionRecord:
    """One position from a self-play game — data needed for training."""

    # Compact move history: flat bytes of (player:i32, q:i32, r:i32) LE triples.
    # Encodes all moves played so far (from initial empty board).
    # Rust's encode_compact_record replays this into (13,33,33) tensors on demand.
    move_history: bytes

    # Sparse policy target: maps action index (flat BOARD_AREA index: q*33 + r + offset)
    # to visit probability. Top-K only (K ≤ 20) to save space.
    policy_target: Dict[int, float]

    # Root Q-value from MCTS (from current player's perspective).
    root_value: float

    # Which player generated this record (0 or 1).
    player: int

    # Game outcome from P0's perspective. None until game ends.
    # 1.0 = P0 wins, -1.0 = P1 wins.
    outcome: Optional[float] = None

    # Unique game identifier for recency-weighted sampling.
    game_id: int = 0

    # Whether this position was generated with full MCTS sims (True) or
    # low-sim PCR (Playout Cap Randomization). PCR samples get lower weight.
    is_full_search: bool = True

    # Turn index within the game (0-based). Used for temperature schedule lookup.
    turn_index: int = 0

    # Lookahead value targets at multiple horizons (KataGo-style).
    # Phase 3 placeholder: just the final outcome. Phase 4 adds EMA lookahead.
    lookahead_values: List[float] = field(default_factory=list)

    def to_value_target(self) -> float:
        """Compute the training value target for this position.

        From the current player's perspective:
          - If current player == P0: target = outcome
          - If current player == P1: target = -outcome
        """
        if self.outcome is None:
            return 0.0
        return self.outcome if self.player == 0 else -self.outcome

    def to_dense_policy(self) -> np.ndarray:
        """Convert sparse policy target to dense (BOARD_AREA,) float32 array."""
        dense = np.zeros(BOARD_AREA, dtype=np.float32)
        for idx, prob in self.policy_target.items():
            if 0 <= idx < BOARD_AREA:
                dense[idx] = prob
        return dense


@dataclass
class GameRecord:
    """Complete record of one self-play game.

    Contains all positions played, plus the final outcome.
    """

    # All positions in this game (one per move, except terminal state).
    positions: List[PositionRecord] = field(default_factory=list)

    # Final game outcome from P0's perspective.
    outcome: float = 0.0

    # Unique game ID (monotonic counter).
    game_id: int = 0

    # Total number of placements in the game.
    game_length: int = 0

    def assign_outcomes(self):
        """Assign the game outcome to all positions."""
        for pos in self.positions:
            pos.outcome = self.outcome

    def to_compact_bytes(self) -> bytes:
        """Serialize the game record into compact bytes for buffer storage.

        Format:
          Header (12 bytes):
            - game_id:   u32 LE
            - outcome:   f32 LE  
            - num_pos:   u32 LE
          Per position (variable):
            - move_history_len: u32 LE
            - move_history: bytes (move_history_len bytes)
            - player: u8
            - is_full_search: u8
            - root_value: f32 LE
            - num_policy_entries: u16 LE
            - per policy entry: (u16 LE action_idx, f32 LE prob)
        """
        parts = bytearray()

        # Header
        parts.extend(struct.pack("<IfI", self.game_id, self.outcome, len(self.positions)))

        for pos in self.positions:
            # Move history
            parts.extend(struct.pack("<I", len(pos.move_history)))
            parts.extend(pos.move_history)

            # Flags
            parts.extend(struct.pack("<BB", pos.player, int(pos.is_full_search)))

            # Root value
            parts.extend(struct.pack("<f", pos.root_value))

            # Policy target (sparse)
            entries = list(pos.policy_target.items())
            parts.extend(struct.pack("<H", len(entries)))
            for idx, prob in entries:
                parts.extend(struct.pack("<Hf", idx, prob))

            # Turn index
            parts.extend(struct.pack("<I", pos.turn_index))

        return bytes(parts)

    @staticmethod
    def from_compact_bytes(data: bytes) -> "GameRecord":
        """Deserialize a game record from compact bytes."""
        offset = 0

        game_id = struct.unpack_from("<I", data, offset)[0]; offset += 4
        outcome = struct.unpack_from("<f", data, offset)[0]; offset += 4
        num_pos = struct.unpack_from("<I", data, offset)[0]; offset += 4

        positions = []
        for _ in range(num_pos):
            # Move history
            mh_len = struct.unpack_from("<I", data, offset)[0]; offset += 4
            move_history = data[offset:offset + mh_len]; offset += mh_len

            # Flags
            player = data[offset]; offset += 1
            is_full = bool(data[offset]); offset += 1

            # Root value
            root_value = struct.unpack_from("<f", data, offset)[0]; offset += 4

            # Policy target
            num_entries = struct.unpack_from("<H", data, offset)[0]; offset += 2
            policy = {}
            for _ in range(num_entries):
                idx = struct.unpack_from("<H", data, offset)[0]; offset += 2
                prob = struct.unpack_from("<f", data, offset)[0]; offset += 4
                policy[idx] = prob

            # Turn index
            turn_idx = struct.unpack_from("<I", data, offset)[0]; offset += 4

            positions.append(PositionRecord(
                move_history=move_history,
                policy_target=policy,
                root_value=root_value,
                player=player,
                outcome=outcome,
                game_id=game_id,
                is_full_search=is_full,
                turn_index=turn_idx,
            ))

        return GameRecord(
            positions=positions,
            outcome=outcome,
            game_id=game_id,
            game_length=num_pos,
        )

    @staticmethod
    def from_game_data(
        move_history_bytes: bytes,
        policy_targets: List[Dict[int, float]],
        root_values: List[float],
        players: List[int],
        outcome: float,
        game_id: int,
        is_full_search: bool = True,
    ) -> "GameRecord":
        """Construct a GameRecord from raw game data.

        Args:
            move_history_bytes: For each position, the compact move history
                up to (but not including) that position's action.
            policy_targets: For each position, sparse policy dict.
            root_values: For each position, root Q-value.
            players: For each position, the player who made the action.
            outcome: Final game outcome from P0's perspective.
            game_id: Monotonic game counter.
            is_full_search: Whether full MCTS sim count was used.
        """
        positions = []
        # Each position encodes the board state BEFORE the move.
        # move_history_bytes is a list of bytes, one per position.
        if isinstance(move_history_bytes, bytes):
            # Single contiguous byte buffer — split by position.
            # Each position's history is the prefix up to that move.
            num_moves = len(policy_targets)
            pos_histories = _split_history_bytes(move_history_bytes, num_moves)
        else:
            pos_histories = move_history_bytes

        for i, (history, policy, rv, player) in enumerate(
            zip(pos_histories, policy_targets, root_values, players)
        ):
            positions.append(PositionRecord(
                move_history=history,
                policy_target=policy,
                root_value=rv,
                player=player,
                outcome=outcome,
                game_id=game_id,
                is_full_search=is_full_search,
                turn_index=i,
            ))

        return GameRecord(
            positions=positions,
            outcome=outcome,
            game_id=game_id,
            game_length=len(positions),
        )


def _split_history_bytes(full_history: bytes, num_positions: int) -> List[bytes]:
    """Split a contiguous move history into per-position prefixes.

    Each triple is 12 bytes (i32 LE × 3). Position i gets the first i triples.
    """
    result = []
    stride = 12  # player:i32, q:i32, r:i32
    for i in range(num_positions):
        result.append(full_history[:i * stride])
    return result


def sparsify_policy(
    dense_policy: np.ndarray,
    top_k: int = 20,
) -> Dict[int, float]:
    """Convert dense policy array to sparse top-K dict.

    Args:
        dense_policy: (BOARD_AREA,) float32 array of visit probabilities.
        top_k: Number of top entries to keep.

    Returns:
        Dict mapping action indices to probabilities, renormalized over top-K.
    """
    if len(dense_policy) == 0:
        return {}
    indices = np.argpartition(-dense_policy, min(top_k, len(dense_policy) - 1))[:top_k]
    values = dense_policy[indices]
    total = values.sum()
    if total > 0:
        values = values / total
    return {int(idx): float(val) for idx, val in zip(indices, values)}


def action_to_board_index(q: int, r: int, offset_q: int = 16, offset_r: int = 16) -> int:
    """Convert axial hex coordinates (q, r) to flat BOARD_AREA index.

    The board tensor uses a 33×33 window centered at the board centroid.
    offset_q and offset_r map tensor index (0,0) to board coordinates.
    Default offsets center the tensor at (16, 16).
    """
    gi = q - offset_q
    gj = r - offset_r
    idx = gi * BOARD_SIZE + gj
    return max(0, min(BOARD_AREA - 1, idx))
