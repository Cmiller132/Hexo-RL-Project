"""Replay reconstruction and board-debug helpers for the dashboard."""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from hexorl.dashboard.db import DashboardStore
from hexorl.selfplay.records import BOARD_SIZE

try:
    import _engine

    HAS_ENGINE = True
except ImportError:  # pragma: no cover - depends on local extension build
    _engine = None
    HAS_ENGINE = False


Move = tuple[int, int, int]


@dataclass(frozen=True)
class ReplayPosition:
    turn_index: int
    current_player: int
    placements_remaining: int
    moves: list[Move]
    stones: list[dict[str, int]]
    is_over: bool
    winner: int | None
    legal_moves: list[dict[str, int]]
    threat_moves: list[dict[str, int]]
    encoding: dict[str, Any]
    overlays: dict[str, Any]


def decode_move_history(history: bytes) -> list[Move]:
    moves: list[Move] = []
    stride = 12
    for offset in range(0, len(history) - stride + 1, stride):
        moves.append(struct.unpack_from("<iii", history, offset))
    return moves


def encode_move_history(moves: Iterable[Move]) -> bytes:
    out = bytearray()
    for player, q, r in moves:
        out.extend(struct.pack("<iii", int(player), int(q), int(r)))
    return bytes(out)


def replay_game(store: DashboardStore, game_id: int) -> dict[str, Any]:
    game_rows = store.rows("SELECT * FROM games WHERE game_id=?", (game_id,))
    if not game_rows:
        raise KeyError(f"Game not found: {game_id}")
    game = game_rows[0]
    position_rows = store.rows(
        "SELECT * FROM positions WHERE game_id=? ORDER BY turn_index", (game_id,)
    )
    final_history = game["final_history_b64"]
    moves = decode_move_history(final_history)
    return {
        "game": _public_game(game),
        "moves": [_move_dict(m) for m in moves],
        "positions": [
            {
                "position_id": row["position_id"],
                "turn_index": row["turn_index"],
                "player": row["player"],
                "root_value": row["root_value"],
                "policy": row.get("policy_json", {}),
                "debug": row.get("debug_json", {}),
            }
            for row in position_rows
        ],
    }


def get_replay_position(
    history: bytes,
    *,
    turn_index: int | None = None,
    near_radius: int = 8,
    constrain_threats: bool = True,
) -> ReplayPosition:
    moves = decode_move_history(history)
    if turn_index is not None:
        moves = moves[: max(0, min(turn_index, len(moves)))]
    partial_history = encode_move_history(moves)
    game = _game_from_moves(moves)
    return _position_from_game(
        game,
        partial_history,
        moves,
        near_radius=near_radius,
        constrain_threats=constrain_threats,
    )


def encode_tensor_for_history(
    history: bytes,
    *,
    near_radius: int = 8,
    constrain_threats: bool = True,
) -> tuple[np.ndarray, int, int, bytes]:
    """Return the canonical Rust encoder tensor for a compact history."""
    moves = decode_move_history(history)
    game = _game_from_moves(moves)
    if not HAS_ENGINE or game is None:
        raise RuntimeError("Rust _engine extension is required for tensor encoding")
    tensor_3d, offset_q, offset_r, legal_bytes = game.encode_board_and_legal(
        near_radius,
        constrain_threats,
    )
    return np.array(tensor_3d, dtype=np.float32), int(offset_q), int(offset_r), legal_bytes


def position_payload(position: ReplayPosition) -> dict[str, Any]:
    return {
        "turn_index": position.turn_index,
        "current_player": position.current_player,
        "placements_remaining": position.placements_remaining,
        "moves": [_move_dict(m) for m in position.moves],
        "stones": position.stones,
        "is_over": position.is_over,
        "winner": position.winner,
        "legal_moves": position.legal_moves,
        "threat_moves": position.threat_moves,
        "encoding": position.encoding,
        "overlays": position.overlays,
    }


def _position_from_game(
    game: Any,
    history: bytes,
    moves: list[Move],
    *,
    near_radius: int,
    constrain_threats: bool,
) -> ReplayPosition:
    if HAS_ENGINE and game is not None:
        current_player = int(getattr(game, "current_player", len(moves) % 2))
        placements_remaining = int(getattr(game, "placements_remaining", _fallback_placements_remaining(moves)))
        is_over = bool(getattr(game, "is_over", False))
        winner = getattr(game, "winner", None)
        stones = _engine_stones(game)
        legal_moves, tensor, offsets, legal_mask = _engine_encoding(
            game, near_radius, constrain_threats
        )
        threat_moves = _engine_threat_moves(game, near_radius)
        encoding = _encoding_summary(tensor, offsets, legal_mask)
    else:
        current_player = len(moves) % 2
        placements_remaining = _fallback_placements_remaining(moves)
        winner = None
        is_over = False
        stones = [_move_dict(m) for m in moves]
        legal_moves = _fallback_legal_moves(moves)
        threat_moves = legal_moves
        encoding = {
            "available": False,
            "channels": [],
            "offset_q": -16,
            "offset_r": -16,
            "legal_mask": _dense_mask(legal_moves),
        }

    return ReplayPosition(
        turn_index=len(moves),
        current_player=current_player,
        placements_remaining=placements_remaining,
        moves=moves,
        stones=stones,
        is_over=is_over,
        winner=None if winner is None else int(winner),
        legal_moves=legal_moves,
        threat_moves=threat_moves,
        encoding=encoding,
        overlays={
            "last_move": _move_dict(moves[-1]) if moves else None,
            "generated_at": time.time(),
            "history_b64_len": len(history),
        },
    )


def _fallback_placements_remaining(moves: list[Move]) -> int:
    if not moves:
        return 1
    return 2 if (len(moves) - 1) % 2 == 0 else 1


def _game_from_moves(moves: list[Move]) -> Any:
    if not HAS_ENGINE:
        return None
    cls = getattr(_engine, "HexGame", None) or getattr(_engine, "PyHexGame")
    game = cls()
    for _player, q, r in moves:
        game.place(int(q), int(r))
    return game


def _engine_stones(game: Any) -> list[dict[str, int]]:
    if hasattr(game, "board_pieces"):
        pieces = game.board_pieces()
        return [
            {"q": int(q), "r": int(r), "player": int(p)}
            for q, r, p in pieces
        ]
    if hasattr(game, "move_history_bytes"):
        return [_move_dict(m) for m in decode_move_history(game.move_history_bytes())]
    return []


def _engine_encoding(
    game: Any,
    near_radius: int,
    constrain_threats: bool,
) -> tuple[list[dict[str, int]], np.ndarray | None, tuple[int, int], list[int]]:
    encoded = game.encode_board_and_legal(near_radius, constrain_threats)
    tensor_3d, offset_q, offset_r, legal_bytes = encoded
    tensor = np.array(tensor_3d, dtype=np.float32)
    legal = _moves_from_bytes(legal_bytes)
    legal_mask = []
    for q, r in legal:
        idx = _flat_index(q, r, offset_q, offset_r)
        if idx >= 0:
            legal_mask.append(idx)
    return (
        [{"q": int(q), "r": int(r)} for q, r in legal],
        tensor,
        (int(offset_q), int(offset_r)),
        legal_mask,
    )


def _engine_threat_moves(game: Any, near_radius: int) -> list[dict[str, int]]:
    if not hasattr(game, "threat_constrained_moves"):
        return []
    try:
        moves = game.threat_constrained_moves(near_radius)
    except TypeError:
        moves = game.threat_constrained_moves()
    if moves is None:
        return []
    return [{"q": int(q), "r": int(r)} for q, r in moves]


def _moves_from_bytes(data: bytes) -> list[tuple[int, int]]:
    if not data:
        return []
    arr = np.frombuffer(data, dtype=np.int32)
    if arr.size % 2 != 0:
        return []
    return [(int(q), int(r)) for q, r in arr.reshape(-1, 2)]


def _encoding_summary(
    tensor: np.ndarray | None,
    offsets: tuple[int, int],
    legal_mask: list[int],
) -> dict[str, Any]:
    if tensor is None:
        return {"available": False, "channels": []}
    channel_names = [
        "own_stones",
        "opp_stones",
        "empty",
        "legal",
        "turn_phase",
        "first_stone",
        "player_color",
        "own_recent",
        "opp_recent",
        "opp_hot",
        "own_hot",
        "centroid_distance",
        "opp_last_turn",
    ]
    channels = []
    for idx, name in enumerate(channel_names):
        channel = tensor[idx]
        channels.append(
            {
                "index": idx,
                "name": name,
                "sum": float(channel.sum()),
                "min": float(channel.min()),
                "max": float(channel.max()),
                "nonzero": int(np.count_nonzero(channel)),
            }
        )
    return {
        "available": True,
        "shape": list(tensor.shape),
        "offset_q": offsets[0],
        "offset_r": offsets[1],
        "channels": channels,
        "legal_mask": legal_mask,
    }


def _fallback_legal_moves(moves: list[Move]) -> list[dict[str, int]]:
    occupied = {(q, r) for _p, q, r in moves}
    if not occupied:
        return [{"q": 0, "r": 0}]
    result = []
    radius = 2
    for q0, r0 in occupied:
        for dq in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                q, r = q0 + dq, r0 + dr
                if (q, r) in occupied:
                    continue
                if max(abs(dq), abs(dr), abs(dq + dr)) <= radius:
                    result.append({"q": q, "r": r})
    result.sort(key=lambda x: (abs(x["q"]) + abs(x["r"]), x["q"], x["r"]))
    return result[:96]


def _dense_mask(moves: list[dict[str, int]]) -> list[int]:
    return [
        idx
        for move in moves
        if (idx := _flat_index(move["q"], move["r"], -16, -16)) >= 0
    ]


def _flat_index(q: int, r: int, offset_q: int, offset_r: int) -> int:
    gi = q - offset_q
    gj = r - offset_r
    if 0 <= gi < BOARD_SIZE and 0 <= gj < BOARD_SIZE:
        return int(gi * BOARD_SIZE + gj)
    return -1


def _move_dict(move: Move) -> dict[str, int]:
    player, q, r = move
    return {"player": int(player), "q": int(q), "r": int(r)}


def _public_game(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "game_id": row["game_id"],
        "run_id": row["run_id"],
        "external_game_id": row["external_game_id"],
        "source": row["source"],
        "epoch": row["epoch"],
        "outcome": row["outcome"],
        "move_count": row["move_count"],
        "payload": row.get("payload_json", {}),
    }


def policy_debug(policy_logits: np.ndarray, legal_mask: list[int], top_k: int = 12) -> dict[str, Any]:
    """Return top-k probabilities and entropy for a legal-masked policy vector."""
    logits = np.asarray(policy_logits, dtype=np.float64).reshape(-1)
    mask = [idx for idx in legal_mask if 0 <= idx < logits.size]
    if not mask:
        return {"top": [], "entropy": 0.0}
    legal_logits = logits[mask]
    legal_logits = legal_logits - np.max(legal_logits)
    probs = np.exp(legal_logits)
    probs = probs / max(float(probs.sum()), 1e-12)
    order = np.argsort(-probs)[:top_k]
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    return {
        "entropy": entropy,
        "top": [
            {
                "action": int(mask[int(i)]),
                "prob": float(probs[int(i)]),
                "logit": float(logits[mask[int(i)]]),
            }
            for i in order
        ],
    }
