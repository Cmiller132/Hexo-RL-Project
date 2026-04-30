"""Replay reconstruction and board-debug helpers for the dashboard."""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from hexorl.contracts.candidates import (
    CANDIDATE_FEATURE_NAMES,
    CANDIDATE_FEATURE_VERSION,
    CandidateContractBuilder,
)
from hexorl.action_contract.tactical_oracle import scan_tactical_oracle_from_history
from hexorl.contracts.history import MoveHistory, encode_move_history as contract_encode_move_history
from hexorl.engine.encoding import encode_board_and_legal
from hexorl.engine.history import game_from_history
from hexorl.engine.legal import decode_legal_bytes
from hexorl.engine.rust import engine_available
from hexorl.dashboard.db import DashboardStore
from hexorl.selfplay.records import BOARD_SIZE

HAS_ENGINE = engine_available()


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
    return list(MoveHistory.decode(history, source="rust").rows)


def encode_move_history(moves: Iterable[Move]) -> bytes:
    return contract_encode_move_history(list(moves))


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
        "positions": [_public_position(row) for row in position_rows],
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
    if not HAS_ENGINE:
        raise RuntimeError("Rust _engine extension is required for tensor encoding")
    tensor_3d, offset_q, offset_r, _legal_rows, legal_bytes = encode_board_and_legal(
        history,
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
        raise RuntimeError("Rust engine is required for dashboard replay positions")

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
    return game_from_history(encode_move_history(moves))


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
    return [(int(q), int(r)) for q, r in decode_legal_bytes(data).tolist()]


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


def _public_position(row: dict[str, Any]) -> dict[str, Any]:
    debug = dict(row.get("debug_json", {}) or {})
    if "policy_weight" not in debug:
        debug["policy_weight"] = 1.0 if debug.get("is_full_search", True) else 0.0
    debug.setdefault("opp_policy_weight", 0.0)
    debug.setdefault("value_weight", 1.0)
    debug.setdefault("regret_weight", 0.0)
    debug.setdefault("final_outcome", debug.get("outcome"))
    selected_action_value = debug.get("selected_action_value")
    if selected_action_value is not None and debug.get("final_outcome") is not None:
        perspective_outcome = float(debug["final_outcome"])
        if int(row["player"]) == 1:
            perspective_outcome = -perspective_outcome
        debug.setdefault(
            "per_step_error",
            (float(selected_action_value) - perspective_outcome) ** 2,
        )
    debug["candidate_rows"] = _candidate_rows_debug(row, debug)
    prior_sources = {
        key: debug.get(key, 0.0)
        for key in (
            "sparse_prior_stage",
            "sparse_prior_root_candidate_count",
            "sparse_prior_leaf_candidate_count",
            "sparse_prior_root_hit_frac",
            "sparse_prior_leaf_hit_frac",
            "fallback_prior_use",
            "fallback_prior_use_on_mcts_top1",
            "fallback_prior_use_on_mcts_top4",
            "fallback_prior_use_on_mcts_top8",
            "pair_prior_candidate_count",
            "pair_prior_hit_frac",
            "pair_fallback_prior_use",
            "pair_fallback_prior_use_on_mcts_top1",
            "pair_fallback_prior_use_on_mcts_top4",
            "pair_fallback_prior_use_on_mcts_top8",
        )
        if key in debug
    }
    return {
        "position_id": row["position_id"],
        "turn_index": row["turn_index"],
        "player": row["player"],
        "root_value": row["root_value"],
        "selected_action_value": debug.get("selected_action_value"),
        "final_outcome": debug.get("final_outcome"),
        "per_step_error": debug.get("per_step_error"),
        "regret_rank": debug.get("regret_rank", 0.0),
        "regret_value": debug.get("regret_value", 0.0),
        "value_weight": debug["value_weight"],
        "policy_weight": debug["policy_weight"],
        "opp_policy_weight": debug["opp_policy_weight"],
        "regret_weight": debug["regret_weight"],
        "policy": row.get("policy_json", {}),
        "policy_target_v2": debug.get("policy_target_v2", []),
        "opp_policy_target_v2": debug.get("opp_policy_target_v2", []),
        "pair_policy_target_v2": debug.get("pair_policy_target_v2", []),
        "prior_sources": prior_sources,
        "debug": debug,
    }


def _candidate_rows_debug(row: dict[str, Any], debug: dict[str, Any], limit: int = 64) -> dict[str, Any]:
    history = row.get("move_history_b64") or b""
    policy_v2 = _policy_v2_from_debug(debug.get("policy_target_v2", []))
    try:
        position = get_replay_position(history, constrain_threats=False)
        legal = [(int(move["q"]), int(move["r"])) for move in position.legal_moves]
        offset_q = int(position.encoding.get("offset_q", -BOARD_SIZE // 2))
        offset_r = int(position.encoding.get("offset_r", -BOARD_SIZE // 2))
        oracle = scan_tactical_oracle_from_history(
            history,
            legal,
            offset_q=offset_q,
            offset_r=offset_r,
        )
        candidates = CandidateContractBuilder().build(
            legal,
            policy_v2,
            offset_q=offset_q,
            offset_r=offset_r,
            budget=min(max(len(legal), 1), 512),
            winning_moves=oracle.win_now_cells,
            forced_block_moves=oracle.forced_block_cells,
            cover_cells=oracle.cover_cells,
            open_four_cells=oracle.open_four_cells,
            open_five_cells=oracle.open_five_cells,
        )
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "feature_version": CANDIDATE_FEATURE_VERSION,
            "feature_names": list(CANDIDATE_FEATURE_NAMES),
            "rows": [],
        }

    rows = []
    active = np.flatnonzero(candidates.mask)
    feature_names = list(CANDIDATE_FEATURE_NAMES)
    for row_idx in active[:limit]:
        row_i = int(row_idx)
        rows.append(
            {
                "row": row_i,
                "q": int(candidates.qr[row_i, 0]),
                "r": int(candidates.qr[row_i, 1]),
                "dense_index": int(candidates.indices[row_i]),
                "target_prob": float(candidates.target[row_i]),
                "features": {
                    name: float(candidates.features[row_i, col])
                    for col, name in enumerate(feature_names)
                },
            }
        )
    return {
        "available": True,
        "feature_version": CANDIDATE_FEATURE_VERSION,
        "feature_names": feature_names,
        "candidate_count": int(active.shape[0]),
        "shown": len(rows),
        "missing_mass": float(candidates.missing_mass),
        "recall_top1": float(candidates.recall_top1),
        "recall_top4": float(candidates.recall_top4),
        "recall_top8": float(candidates.recall_top8),
        "recall_winning_move": float(candidates.recall_winning_move),
        "recall_forced_block": float(candidates.recall_forced_block),
        "recall_two_placement_cover": float(candidates.recall_two_placement_cover),
        "rows": rows,
    }


def _policy_v2_from_debug(value: Any) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        q, r, prob = item
        try:
            out.append((int(q), int(r), float(prob)))
        except (TypeError, ValueError):
            continue
    return out


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
