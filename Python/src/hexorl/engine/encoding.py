"""Python-facing Rust encoding calls."""

from __future__ import annotations

import numpy as np

from hexorl.contracts.history import MoveHistory
from hexorl.engine.history import game_from_history
from hexorl.engine.legal import decode_legal_bytes
from hexorl.engine.rust import engine_module


def encode_compact_record(history: bytes | MoveHistory, near_radius: int = 8) -> np.ndarray:
    payload = history.encode() if isinstance(history, MoveHistory) else bytes(history)
    module = engine_module(required=True)
    return np.asarray(module.encode_compact_record(payload, int(near_radius)), dtype=np.float32)


def apply_d6_symmetry(tensor: np.ndarray, sym_idx: int) -> np.ndarray:
    module = engine_module(required=True)
    return np.asarray(module.apply_d6_symmetry(np.asarray(tensor), int(sym_idx)))


def encode_board_and_legal(history: bytes | MoveHistory, near_radius: int = 8, constrain_threats: bool = True):
    game = game_from_history(history)
    tensor_3d, offset_q, offset_r, legal_bytes = game.encode_board_and_legal(int(near_radius), bool(constrain_threats))
    return (
        np.asarray(tensor_3d, dtype=np.float32),
        int(offset_q),
        int(offset_r),
        decode_legal_bytes(legal_bytes),
        legal_bytes,
    )
