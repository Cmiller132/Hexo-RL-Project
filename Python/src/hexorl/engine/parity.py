"""Parity helpers for engine/contract tests."""

from __future__ import annotations

import numpy as np

from hexorl.contracts.history import MoveHistory
from hexorl.contracts.symmetry import transform_history as python_transform_history
from hexorl.engine.encoding import apply_d6_symmetry, encode_compact_record
from hexorl.engine.legal import LegalTableProvider


def legal_rows_for_history(history: bytes | MoveHistory) -> np.ndarray:
    return LegalTableProvider().from_history(history).rows


def transformed_history_bytes(history: bytes | MoveHistory, sym_idx: int) -> bytes:
    return python_transform_history(history, sym_idx)


def rust_tensor_after_symmetry(history: bytes | MoveHistory, sym_idx: int) -> np.ndarray:
    tensor = encode_compact_record(history)[-1]
    return apply_d6_symmetry(tensor, sym_idx)
