"""Single Python owner for Hex D6 transforms."""

from __future__ import annotations

import numpy as np

from hexorl.contracts.coordinates import BOARD_SIZE
from hexorl.contracts.history import MoveHistory
from hexorl.contracts.legal import LegalActionTable
from hexorl.contracts.validation import ContractValidationError


def transform_qr(qr: tuple[int, int], sym_idx: int) -> tuple[int, int]:
    q, r = int(qr[0]), int(qr[1])
    sym = int(sym_idx) % 12
    if sym == 0:
        return (q, r)
    if sym == 1:
        return (-r, q + r)
    if sym == 2:
        return (-q - r, q)
    if sym == 3:
        return (-q, -r)
    if sym == 4:
        return (r, -q - r)
    if sym == 5:
        return (q + r, -q)
    if sym == 6:
        return (r, q)
    if sym == 7:
        return (-q, q + r)
    if sym == 8:
        return (-q - r, r)
    if sym == 9:
        return (-r, -q)
    if sym == 10:
        return (q, -q - r)
    return (q + r, -r)


def transform_history(
    history: bytes | MoveHistory,
    sym_idx: int,
    *,
    source: str = "rust",
    allow_fixture: bool = False,
) -> bytes:
    contract = history if isinstance(history, MoveHistory) else MoveHistory.decode(
        history,
        source=source,
        allow_fixture=allow_fixture,
    )
    rows = tuple((player, *transform_qr((q, r), sym_idx)) for player, q, r in contract.rows)
    return MoveHistory.from_rows(rows, source=contract.source, radius=contract.radius, allow_fixture=contract.allow_fixture).encode()


def transform_legal_table(table: LegalActionTable, sym_idx: int) -> LegalActionTable:
    rows = [transform_qr((int(q), int(r)), sym_idx) for q, r in table.rows.tolist()]
    return LegalActionTable.from_rows(
        rows,
        source=table.source,
        radius=table.radius,
        occupied_count=table.occupied_count,
        current_player=table.current_player,
        placements_remaining=table.placements_remaining,
        history_hash=table.history_hash,
        allow_fixture=table.allow_fixture,
    )


def transform_policy_target(target, sym_idx: int):
    merged: dict[tuple[int, int], float] = {}
    for q, r, prob in target:
        if float(prob) <= 0.0:
            continue
        qr = transform_qr((int(q), int(r)), sym_idx)
        merged[qr] = merged.get(qr, 0.0) + float(prob)
    return [(q, r, prob) for (q, r), prob in merged.items()]


def transform_pair_policy_target(target, sym_idx: int):
    out = []
    for first, second, prob in target:
        if float(prob) <= 0.0:
            continue
        out.append((transform_qr(first, sym_idx), transform_qr(second, sym_idx), float(prob)))
    return out


def transform_dense_policy(policy: np.ndarray, sym_idx: int) -> np.ndarray:
    flat = np.asarray(policy)
    if flat.shape != (BOARD_SIZE * BOARD_SIZE,):
        raise ContractValidationError("dense policy must have shape (1089,)", owner="contracts.symmetry")
    grid = flat.reshape(BOARD_SIZE, BOARD_SIZE)
    transformed = apply_tensor_symmetry(grid[None, :, :], sym_idx)[0].reshape(-1)
    total = float(transformed.sum())
    if total > 0.0:
        transformed = transformed / total
    return transformed.astype(flat.dtype, copy=False)


def transform_axis_label(axis_label: int, sym_idx: int) -> int:
    if int(axis_label) < 0:
        return int(axis_label)
    axes = [(1, 0), (0, 1), (1, -1)]
    tq, tr = transform_qr(axes[int(axis_label) % 3], sym_idx)
    for idx, axis in enumerate(axes):
        if (tq, tr) == axis or (tq, tr) == (-axis[0], -axis[1]):
            return idx
    return int(axis_label)


def transform_axis_maps(axis_maps: np.ndarray, sym_idx: int) -> np.ndarray:
    maps = np.asarray(axis_maps)
    if maps.shape != (6, BOARD_SIZE, BOARD_SIZE):
        raise ContractValidationError("axis maps must have shape (6, 33, 33)", owner="contracts.symmetry")
    spatial = apply_tensor_symmetry(maps, sym_idx)
    out = np.zeros_like(spatial)
    for src_axis in range(3):
        dst_axis = transform_axis_label(src_axis, sym_idx)
        out[dst_axis] += spatial[src_axis]
        out[dst_axis + 3] += spatial[src_axis + 3]
    return out


def apply_tensor_symmetry(tensor: np.ndarray, sym_idx: int) -> np.ndarray:
    arr = np.asarray(tensor)
    if arr.ndim != 3 or arr.shape[1:] != (BOARD_SIZE, BOARD_SIZE):
        raise ContractValidationError("tensor symmetry expects shape (C, 33, 33)", owner="contracts.symmetry")
    half = BOARD_SIZE // 2
    yi, xi = np.mgrid[0:BOARD_SIZE, 0:BOARD_SIZE]
    qi = yi - half
    rj = xi - half
    q_t, r_t = _transform_qr_arrays(qi, rj, int(sym_idx) % 12)
    ti = q_t + half
    tj = r_t + half
    valid = (ti >= 0) & (ti < BOARD_SIZE) & (tj >= 0) & (tj < BOARD_SIZE)
    result = np.zeros_like(arr)
    for channel in range(arr.shape[0]):
        result[channel, ti[valid], tj[valid]] = arr[channel, yi[valid], xi[valid]]
    return result


def compose_symmetries(first: int, second: int) -> int:
    probe = (2, -1)
    target = transform_qr(transform_qr(probe, first), second)
    for sym in range(12):
        if transform_qr(probe, sym) == target and transform_qr((1, 2), sym) == transform_qr(transform_qr((1, 2), first), second):
            return sym
    raise ContractValidationError("failed to compose D6 symmetries", owner="contracts.symmetry")


def inverse_symmetry(sym_idx: int) -> int:
    sym = int(sym_idx) % 12
    for candidate in range(12):
        if compose_symmetries(sym, candidate) == 0 and compose_symmetries(candidate, sym) == 0:
            return candidate
    raise ContractValidationError("failed to invert D6 symmetry", owner="contracts.symmetry")


def _transform_qr_arrays(q, r, sym: int):
    if sym == 0:
        return q, r
    if sym == 1:
        return -r, q + r
    if sym == 2:
        return -q - r, q
    if sym == 3:
        return -q, -r
    if sym == 4:
        return r, -q - r
    if sym == 5:
        return q + r, -q
    if sym == 6:
        return r, q
    if sym == 7:
        return -q, q + r
    if sym == 8:
        return -q - r, r
    if sym == 9:
        return -r, -q
    if sym == 10:
        return q, -q - r
    return q + r, -r
