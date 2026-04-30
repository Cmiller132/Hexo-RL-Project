"""Coordinate constants and canonical row helpers."""

from __future__ import annotations

BOARD_SIZE = 33
BOARD_AREA = BOARD_SIZE * BOARD_SIZE
BOARD_HALF = BOARD_SIZE // 2
PLACEMENT_RADIUS = 8
HISTORY_STRIDE = 12


def dense_index(q: int, r: int, *, offset_q: int = -BOARD_HALF, offset_r: int = -BOARD_HALF) -> int:
    gi = int(q) - int(offset_q)
    gj = int(r) - int(offset_r)
    if 0 <= gi < BOARD_SIZE and 0 <= gj < BOARD_SIZE:
        return int(gi * BOARD_SIZE + gj)
    return -1


def hex_distance(a: tuple[int, int], b: tuple[int, int] = (0, 0)) -> int:
    dq = int(a[0]) - int(b[0])
    dr = int(a[1]) - int(b[1])
    return max(abs(dq), abs(dr), abs(dq + dr))


def canonical_legal_sort(rows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(rows, key=lambda qr: (hex_distance(qr), qr[0], qr[1]))
