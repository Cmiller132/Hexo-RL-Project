"""Python-hackable axis policy target design primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

from hexorl.selfplay.records import BOARD_AREA, BOARD_SIZE


AXES: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, -1))
DEFAULT_OFFSET = -16


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    default: float
    min: float
    max: float
    step: float = 0.05
    description: str = ""


@dataclass
class AxisPolicyInput:
    """Position snapshot consumed by axis target prototypes."""

    stones: list[dict[str, int]]
    legal_moves: list[dict[str, int]]
    current_player: int = 0
    offset_q: int = DEFAULT_OFFSET
    offset_r: int = DEFAULT_OFFSET
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def own_stones(self) -> set[tuple[int, int]]:
        return {
            (int(s["q"]), int(s["r"]))
            for s in self.stones
            if int(s.get("player", 0)) == self.current_player
        }

    @property
    def opp_stones(self) -> set[tuple[int, int]]:
        return {
            (int(s["q"]), int(s["r"]))
            for s in self.stones
            if int(s.get("player", 0)) != self.current_player
        }

    @property
    def legal_set(self) -> set[tuple[int, int]]:
        return {(int(m["q"]), int(m["r"])) for m in self.legal_moves}


@dataclass
class AxisPolicyResult:
    prototype_id: str
    parameters: dict[str, float]
    axis_maps: np.ndarray
    combined_policy: np.ndarray
    debug_terms: dict[str, Any]
    offset_q: int = DEFAULT_OFFSET
    offset_r: int = DEFAULT_OFFSET
    current_player: int = 0

    def to_json(self) -> dict[str, Any]:
        cells = []
        axis_count = min(3, self.axis_maps.shape[0])
        for i in range(self.axis_maps.shape[1]):
            for j in range(self.axis_maps.shape[2]):
                axis_values = [float(self.axis_maps[axis, i, j]) for axis in range(axis_count)]
                if not any(abs(value) > 1e-7 for value in axis_values):
                    continue
                score = max(axis_values, key=lambda value: abs(value))
                owner = int(self.current_player) if score >= 0 else 1 - int(self.current_player)
                cells.append(
                    {
                        "q": int(i + self.offset_q),
                        "r": int(j + self.offset_r),
                        "score": float(score),
                        "owner": owner,
                        "axes": axis_values,
                    }
                )
        cells.sort(key=lambda cell: abs(float(cell["score"])), reverse=True)
        return {
            "prototype_id": self.prototype_id,
            "parameters": self.parameters,
            "offset_q": self.offset_q,
            "offset_r": self.offset_r,
            "current_player": self.current_player,
            "axis_summaries": [
                {
                    "axis": axis,
                    "sum": float(self.axis_maps[axis].sum()),
                    "min": float(self.axis_maps[axis].min()),
                    "max": float(self.axis_maps[axis].max()),
                    "nonzero": int(np.count_nonzero(self.axis_maps[axis])),
                }
                for axis in range(3)
            ],
            "cells": cells,
            "debug_terms": self.debug_terms,
        }


class AxisPolicyPrototype(Protocol):
    prototype_id: str
    label: str
    description: str
    parameters: tuple[ParameterSpec, ...]

    def compute(
        self,
        position: AxisPolicyInput,
        parameters: Mapping[str, float] | None = None,
    ) -> AxisPolicyResult:
        ...


def merge_parameters(
    specs: tuple[ParameterSpec, ...],
    overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    values = {spec.name: float(spec.default) for spec in specs}
    for key, value in (overrides or {}).items():
        values[key] = float(value)
    return values


def empty_axis_maps() -> np.ndarray:
    return np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)


def normalize_policy(
    scores: np.ndarray,
    legal_moves: set[tuple[int, int]],
    offset_q: int,
    offset_r: int,
    *,
    fallback_uniform: bool = True,
) -> np.ndarray:
    """Legal-mask and normalize a flat or board-shaped score array."""
    flat_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    masked = np.zeros(BOARD_AREA, dtype=np.float64)
    for q, r in legal_moves:
        idx = flat_index(q, r, offset_q, offset_r)
        if idx >= 0:
            masked[idx] = max(float(flat_scores[idx]), 0.0)
    total = float(masked.sum())
    if fallback_uniform and total <= 0.0 and legal_moves:
        for q, r in legal_moves:
            idx = flat_index(q, r, offset_q, offset_r)
            if idx >= 0:
                masked[idx] = 1.0
        total = float(masked.sum())
    if total > 0.0:
        masked /= total
    return masked.astype(np.float32)


def flat_index(q: int, r: int, offset_q: int, offset_r: int) -> int:
    i = int(q) - int(offset_q)
    j = int(r) - int(offset_r)
    if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
        return i * BOARD_SIZE + j
    return -1


def board_index(q: int, r: int, offset_q: int, offset_r: int) -> tuple[int, int] | None:
    i = int(q) - int(offset_q)
    j = int(r) - int(offset_r)
    if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
        return i, j
    return None


def line_count(stones: set[tuple[int, int]], q: int, r: int, dq: int, dr: int) -> int:
    count = 0
    nq, nr = q + dq, r + dr
    while (nq, nr) in stones:
        count += 1
        nq += dq
        nr += dr
    return count


def distance_to_stones(q: int, r: int, stones: set[tuple[int, int]]) -> int:
    if not stones:
        return 0
    return min(hex_distance(q, r, sq, sr) for sq, sr in stones)


def hex_distance(q1: int, r1: int, q2: int, r2: int) -> int:
    dq = q1 - q2
    dr = r1 - r2
    return max(abs(dq), abs(dr), abs(dq + dr))
