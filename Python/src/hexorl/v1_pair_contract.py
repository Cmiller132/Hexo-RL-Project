"""Shared V1 pair-action contracts.

The V1 pair model, runtime, replay, and training code all consume the feature
order defined here. Keep this module dependency-light so it can be imported by
graph batching and replay serialization without creating runtime cycles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


V1_PAIR_FEATURE_SCHEMA_VERSION = 2
V1_PAIR_FEATURE_NAMES: tuple[str, ...] = (
    "axial_distance_norm",
    "same_axis",
    "same_line",
    "same_window",
    "terminal_exact_win",
    "terminal_equivalent_win",
    "terminal_exact_cover",
    "covers_all_opponent_win_requirements",
    "impossible_to_cover",
    "phase_full_turn",
    "phase_known_first",
    "phase_both_legal",
)
V1_PAIR_FEATURE_DIM = len(V1_PAIR_FEATURE_NAMES)
V1_TERMINAL_TACTICAL_TARGET_DIM = 8


TACTICAL_STATUS_QUIET = "quiet"
TACTICAL_STATUS_HOT_COMPLETION = "hot_completion_available"
TACTICAL_STATUS_HOT_COVER = "hot_cover_required"
TACTICAL_STATUS_HOT_COVER_IMPOSSIBLE = "hot_cover_impossible"
TACTICAL_STATUSES = frozenset(
    {
        TACTICAL_STATUS_QUIET,
        TACTICAL_STATUS_HOT_COMPLETION,
        TACTICAL_STATUS_HOT_COVER,
        TACTICAL_STATUS_HOT_COVER_IMPOSSIBLE,
    }
)


PairCoord = tuple[int, int]
PairKey = tuple[PairCoord, PairCoord]


def canonical_pair_key(first: Any, second: Any) -> PairKey:
    a = _coord(first)
    b = _coord(second)
    if a == b:
        raise ValueError(f"duplicate coordinates are illegal for a V1 pair: {a}")
    return (a, b) if a <= b else (b, a)


def hex_distance(first: PairCoord, second: PairCoord) -> int:
    dq = int(first[0]) - int(second[0])
    dr = int(first[1]) - int(second[1])
    return max(abs(dq), abs(dr), abs(dq + dr))


def same_axis(first: PairCoord, second: PairCoord) -> bool:
    return (
        int(first[0]) == int(second[0])
        or int(first[1]) == int(second[1])
        or int(first[0]) + int(first[1]) == int(second[0]) + int(second[1])
    )


def same_line(first: PairCoord, second: PairCoord) -> bool:
    return same_axis(first, second)


def same_window(first: PairCoord, second: PairCoord) -> bool:
    return same_line(first, second) and hex_distance(first, second) <= 5


@dataclass(frozen=True)
class V1TerminalTacticalPayload:
    status: str = TACTICAL_STATUS_QUIET
    winning_single_cells: tuple[PairCoord, ...] = ()
    hot_completion_pairs: tuple[PairKey, ...] = ()
    terminal_equivalent_pairs: tuple[PairKey, ...] = ()
    opponent_win_requirements: tuple[PairCoord, ...] = ()
    hot_cover_pairs: tuple[PairKey, ...] = ()
    impossible_to_cover: bool = False
    pair_row_schema_version: int = 1

    def __post_init__(self) -> None:
        status = str(self.status or TACTICAL_STATUS_QUIET)
        if status not in TACTICAL_STATUSES:
            raise ValueError(f"unsupported V1 tactical status: {status!r}")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "winning_single_cells", _coords(self.winning_single_cells))
        object.__setattr__(self, "hot_completion_pairs", _pairs(self.hot_completion_pairs))
        object.__setattr__(
            self,
            "terminal_equivalent_pairs",
            _pairs(self.terminal_equivalent_pairs),
        )
        object.__setattr__(
            self,
            "opponent_win_requirements",
            _coords(self.opponent_win_requirements),
        )
        object.__setattr__(self, "hot_cover_pairs", _pairs(self.hot_cover_pairs))
        object.__setattr__(self, "impossible_to_cover", bool(self.impossible_to_cover))
        object.__setattr__(self, "pair_row_schema_version", int(self.pair_row_schema_version or 1))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "V1TerminalTacticalPayload":
        if not data:
            return cls()
        return cls(
            status=str(data.get("status", TACTICAL_STATUS_QUIET)),
            winning_single_cells=tuple(data.get("winning_single_cells", ())),
            hot_completion_pairs=_pair_rows_or_pairs(data.get("hot_completion_pairs", ())),
            terminal_equivalent_pairs=_pair_rows_or_pairs(
                data.get("terminal_equivalent_pairs", ())
            ),
            opponent_win_requirements=tuple(data.get("opponent_win_requirements", ())),
            hot_cover_pairs=_pair_rows_or_pairs(data.get("hot_cover_pairs", ())),
            impossible_to_cover=bool(data.get("impossible_to_cover", False)),
            pair_row_schema_version=int(data.get("pair_row_schema_version", 1) or 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "winning_single_cells": [list(cell) for cell in self.winning_single_cells],
            "hot_completion_pairs": _pairs_to_json(self.hot_completion_pairs),
            "terminal_equivalent_pairs": _pairs_to_json(self.terminal_equivalent_pairs),
            "opponent_win_requirements": [
                list(cell) for cell in self.opponent_win_requirements
            ],
            "hot_cover_pairs": _pairs_to_json(self.hot_cover_pairs),
            "impossible_to_cover": self.impossible_to_cover,
            "pair_row_schema_version": self.pair_row_schema_version,
        }


def v1_pair_features_for_candidates(
    candidates: Sequence[Any],
    tactical_payload: Mapping[str, Any] | V1TerminalTacticalPayload | None = None,
) -> np.ndarray:
    tactical = (
        tactical_payload
        if isinstance(tactical_payload, V1TerminalTacticalPayload)
        else V1TerminalTacticalPayload.from_mapping(tactical_payload)
    )
    hot_completion = set(tactical.hot_completion_pairs)
    terminal_equivalent = set(tactical.terminal_equivalent_pairs)
    hot_cover = set(tactical.hot_cover_pairs)
    out = np.zeros((len(candidates), V1_PAIR_FEATURE_DIM), dtype=np.float32)
    for row, candidate in enumerate(candidates):
        pair_key = canonical_pair_key(*getattr(candidate, "pair_key"))
        first, second = pair_key
        terminal_exact_win = bool(getattr(candidate, "terminal_exact_flag", False)) and (
            pair_key in hot_completion or pair_key not in hot_cover
        )
        terminal_equiv = bool(getattr(candidate, "terminal_equivalence_flag", False)) or (
            pair_key in terminal_equivalent
        )
        terminal_cover = pair_key in hot_cover or "opponent_hot_cover" in str(
            getattr(candidate, "candidate_selection_reason", "")
        )
        covers_all = terminal_cover and bool(tactical.opponent_win_requirements)
        out[row, 0] = min(float(hex_distance(first, second)), 32.0) / 32.0
        out[row, 1] = 1.0 if same_axis(first, second) else 0.0
        out[row, 2] = 1.0 if same_line(first, second) else 0.0
        out[row, 3] = 1.0 if same_window(first, second) else 0.0
        out[row, 4] = 1.0 if terminal_exact_win else 0.0
        out[row, 5] = 1.0 if terminal_equiv else 0.0
        out[row, 6] = 1.0 if terminal_cover else 0.0
        out[row, 7] = 1.0 if covers_all else 0.0
        out[row, 8] = 1.0 if tactical.impossible_to_cover else 0.0
        out[row, 9] = 1.0
        out[row, 10] = 0.0
        out[row, 11] = 1.0
    return out


def v1_pair_features_from_qr(
    pair_qr: np.ndarray,
    *,
    placements_remaining: float,
) -> np.ndarray:
    pairs = np.asarray(pair_qr, dtype=np.int32).reshape(-1, 4)
    out = np.zeros((pairs.shape[0], V1_PAIR_FEATURE_DIM), dtype=np.float32)
    full_turn = float(placements_remaining) >= 1.5
    known_first = 0.0 < float(placements_remaining) < 1.5
    for row, (q1, r1, q2, r2) in enumerate(pairs.tolist()):
        first = (int(q1), int(r1))
        second = (int(q2), int(r2))
        out[row, 0] = min(float(hex_distance(first, second)), 32.0) / 32.0
        out[row, 1] = 1.0 if same_axis(first, second) else 0.0
        out[row, 2] = 1.0 if same_line(first, second) else 0.0
        out[row, 3] = 1.0 if same_window(first, second) else 0.0
        out[row, 9] = 1.0 if full_turn else 0.0
        out[row, 10] = 1.0 if known_first else 0.0
        out[row, 11] = 1.0
    return out


def terminal_tactical_target_vector(
    tactical_payload: Mapping[str, Any] | V1TerminalTacticalPayload | None,
    *,
    selected_pair: PairKey | None = None,
) -> np.ndarray:
    tactical = (
        tactical_payload
        if isinstance(tactical_payload, V1TerminalTacticalPayload)
        else V1TerminalTacticalPayload.from_mapping(tactical_payload)
    )
    selected = None if selected_pair is None else canonical_pair_key(*selected_pair)
    out = np.zeros(V1_TERMINAL_TACTICAL_TARGET_DIM, dtype=np.float32)
    out[0] = 1.0 if tactical.status == TACTICAL_STATUS_HOT_COMPLETION else 0.0
    out[1] = 1.0 if tactical.terminal_equivalent_pairs else 0.0
    out[2] = 1.0 if tactical.status == TACTICAL_STATUS_HOT_COVER else 0.0
    out[3] = 1.0 if tactical.status == TACTICAL_STATUS_HOT_COVER_IMPOSSIBLE else 0.0
    out[4] = 1.0 if tactical.winning_single_cells else 0.0
    out[5] = 1.0 if tactical.hot_cover_pairs else 0.0
    out[6] = 1.0 if tactical.impossible_to_cover else 0.0
    if selected is not None:
        tactical_pairs = (
            set(tactical.hot_completion_pairs)
            | set(tactical.terminal_equivalent_pairs)
            | set(tactical.hot_cover_pairs)
        )
        out[7] = 1.0 if selected in tactical_pairs else 0.0
    return out


def _coord(value: Any) -> PairCoord:
    return (int(value[0]), int(value[1]))


def _coords(values: Sequence[Any]) -> tuple[PairCoord, ...]:
    return tuple(sorted({_coord(value) for value in values}))


def _pairs(values: Sequence[Any]) -> tuple[PairKey, ...]:
    return tuple(sorted({canonical_pair_key(*value) for value in values}))


def _pair_rows_or_pairs(values: Sequence[Any]) -> tuple[PairKey, ...]:
    pairs: list[PairKey] = []
    for item in values:
        if isinstance(item, Mapping):
            if "first" in item and "second" in item:
                pairs.append(canonical_pair_key(item["first"], item["second"]))
            elif "pair_key" in item:
                key = item["pair_key"]
                pairs.append(canonical_pair_key(key[0], key[1]))
            elif all(key in item for key in ("first_q", "first_r", "second_q", "second_r")):
                pairs.append(
                    canonical_pair_key(
                        (item["first_q"], item["first_r"]),
                        (item["second_q"], item["second_r"]),
                    )
                )
            continue
        if (
            isinstance(item, Sequence)
            and not isinstance(item, (str, bytes, bytearray))
            and len(item) >= 7
            and not isinstance(item[0], (Sequence, Mapping))
        ):
            pairs.append(canonical_pair_key((item[3], item[4]), (item[5], item[6])))
            continue
        pairs.append(canonical_pair_key(item[0], item[1]))
    return tuple(sorted(set(pairs)))


def _pairs_to_json(pairs: Sequence[PairKey]) -> list[list[list[int]]]:
    return [[list(first), list(second)] for first, second in pairs]
