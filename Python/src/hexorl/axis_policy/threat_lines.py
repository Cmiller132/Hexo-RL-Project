"""Threat-focused line strength prototypes for the Axis Lab."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from hexorl.axis_policy.core import (
    AXES,
    AxisPolicyInput,
    AxisPolicyResult,
    ParameterSpec,
    board_index,
    empty_axis_maps,
    merge_parameters,
    normalize_policy,
)
from hexorl.selfplay.records import BOARD_SIZE

WIN_LENGTH = 6


class ThreatWindowStrengthPrototype:
    prototype_id = "threat_window_strength"
    label = "Threat Window Strength"
    description = "Signed per-axis line strength from pure 6-cell windows, focused on 3/4/5-stone threats."
    parameters = (
        ParameterSpec("three", 0.15, 0.0, 1.0, 0.01, "Weight for a pure 3-stone window."),
        ParameterSpec("four", 0.55, 0.0, 3.0, 0.01, "Weight for a pure 4-stone hot window."),
        ParameterSpec("five", 1.4, 0.0, 6.0, 0.05, "Weight for a pure 5-stone immediate threat."),
        ParameterSpec("own_weight", 1.0, 0.0, 4.0, 0.05, "Current-player threat multiplier."),
        ParameterSpec("opp_weight", 1.15, 0.0, 4.0, 0.05, "Opponent-threat multiplier."),
        ParameterSpec("block_visibility", 0.9, 0.0, 3.0, 0.05, "How strongly opponent threats appear in top-cell display."),
    )

    def compute(
        self,
        position: AxisPolicyInput,
        parameters: Mapping[str, float] | None = None,
    ) -> AxisPolicyResult:
        params = merge_parameters(self.parameters, parameters)
        maps = empty_axis_maps()
        own = position.own_stones
        opp = position.opp_stones
        urgency = _urgency(params)
        debug = {"own_windows": 0, "opp_windows": 0, "contested_skipped": 0}

        for axis, (dq, dr) in enumerate(AXES):
            for gi in range(BOARD_SIZE):
                for gj in range(BOARD_SIZE):
                    q = gi + position.offset_q
                    r = gj + position.offset_r
                    total = 0.0
                    for cells in _windows_containing(q, r, dq, dr):
                        own_count, opp_count = _counts(cells, own, opp)
                        if own_count and opp_count:
                            debug["contested_skipped"] += 1
                            continue
                        if own_count >= 3:
                            total += params["own_weight"] * urgency[own_count]
                            debug["own_windows"] += 1
                        elif opp_count >= 3:
                            total -= params["opp_weight"] * urgency[opp_count]
                            debug["opp_windows"] += 1
                    maps[axis, gi, gj] = total

        combined = _signed_legal_display(
            maps,
            position.legal_set,
            position.offset_q,
            position.offset_r,
            block_visibility=params["block_visibility"],
        )
        return AxisPolicyResult(
            self.prototype_id,
            params,
            maps,
            combined,
            {
                **debug,
                "target_kind": "signed_threat_line_field",
                "negative_values": "opponent pure-window strength",
                "positive_values": "current-player pure-window strength",
            },
            position.offset_q,
            position.offset_r,
        )


class ForcingCellPrototype:
    prototype_id = "forcing_cells"
    label = "Forcing Cells"
    description = "Scores legal cells that win now, complete a two-stone turn threat, block wins, or create hot windows."
    parameters = (
        ParameterSpec("win_now", 8.0, 0.0, 30.0, 0.25, "Own 5-window completion."),
        ParameterSpec("turn_win", 3.5, 0.0, 20.0, 0.25, "Own 4-window cell when two placements are available."),
        ParameterSpec("block_now", 7.0, 0.0, 30.0, 0.25, "Opponent 5-window block."),
        ParameterSpec("block_turn", 3.0, 0.0, 20.0, 0.25, "Opponent 4-window block pressure."),
        ParameterSpec("create_hot", 1.2, 0.0, 10.0, 0.1, "Own 3-window cell that creates a hot window."),
    )

    def compute(
        self,
        position: AxisPolicyInput,
        parameters: Mapping[str, float] | None = None,
    ) -> AxisPolicyResult:
        params = merge_parameters(self.parameters, parameters)
        maps = empty_axis_maps()
        own = position.own_stones
        opp = position.opp_stones
        remaining = int(position.metadata.get("placements_remaining", 2))
        debug = {"win_now": 0, "turn_win": 0, "block_now": 0, "block_turn": 0, "create_hot": 0}

        for q, r in position.legal_set:
            ij = board_index(q, r, position.offset_q, position.offset_r)
            if ij is None:
                continue
            for axis, (dq, dr) in enumerate(AXES):
                score = 0.0
                for cells in _windows_containing(q, r, dq, dr):
                    own_count, opp_count = _counts(cells, own, opp)
                    if opp_count == 0:
                        after = own_count + 1
                        if after >= 6:
                            score += params["win_now"]
                            debug["win_now"] += 1
                        elif own_count == 4 and remaining >= 2:
                            score += params["turn_win"]
                            debug["turn_win"] += 1
                        elif own_count == 3:
                            score += params["create_hot"]
                            debug["create_hot"] += 1
                    if own_count == 0:
                        if opp_count >= 5:
                            score += params["block_now"]
                            debug["block_now"] += 1
                        elif opp_count == 4:
                            score += params["block_turn"]
                            debug["block_turn"] += 1
                maps[axis, ij[0], ij[1]] = score

        combined = normalize_policy(
            maps.max(axis=0),
            position.legal_set,
            position.offset_q,
            position.offset_r,
        )
        return AxisPolicyResult(
            self.prototype_id,
            params,
            maps,
            combined,
            {"target_kind": "legal_cell_forcing_strength", "placements_remaining": remaining, **debug},
            position.offset_q,
            position.offset_r,
        )


class MultiLineThreatPrototype:
    prototype_id = "multi_line_threats"
    label = "Multi-Line Threats"
    description = "Rewards cells that touch multiple independent threat windows, emphasizing forks and double blocks."
    parameters = (
        ParameterSpec("three", 0.35, 0.0, 3.0, 0.05, "Base score for a resulting 4-stone hot window."),
        ParameterSpec("four", 1.1, 0.0, 6.0, 0.05, "Base score for a resulting 5-stone window."),
        ParameterSpec("five", 3.0, 0.0, 12.0, 0.1, "Base score for an immediate completion/block."),
        ParameterSpec("fork_bonus", 1.8, 0.0, 8.0, 0.1, "Multiplier for multiple windows on one cell."),
        ParameterSpec("block_weight", 1.1, 0.0, 4.0, 0.05, "Multiplier for opponent multi-line blocks."),
    )

    def compute(
        self,
        position: AxisPolicyInput,
        parameters: Mapping[str, float] | None = None,
    ) -> AxisPolicyResult:
        params = merge_parameters(self.parameters, parameters)
        maps = empty_axis_maps()
        own = position.own_stones
        opp = position.opp_stones
        debug = {"own_forks": 0, "block_forks": 0}

        for q, r in position.legal_set:
            ij = board_index(q, r, position.offset_q, position.offset_r)
            if ij is None:
                continue
            for axis, (dq, dr) in enumerate(AXES):
                own_hits: list[float] = []
                block_hits: list[float] = []
                for cells in _windows_containing(q, r, dq, dr):
                    own_count, opp_count = _counts(cells, own, opp)
                    if opp_count == 0 and own_count >= 3:
                        own_hits.append(_after_place_weight(own_count, params))
                    if own_count == 0 and opp_count >= 3:
                        block_hits.append(_after_place_weight(opp_count, params) * params["block_weight"])
                score = _fork_score(own_hits, params["fork_bonus"]) + _fork_score(block_hits, params["fork_bonus"])
                if len(own_hits) >= 2:
                    debug["own_forks"] += 1
                if len(block_hits) >= 2:
                    debug["block_forks"] += 1
                maps[axis, ij[0], ij[1]] = score

        combined = normalize_policy(
            maps.sum(axis=0),
            position.legal_set,
            position.offset_q,
            position.offset_r,
        )
        return AxisPolicyResult(
            self.prototype_id,
            params,
            maps,
            combined,
            {"target_kind": "legal_cell_multi_line_strength", **debug},
            position.offset_q,
            position.offset_r,
        )


def _windows_containing(q: int, r: int, dq: int, dr: int) -> list[list[tuple[int, int]]]:
    return [
        [(q - dq * off + dq * step, r - dr * off + dr * step) for step in range(WIN_LENGTH)]
        for off in range(WIN_LENGTH)
    ]


def _counts(
    cells: list[tuple[int, int]],
    own: set[tuple[int, int]],
    opp: set[tuple[int, int]],
) -> tuple[int, int]:
    own_count = sum(1 for cell in cells if cell in own)
    opp_count = sum(1 for cell in cells if cell in opp)
    return own_count, opp_count


def _urgency(params: Mapping[str, float]) -> list[float]:
    return [0.0, 0.0, 0.0, params["three"], params["four"], params["five"], params["five"]]


def _after_place_weight(count_before: int, params: Mapping[str, float]) -> float:
    after = count_before + 1
    if after >= 6:
        return params["five"]
    if after == 5:
        return params["four"]
    if after == 4:
        return params["three"]
    return 0.0


def _fork_score(values: list[float], fork_bonus: float) -> float:
    if not values:
        return 0.0
    total = float(sum(values))
    if len(values) >= 2:
        total *= 1.0 + fork_bonus * (len(values) - 1)
    return total


def _signed_legal_display(
    maps: np.ndarray,
    legal_moves: set[tuple[int, int]],
    offset_q: int,
    offset_r: int,
    *,
    block_visibility: float,
) -> np.ndarray:
    positive = np.maximum(maps.max(axis=0), 0.0)
    negative = np.maximum((-maps).max(axis=0), 0.0) * block_visibility
    return normalize_policy(np.maximum(positive, negative), legal_moves, offset_q, offset_r)
