"""Multi-axis line strength prototypes for the Axis Lab."""

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
    description = "Signed per-axis strength from pure 6-cell windows for both players."
    parameters = (
        ParameterSpec("three", 0.15, 0.0, 1.0, 0.01, "Weight for a pure 3-stone window."),
        ParameterSpec("four", 0.55, 0.0, 3.0, 0.01, "Weight for a pure 4-stone hot window."),
        ParameterSpec("five", 1.4, 0.0, 6.0, 0.05, "Weight for a pure 5-stone immediate threat."),
        ParameterSpec("own_weight", 1.0, 0.0, 4.0, 0.05, "Current-player threat multiplier."),
        ParameterSpec("opp_weight", 1.15, 0.0, 4.0, 0.05, "Opponent-threat multiplier."),
        ParameterSpec("opponent_visibility", 1.0, 0.0, 3.0, 0.05, "How strongly opponent axis strength appears in the display overlay."),
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
            opponent_visibility=params["opponent_visibility"],
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
            position.current_player,
        )


class AxisDevelopmentPrototype:
    prototype_id = "axis_development"
    label = "Axis Development"
    description = "Signed legal-cell potential for building strength across several pure axes, shown for both players."
    parameters = (
        ParameterSpec("one", 0.02, 0.0, 0.2, 0.005, "Marginal value for starting a pure axis window."),
        ParameterSpec("two", 0.06, 0.0, 0.5, 0.005, "Marginal value for a two-stone pure axis window."),
        ParameterSpec("three", 0.18, 0.0, 1.2, 0.01, "Marginal value for a developing three-stone window."),
        ParameterSpec("four", 0.58, 0.0, 3.0, 0.01, "Marginal value for creating/strengthening a hot window."),
        ParameterSpec("five", 1.25, 0.0, 6.0, 0.05, "Marginal value near a completed line."),
        ParameterSpec("multi_axis_bonus", 0.55, 0.0, 4.0, 0.05, "Bonus when a cell improves more than one axis."),
        ParameterSpec("opp_weight", 1.0, 0.0, 4.0, 0.05, "Opponent-axis strength multiplier."),
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
        weights = _development_weights(params)
        debug = {"own_multi_axis_cells": 0, "opp_multi_axis_cells": 0}

        for q, r in position.legal_set:
            ij = board_index(q, r, position.offset_q, position.offset_r)
            if ij is None:
                continue
            own_active_axes = 0
            opp_active_axes = 0
            pending: list[tuple[int, float, float]] = []
            for axis, (dq, dr) in enumerate(AXES):
                own_axis = 0.0
                opp_axis = 0.0
                for cells in _windows_containing(q, r, dq, dr):
                    own_count, opp_count = _counts(cells, own, opp)
                    if opp_count == 0:
                        own_axis += _marginal_gain(own_count, weights)
                    if own_count == 0:
                        opp_axis += _marginal_gain(opp_count, weights) * params["opp_weight"]
                if own_axis > 0:
                    own_active_axes += 1
                if opp_axis > 0:
                    opp_active_axes += 1
                pending.append((axis, own_axis, opp_axis))
            own_bonus = 1.0 + params["multi_axis_bonus"] * max(0, own_active_axes - 1)
            opp_bonus = 1.0 + params["multi_axis_bonus"] * max(0, opp_active_axes - 1)
            if own_active_axes >= 2:
                debug["own_multi_axis_cells"] += 1
            if opp_active_axes >= 2:
                debug["opp_multi_axis_cells"] += 1
            for axis, own_axis, opp_axis in pending:
                maps[axis, ij[0], ij[1]] = own_axis * own_bonus - opp_axis * opp_bonus

        combined = _signed_legal_display(
            maps,
            position.legal_set,
            position.offset_q,
            position.offset_r,
            opponent_visibility=1.0,
        )
        return AxisPolicyResult(
            self.prototype_id,
            params,
            maps,
            combined,
            {
                "target_kind": "signed_multi_axis_development",
                "positive_values": "cell builds current-player pure axes",
                "negative_values": "cell belongs to opponent pure-axis development",
                **debug,
            },
            position.offset_q,
            position.offset_r,
            position.current_player,
        )


class MultiLineThreatPrototype:
    prototype_id = "multi_line_threats"
    label = "Multi-Line Threats"
    description = "Signed fork pressure: cells that participate in multiple own or opponent threat-building windows."
    parameters = (
        ParameterSpec("three", 0.35, 0.0, 3.0, 0.05, "Base score for a resulting 4-stone hot window."),
        ParameterSpec("four", 1.1, 0.0, 6.0, 0.05, "Base score for a resulting 5-stone window."),
        ParameterSpec("five", 3.0, 0.0, 12.0, 0.1, "Base score for an immediate completion/block."),
        ParameterSpec("fork_bonus", 1.8, 0.0, 8.0, 0.1, "Multiplier for multiple windows on one cell."),
        ParameterSpec("opp_weight", 1.0, 0.0, 4.0, 0.05, "Opponent multi-line strength multiplier."),
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
        debug = {"own_forks": 0, "opp_forks": 0}

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
                        block_hits.append(_after_place_weight(opp_count, params) * params["opp_weight"])
                score = _fork_score(own_hits, params["fork_bonus"]) - _fork_score(block_hits, params["fork_bonus"])
                if len(own_hits) >= 2:
                    debug["own_forks"] += 1
                if len(block_hits) >= 2:
                    debug["opp_forks"] += 1
                maps[axis, ij[0], ij[1]] = score

        combined = _signed_legal_display(
            maps,
            position.legal_set,
            position.offset_q,
            position.offset_r,
            opponent_visibility=1.0,
        )
        return AxisPolicyResult(
            self.prototype_id,
            params,
            maps,
            combined,
            {
                "target_kind": "signed_multi_line_fork_strength",
                "positive_values": "current-player fork pressure",
                "negative_values": "opponent fork pressure",
                **debug,
            },
            position.offset_q,
            position.offset_r,
            position.current_player,
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


def _development_weights(params: Mapping[str, float]) -> list[float]:
    return [
        0.0,
        params["one"],
        params["two"],
        params["three"],
        params["four"],
        params["five"],
        params["five"],
    ]


def _marginal_gain(count_before: int, weights: list[float]) -> float:
    before = min(max(count_before, 0), 6)
    after = min(before + 1, 6)
    return max(float(weights[after] - weights[before]), 0.0)


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
    opponent_visibility: float,
) -> np.ndarray:
    positive = np.maximum(maps.max(axis=0), 0.0)
    negative = np.maximum((-maps).max(axis=0), 0.0) * opponent_visibility
    return normalize_policy(
        np.maximum(positive, negative),
        legal_moves,
        offset_q,
        offset_r,
        fallback_uniform=False,
    )
