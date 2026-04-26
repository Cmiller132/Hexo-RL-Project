"""Multi-axis development target prototype for the Axis Lab."""

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

WIN_LENGTH = 6


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

        combined = normalize_policy(
            np.maximum(maps.max(axis=0), np.maximum((-maps).max(axis=0), 0.0)),
            position.legal_set,
            position.offset_q,
            position.offset_r,
            fallback_uniform=False,
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
