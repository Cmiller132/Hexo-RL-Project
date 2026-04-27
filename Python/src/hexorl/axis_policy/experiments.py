"""Experimental axis-lab diagnostics for long-term structure building."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from hexorl.axis_policy.core import (
    AXES,
    AxisPolicyInput,
    AxisPolicyResult,
    ParameterSpec,
    board_index,
    merge_parameters,
    normalize_policy,
    window_in_bounds,
)
from hexorl.selfplay.records import BOARD_SIZE

WIN_LENGTH = 6
DELTA_CORE_PARAMETERS = (
    ParameterSpec("w1", 0.02, 0.0, 0.2, 0.005, "Strength after placing into an empty pure window."),
    ParameterSpec("w2", 0.06, 0.0, 0.4, 0.005, "Strength after reaching two stones in a pure window."),
    ParameterSpec("w3", 0.15, 0.0, 0.8, 0.01, "Strength after reaching three stones in a pure window."),
    ParameterSpec("w4", 1.0, 0.0, 3.0, 0.01, "Strength after reaching four stones in a pure window."),
    ParameterSpec("w5", 1.0, 0.0, 3.0, 0.01, "Strength after reaching five stones in a pure window."),
    ParameterSpec("opp_weight", 1.0, 0.0, 2.0, 0.05, "Opponent delta multiplier."),
    ParameterSpec("existing_credit", 0.25, 0.0, 1.0, 0.01, "How much pre-existing window strength is subtracted."),
    ParameterSpec("tail_weight", 0.18, 0.0, 1.0, 0.01, "How much overlapping non-best windows contribute."),
)


class DeltaForkPrototype:
    prototype_id = "exp_delta_fork"
    label = "Experiment: Delta Fork"
    description = "Legal-cell marginal axis gain if the side placed here, with a multi-axis fork boost. Diagnostic only."
    parameters = (
        ParameterSpec("w1", 0.02, 0.0, 0.2, 0.005, "Strength after placing into an empty pure window."),
        ParameterSpec("w2", 0.06, 0.0, 0.4, 0.005, "Strength after reaching two stones in a pure window."),
        ParameterSpec("w3", 0.15, 0.0, 0.8, 0.01, "Strength after reaching three stones in a pure window."),
        ParameterSpec("w4", 1.0, 0.0, 3.0, 0.01, "Strength after reaching four stones in a pure window."),
        ParameterSpec("w5", 1.0, 0.0, 3.0, 0.01, "Strength after reaching five stones in a pure window."),
        ParameterSpec("fork_bonus", 0.45, 0.0, 2.0, 0.05, "Multiplier added for each extra active axis."),
        ParameterSpec("opp_weight", 1.0, 0.0, 2.0, 0.05, "Opponent delta multiplier."),
        ParameterSpec("existing_credit", 0.25, 0.0, 1.0, 0.01, "How much pre-existing window strength is subtracted."),
        ParameterSpec("tail_weight", 0.18, 0.0, 1.0, 0.01, "How much overlapping non-best windows contribute."),
    )

    def compute(
        self,
        position: AxisPolicyInput,
        parameters: Mapping[str, float] | None = None,
    ) -> AxisPolicyResult:
        params = merge_parameters(self.parameters, parameters)
        strength = _strength_array(params)
        maps = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        own = position.own_stones
        opp = position.opp_stones
        candidates, candidate_source = _delta_candidate_cells(position)
        debug = {
            "own_fork_cells": 0,
            "opp_fork_cells": 0,
            "legal_cells_scored": 0,
            "candidate_source": candidate_source,
            "candidate_cells": len(candidates),
        }

        for q, r in candidates:
            ij = board_index(q, r, position.offset_q, position.offset_r)
            if ij is None:
                continue
            own_axes: list[float] = []
            opp_axes: list[float] = []
            for dq, dr in AXES:
                own_axes.append(
                    _marginal_axis_gain(
                        q,
                        r,
                        dq,
                        dr,
                        own,
                        opp,
                        strength,
                        params["existing_credit"],
                        params["tail_weight"],
                        position.offset_q,
                        position.offset_r,
                    )
                )
                opp_axes.append(
                    _marginal_axis_gain(
                        q,
                        r,
                        dq,
                        dr,
                        opp,
                        own,
                        strength,
                        params["existing_credit"],
                        params["tail_weight"],
                        position.offset_q,
                        position.offset_r,
                    )
                    * params["opp_weight"]
                )
            own_active = _active_count(own_axes)
            opp_active = _active_count(opp_axes)
            if own_active >= 2:
                debug["own_fork_cells"] += 1
            if opp_active >= 2:
                debug["opp_fork_cells"] += 1
            own_mult = 1.0 + params["fork_bonus"] * max(0, own_active - 1)
            opp_mult = 1.0 + params["fork_bonus"] * max(0, opp_active - 1)
            for axis in range(3):
                maps[axis, ij[0], ij[1]] = own_axes[axis] * own_mult
                maps[axis + 3, ij[0], ij[1]] = opp_axes[axis] * opp_mult
            debug["legal_cells_scored"] += 1

        display = np.maximum(maps[:3].max(axis=0), maps[3:].max(axis=0))
        combined = normalize_policy(
            display,
            candidates,
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
                **debug,
                "target_kind": "diagnostic_legal_delta_not_training_target",
                "reading": "value is marginal pure-window strength gained by placing on this legal cell",
                "strength_weights": strength.tolist(),
            },
            position.offset_q,
            position.offset_r,
            position.current_player,
        )


class DeltaNormPrototype:
    prototype_id = "exp_delta_norm"
    label = "Experiment: Delta Norm"
    description = "Legal-cell placement delta scored by vector norm, so multiple strong axes raise the displayed value smoothly."
    parameters = (
        *DELTA_CORE_PARAMETERS,
        ParameterSpec("p_norm", 2.0, 1.0, 4.0, 0.1, "Norm exponent. 2 is Euclidean; larger values move toward max-axis."),
    )

    def compute(
        self,
        position: AxisPolicyInput,
        parameters: Mapping[str, float] | None = None,
    ) -> AxisPolicyResult:
        params = merge_parameters(self.parameters, parameters)
        return _compute_delta_variant(position, params, "norm", self.prototype_id)


class DeltaSoftStrongPrototype:
    prototype_id = "exp_delta_soft_strong"
    label = "Experiment: Delta Soft Strong"
    description = "Legal-cell placement delta with a smooth strong-axis gate; tiny incidental axes do not count as full forks."
    parameters = (
        *DELTA_CORE_PARAMETERS,
        ParameterSpec("strong_threshold", 0.75, 0.0, 3.0, 0.01, "Axis value where the smooth strong-axis gate is half active."),
        ParameterSpec("fork_bonus", 0.65, 0.0, 3.0, 0.05, "Multiplier added as multiple axes become strongly active."),
    )

    def compute(
        self,
        position: AxisPolicyInput,
        parameters: Mapping[str, float] | None = None,
    ) -> AxisPolicyResult:
        params = merge_parameters(self.parameters, parameters)
        return _compute_delta_variant(position, params, "soft_strong", self.prototype_id)


class DeltaBalancePrototype:
    prototype_id = "exp_delta_balance"
    label = "Experiment: Delta Balance"
    description = "Legal-cell placement delta with a smooth ratio bonus when the second and third axes are comparable to the best axis."
    parameters = (
        *DELTA_CORE_PARAMETERS,
        ParameterSpec("balance_bonus", 0.75, 0.0, 3.0, 0.05, "Bonus when the second-best axis approaches the best axis."),
        ParameterSpec("third_axis_bonus", 0.25, 0.0, 2.0, 0.05, "Smaller bonus when the third axis also has meaningful strength."),
    )

    def compute(
        self,
        position: AxisPolicyInput,
        parameters: Mapping[str, float] | None = None,
    ) -> AxisPolicyResult:
        params = merge_parameters(self.parameters, parameters)
        return _compute_delta_variant(position, params, "balance", self.prototype_id)


class CrossAxisPivotPrototype:
    prototype_id = "exp_cross_axis_pivot"
    label = "Experiment: Cross-Axis Pivot"
    description = "Dense axis field that boosts cells participating in multiple strong axes. Diagnostic only."
    parameters = (
        ParameterSpec("w1", 0.02, 0.0, 0.2, 0.005, "Strength for one stone in a pure 6-cell window."),
        ParameterSpec("w2", 0.06, 0.0, 0.4, 0.005, "Strength for two stones in a pure 6-cell window."),
        ParameterSpec("w3", 0.15, 0.0, 0.8, 0.01, "Strength for three stones in a pure 6-cell window."),
        ParameterSpec("w4", 1.0, 0.0, 3.0, 0.01, "Strength for four stones in a pure 6-cell window."),
        ParameterSpec("w5", 1.0, 0.0, 3.0, 0.01, "Strength for five stones in a pure 6-cell window."),
        ParameterSpec("active_threshold", 0.08, 0.0, 1.0, 0.01, "Axis value needed to count as active."),
        ParameterSpec("pivot_bonus", 0.35, 0.0, 2.0, 0.05, "Multiplier added for each extra active axis."),
        ParameterSpec("reserve_bonus", 0.25, 0.0, 2.0, 0.05, "Adds a small second-axis reserve value into active axes."),
        ParameterSpec("tail_weight", 0.18, 0.0, 1.0, 0.01, "How much overlapping non-best windows contribute."),
    )

    def compute(
        self,
        position: AxisPolicyInput,
        parameters: Mapping[str, float] | None = None,
    ) -> AxisPolicyResult:
        params = merge_parameters(self.parameters, parameters)
        strength = _strength_array(params)
        maps = _dense_axis_maps(position, strength, params["tail_weight"])
        debug = {"own_pivot_cells": 0, "opp_pivot_cells": 0}

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                own_axes = [float(maps[axis, i, j]) for axis in range(3)]
                opp_axes = [float(maps[axis + 3, i, j]) for axis in range(3)]
                own_boosted, own_active = _boost_cross_axis(
                    own_axes,
                    params["active_threshold"],
                    params["pivot_bonus"],
                    params["reserve_bonus"],
                )
                opp_boosted, opp_active = _boost_cross_axis(
                    opp_axes,
                    params["active_threshold"],
                    params["pivot_bonus"],
                    params["reserve_bonus"],
                )
                if own_active >= 2:
                    debug["own_pivot_cells"] += 1
                if opp_active >= 2:
                    debug["opp_pivot_cells"] += 1
                for axis in range(3):
                    maps[axis, i, j] = own_boosted[axis]
                    maps[axis + 3, i, j] = opp_boosted[axis]

        display = np.maximum(maps[:3].max(axis=0), maps[3:].max(axis=0))
        combined = normalize_policy(
            display,
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
                **debug,
                "target_kind": "diagnostic_dense_pivot_not_training_target",
                "reading": "base dense axis strength boosted where several axes are simultaneously alive",
                "strength_weights": strength.tolist(),
            },
            position.offset_q,
            position.offset_r,
            position.current_player,
        )


def _strength_array(params: Mapping[str, float]) -> np.ndarray:
    return np.array(
        [0.0, params["w1"], params["w2"], params["w3"], params["w4"], params["w5"], params["w5"]],
        dtype=np.float32,
    )


def _compute_delta_variant(
    position: AxisPolicyInput,
    params: Mapping[str, float],
    mode: str,
    prototype_id: str,
) -> AxisPolicyResult:
    strength = _strength_array(params)
    maps = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    own = position.own_stones
    opp = position.opp_stones
    debug = {
        "legal_cells_scored": 0,
        "own_multi_axis_cells": 0,
        "opp_multi_axis_cells": 0,
        "variant": mode,
    }
    candidates, candidate_source = _delta_candidate_cells(position)
    debug["candidate_source"] = candidate_source
    debug["candidate_cells"] = len(candidates)

    for q, r in candidates:
        ij = board_index(q, r, position.offset_q, position.offset_r)
        if ij is None:
            continue
        own_axes, opp_axes = _placement_delta_axes(
            q,
            r,
            own,
            opp,
            strength,
            params,
            position.offset_q,
            position.offset_r,
        )
        if _active_count(own_axes, float(params.get("strong_threshold", 0.75))) >= 2:
            debug["own_multi_axis_cells"] += 1
        if _active_count(opp_axes, float(params.get("strong_threshold", 0.75))) >= 2:
            debug["opp_multi_axis_cells"] += 1
        own_shaped = _shape_delta_axes(own_axes, params, mode)
        opp_shaped = _shape_delta_axes(opp_axes, params, mode)
        for axis in range(3):
            maps[axis, ij[0], ij[1]] = own_shaped[axis]
            maps[axis + 3, ij[0], ij[1]] = opp_shaped[axis]
        debug["legal_cells_scored"] += 1

    display = np.maximum(maps[:3].max(axis=0), maps[3:].max(axis=0))
    combined = normalize_policy(
        display,
        candidates,
        position.offset_q,
        position.offset_r,
        fallback_uniform=False,
    )
    return AxisPolicyResult(
        prototype_id,
        dict(params),
        maps,
        combined,
        {
            **debug,
            "target_kind": "diagnostic_legal_delta_variant_not_training_target",
            "reading": "value is placement-created pure-window strength, shaped to expose multi-axis structure",
            "strength_weights": strength.tolist(),
        },
        position.offset_q,
        position.offset_r,
        position.current_player,
    )


def _placement_delta_axes(
    q: int,
    r: int,
    own: set[tuple[int, int]],
    opp: set[tuple[int, int]],
    strength: np.ndarray,
    params: Mapping[str, float],
    offset_q: int,
    offset_r: int,
) -> tuple[list[float], list[float]]:
    own_axes: list[float] = []
    opp_axes: list[float] = []
    for dq, dr in AXES:
        own_axes.append(
            _marginal_axis_gain(
                q,
                r,
                dq,
                dr,
                own,
                opp,
                strength,
                params["existing_credit"],
                params["tail_weight"],
                offset_q,
                offset_r,
            )
        )
        opp_axes.append(
            _marginal_axis_gain(
                q,
                r,
                dq,
                dr,
                opp,
                own,
                strength,
                params["existing_credit"],
                params["tail_weight"],
                offset_q,
                offset_r,
            )
            * params["opp_weight"]
        )
    return own_axes, opp_axes


def _delta_candidate_cells(position: AxisPolicyInput) -> tuple[set[tuple[int, int]], str]:
    legal = position.legal_set
    if legal:
        return legal, "legal_moves"
    occupied = position.own_stones | position.opp_stones
    candidates: set[tuple[int, int]] = set()
    for q, r in occupied:
        for dq, dr in AXES:
            for step in range(-(WIN_LENGTH - 1), WIN_LENGTH):
                cell = (q + dq * step, r + dr * step)
                if cell not in occupied:
                    candidates.add(cell)
    return candidates, "terminal_hypothetical_line_cells"


def _shape_delta_axes(
    axes: Sequence[float],
    params: Mapping[str, float],
    mode: str,
) -> list[float]:
    values = [max(float(value), 0.0) for value in axes[:3]]
    best = max(values, default=0.0)
    if best <= 0.0:
        return values
    if mode == "norm":
        p_norm = max(float(params["p_norm"]), 1.0)
        scalar = sum(value**p_norm for value in values) ** (1.0 / p_norm)
        return _scale_max_to(values, scalar)
    if mode == "soft_strong":
        threshold = max(float(params["strong_threshold"]), 1e-6)
        gates = [value / (value + threshold) if value > 0.0 else 0.0 for value in values]
        active_mass = sum(gates)
        multiplier = 1.0 + float(params["fork_bonus"]) * max(0.0, active_mass - 1.0)
        return [value * multiplier for value in values]
    if mode == "balance":
        ordered = sorted(values, reverse=True)
        second_ratio = ordered[1] / best if len(ordered) > 1 else 0.0
        third_ratio = ordered[2] / best if len(ordered) > 2 else 0.0
        multiplier = 1.0 + float(params["balance_bonus"]) * second_ratio
        multiplier += float(params["third_axis_bonus"]) * third_ratio
        return [value * multiplier for value in values]
    return values


def _scale_max_to(values: Sequence[float], scalar: float) -> list[float]:
    best = max(values, default=0.0)
    if best <= 0.0:
        return [float(value) for value in values]
    scale = float(scalar) / best
    return [float(value) * scale for value in values]


def _dense_axis_maps(position: AxisPolicyInput, strength: np.ndarray, tail_weight: float) -> np.ndarray:
    maps = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    own = position.own_stones
    opp = position.opp_stones
    for axis, (dq, dr) in enumerate(AXES):
        for gi in range(BOARD_SIZE):
            for gj in range(BOARD_SIZE):
                q = gi + position.offset_q
                r = gj + position.offset_r
                own_values: list[float] = []
                opp_values: list[float] = []
                for cells in _windows_containing(q, r, dq, dr):
                    if not window_in_bounds(cells, position.offset_q, position.offset_r):
                        continue
                    own_count = _count(cells, own)
                    opp_count = _count(cells, opp)
                    if own_count and opp_count:
                        continue
                    own_values.append(float(strength[own_count]))
                    opp_values.append(float(strength[opp_count]))
                maps[axis, gi, gj] = _best_plus_tail(own_values, tail_weight)
                maps[axis + 3, gi, gj] = _best_plus_tail(opp_values, tail_weight)
    return maps


def _marginal_axis_gain(
    q: int,
    r: int,
    dq: int,
    dr: int,
    side: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
    strength: np.ndarray,
    existing_credit: float,
    tail_weight: float,
    offset_q: int,
    offset_r: int,
) -> float:
    gains: list[float] = []
    for cells in _windows_containing(q, r, dq, dr):
        if not window_in_bounds(cells, offset_q, offset_r):
            continue
        if any(cell in blockers for cell in cells):
            continue
        before = min(_count(cells, side), WIN_LENGTH)
        after = min(before + 1, WIN_LENGTH)
        gains.append(max(float(strength[after] - existing_credit * strength[before]), 0.0))
    return _best_plus_tail(gains, tail_weight)


def _windows_containing(q: int, r: int, dq: int, dr: int) -> list[list[tuple[int, int]]]:
    return [
        [(q - dq * off + dq * step, r - dr * off + dr * step) for step in range(WIN_LENGTH)]
        for off in range(WIN_LENGTH)
    ]


def _count(cells: Iterable[tuple[int, int]], stones: set[tuple[int, int]]) -> int:
    return sum(1 for cell in cells if cell in stones)


def _best_plus_tail(values: Sequence[float], tail_weight: float) -> float:
    positives = sorted((float(value) for value in values if value > 0.0), reverse=True)
    if not positives:
        return 0.0
    return positives[0] + float(tail_weight) * sum(positives[1:])


def _active_count(values: Sequence[float], threshold: float = 1e-7) -> int:
    return sum(1 for value in values if value > threshold)


def _boost_cross_axis(
    axes: Sequence[float],
    threshold: float,
    pivot_bonus: float,
    reserve_bonus: float,
) -> tuple[list[float], int]:
    values = [max(float(value), 0.0) for value in axes[:3]]
    active = _active_count(values, threshold)
    if active < 2:
        return values, active
    sorted_values = sorted(values, reverse=True)
    reserve = sorted_values[1] if len(sorted_values) > 1 else 0.0
    multiplier = 1.0 + pivot_bonus * max(0, active - 1)
    boosted = [
        value * multiplier + reserve_bonus * reserve if value > threshold else value
        for value in values
    ]
    return boosted, active
