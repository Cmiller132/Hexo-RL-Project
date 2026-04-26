"""Perspective-indexed dual-player axis strength targets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Mapping

import numpy as np

from hexorl.axis_policy.core import (
    AXES,
    AxisPolicyInput,
    AxisPolicyResult,
    ParameterSpec,
    merge_parameters,
    normalize_policy,
)
from hexorl.selfplay.records import BOARD_SIZE

WIN_LENGTH = 6


BASE_PARAMS = (
    ParameterSpec("w1", 0.02, 0.0, 0.2, 0.005, "Strength for one stone in a pure 6-cell window."),
    ParameterSpec("w2", 0.06, 0.0, 0.4, 0.005, "Strength for two stones in a pure 6-cell window."),
    ParameterSpec("w3", 0.15, 0.0, 0.8, 0.01, "Strength for three stones in a pure 6-cell window."),
    ParameterSpec("w4", 1.0, 0.0, 3.0, 0.01, "Strength for four stones in a pure 6-cell window."),
    ParameterSpec("w5", 1.0, 0.0, 3.0, 0.01, "Strength for five stones in a pure 6-cell window."),
)

TAIL_PARAM = ParameterSpec(
    "tail_weight",
    0.18,
    0.0,
    1.0,
    0.01,
    "How much overlapping non-best windows contribute.",
)


class _DualAxisStrengthBase:
    prototype_id = "dual_axis_strength"
    label = "Dual Axis Strength"
    description = "Perspective-indexed 6-plane dense target: own axes and opponent axes are kept separate."
    parameters = BASE_PARAMS
    aggregation = "sum"

    def compute(
        self,
        position: AxisPolicyInput,
        parameters: Mapping[str, float] | None = None,
    ) -> AxisPolicyResult:
        params = merge_parameters(self.parameters, parameters)
        strength = _strength_array(params)
        maps, debug = _compute_maps(
            position,
            strength,
            aggregation=self.aggregation,
            tail_weight=float(params.get("tail_weight", 0.0)),
        )
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
                "target_kind": self.prototype_id,
                "aggregation": self.aggregation,
                "channel_layout": [
                    "own_axis_0",
                    "own_axis_1",
                    "own_axis_2",
                    "opp_axis_0",
                    "opp_axis_1",
                    "opp_axis_2",
                ],
                "strength_weights": strength.tolist(),
            },
            position.offset_q,
            position.offset_r,
            position.current_player,
        )


class DualAxisStrengthPrototype(_DualAxisStrengthBase):
    prototype_id = "dual_axis_strength"
    label = "Dual Axis Strength"
    description = "Main candidate: raw summed six-plane own/opponent axis strength with 4/5 treated equally."


class DualAxisStrengthTailPrototype(_DualAxisStrengthBase):
    prototype_id = "dual_axis_strength_tail"
    label = "Dual Axis Strength - Best+Tail"
    description = "Six-plane target with best-window-plus-tail aggregation to reduce shifted-window overcounting."
    parameters = (*BASE_PARAMS, TAIL_PARAM)
    aggregation = "best_tail"


class DualAxisStrengthLegacyWeightsPrototype(_DualAxisStrengthBase):
    prototype_id = "dual_axis_strength_legacy_weights"
    label = "Dual Axis Strength - Legacy Weights"
    description = "Six-plane own/opponent target using the older gentler 4-window weight for comparison."
    parameters = (
        ParameterSpec("w1", 0.02, 0.0, 0.2, 0.005, "Strength for one stone in a pure 6-cell window."),
        ParameterSpec("w2", 0.06, 0.0, 0.4, 0.005, "Strength for two stones in a pure 6-cell window."),
        ParameterSpec("w3", 0.15, 0.0, 0.8, 0.01, "Strength for three stones in a pure 6-cell window."),
        ParameterSpec("w4", 0.45, 0.0, 3.0, 0.01, "Strength for four stones in a pure 6-cell window."),
        ParameterSpec("w5", 1.0, 0.0, 3.0, 0.01, "Strength for five stones in a pure 6-cell window."),
    )


class DualAxisStrengthHotPrototype(_DualAxisStrengthBase):
    prototype_id = "dual_axis_strength_hot"
    label = "Dual Axis Strength - Hot Focus"
    description = "Six-plane comparison target that suppresses early loose structure and emphasizes 4/5 windows."
    parameters = (
        ParameterSpec("w1", 0.0, 0.0, 0.2, 0.005, "Strength for one stone in a pure 6-cell window."),
        ParameterSpec("w2", 0.02, 0.0, 0.4, 0.005, "Strength for two stones in a pure 6-cell window."),
        ParameterSpec("w3", 0.08, 0.0, 0.8, 0.01, "Strength for three stones in a pure 6-cell window."),
        ParameterSpec("w4", 1.0, 0.0, 3.0, 0.01, "Strength for four stones in a pure 6-cell window."),
        ParameterSpec("w5", 1.0, 0.0, 3.0, 0.01, "Strength for five stones in a pure 6-cell window."),
    )


def _strength_array(params: Mapping[str, float]) -> np.ndarray:
    return np.array(
        [0.0, params["w1"], params["w2"], params["w3"], params["w4"], params["w5"], params["w5"]],
        dtype=np.float32,
    )


def _compute_maps(
    position: AxisPolicyInput,
    strength: np.ndarray,
    *,
    aggregation: str,
    tail_weight: float,
) -> tuple[np.ndarray, dict[str, int]]:
    maps = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    own = position.own_stones
    opp = position.opp_stones
    debug = {"own_windows": 0, "opp_windows": 0, "contested_skipped": 0}

    for axis, (dq, dr) in enumerate(AXES):
        own_plane = axis
        opp_plane = axis + 3
        for gi in range(BOARD_SIZE):
            for gj in range(BOARD_SIZE):
                q = gi + position.offset_q
                r = gj + position.offset_r
                own_values: list[float] = []
                opp_values: list[float] = []
                for off in range(WIN_LENGTH):
                    wq = q - dq * off
                    wr = r - dr * off
                    own_count = 0
                    opp_count = 0
                    for step in range(WIN_LENGTH):
                        cell = (wq + dq * step, wr + dr * step)
                        own_count += int(cell in own)
                        opp_count += int(cell in opp)
                    if own_count and opp_count:
                        debug["contested_skipped"] += 1
                        continue
                    if own_count:
                        own_values.append(float(strength[own_count]))
                        debug["own_windows"] += 1
                    elif opp_count:
                        opp_values.append(float(strength[opp_count]))
                        debug["opp_windows"] += 1
                maps[own_plane, gi, gj] = _aggregate(own_values, aggregation, tail_weight)
                maps[opp_plane, gi, gj] = _aggregate(opp_values, aggregation, tail_weight)
    return maps, debug


def _aggregate(values: Sequence[float], aggregation: str, tail_weight: float) -> float:
    positives = sorted((float(value) for value in values if value > 0.0), reverse=True)
    if not positives:
        return 0.0
    if aggregation == "best_tail":
        return positives[0] + float(tail_weight) * sum(positives[1:])
    return sum(positives)
