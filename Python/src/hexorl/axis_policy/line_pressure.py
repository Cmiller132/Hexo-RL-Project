"""Axis target prototype based on contiguous line extension pressure."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from hexorl.axis_policy.core import (
    AXES,
    AxisPolicyInput,
    AxisPolicyResult,
    ParameterSpec,
    board_index,
    distance_to_stones,
    empty_axis_maps,
    line_count,
    merge_parameters,
    normalize_policy,
)


class LinePressurePrototype:
    prototype_id = "line_pressure"
    label = "Line Pressure"
    description = "Rewards cells that extend own runs and lightly blocks opponent runs."
    parameters = (
        ParameterSpec("own_weight", 1.0, 0.0, 4.0, 0.05, "Weight for extending current-player runs."),
        ParameterSpec("opp_weight", 0.65, 0.0, 4.0, 0.05, "Weight for blocking opponent runs."),
        ParameterSpec("length_power", 1.35, 0.5, 3.0, 0.05, "Nonlinear reward for longer contiguous runs."),
        ParameterSpec("axis_sharpness", 0.35, 0.0, 3.0, 0.05, "Extra reward when one axis dominates."),
        ParameterSpec("distance_falloff", 0.75, 0.0, 2.5, 0.05, "Downweight cells far from existing stones."),
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
        all_stones = own | opp

        for q, r in position.legal_set:
            if all_stones:
                proximity = 1.0 / (1.0 + params["distance_falloff"] * distance_to_stones(q, r, all_stones))
            else:
                proximity = 1.0
            axis_scores = []
            for axis, (dq, dr) in enumerate(AXES):
                own_len = (
                    line_count(own, q, r, dq, dr)
                    + line_count(own, q, r, -dq, -dr)
                )
                opp_len = (
                    line_count(opp, q, r, dq, dr)
                    + line_count(opp, q, r, -dq, -dr)
                )
                score = (
                    params["own_weight"] * ((own_len + 1.0) ** params["length_power"])
                    + params["opp_weight"] * ((opp_len + 1.0) ** params["length_power"])
                ) * proximity
                axis_scores.append(score)
                ij = board_index(q, r, position.offset_q, position.offset_r)
                if ij is not None:
                    maps[axis, ij[0], ij[1]] = score
            if params["axis_sharpness"] > 0:
                best = int(np.argmax(axis_scores))
                ij = board_index(q, r, position.offset_q, position.offset_r)
                if ij is not None:
                    maps[best, ij[0], ij[1]] *= 1.0 + params["axis_sharpness"]

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
            {
                "own_stones": len(own),
                "opp_stones": len(opp),
                "legal_moves": len(position.legal_set),
                "falloff_applied": bool(all_stones),
            },
            position.offset_q,
            position.offset_r,
        )
