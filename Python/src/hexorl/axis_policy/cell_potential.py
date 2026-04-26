"""Axis target prototype based on local cell potential and centrality."""

from __future__ import annotations

import math
from typing import Mapping

from hexorl.axis_policy.core import (
    AXES,
    AxisPolicyInput,
    AxisPolicyResult,
    ParameterSpec,
    board_index,
    distance_to_stones,
    empty_axis_maps,
    hex_distance,
    merge_parameters,
    normalize_policy,
)


class CellPotentialPrototype:
    prototype_id = "cell_potential"
    label = "Cell Potential"
    description = "Smooth potential field favoring nearby, central, multi-axis cells."
    parameters = (
        ParameterSpec("centrality", 0.45, 0.0, 3.0, 0.05, "Reward for staying near the origin."),
        ParameterSpec("own_proximity", 1.1, 0.0, 4.0, 0.05, "Reward for nearby own stones."),
        ParameterSpec("opp_proximity", 0.55, 0.0, 4.0, 0.05, "Reward for contesting nearby opponent stones."),
        ParameterSpec("axis_spread", 0.35, 0.0, 3.0, 0.05, "Reward for cells aligned with populated axes."),
        ParameterSpec("falloff", 0.32, 0.05, 2.0, 0.05, "Exponential distance falloff."),
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
            center_dist = hex_distance(q, r, 0, 0)
            own_dist = distance_to_stones(q, r, own)
            opp_dist = distance_to_stones(q, r, opp)
            base = (
                params["centrality"] * math.exp(-params["falloff"] * center_dist)
                + params["own_proximity"] * math.exp(-params["falloff"] * own_dist)
                + params["opp_proximity"] * math.exp(-params["falloff"] * opp_dist)
            )
            for axis, (dq, dr) in enumerate(AXES):
                axis_neighbors = (
                    ((q + dq, r + dr) in all_stones)
                    + ((q - dq, r - dr) in all_stones)
                )
                score = base + params["axis_spread"] * float(axis_neighbors)
                ij = board_index(q, r, position.offset_q, position.offset_r)
                if ij is not None:
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
            {"own_stones": len(own), "opp_stones": len(opp), "legal_moves": len(position.legal_set)},
        )
