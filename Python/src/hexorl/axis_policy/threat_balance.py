"""Axis target prototype balancing immediate attack and defense."""

from __future__ import annotations

from typing import Mapping

from hexorl.axis_policy.core import (
    AXES,
    AxisPolicyInput,
    AxisPolicyResult,
    ParameterSpec,
    board_index,
    empty_axis_maps,
    line_count,
    merge_parameters,
    normalize_policy,
)


class ThreatBalancePrototype:
    prototype_id = "threat_balance"
    label = "Threat Balance"
    description = "Aggressively boosts cells that make or stop 4/5-like line pressure."
    parameters = (
        ParameterSpec("make_four", 2.5, 0.0, 8.0, 0.1, "Bonus for own line length reaching four."),
        ParameterSpec("make_five", 6.0, 0.0, 16.0, 0.1, "Bonus for own line length reaching five."),
        ParameterSpec("block_four", 2.0, 0.0, 8.0, 0.1, "Bonus for blocking opponent length four."),
        ParameterSpec("block_five", 7.0, 0.0, 16.0, 0.1, "Bonus for blocking opponent length five."),
        ParameterSpec("base", 0.2, 0.0, 2.0, 0.05, "Baseline legal-cell mass."),
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
        debug = {"own_four": 0, "own_five": 0, "opp_four": 0, "opp_five": 0}

        for q, r in position.legal_set:
            for axis, (dq, dr) in enumerate(AXES):
                own_len = (
                    line_count(own, q, r, dq, dr)
                    + line_count(own, q, r, -dq, -dr)
                    + 1
                )
                opp_len = (
                    line_count(opp, q, r, dq, dr)
                    + line_count(opp, q, r, -dq, -dr)
                    + 1
                )
                score = params["base"]
                if own_len >= 4:
                    score += params["make_four"]
                    debug["own_four"] += 1
                if own_len >= 5:
                    score += params["make_five"]
                    debug["own_five"] += 1
                if opp_len >= 4:
                    score += params["block_four"]
                    debug["opp_four"] += 1
                if opp_len >= 5:
                    score += params["block_five"]
                    debug["opp_five"] += 1
                ij = board_index(q, r, position.offset_q, position.offset_r)
                if ij is not None:
                    maps[axis, ij[0], ij[1]] = score

        combined = normalize_policy(
            maps.sum(axis=0),
            position.legal_set,
            position.offset_q,
            position.offset_r,
        )
        debug["legal_moves"] = len(position.legal_set)
        return AxisPolicyResult(self.prototype_id, params, maps, combined, debug)
