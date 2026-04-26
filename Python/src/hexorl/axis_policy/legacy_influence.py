"""Legacy Hexagon-style signed axis influence target."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from hexorl.axis_policy.core import (
    AXES,
    AxisPolicyInput,
    AxisPolicyResult,
    ParameterSpec,
    empty_axis_maps,
    merge_parameters,
    normalize_policy,
)
from hexorl.selfplay.records import BOARD_SIZE


class LegacyAxisInfluencePrototype:
    prototype_id = "legacy_axis_influence"
    label = "Legacy Axis Influence"
    description = "Signed Hexagon-style target: pure 6-cell windows add own urgency and subtract opponent urgency per axis."
    parameters = (
        ParameterSpec("w1", 0.02, 0.0, 0.2, 0.005, "Urgency for one stone in a pure 6-cell window."),
        ParameterSpec("w2", 0.06, 0.0, 0.4, 0.005, "Urgency for two stones in a pure 6-cell window."),
        ParameterSpec("w3", 0.15, 0.0, 0.8, 0.01, "Urgency for three stones in a pure 6-cell window."),
        ParameterSpec("w4", 0.45, 0.0, 2.0, 0.01, "Urgency for four stones in a pure 6-cell window."),
        ParameterSpec("w5", 1.0, 0.0, 4.0, 0.05, "Urgency for five stones in a pure 6-cell window."),
        ParameterSpec("positive_temperature", 1.0, 0.1, 4.0, 0.05, "Sharpness for the derived positive top-cell overlay."),
    )

    def compute(
        self,
        position: AxisPolicyInput,
        parameters: Mapping[str, float] | None = None,
    ) -> AxisPolicyResult:
        params = merge_parameters(self.parameters, parameters)
        urgency = np.array(
            [0.0, params["w1"], params["w2"], params["w3"], params["w4"], params["w5"], params["w5"]],
            dtype=np.float32,
        )
        maps = empty_axis_maps()
        own = position.own_stones
        opp = position.opp_stones

        for axis, (dq, dr) in enumerate(AXES):
            for gi in range(BOARD_SIZE):
                for gj in range(BOARD_SIZE):
                    q = gi + position.offset_q
                    r = gj + position.offset_r
                    total = 0.0
                    for off in range(6):
                        wq = q - dq * off
                        wr = r - dr * off
                        own_count = 0
                        opp_count = 0
                        for step in range(6):
                            cell = (wq + dq * step, wr + dr * step)
                            own_count += int(cell in own)
                            opp_count += int(cell in opp)
                        if own_count and opp_count:
                            continue
                        total += float(urgency[own_count] - urgency[opp_count])
                    maps[axis, gi, gj] = total

        positive = np.maximum(maps.max(axis=0), 0.0)
        if params["positive_temperature"] != 1.0:
            positive = np.power(positive, params["positive_temperature"])
        combined = normalize_policy(
            positive,
            position.legal_set,
            position.offset_q,
            position.offset_r,
        )
        debug = {
            "target_kind": "signed_axis_field",
            "value_range": "negative=opponent pure-window urgency, positive=current-player pure-window urgency",
            "legacy_urgency": urgency.tolist(),
            "own_stones": len(own),
            "opp_stones": len(opp),
            "legal_moves": len(position.legal_set),
        }
        return AxisPolicyResult(
            self.prototype_id,
            params,
            maps,
            combined,
            debug,
            position.offset_q,
            position.offset_r,
        )
