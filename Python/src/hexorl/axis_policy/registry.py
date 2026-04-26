"""Prototype registry for the axis policy design suite."""

from __future__ import annotations

from typing import Mapping

from hexorl.axis_policy.core import AxisPolicyInput, AxisPolicyPrototype
from hexorl.axis_policy.dual_strength import DualAxisStrengthPrototype
from hexorl.axis_policy.experiments import CrossAxisPivotPrototype, DeltaForkPrototype
from hexorl.axis_policy.legacy_influence import LegacyAxisInfluencePrototype


def builtins() -> list[AxisPolicyPrototype]:
    return [
        DualAxisStrengthPrototype(),
        LegacyAxisInfluencePrototype(),
        DeltaForkPrototype(),
        CrossAxisPivotPrototype(),
    ]


def get_prototype(prototype_id: str) -> AxisPolicyPrototype:
    for proto in builtins():
        if proto.prototype_id == prototype_id:
            return proto
    raise KeyError(f"Unknown axis policy prototype: {prototype_id}")


def describe_prototypes() -> list[dict]:
    return [
        {
            "id": proto.prototype_id,
            "label": proto.label,
            "description": proto.description,
            "parameters": [
                {
                    "name": spec.name,
                    "default": spec.default,
                    "min": spec.min,
                    "max": spec.max,
                    "step": spec.step,
                    "description": spec.description,
                }
                for spec in proto.parameters
            ],
        }
        for proto in builtins()
    ]


def evaluate_all(
    position: AxisPolicyInput,
    parameter_overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> list[dict]:
    overrides = parameter_overrides or {}
    return [
        proto.compute(position, overrides.get(proto.prototype_id)).to_json()
        for proto in builtins()
    ]
