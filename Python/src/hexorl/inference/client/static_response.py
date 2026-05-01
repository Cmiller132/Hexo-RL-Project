"""Static client responses for zero-row requests."""

from __future__ import annotations

import numpy as np


class StaticResponse:
    def __init__(self, heads: dict[str, np.ndarray]):
        self.head_outputs = heads
        self.telemetry = {"wait_ms": 0.0}

    def require_ok(self) -> None:
        return


def zero_count_response(operation_name: str, count: int = 0) -> StaticResponse:
    empty = np.empty(0, dtype=np.float32)
    if operation_name == "sparse_place_value":
        return StaticResponse(
            {
                "policy": empty,
                "value": empty,
                "sparse_policy": np.empty((count, 0), dtype=np.float32),
            }
        )
    if operation_name == "pair_policy":
        return StaticResponse(
            {
                "policy": empty,
                "value": empty,
                "sparse_policy": np.empty((count, 0), dtype=np.float32),
                "pair_policy": np.empty((count, 0), dtype=np.float32),
            }
        )
    return StaticResponse({"policy": empty, "value": empty})


__all__ = ["StaticResponse", "zero_count_response"]
