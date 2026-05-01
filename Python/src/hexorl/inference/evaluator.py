"""Evaluator protocol shared by local and remote inference."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from hexorl.inference.protocol import InferenceProtocolManifest, InferenceResponse


class Evaluator(Protocol):
    manifest: InferenceProtocolManifest

    def evaluate(self, op: str, payload: dict[str, np.ndarray]) -> InferenceResponse: ...

    def close(self) -> None: ...

