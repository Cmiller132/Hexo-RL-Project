"""Shared adapter validation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hexorl.inference.protocol import (
    InferencePayloadValidationError,
    InferenceProtocolManifest,
    InferenceRequest,
    InferenceResponse,
)


@dataclass(frozen=True)
class AdapterCapacity:
    max_batch_size: int
    max_candidate_rows: int
    max_pair_rows: int
    max_graph_tokens: int
    max_graph_relations: int


class InferenceAdapter:
    name = "base"
    version = 1

    def __init__(self, manifest: InferenceProtocolManifest):
        self._manifest = manifest

    def manifest(self) -> InferenceProtocolManifest:
        return self._manifest

    def capacity(self) -> AdapterCapacity:
        return AdapterCapacity(
            max_batch_size=self._manifest.max_batch_size,
            max_candidate_rows=self._manifest.max_candidate_rows,
            max_pair_rows=self._manifest.max_pair_rows,
            max_graph_tokens=self._manifest.max_graph_tokens,
            max_graph_relations=self._manifest.max_graph_relations,
        )

    def validate_request(self, request: InferenceRequest) -> None:
        if request.manifest_hash != self._manifest.hash():
            raise InferencePayloadValidationError("request manifest hash does not match adapter manifest")

    def collate(self, requests: list[InferenceRequest]):
        return requests

    def forward(self, model, batch):
        return model(batch)

    def decode(self, batch, outputs):
        return outputs

    def assert_response(self, response: InferenceResponse) -> None:
        response.require_ok()


def require_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(np.asarray(array)).all():
        raise InferencePayloadValidationError(f"{name} contains non-finite values")


def require_shape(name: str, array: np.ndarray, expected_suffix: tuple[int, ...]) -> None:
    arr = np.asarray(array)
    if arr.ndim < len(expected_suffix) or tuple(arr.shape[-len(expected_suffix):]) != expected_suffix:
        raise InferencePayloadValidationError(
            f"{name} shape {arr.shape} does not end with {expected_suffix}"
        )
