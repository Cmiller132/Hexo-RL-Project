"""Pair-scoring adapter.

Pair row generation remains owned by PairStrategy and callers. This adapter only
validates supplied candidate-index pair rows and transports pair logits.
"""

from __future__ import annotations

import numpy as np

from hexorl.inference.adapters.base import InferenceAdapter, require_finite, require_shape
from hexorl.inference.protocol import InferencePayloadValidationError, InferenceRequest, InferenceRequestKind


class PairScoringAdapter(InferenceAdapter):
    name = "pair_scoring"

    def validate_request(self, request: InferenceRequest) -> None:
        super().validate_request(request)
        if request.request_kind not in (
            InferenceRequestKind.PAIR_SCORING,
            InferenceRequestKind.SPARSE_PAIR_POLICY_VALUE,
            InferenceRequestKind.GRAPH_PAIR_POLICY_VALUE,
        ):
            raise InferencePayloadValidationError(f"pair adapter cannot handle {request.request_kind.value}")
        payload = request.payload
        tensor = np.asarray(payload["tensor"])
        count = int(payload["count"])
        indices = np.asarray(payload["candidate_indices"])
        features = np.asarray(payload["candidate_features"])
        mask = np.asarray(payload["candidate_mask"])
        pair_indices = np.asarray(payload["pair_candidate_indices"])
        pair_mask = np.asarray(payload["pair_candidate_mask"])
        if count <= 0 or count > self.capacity().max_batch_size:
            raise InferencePayloadValidationError(f"count {count} exceeds pair adapter capacity")
        if pair_indices.shape[:2] != pair_mask.shape[:2]:
            raise InferencePayloadValidationError("pair indices and pair mask must share (B, P)")
        if pair_indices.ndim != 3 or pair_indices.shape[2] != 2:
            raise InferencePayloadValidationError("pair indices must have shape (B, P, 2)")
        if pair_indices.shape[1] > self.capacity().max_pair_rows:
            raise InferencePayloadValidationError("pair rows exceed manifest capacity")
        if indices.shape[:2] != mask.shape[:2] or features.shape[:2] != indices.shape[:2]:
            raise InferencePayloadValidationError("candidate indices/features/mask must share (B, K)")
        require_shape("tensor", tensor[:count], (13, 33, 33))
        require_finite("tensor", tensor[:count])
        require_finite("candidate_features", features[:count])
