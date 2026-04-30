"""Sparse action-keyed inference adapter."""

from __future__ import annotations

import numpy as np

from hexorl.inference.adapters.base import InferenceAdapter, require_finite, require_shape
from hexorl.inference.protocol import InferencePayloadValidationError, InferenceRequest, InferenceRequestKind


class SparsePolicyValueAdapter(InferenceAdapter):
    name = "sparse"

    def validate_request(self, request: InferenceRequest) -> None:
        super().validate_request(request)
        if request.request_kind != InferenceRequestKind.SPARSE_POLICY_VALUE:
            raise InferencePayloadValidationError(f"sparse adapter cannot handle {request.request_kind.value}")
        tensor = np.asarray(request.payload["tensor"])
        count = int(request.payload["count"])
        indices = np.asarray(request.payload["candidate_indices"])
        features = np.asarray(request.payload["candidate_features"])
        mask = np.asarray(request.payload["candidate_mask"])
        if count <= 0 or count > self.capacity().max_batch_size:
            raise InferencePayloadValidationError(f"count {count} exceeds sparse adapter capacity")
        if indices.shape[:2] != mask.shape[:2] or features.shape[:2] != indices.shape[:2]:
            raise InferencePayloadValidationError("candidate indices/features/mask must share (B, K)")
        if indices.shape[0] < count:
            raise InferencePayloadValidationError("candidate rows do not cover request count")
        if indices.shape[1] > self.capacity().max_candidate_rows:
            raise InferencePayloadValidationError("candidate rows exceed manifest capacity")
        require_shape("tensor", tensor[:count], (13, 33, 33))
        require_finite("tensor", tensor[:count])
        require_finite("candidate_features", features[:count])
