"""Dense policy/value inference adapter."""

from __future__ import annotations

import numpy as np

from hexorl.inference.adapters.base import InferenceAdapter, require_finite, require_shape
from hexorl.inference.protocol import InferencePayloadValidationError, InferenceRequest, InferenceRequestKind


class DensePolicyValueAdapter(InferenceAdapter):
    name = "dense"

    def validate_request(self, request: InferenceRequest) -> None:
        super().validate_request(request)
        if request.request_kind not in (
            InferenceRequestKind.DENSE_POLICY_VALUE,
            InferenceRequestKind.REGRET_RANK_POLICY_VALUE,
        ):
            raise InferencePayloadValidationError(f"dense adapter cannot handle {request.request_kind.value}")
        tensor = np.asarray(request.payload["tensor"])
        count = int(request.payload["count"])
        if count < 0 or count > self.capacity().max_batch_size:
            raise InferencePayloadValidationError(f"count {count} exceeds dense adapter capacity")
        require_shape("tensor", tensor[:count], (13, 33, 33))
        require_finite("tensor", tensor[:count])
