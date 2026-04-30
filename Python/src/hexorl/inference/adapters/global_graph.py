"""Global graph inference adapter."""

from __future__ import annotations

import numpy as np

from hexorl.inference.adapters.base import InferenceAdapter, require_finite
from hexorl.inference.protocol import InferencePayloadValidationError, InferenceRequest, InferenceRequestKind


class GlobalGraphPolicyValueAdapter(InferenceAdapter):
    name = "global_graph"

    def validate_request(self, request: InferenceRequest) -> None:
        super().validate_request(request)
        if request.request_kind not in (
            InferenceRequestKind.GLOBAL_GRAPH_POLICY_VALUE,
            InferenceRequestKind.GRAPH_PAIR_POLICY_VALUE,
        ):
            raise InferencePayloadValidationError(f"global graph adapter cannot handle {request.request_kind.value}")
        graph_batch = request.payload["graph_batch"]
        token_features = np.asarray(graph_batch.token_features)
        legal_qr = np.asarray(graph_batch.legal_qr)
        pair_rows = np.asarray(graph_batch.pair_token_indices)
        if token_features.shape[0] > self.capacity().max_graph_tokens:
            raise InferencePayloadValidationError("graph tokens exceed manifest capacity")
        if legal_qr.shape[0] > self._manifest.max_legal_rows:
            raise InferencePayloadValidationError("graph legal rows exceed manifest capacity")
        if pair_rows.shape[0] > self.capacity().max_pair_rows:
            raise InferencePayloadValidationError("graph pair rows exceed manifest capacity")
        require_finite("graph token features", token_features)
