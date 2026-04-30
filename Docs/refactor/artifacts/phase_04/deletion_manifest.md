# Phase 04 Deletion Manifest

Deleted or disconnected runtime paths:

- Removed `InferenceClient.submit_sparse`.
- Removed `InferenceClient.submit_sparse_pair`.
- Removed `InferenceClient.submit_graph`.
- Removed `InferenceClient.submit_regret_rank`.
- Replaced client-side mode-specific shared-memory packing with `ShmTransport.round_trip`.
- Replaced shared-memory `req_mode` with typed `req_kind`.
- Replaced server graph/dense dispatch by request mode with dispatch by `InferenceRequestKind`.
- Removed server-side non-finite sanitization; invalid outputs now raise `InferenceOutputValidationError`.
- Migrated self-play runtime callers to `evaluate_sparse`, `evaluate_pair_scoring`, `evaluate_global_graph`, and `evaluate_regret_rank`.

No compatibility shim was added under `Python/src/hexorl/`.
