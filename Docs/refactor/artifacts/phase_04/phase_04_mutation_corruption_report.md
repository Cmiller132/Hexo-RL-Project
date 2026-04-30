# Phase 04 Mutation And Corruption Report

Negative coverage:

- Manifest mismatch raises `InferenceProtocolMismatch` before enqueue.
- Non-finite dense payloads are rejected by adapter validation.
- Candidate shape mismatches are rejected by sparse adapter validation.
- Pair shape mismatches are rejected by pair-scoring adapter validation.
- Non-finite model outputs are rejected by server boundary validation rather than sanitized.
- Transport timeout leaves state `failed` and reports request context.

Evidence:

- `Python/tests/inference/test_protocol_mismatch.py`
- `Python/tests/inference/test_sparse_adapter_roundtrip.py`
- `Python/tests/inference/test_pair_scoring_adapter_roundtrip.py`
- `Python/tests/inference/test_shm_transport_timeouts.py`
- `Python/tests/test_inference_server.py`
