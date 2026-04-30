# Phase 04 CI Routing Plan

Tier 0 preflight:

- `python -m compileall -q Python/src/hexorl`
- Protocol unit tests under `Python/tests/inference/test_protocol_*.py`
- Import audits for architecture-string dispatch and private submit paths.

Tier 1 deterministic CPU:

- `Python/tests/inference/test_protocol_manifest.py`
- `Python/tests/inference/test_protocol_handshake.py`
- `Python/tests/inference/test_protocol_mismatch.py`
- `Python/tests/inference/test_shm_transport_lifecycle.py`
- `Python/tests/inference/test_shm_transport_timeouts.py`
- `Python/tests/inference/test_inference_no_indefinite_waits.py`
- Adapter round-trip tests for dense, sparse, global graph, and pair scoring.

Tier 2 integration/performance:

- Existing inference server integration tests.
- Synthetic batching/backpressure profile.
- Self-play-shaped request workload with bounded waits and telemetry.

Phase-closing invariants cannot be skipped, xfailed, flaky-only, or manual-only.
