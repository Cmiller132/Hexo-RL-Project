# Phase 04 Handshake Matrix

| Case | Result | Evidence |
|---|---|---|
| Matching protocol/schema/manifest/kind | accepted before enqueue | `Python/tests/inference/test_protocol_handshake.py` |
| Manifest capacity mismatch | `InferenceProtocolMismatch` before enqueue | `Python/tests/inference/test_protocol_mismatch.py` |
| Unsupported request kind | `InferenceProtocolMismatch` by `negotiate_protocol` | covered by request-kind support check in `protocol.py` |
| Client lifecycle | `created -> handshaking -> ready -> draining -> ready` | `Python/tests/inference/test_shm_transport_lifecycle.py` |

Handshake implementation: `InferenceClient.connect()` builds a manifest, enters `HANDSHAKING`, calls `negotiate_protocol`, and marks the transport ready only after acceptance.
