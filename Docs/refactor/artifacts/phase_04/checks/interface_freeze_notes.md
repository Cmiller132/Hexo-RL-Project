# Phase 04 Interface Freeze Notes

Frozen public surfaces before implementation:

- `Python/src/hexorl/inference/protocol.py`
  - `InferenceProtocolManifest`
  - request envelope
  - response contract
  - handshake request/response
  - `InferenceProtocolMismatch`
- `Python/src/hexorl/inference/shm_transport.py`
  - transport lifecycle states: `created`, `handshaking`, `ready`, `draining`, `closed`, `failed`
  - bounded wait/deadline/heartbeat/backpressure ownership
  - slot request/response generation counters
- `Python/src/hexorl/inference/batching.py`
  - compatible request grouping by request kind, protocol, schema, and adapter capability
  - fill-rate, wait, forward, decode, queue-depth telemetry
- `Python/src/hexorl/inference/adapters/*`
  - `manifest()`
  - `validate_request(request)`
  - `capacity()`
  - `collate(requests)`
  - `forward(model, batch)`
  - `decode(batch, model_outputs)`
  - `assert_response(response)`
- `Python/src/hexorl/inference/telemetry.py`
  - response telemetry schema
  - protocol mismatch event payload
  - timeout/backpressure/cleanup failure ownership labels

Rust boundary:

- `crates/hexgame-py/src/protocol.rs` remains the canonical legal/history/pair row encoding owner.
- Phase 04 Python protocol objects record FFI protocol version and row encoding identity, but do not redefine row byte layouts.

Initial request kinds:

- `dense_policy_value`
- `sparse_policy_value`
- `global_graph_policy_value`
- `pair_scoring`
- `sparse_pair_policy_value`
- `graph_pair_policy_value`
- `regret_rank_policy_value`
