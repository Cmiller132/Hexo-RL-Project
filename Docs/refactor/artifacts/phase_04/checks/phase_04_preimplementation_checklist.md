# Phase 04 Preimplementation Checklist

## Assignment Frame

Goal

Implement a typed, versioned inference boundary where every client/server session negotiates an `InferenceProtocolManifest`, dispatches by request kind, validates typed request/response contracts, and uses bounded transport/batching lifecycle ownership.

Success criteria

- V2-040: `InferenceProtocolManifest`, request envelopes, response contracts, fake request-kind extension proof, and Rust FFI row identity fields exist and are consumed before enqueue.
- V2-041: Server dispatch uses negotiated request kind and manifest, not model architecture strings.
- V2-042: Protocol/schema/contract/capacity/head/Rust-row mismatches fail before enqueue with structured errors and telemetry.
- V2-043: Transport owns pack, ready, wait, timeout, decode, reset, heartbeat, cleanup, and slot generations.
- V2-044: Responses include protocol, contract, model, count, timing, warning, FFI source, and row hash telemetry.
- V2-045: Stale/corrupt/mutated payloads fail before policy/search consumption.
- V2-046: Batching/backpressure remains GPU-batchable and bounded under synthetic and self-play-shaped load.

Constraints

- No architecture-string inference dispatch.
- No raw ad-hoc payload dict crosses the public client/server boundary.
- No indefinite IPC wait, process join, queue get, queue put, or server poll.
- No private dense/sparse/graph/pair tensor rebuild path in migrated runtime imports.
- Pair-capable adapters may score supplied rows only; `PairStrategy` owns pair row generation and MCTS consumption.
- Rust row encodings remain owned by `crates/hexgame-py/src/protocol.rs`.

Required evidence

- Protocol manifest examples for dense, sparse, global graph, and pair scoring.
- Handshake matrix and mismatch telemetry.
- Timeout audit for every inference wait.
- Import audit for architecture dispatch, submit variants, queue waits, joins, and pair heuristics.
- Response telemetry snapshot and inference debug bundle.
- Mutation/corruption report.
- Batching/backpressure profile.
- Mandatory inference protocol, transport, adapter, dispatch, telemetry, and audit tests.

Stop rules

- Stop before coding if a typed protocol cannot be inserted without keeping old submit lifecycles as runtime fallbacks.
- Stop if a deletion would remove the only inference path before a protocol path is consumed.
- Stop if Rust row identity/hash evidence cannot be validated before enqueue.
- Stop if a required wait cannot be bounded deterministically.
- Stop if a phase-closing inference test would need to be skipped, xfailed, or made manual-only.

## Matrix Rows

| Row | Requirement | Status |
|---|---|---|
| V2-040 | Protocol manifest/envelope/payload schemas and Rust FFI identities | implemented and tested |
| V2-041 | Dispatch by request kind/protocol | implemented and tested |
| V2-042 | Structured fail-fast mismatch and bounded waits | implemented and tested |
| V2-043 | Transport lifecycle and slot generations | implemented and tested |
| V2-044 | Response telemetry with FFI protocol/row hashes | implemented and tested |
| V2-045 | Corruption/mutation/stale payload validation | implemented and tested |
| V2-046 | Batching/backpressure performance evidence | implemented and tested |
