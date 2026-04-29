# Phase 04 — Inference Protocol and Adapterization

## Purpose
Standardize inference communication and tensor assembly through explicit protocol contracts and per-family adapters.

## Target Modules
- `inference/protocol.py`
- `inference/client.py`, `server.py`, `batching.py`, `shm_transport.py`, `telemetry.py`
- `inference/adapters/{dense,sparse,global_graph,pair_scoring}.py`

## V2 Requirements
- No ad-hoc payload dicts across inference boundaries.
- Protocol is versioned and compatible across client/server lifecycle.
- Tensorization logic owned by adapters, not worker/dashboard duplications.

## Parallel Subagent Work
- S1: request/response schema versions and compatibility policy.
- S2: transport lifecycle and backpressure semantics.
- S3: adapter implementations mapped to model capabilities.
- S4: training-side interoperability expectations.
- S5: protocol fuzzing + perf telemetry baselines.

## Mandatory Tests
- Serialization/deserialization compatibility tests.
- Client/server integration with mixed family requests.
- Throughput and latency smoke versus phase-00 baseline.
- Error path tests (timeouts, oversized batches, invalid version).

## Exit Criteria
- Inference consumers use shared protocol objects and adapters only.
- No private tensor rebuild paths remain in migrated scope.
