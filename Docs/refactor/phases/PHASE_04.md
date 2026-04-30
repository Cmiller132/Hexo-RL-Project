# Phase 04 - Inference Protocol and Adapters

## Purpose
Make inference a typed, versioned boundary between runtime consumers and model-family adapters.

Phase 04 removes architecture-string dispatch, private submit lifecycles, hidden tensor rebuilds, and unbounded IPC waits. The inference server accepts explicit request kinds and protocol manifests; adapters map shared contracts to model tensors and model outputs back to response contracts. Search still decides what to consume. In particular, pair-capable adapters may expose pair outputs, but `PairStrategy` controls whether pair rows are generated, scored, or used by MCTS.

Source of truth: `Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_V2_20260429.md`.

## Target Modules
- `Python/src/hexorl/inference/protocol.py`
- `Python/src/hexorl/inference/client.py`
- `Python/src/hexorl/inference/server.py`
- `Python/src/hexorl/inference/batching.py`
- `Python/src/hexorl/inference/shm_transport.py`
- `Python/src/hexorl/inference/telemetry.py`
- `Python/src/hexorl/inference/adapters/dense.py`
- `Python/src/hexorl/inference/adapters/sparse.py`
- `Python/src/hexorl/inference/adapters/global_graph.py`
- `Python/src/hexorl/inference/adapters/pair_scoring.py`
- `crates/hexgame-py/src/protocol.rs` as the Rust/PyO3 row-protocol owner consumed by inference-facing contracts

`inference/protocol.py` owns the Python inference protocol. It does not own Rust FFI row encodings. Legal rows, board-piece rows, compact history rows, and pair-row byte/array layouts remain owned by `crates/hexgame-py/src/protocol.rs` and are surfaced through validated engine/contracts objects.

## Required Protocol Objects

Add `InferenceProtocolManifest` in `inference/protocol.py`.

The protocol must use a stable base envelope plus request-kind payload schemas. The envelope owns identity, versioning, deadlines, tracing, manifest hashes, and transport metadata. Request-kind payload schemas own dense, sparse, graph, pair, or future family-specific payload details. Adding a new request kind must not require rewriting transport lifecycle code.

Required manifest fields:

```text
protocol_version
request_kind
request_schema_version
response_schema_version
model_family
model_spec_version
input_contract
output_contract
action_contract
graph_schema_version
relation_schema_version
candidate_contract_version
pair_action_contract_version
ffi_protocol_version
legal_row_encoding
history_row_encoding
pair_row_encoding
heads
adapter_name
adapter_version
transport
max_batch_size
max_legal_rows
max_candidate_rows
max_pair_rows
max_graph_tokens
max_graph_relations
timeout_ms
heartbeat_interval_ms
created_by_git_sha
config_hash
```

`request_kind` is the dispatch key. Valid initial kinds:

```text
dense_policy_value
sparse_policy_value
global_graph_policy_value
pair_scoring
sparse_pair_policy_value
graph_pair_policy_value
regret_rank_policy_value
```

The concrete names should be aligned with current runtime modes during implementation, including dense crop, sparse candidate, sparse pair, global graph, graph pair, and regret-rank only if regret-rank remains a runtime path.

Request contracts must include:

```text
request_id
trace_id
protocol_version
request_kind
request_schema_version
required_output_contract
manifest_hash
position_identity
history_hash
legal_table_hash
legal_row_hash
pair_row_hash
adapter_capability_request
slot_request_generation
payload_refs
deadline_ms
payload_schema_version
payload_kind
```

Response contracts must include:

```text
request_id
trace_id
protocol_version
request_kind
response_schema_version
manifest_hash
status
slot_response_generation
output_contract
head_outputs
telemetry
warnings
error_code
```

No raw ad-hoc payload dict crosses the client/server boundary. Dicts are acceptable only as serialization internals behind typed protocol objects.

A fake request kind must be registerable through the manifest and payload-schema path without adding architecture-string dispatch or changing shared transport lifecycle code.

## Handshake and Negotiation

Client startup performs one explicit handshake with the server before any inference request is submitted.

Handshake fields:

```text
client_protocol_version
client_supported_request_kinds
client_supported_request_schema_versions
client_supported_response_schema_versions
client_supported_transports
client_required_contracts
client_capacity_request
server_protocol_version
server_supported_request_kinds
server_supported_request_schema_versions
server_supported_response_schema_versions
server_supported_transports
server_declared_contracts
server_capacity_limits
server_model_manifest
selected_request_kind
selected_request_schema_version
selected_response_schema_version
selected_transport
selected_capacity
selected_heads
negotiated_manifest_hash
```

Negotiation rules:

- Protocol version must match exactly for Phase 04.
- Request kind must be present in both client and server support lists.
- Request and response schema versions must match exactly. Schema migration tools, if ever needed, live outside runtime and are not part of Phase 04 inference dispatch.
- Required input, output, action, graph, relation, candidate, and pair contract versions must match the selected adapter.
- Selected capacity is the minimum of client request, server limit, and checkpoint manifest limit.
- Head negotiation is explicit. A server may expose more heads than the client requests, but it must not silently omit requested heads.
- Pair heads are capability outputs only. Negotiating `policy_pair_first`, `policy_pair_second`, or `policy_pair_joint` does not enable pair row generation or MCTS consumption.

Fail-fast behavior:

- Any protocol, request kind, schema, contract, capacity, or head mismatch raises `InferenceProtocolMismatch` before the first request is enqueued.
- Mismatches are logged through `inference_protocol_mismatch` telemetry with the client manifest, server manifest, selected checkpoint manifest, and mismatch field.
- There is no fallback to architecture-prefix dispatch, legacy submit methods, dense tensor reconstruction, or Python legal/candidate/pair rebuilding.
- Inference requests that carry Rust-derived legal, history, candidate, or pair identities must fail before enqueue if the identity cannot be traced to validated contracts built from the centralized engine/FFI protocol.

Detailed verification:

- Inference tests must assume request packing, shared-memory transport, tensor collation, model forward, scatter, and decode can each corrupt, reorder, or stale-read data.
- Request packing must preserve trace id, history hash, legal row ids, candidate row ids, pair row ids, graph token ids, graph relation ids, schema versions, caps, and masks.
- Transport buffers must not allow stale request data, stale response data, stale ready flags, stale slot generations, or post-validation mutation to appear as a valid response. Ready flags alone are insufficient; per-slot request and response sequence counters are required before response read.
- Response decoding must validate output shape, row identity, protocol version, model family, output contract, masks, count fields, and non-finite values before returning to `PolicyProvider`.
- Non-finite model outputs are rejected at the protocol boundary. They must not be sanitized and allowed to continue into policy/search.
- Negative tests must corrupt protocol versions, request kind, row counts, masks, token counts, pair counts, output shapes, stale ready flags, stale slot generations, stale trace ids, and non-finite outputs.
- Negative tests must include Rust-boundary mismatch cases: stale legal-table hash, stale compact-history hash, malformed legal/history protocol bytes when constructing fixtures, and pair-row identities that cannot be traced back to `PairActionTable`.
- A single-position inference debug payload must show packed request metadata, transport lifecycle timings, raw model output metadata, decoded output metadata, response hashes, and validation failures.

## Transport Lifecycle Ownership

`inference/shm_transport.py` owns IPC setup, shared-memory allocation, queue wiring, heartbeat, backpressure, deadlines, teardown, and orphan cleanup. If implementation keeps the current `inference/shm_queue.py`, `client.py`, and `server.py` split, this phase must either move ownership into `shm_transport.py` or explicitly document the one surviving transport owner and delete duplicated lifecycle state elsewhere.

`inference/batching.py` owns batching policy. It may be implemented under another phase-approved module name only if there is still one batching owner and the protocol/docs are updated before implementation.

Batching requirements:

- batch by compatible request kind, protocol, schema, and adapter capability
- preserve GPU batching by avoiding per-leaf model forward calls
- support adaptive microbatch wait, max batch size, max in-flight per worker, and fairness across producers
- expose high/low watermarks, queue depth, fill rate, wait time, forward time, decode time, and retryable backpressure status
- fail or throttle explicitly under saturation instead of creating unbounded waits

Required lifecycle states:

```text
created
handshaking
ready
draining
closed
failed
```

Rules:

- Client and server do not duplicate lifecycle state machines inside submit methods.
- All requests carry a deadline derived from `timeout_ms`.
- IPC waits are bounded. No `get`, `put`, process join, shared-memory wait, or server poll may block indefinitely.
- Timeout errors include request id, trace id, request kind, queue depth, last heartbeat, and transport state.
- Server shutdown drains accepted requests or marks them failed; it does not leave callers waiting on responses that will never arrive.
- Backpressure is explicit. Oversized batches and saturated queues fail or return retryable status according to the manifest; they do not silently stall.
- Transport cleanup owns shared-memory unlink/close and process ownership accounting.
- Batching telemetry must be attached to Phase 04 artifacts for synthetic load and at least one self-play-shaped workload.

## Adapter Requirements

Adapters are the only inference-side owners of tensor assembly and response decoding.

Required adapter interface:

```text
manifest()
validate_request(request)
capacity()
collate(requests)
forward(model, batch)
decode(batch, model_outputs)
assert_response(response)
```

Adapter capability mapping:

- Dense adapter maps `DENSE_PLACE_POLICY` and value heads from `PositionContract`, `LegalActionTable`, and dense crop/input contracts.
- Sparse adapter maps `SPARSE_PLACE_POLICY` and value heads from `CandidateTable` plus legal/action contracts.
- Global graph adapter maps `GLOBAL_PLACE_POLICY`, graph inputs, legal rows, and any declared graph heads from `GraphSemanticContract` and tensorized graph contracts.
- Pair scoring adapter maps declared pair heads from canonical `PairActionTable` rows only.

Pair ownership rule:

- Adapters may validate and score pair rows supplied in a request.
- Adapters must not decide whether to build pair rows, enumerate full pairs, score leaf pairs, or feed pair priors to search.
- `PairStrategy` owns pair-row generation policy, caps, pair consumption, and MCTS influence.
- `global_xattn` and other global graph families still default to zero pair scoring unless `PairStrategy` explicitly requests pair work.

Response telemetry assertions:

- Every response includes protocol version, schema version, manifest hash, adapter name/version, model family, request kind, selected heads, batch size, legal row count, candidate row count, pair row count, graph token count, graph relation count, and timing spans.
- Required spans include `ipc_pack_ms`, `ipc_wait_ms`, `queue_wait_ms`, `collate_ms`, `model_forward_ms`, `scatter_ms`, and `decode_ms`.
- Pair responses also include `pair_chunk_count`, `pair_chunk_forward_ms`, `pair_rows_requested`, `pair_rows_scored`, and the active pair strategy name supplied by the caller.
- Response assertions reject missing heads, unexpected heads, non-finite logits/values, shape mismatches, legal-row count mismatches, pair-row count mismatches, graph count mismatches, stale manifest hashes, and mismatched request ids.
- Response telemetry can identify pack, transport, collate, model-forward, scatter, decode, validation, timeout, and cleanup failures separately.
- Corrupt or stale transport/request/response payloads fail before policy/search consumes them.

## Required Deletions

Delete or fully disconnect these old paths during this phase:

```text
client lifecycle duplication across submit methods
old submit_* methods that own private queue/shared-memory setup
submit_graph, submit_sparse_pair, req_mode, or equivalent mode-specific submit paths that bypass typed protocol validation
server architecture.startswith("global_") dispatch
server dispatch by model architecture string
hidden fixed-cap assumptions not declared by protocol manifest
private dense tensor rebuild paths in worker/dashboard/inference submit code
private sparse candidate tensor rebuild paths in worker/dashboard/inference submit code
private global graph tensor rebuild paths in worker/dashboard/inference submit code
private pair tensor rebuild paths in worker/dashboard/inference submit code
implicit pair scoring triggered by pair head presence inside inference
indefinite IPC waits, joins, queue gets, queue puts, and server polling loops
server-side non-finite sanitization that hides invalid model outputs instead of returning structured protocol errors
```

Short-lived compatibility wrappers are allowed only inside the migration branch for tests. They must not be imported by migrated runtime paths.

## Parallel Subagent Work
- S1: Define `InferenceProtocolManifest`, request contracts, response contracts, handshake negotiation, and mismatch errors.
- S2: Move shared-memory, queue, heartbeat, timeout, backpressure, shutdown, and cleanup ownership into `shm_transport.py`.
- S3: Implement dense, sparse, global graph, and pair scoring adapters using model-family capabilities from Phase 03.
- S4: Cut client/server dispatch from architecture strings to request kinds and adapter manifests.
- S5: Add protocol integration tests, import audits, timeout tests, telemetry assertions, and artifact generation.

## Mandatory Tests

Protocol and handshake:

```text
pytest Python/tests/inference/test_protocol_manifest.py
pytest Python/tests/inference/test_protocol_handshake.py
pytest Python/tests/inference/test_protocol_mismatch.py
```

Required cases:

- Protocol version mismatch fails before enqueue.
- Request kind mismatch fails before enqueue.
- Request schema mismatch fails before enqueue.
- Response schema mismatch fails before enqueue.
- Input/output/action contract mismatch fails before enqueue.
- Capacity mismatch selects bounded minimum or fails if required minimum cannot be met.
- Requested head missing from server manifest fails before enqueue.
- Extra unrequested server head is ignored unless the adapter marks it required.
- Fake request-kind registration works through the protocol envelope without transport lifecycle rewrites or architecture dispatch.

Transport lifecycle:

```text
pytest Python/tests/inference/test_shm_transport_lifecycle.py
pytest Python/tests/inference/test_shm_transport_timeouts.py
pytest Python/tests/inference/test_inference_no_indefinite_waits.py
```

Required cases:

- Queue `get`/`put` calls use finite timeouts.
- Server crash returns bounded failure to the client.
- Saturated queue produces retryable or failed status, not a hang.
- Shutdown drains or fails accepted requests.
- Shared-memory segments are closed/unlinked by transport ownership.
- Stale ready flags, stale trace ids, stale response buffers, and reused shared-memory contents fail validation.
- Post-validation mutation of request/response metadata is detected before policy/search consumption.
- Batching/backpressure tests cover queue saturation, fairness, timeout, retryable failure, and batch fill-rate telemetry.

Adapters and responses:

```text
pytest Python/tests/inference/test_dense_adapter_roundtrip.py
pytest Python/tests/inference/test_sparse_adapter_roundtrip.py
pytest Python/tests/inference/test_global_graph_adapter_roundtrip.py
pytest Python/tests/inference/test_pair_scoring_adapter_roundtrip.py
pytest Python/tests/inference/test_response_telemetry.py
```

Required cases:

- Dense adapter returns one place logit per legal row or declared dense action contract.
- Sparse adapter returns one sparse logit per candidate/legal row mapping.
- Global graph adapter returns exactly one `policy_place` logit per legal action row.
- Pair scoring adapter returns exactly one pair logit per supplied canonical `PairActionTable` row.
- Pair adapter does not generate pair rows by itself.
- Response telemetry contains protocol, schema, manifest, count, timing, and head metadata.
- Non-finite, wrong-shape, wrong-row-count, and stale-manifest responses fail assertions.
- Corrupt masks, row ids, token ids, relation ids, pair row ids, request ids, trace ids, and schema versions fail assertions.
- Single-position inference debug payload localizes pack, transport, collate, forward, scatter, decode, and validation failures.

Dispatch and import audits:

```text
pytest Python/tests/inference/test_server_dispatch_by_request_kind.py
pytest Python/tests/inference/test_no_private_inference_rebuilds.py
pytest Python/tests/inference/test_no_architecture_string_dispatch.py
```

Required `rg` audit checks:

```text
rg "architecture\\.startswith|startswith\\(\"global_\"\\)" Python/src/hexorl/inference
rg "submit_.*global|submit_.*dense|submit_.*sparse|submit_.*pair" Python/src/hexorl/inference
rg "Queue\\.get\\(|Queue\\.put\\(|\\.join\\(" Python/src/hexorl/inference
rg "pair_head_present|pair_prior_mix" Python/src/hexorl/inference
```

The first, second, and fourth searches must return no migrated runtime dispatch paths. The third search must be reviewed so every wait has a finite timeout or documented bounded wrapper.

## Required Artifacts

Produce these artifacts before closing Phase 04:

```text
Docs/refactor/artifacts/phase_04_protocol_manifest_examples.md
Docs/refactor/artifacts/phase_04_handshake_matrix.md
Docs/refactor/artifacts/phase_04_timeout_audit.md
Docs/refactor/artifacts/phase_04_import_audit.md
Docs/refactor/artifacts/phase_04_response_telemetry_snapshot.md
Docs/refactor/artifacts/phase_04_batching_backpressure_profile.md
Docs/refactor/artifacts/phase_04_inference_debug_bundle.md
Docs/refactor/artifacts/phase_04_mutation_corruption_report.md
```

Artifact contents:

- Manifest examples for dense, sparse, global graph, and pair scoring requests.
- Request-kind extension example proving envelope/payload separation.
- Handshake compatibility matrix covering request kind, version, contract, capacity, and head negotiation.
- Timeout audit listing every inference wait site and its finite deadline behavior.
- Batching/backpressure profile with batch fill rate, queue depth, p50/p95 waits, timeout/retryable counts, and GPU utilization or proxy timing.
- Import audit proving worker/dashboard/training do not call private inference tensor rebuild paths.
- Telemetry snapshot showing response schema/protocol fields and timing spans.
- Inference debug bundle showing one traced position through pack, transport, model output, decode, response validation, and failure ownership.
- Mutation/corruption report covering stale buffers, stale ids, wrong row counts, bad masks, wrong shapes, and non-finite outputs.

## Hard Exit Gates

Phase 04 is not complete until all gates pass:

```text
InferenceProtocolManifest is required for every client/server session.
Base envelope plus request-kind payload schemas support extension without architecture dispatch.
Server dispatches by request_kind and negotiated manifest, not architecture string.
Protocol, schema, contract, capacity, and head mismatches fail before enqueue.
No inference wait can block indefinitely.
Transport owns shared-memory, queue, heartbeat, backpressure, shutdown, and cleanup lifecycle.
Batching/backpressure preserves bounded waits and GPU-batchable request groups.
Dense, sparse, global graph, and pair scoring adapters round-trip through the inference server.
All responses include and pass telemetry/schema/protocol assertions.
Corrupt, stale, or mutated inference payloads fail before policy/search consumes them.
Inference debug payload can localize failures to pack, transport, collate, model forward, scatter, decode, or response validation.
Pair-capable adapters expose capabilities only; PairStrategy controls pair row generation and consumption.
No private worker/dashboard/inference tensor rebuild path remains in migrated runtime imports.
Old submit lifecycle paths are deleted or disconnected from runtime imports.
Import audits prove no architecture-prefix inference dispatch remains.
Mandatory tests and artifact files are present.
```
