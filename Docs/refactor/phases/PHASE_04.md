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

## Required Protocol Objects

Add `InferenceProtocolManifest` in `inference/protocol.py`.

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
```

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
adapter_capability_request
payload_refs
deadline_ms
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
output_contract
head_outputs
telemetry
warnings
error_code
```

No raw ad-hoc payload dict crosses the client/server boundary. Dicts are acceptable only as serialization internals behind typed protocol objects.

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

## Transport Lifecycle Ownership

`inference/shm_transport.py` owns IPC setup, shared-memory allocation, queue wiring, heartbeat, backpressure, deadlines, teardown, and orphan cleanup.

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

## Required Deletions

Delete or fully disconnect these old paths during this phase:

```text
client lifecycle duplication across submit methods
old submit_* methods that own private queue/shared-memory setup
server architecture.startswith("global_") dispatch
server dispatch by model architecture string
hidden fixed-cap assumptions not declared by protocol manifest
private dense tensor rebuild paths in worker/dashboard/inference submit code
private sparse candidate tensor rebuild paths in worker/dashboard/inference submit code
private global graph tensor rebuild paths in worker/dashboard/inference submit code
private pair tensor rebuild paths in worker/dashboard/inference submit code
implicit pair scoring triggered by pair head presence inside inference
indefinite IPC waits, joins, queue gets, queue puts, and server polling loops
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
```

Artifact contents:

- Manifest examples for dense, sparse, global graph, and pair scoring requests.
- Handshake compatibility matrix covering request kind, version, contract, capacity, and head negotiation.
- Timeout audit listing every inference wait site and its finite deadline behavior.
- Import audit proving worker/dashboard/training do not call private inference tensor rebuild paths.
- Telemetry snapshot showing response schema/protocol fields and timing spans.

## Hard Exit Gates

Phase 04 is not complete until all gates pass:

```text
InferenceProtocolManifest is required for every client/server session.
Server dispatches by request_kind and negotiated manifest, not architecture string.
Protocol, schema, contract, capacity, and head mismatches fail before enqueue.
No inference wait can block indefinitely.
Transport owns shared-memory, queue, heartbeat, backpressure, shutdown, and cleanup lifecycle.
Dense, sparse, global graph, and pair scoring adapters round-trip through the inference server.
All responses include and pass telemetry/schema/protocol assertions.
Pair-capable adapters expose capabilities only; PairStrategy controls pair row generation and consumption.
No private worker/dashboard/inference tensor rebuild path remains in migrated runtime imports.
Old submit lifecycle paths are deleted or disconnected from runtime imports.
Import audits prove no architecture-prefix inference dispatch remains.
Mandatory tests and artifact files are present.
```
