# Inference Inventory

Current inference semantics are distributed across fixed shared-memory arrays,
raw model-output keys, and response head flags. Stage 4 must move semantics to
protocol/adapters while keeping shared memory as transport.

## Inference Paths

| Request kind | Input tensors | Row tables | Shared-memory fields | Raw output keys | Response fields | Head flags | Row identity carried or lost | Value decoding | Runtime consumers | New protocol responsibility | New adapter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense crop batch | dense `(B,13,33,33)` | dense board rows implicit `0..1088` | `req_tensor`, `req_count`, `res_policy`, `res_value`, `res_regret_rank` | `policy`, `value`, optional `regret_rank` | flattened dense policy logits, scalar values, regret rank | none | dense row identity implicit and not hashed | `HexNet.bins_to_value`, clamp `[-1,1]` | self-play leaves, dense root, dashboard/eval | carry output schema id, dense row-table id, value decoder id | dense inference adapter |
| sparse candidate batch | dense tensor plus candidate row data | candidate qr/index/features/mask | `req_candidate_count`, `req_candidate_indices`, `req_candidate_features`, `req_candidate_mask`, `res_sparse_logits` | `policy`, `value`, `sparse_policy` | dense policy, value, sparse logits | none | candidate qr not returned in response; caller retains it | same as dense | sparse MCTS root/leaf | response must bind sparse logits to candidate row hash and feature schema | sparse candidate adapter |
| crop pair candidate batch | dense tensor plus candidate rows plus pair row indices/mask | candidate rows and pair candidate rows | candidate fields plus `req_pair_count`, `req_pair_indices`, `req_pair_mask`, `res_pair_logits` | `policy`, `value`, `sparse_policy`, `pair_policy` | dense policy, value, sparse logits, pair logits | none | pair row identity caller-retained only | same as dense | diagnostic pair strategy | bind pair logits to candidate row hash and pair row hash | crop pair adapter |
| graph global request | graph token/action/pair/relation arrays | token rows, legal rows, optional opponent legal rows, pair rows | `req_graph_meta`, graph token/legal/opp/pair slots, sparse relation edge slots, `res_graph_meta`, graph logits slots | `policy_place`, `value`, optional `opp_policy`, `policy_pair_first`, `policy_pair_joint`, `policy_pair_second`, `regret_rank` | keyed graph logits and metadata | `GRAPH_HEAD_*` bit flags | legal qr returned; no row hash; pair row identity inferred from request slot | `HexNet.bins_to_value`, clamp `[-1,1]` | global MCTS root and pair strategy | schema version, relation schema, row hashes, pair phase, requested outputs, returned outputs | global graph adapter |
| graph pair chunk request | graph request with chunked pair rows | same legal table, pair chunk table | same graph slots with chunked pair count | `policy_pair_joint` or `policy_pair_second` | pair logits plus graph value/policy | same flags | legal table persists in caller; pair chunk identity not hashed | value decoded though pair request may not consume it | pair strategy chunk scorer | validate pair chunk row hash against base graph row hash and phase | graph pair chunk adapter |
| regret-rank diagnostic | dense crop via `submit_regret_rank` | dense board rows | dense slots plus `res_regret_rank` | `regret_rank` if model returns it | regret rank array | none | n/a | n/a | RGSC diagnostics/restart | declare diagnostic output contract | diagnostic adapter |

## Transport Constraints To Preserve

- Dense crop tensor: `13x33x33` float32 per position.
- Candidate capacity: `MAX_CANDIDATES = 512`.
- Pair candidate capacity: `MAX_PAIR_CANDIDATES = 512`.
- Graph capacities: `MAX_GRAPH_TOKENS = 4096`, `MAX_GRAPH_ACTIONS = 8192`,
  `MAX_GRAPH_PAIRS = 4096`, `MAX_GRAPH_RELATION_EDGES = 524288`.
- Graph relation IPC is sparse edge/type/bias overlay only. The inference
  server derives dense geometry relations and relation bias once before model
  forward; dense relation matrices are not shared-memory request slots.
- Graph IPC currently supports exactly one graph position per worker request.
- Graph request metadata currently holds only schema versions, counts, and
  capacity constants. Stage 4 needs a schema change to include protocol version,
  output request mask, row hashes, pair phase, and value decoder id.

## Protocol Decisions

- Model outputs are decoded by architecture-selected adapters, not shared-memory
  arrays.
- Shared memory carries protocol facts. It does not define what a head means.
- Response validation must reject:
  - missing requested runtime output;
  - row hash mismatch;
  - schema version mismatch;
  - value decoder mismatch;
  - pair logits returned for a pair phase different from requested phase;
  - pair output present without an active pair strategy request.
