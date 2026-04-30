# Phase 04 Protocol Manifest Examples

Canonical owner: `Python/src/hexorl/inference/protocol.py`.

Example manifest fields are produced by `default_protocol_manifest(max_batch_size=8, timeout_ms=250.0)`. The manifest includes every Phase 04 required key: protocol/request/response schema versions, model family/spec, input/output/action contracts, graph/relation/candidate/pair versions, Rust row encodings, heads, adapter name/version, transport, capacities, timeout/heartbeat, git sha, and config hash.

Initial request kinds are:

- `dense_policy_value`
- `sparse_policy_value`
- `global_graph_policy_value`
- `pair_scoring`
- `sparse_pair_policy_value`
- `graph_pair_policy_value`
- `regret_rank_policy_value`

Request examples:

- Dense: payload `tensor`, `count`; hashes `position_hash`, `manifest_hash`; adapter `dense_policy_value`.
- Sparse: dense payload plus `candidate_indices`, `candidate_features`, `candidate_mask`; hash `legal_hash`.
- Pair scoring: sparse payload plus supplied `pair_candidate_indices`, `pair_candidate_mask`; hash `pair_hash`. Pair row generation is outside the adapter.
- Global graph: payload `graph_batch`; hashes graph legal rows and graph pair token indices.

Validation evidence: `Python/tests/inference/test_protocol_manifest.py`, `test_dense_adapter_roundtrip.py`, `test_sparse_adapter_roundtrip.py`, `test_pair_scoring_adapter_roundtrip.py`, and `test_global_graph_adapter_roundtrip.py`.
