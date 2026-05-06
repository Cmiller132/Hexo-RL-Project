# Golden Test Plan

Stage 1 does not add importable runtime modules. This file names the golden
tests and replacement tests that close the proof harness before and during the
cutover stages.

The per-test trust inventory in
`Docs/artifacts/model_architecture/test_trust_audit.md` is the authoritative
Stage 2 worklist. This file summarizes the closure plan and keeps the golden
rules visible.

## Golden Rules To Preserve

- Legal rows from Rust/global graph must align exactly with MCTS legal rows.
- Row-table count equality is not enough; row identity and order matter.
- Graph legal, opponent legal, pair, token, and relation rows are separate
  contracts.
- Pair targets reject duplicate cells and illegal rows.
- First-placement joint pair rows are unordered canonical pairs.
- Second-placement pair rows are ordered known-first rows.
- Pair output presence does not enable pair MCTS influence.
- Config rejects unsupported architecture ids and invalid head/horizon
  combinations.
- Self-play inference requires a policy capability and a value capability.
- Lookahead heads must match configured horizons and active weights.
- Sparse candidate overflow disables affected trainable sparse/pair signals
  explicitly.
- Truncated/no-outcome values and missing selected-action regret rows are
  weighted out explicitly.

## Existing Tests To Keep As Golden After Harness Repair

- `Python/tests/test_config_and_guardrails.py`
  - config lookahead validation;
  - architecture id validation;
  - `graph` alias classification until Stage 2 decides final behavior;
  - pair strategy requires explicit strategy/cap;
  - pair heads do not enable pair scoring without strategy;
  - sparse policy config contracts;
  - hex convolution mask invariants.
- `Python/tests/test_tactical_oracle.py`
  - production tactical oracle requirement;
  - Python fallback only when explicit;
  - candidate builder includes critical cells.
- `Python/tests/test_training_data_pipeline.py`
  - turn-boundary/value perspective tests;
  - opponent policy target selection;
  - regret target and weight tests;
  - compact replay preservation tests;
  - pair candidate duplicate/illegal tests;
  - candidate feature schema tests;
  - low-PCR/truncated weight tests.
- `Python/tests/test_global_graph_contract.py`
  - row alignment helpers;
  - graph pair target ordering and duplicate/illegal checks;
  - graph output head gating;
  - graph family coverage.
- `Python/tests/test_inference_server.py`
  - dense and sparse inference server behavior;
  - graph keyed-logit behavior after graph fixture harness repair.

## Existing Tests To Rewrite

- Tests that depend on `build_graph_batch_from_history` but do not have `_engine`
  available need either a built Rust extension in CI or a contract-local graph
  fixture that does not weaken production runtime behavior.
- `test_compute_losses_skips_missing_targets_and_handles_batch_one` must be
  rewritten in Stage 3. Missing trainable targets should become hard errors; a
  separate diagnostic/optional-output test may cover intentionally omitted
  non-trainable heads.
- Tests relying on `graph` alias normalization should be rewritten after Stage 2
  chooses the final alias behavior.
- Tests that assert fallback aliases, including opponent-policy or pair-first
  fallback targets, must be rewritten to assert explicit target contracts.
- Tests that rely on lookahead fallback to value must be rewritten to assert a
  missing-lookahead hard error.

## Existing Tests To Delete

No existing test is deleted in Stage 1. Deletion candidates are limited to tests
whose only purpose is preserving legacy alias/fallback behavior after the
replacement tests exist.

## New Contract Tests To Add

Stage 2:

- registry resolves all kept architecture ids and rejects deleted aliases;
- config cannot disable self-play required policy/value outputs;
- `lookahead_*` expands from configured horizons;
- config override enables/disables only supported optional outputs;
- retained `hexorl/model` implementation is imported only by approved recipes.

Stage 3:

- output/target row hash mismatch fails before loss computation;
- trainable missing target, mask, weight, or phase fails;
- zero target mass fails unless explicitly weighted/non-trainable;
- `policy_pair_first` marginalizes unordered pair mass over both cells;
- `policy_pair_second` requires `second_placement_known_first`;
- lookahead target missing fails; no fallback to value;
- dense, sparse, graph, and pair namespaces cannot consume each other's targets.

Stage 4:

- inference request/response row hashes match for dense, candidate, legal, pair,
  and graph-token rows;
- shared-memory metadata carries protocol version, output mask, row hashes,
  value decoder, and pair phase;
- pair output returned without a matching strategy request is ignored or
  rejected according to the adapter contract;
- global graph pair chunk row hash mismatch fails;
- value decoder mismatch fails before MCTS consumption.
