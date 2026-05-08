# V1 Pair Model Completion Verification

## Intent

This document records the source-level V1 pair cutover fixes applied on May 8, 2026, and the verification steps for environments with and without the Rust `_engine` extension.

The implementation intent is:

- Make `sampled_joint_pair_v1` use one canonical pair-feature contract across runtime, replay, graph batching, and training.
- Treat V1 replay as a schema break: schema-2 metadata must carry explicit terminal tactical payloads.
- Keep terminal tactical labels faithful to the Rust tactical payload instead of reconstructing them from selected candidate masks.
- Make V1 candidate admission model-informed before final pair scoring.
- Reuse trained V1 pair projection parameters to expose runtime-only legal
  proposal/completion projections without adding new training targets.
- Drive normal two-placement self-play through the Rust expansion protocol:
  Rust owns recursive pair selection, node stats, widening, backup, and action
  choice; Python supplies neural expansion batches.
- Preserve tactical-protected support even when it exceeds ordinary candidate budget.
- Remove unused proposal-score target arrays so `pair_proposal_score` is trained through the ranking target intentionally.
- Prevent silent V1 value fallback by requiring a `value` output from graph inference.
- Quarantine pre-V1 pair-to-single projection authority outside the flagship
  `sampled_joint_pair_v1` path.

## Source-Level Fixes

### Contract And Schema

- Added `Python/src/hexorl/v1_pair_contract.py` as the canonical V1 feature and tactical payload contract.
- Defined the ordered 12-field V1 pair feature schema:
  `axial_distance_norm`, `same_axis`, `same_line`, `same_window`,
  `terminal_exact_win`, `terminal_equivalent_win`, `terminal_exact_cover`,
  `covers_all_opponent_win_requirements`, `impossible_to_cover`,
  `phase_full_turn`, `phase_known_first`, `phase_both_legal`.
- Bumped V1 replay schema to `2` and compact metadata version to `3`.
- Required schema-2 `terminal_tactical_payload` in dict metadata loads.
- Added Rust/PyO3 8-field `PairRowV1` tuple parsing for tactical payload rows.

### Runtime Admission And Scoring

- V1 self-play now runs a proposal graph pass before bounded pair scoring.
- Selector admission consumes learned `cell_marginal_logits`,
  `legal_proposal_embeddings`, `legal_completion_query`, and
  `legal_completion_key`.
- `global_pair_biaffine_0` emits runtime-only legal projection tensors shaped
  by LEGAL rows. They reuse the same learned V1 pair projections that train
  `pair_proposal_score` and `pair_completion_logits`; the loss planner ignores
  these runtime-only tensors as non-head outputs.
- If proposal inference returns legal-by-legal `pair_completion_logits` or `pair_proposal_score`, those matrices feed anchor-completion admission.
- Removed live `_v1_legal_embedding_features` heuristic use from `sampled_joint_pair_v1`.
- Final scoring uses canonical pair features from `v1_pair_contract.py`.
- `sampled_joint_pair_v1` graph value decoding now raises when `value` is missing or empty.

### Pair-Native Search

- Added Rust `run_search_step(max_expansions)`,
  `complete_expansion(node_key, value, pair_qr, pair_logits, correction_weights,
  correction_modes)`, and `select_root_action()` to `V1PairSearchEngine`.
- Added recursive V1 search nodes keyed by stable node keys. Nodes store game
  state, legal row identity, tactical payload, visit/value stats, terminal
  value, and cached pair reservoir state.
- `run_search_step` now consumes the configured simulation budget by descending
  from the root through expanded interior nodes with PUCT and progressive
  widening until it reaches a terminal or unexpanded leaf.
- `complete_expansion` now attaches exactly one reservoir to the requested node,
  widens it from cache, marks the node expanded, and backs up the supplied
  current-player value through the pending simulation path.
- `select_root_action` rejects normal full-turn roots until pending expansions
  are complete and the recursive simulation budget has been satisfied.
- Python V1 self-play now loops on Rust expansion requests for normal
  two-placement roots and uses the same proposal/scoring path for interior
  full-turn nodes.
- Removed the dormant Rust shallow nonterminal `evaluate_pair(...)=0.0`
  simulation path; normal pair roots now require expansion completion before
  selection.
- Interior reservoir build, widening, expanded-node counts, scoring-pass counts,
  and zero refill events are surfaced through replay telemetry.

### Tactical Fidelity And Support

- Replay stores `terminal_tactical_payload`.
- `terminal_tactical_v1` targets are built from the tactical payload.
- Selector support flags now include `terminal_cover`, `covers_all_opponent_win_requirements`, and `impossible_to_cover`.
- Hot-cover candidates from an impossible-cover tactical payload carry impossible-cover support.
- Tactical-protected candidates are not evicted by ordinary candidate budget.
- Final graph context capacity expands to admitted protected-pair cell requirements.

### Training Cleanup

- Removed unused `v1_pair_proposal_score_target` arrays and helper scoring.
- `pair_proposal_score` remains intentionally wired to `v1_pair_ranking_target`.
- Unsampled legal pairs remain excluded from policy, ranking, Q, and negative masks.

### Legacy Quarantine

- Moved `pair_logits_to_action_logits` out of `pair_strategy.py` into
  `hexorl.search.legacy_pair_projection`.
- `build_pair_strategy(...)` rejects `root_pair_mcts` and `full_pair_mcts` by
  default; explicit offline baseline code must call
  `build_legacy_pair_baseline_strategy(...)`.
- Normal autotune/config surfaces now expose `none` and `sampled_joint_pair_v1`
  only.

## Local Verification Run

Run from repository root with:

```powershell
$env:PYTHONPATH='Python/src'
```

Focused V1 suite:

```powershell
python -m pytest `
  Python\tests\test_v1_pair_contract.py `
  Python\tests\test_v1_pair_candidate_selector.py `
  Python\tests\test_v1_pair_targets.py `
  Python\tests\test_v1_pair_training_losses.py `
  Python\tests\test_v1_pair_biaffine_model.py `
  Python\tests\test_v1_pair_ci_audit_gates.py `
  Python\tests\test_v1_selfplay_worker_runtime.py::test_v1_proposal_matrix_reader_accepts_legal_by_legal_outputs `
  Python\tests\test_v1_selfplay_worker_runtime.py::test_sampled_joint_pair_v1_worker_uses_pair_native_runtime `
  -q
```

Observed result in this environment:

```text
34 passed
```

Replay and training metadata checks:

```powershell
python -m pytest `
  Python\tests\test_training_data_pipeline.py::test_v1_pair_search_metadata_roundtrips_through_compact_record_and_ring `
  Python\tests\test_training_data_pipeline.py::test_v1_pair_search_metadata_compact_blob_rejects_legacy_json `
  Python\tests\test_training_data_pipeline.py::test_v1_pair_search_metadata_schema_two_requires_tactical_payload `
  Python\tests\test_training_data_pipeline.py::test_replay_memory_estimate_accounts_for_compressed_v1_metadata `
  Python\tests\test_training_data_pipeline.py::test_v1_support_type_is_explicit_and_validated `
  Python\tests\test_training_data_pipeline.py::test_v1_unsampled_legal_pairs_are_not_implicit_negatives `
  Python\tests\test_training_data_pipeline.py::test_v1_metadata_rejects_legacy_pair_policy_target_mixing `
  Python\tests\test_training_data_pipeline.py::test_process_game_record_keeps_v1_metadata_out_of_legacy_pair_completeness `
  Python\tests\test_training_data_pipeline.py::test_prepare_dense_training_batch_masks_legacy_pair_weight_for_v1_schema_marker `
  -q
```

Observed result in this environment:

```text
9 passed
```

Compile checks:

```powershell
python -m py_compile `
  Python\src\hexorl\v1_pair_contract.py `
  Python\src\hexorl\graph\batch.py `
  Python\src\hexorl\selfplay\records.py `
  Python\src\hexorl\selfplay\worker.py `
  Python\src\hexorl\train\v1_pair_targets.py `
  Python\src\hexorl\train\loss_plan.py `
  Python\src\hexorl\buffer\sampler.py `
  Python\src\hexorl\search\pair_candidate_selector_v1.py `
  Python\src\hexorl\search\pair_strategy.py `
  Python\src\hexorl\search\legacy_pair_projection.py `
  Python\src\hexorl\inference\protocol.py `
  Python\src\hexorl\inference\adapters.py `
  Python\src\hexorl\inference\server.py `
  Python\src\hexorl\inference\shm_queue.py `
  Python\src\hexorl\models\families\global_graph.py `
  Python\src\hexorl\config\schema.py `
  Python\src\hexorl\autotune\recipes.py `
  Python\src\hexorl\tuning\optuna_tuning.py `
  Python\src\hexorl\dashboard\app.py `
  Python\tests\test_v1_selfplay_worker_runtime.py `
  Python\tests\test_v1_pair_search_ffi.py
```

Observed result in this environment:

```text
exit 0
```

Source audits:

```powershell
rg -n "v1_pair_proposal_score_target|pair_proposal_score_target|_proposal_score\(|source_score\(|_v1_pair_features_from_candidates" Python\src Python\tests
rg -n "_v1_legal_embedding_features" Python\src\hexorl\selfplay\worker.py
rg -n "simulate_candidate\(|evaluate_pair\(" crates\hexgame-core\src
```

Observed result in this environment:

```text
no matches
```

Flagship V1 projection audit:

```powershell
$env:PYTHONPATH='Python/src'
@'
from pathlib import Path
source = Path('Python/src/hexorl/selfplay/worker.py').read_text(encoding='utf-8')
start = source.index('    def _v1_build_and_score_root')
end = source.index('    def _play_one_game(', start)
segment = source[start:end]
for token in ['pair_logits_to_action_logits','apply_root_pair_priors','apply_root_pair_first_priors','apply_root_pair_second_priors','apply_root_pair_rows']:
    assert token not in segment, token
print('flagship V1 segment legacy projection audit passed')
'@ | python -
```

Observed result:

```text
flagship V1 segment legacy projection audit passed
```

## Rust-Capable Verification

This environment did not have `cargo`, `rustc`, or a built `_engine` extension. A Rust-capable environment should run:

```powershell
cargo test -p hexgame-core v1_pair_search
cargo test -p hexgame-core terminal_tactical_payload_reports_v1_shapes_and_statuses
$env:PYTHONPATH='Python/src'
python -m pytest Python\tests\test_v1_pair_search_ffi.py -q
python -m pytest Python\tests\test_engine_smoke.py -q
```

The intent of those checks is to verify:

- Rust `TerminalTacticalSetV1` payload shape and status coverage.
- PyO3 roundtrip of Rust `PairRowV1` rows into Python tactical replay payloads.
- V1 root admission/search/apply identity checks against Rust legal row tables.
- Expansion request/complete/select lifecycle exposed through PyO3.
- Nonterminal neural value backup changing root choice.
- One reservoir per expanded full-turn node.
- Widening reveal from cache without rescoring.
- Recursive descent past depth 1 after an expanded child reveals pair actions.
- Normal root selection rejection before pending expansions and simulation
  budget completion.
- Stale child legal/pair row identity rejection.

## Known Verification Boundary

The current source implements recursive Rust V1 pair search, the PyO3 expansion
protocol, and Python V1 runtime wiring, but this local environment could not
execute Rust-backed tests because `cargo`, `rustc`, and `_engine` were
unavailable. The full acceptance bar for final cutover still requires
Rust-backed execution evidence, search trace artifacts, performance profiles,
reservoir refill serialization with a non-default refill configuration, and
equal-wall-clock arena evaluation.
