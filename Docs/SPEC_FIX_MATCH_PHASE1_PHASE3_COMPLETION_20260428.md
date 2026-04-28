# Spec-Fix Match Plan: Phase 1-3 Completion

Date: 2026-04-28

This document is the companion plan to:

```text
Docs/SPEC_FIX_MATCH_GLOBAL_GRAPH_MODEL_20260428.md
```

That graph-model document covers the true `global_graph_option1` architecture.
This document covers everything else required to make the current workspace
fully match the intent of the Phase 1, Phase 2, Phase 3, and model-head specs:

```text
Docs/PHASE1_RESTNET_ACTION_CONTRACT_SCOUT_20260427.md
Docs/PHASE2_GRAPH_AND_OVERNIGHT_AUTOTUNE_20260428.md
Docs/AUTOTUNING_METHODS_AND_48H_PLAN_20260427.md
Docs/OVERNIGHT_PHASE3_AUTOTUNE_MONITOR_20260428.md
Docs/IMPLEMENTATION_SPEC_AUDIT_20260428.md
Docs/MODEL_HEAD_TARGET_AND_D6_FIXES_20260428.md
```

The goal is not legacy parity. The goal is a complete, cohesive production
training system where every enabled feature is real, testable, observable, and
safe enough to tune.

## Handoff Instructions For Implementation Agents

Treat this document as the root implementation brief. An agent implementing the
finished Phase 1-3 plan should read the linked documents in the order below,
then implement the acceptance gates in this document and in the global graph
spec. Do not treat any partial/scout implementation as complete because it
trains or runs.

### Required Reading Order

1. [Docs/game.md](game.md)
   - Source of truth for Hexo rules: one-stone opening, two-placement turns,
     infinite hex board, placement radius, six-in-a-row win condition, D6
     symmetry, 4/5-window threat equivalence, pair-turn defense, and axis
     forking.
   - Any model, target, search, graph, D6, replay, or dashboard behavior that
     contradicts this file is wrong.

2. [Docs/SPEC_FIX_MATCH_PHASE1_PHASE3_COMPLETION_20260428.md](SPEC_FIX_MATCH_PHASE1_PHASE3_COMPLETION_20260428.md)
   - This file. Use it as the root task list, milestone order, and acceptance
     definition for everything outside the true global graph model.

3. [Docs/SPEC_FIX_MATCH_GLOBAL_GRAPH_MODEL_20260428.md](SPEC_FIX_MATCH_GLOBAL_GRAPH_MODEL_20260428.md)
   - Required companion spec for the finished `global_graph_option1` /
     `global_graph768_champion` architecture.
   - Implement it alongside this document. The Phase 1-3 plan is not finished
     until this graph contract is also satisfied.
   - It now defines the graph alternatives to compare:
     `global_xattn_0`, `global_line_window_0`, `global_pair_twostage_0`,
     `global_graph_full_0`, optional `global_hybrid_action_0`, and final
     `global_graph768_champion`.

4. [Docs/PHASE1_RESTNET_ACTION_CONTRACT_SCOUT_20260427.md](PHASE1_RESTNET_ACTION_CONTRACT_SCOUT_20260427.md)
   - Original Phase 1 intent: strongest crop baseline, ResTNet crop scout, and
     candidate/action-keyed policy scout.
   - Important caveat: current ResTNet is only a crop scout until hex-masked
     convolutions, relative bias, D6-aware coordinates, and real attention
     tests are implemented.

5. [Docs/TRANSFORMER_ARCHITECTURE_ABLATIONS_FOR_HEXO_20260427.md](TRANSFORMER_ARCHITECTURE_ABLATIONS_FOR_HEXO_20260427.md)
   - Original Phase 2 architecture intent.
   - Use the global graph spec above as the corrected no-compromise version
     when there is any conflict.

6. [Docs/AUTOTUNING_METHODS_AND_48H_PLAN_20260427.md](AUTOTUNING_METHODS_AND_48H_PLAN_20260427.md)
   - Original Phase 3 tuning and champion-selection plan.
   - Important caveat: current code is ASHA-style + PBT fallback. True BOHB
     and true PB2 remain required for spec completion.

7. [Docs/RGSC_IMPLEMENTATION.md](RGSC_IMPLEMENTATION.md)
   - Current regret/RGSC status and the exact requirements to move from regret
     auxiliary heads plus regret-biased replay to full RGSC search control.

8. [Docs/MODEL_HEAD_TARGET_AND_D6_FIXES_20260428.md](MODEL_HEAD_TARGET_AND_D6_FIXES_20260428.md)
   - Documents the fixed target/D6 issues from the review pass.
   - Do not regress these fixes while implementing the broader plan.

9. [Docs/IMPLEMENTATION_SPEC_AUDIT_20260428.md](IMPLEMENTATION_SPEC_AUDIT_20260428.md)
   - Audit context for earlier implementation gaps. Use it as a checklist for
     avoiding repeated half-implementations.

10. [Docs/PHASE2_GRAPH_AND_OVERNIGHT_AUTOTUNE_20260428.md](PHASE2_GRAPH_AND_OVERNIGHT_AUTOTUNE_20260428.md)
    and [Docs/OVERNIGHT_PHASE3_AUTOTUNE_MONITOR_20260428.md](OVERNIGHT_PHASE3_AUTOTUNE_MONITOR_20260428.md)
    - Historical run notes and practical failure modes.
    - Use these for host/runtime lessons only. They did not validate BOHB or
      PB2, and they did not finish the true global graph model.

11. [Docs/ABLATION_RESULTS_20260427.md](ABLATION_RESULTS_20260427.md),
    [Docs/ABLATION_RESULTS_POST_OPT_20260427.md](ABLATION_RESULTS_POST_OPT_20260427.md),
    [Docs/PERFORMANCE_PROFILE_20260427.md](PERFORMANCE_PROFILE_20260427.md), and
    [Docs/OPTIMIZATION_OPTIONS_20260427.md](OPTIMIZATION_OPTIONS_20260427.md)
    - Supporting performance and ablation context.
    - Use them to understand why the plan prioritizes throughput telemetry,
      runtime sweeps, graph/candidate action contracts, and scorecard gates.

12. [Docs/sources.md](sources.md)
    - Paper/source map for ResTNet, RGSC/regret, PB2, ASHA, BOHB, Mish, and
      RepVGG.
    - Use this when checking whether an implementation really matches the
      source method or is only inspired by it.

13. [Docs/TUNING_ORCHESTRATOR_GUIDE_20260428.md](TUNING_ORCHESTRATOR_GUIDE_20260428.md)
    - Operational guide for the 30-minute autonomous tuning orchestrator after
      the audit fixes and spec-match implementation are in place.
    - Use it when supervising ASHA/BOHB/PB2, interpreting crashes, deciding
      whether to patch/quarantine/restart, and judging whether model strength
      is real or a bug artifact.

### Implementation Rules

- Keep the project cohesive. Do not add legacy compatibility branches unless a
  current production path requires them.
- Prefer deleting or renaming misleading partial paths over letting them appear
  complete.
- Every enabled config feature must have:
  - target generation;
  - model outputs;
  - loss wiring;
  - metrics;
  - dashboard/debug visibility when useful;
  - tests proving it is active.
- Every scout/fallback must be named as a scout/fallback in configs, reports,
  and dashboard labels.
- A run finishing without crashing is not an acceptance gate. The tests and
  scorecard gates below are the acceptance gates.
- A model improving in Elo is not sufficient if bug sentinels, invariant
  suites, D6 checks, replay checks, or independent tactical oracles disagree.

## Executive Contract

The workspace is considered fully up to the Phase 1-3 spec only when:

1. Every model target is generated on correct Hexo turn boundaries and correct
   player perspective.
2. D6 augmentation is correct for dense, sparse, pair, and `graph_hybrid_0`
   paths, with exhaustive action-identity tests.
3. The sparse/action-keyed policy path cannot leak training labels, silently
   cap decisive actions, or hide fallback-prior use.
4. Sparse-prior MCTS stages are measurable and cannot silently become no-ops.
5. Pair policy is fully observable and fully consumed by MCTS for two-placement
   turns. Shadow logging may exist as a diagnostic, but it is not the finished
   contract.
6. Phase 3 scorecards use real tactical, outside-window, league, and candidate
   safety metrics.
7. Dashboard/debug views expose the same data contracts used by training and
   MCTS.
8. Config validation rejects stale, ignored, or misleading settings.
9. `graph_hybrid_0` remains honestly named as a crop-compatible scout, while
   true graph work is completed through the separate global graph spec.

Implementation order in this document is only sequencing. It is not permission
to stop at an intermediate model, diagnostic-only pair policy, PBT-only
scheduler, crop-only tactical oracle, or dashboard-only visibility. A finished
P1/P2/P3 workspace must implement every acceptance gate in this document plus
the true global graph contract in
`Docs/SPEC_FIX_MATCH_GLOBAL_GRAPH_MODEL_20260428.md`.

## Current Snapshot

The 2026-04-28 implementation audit already fixed several concrete bugs:

- candidate features no longer include target probability or target-present
  labels;
- sparse candidate copied width and missing-mass accounting now expose target
  truncation;
- configured lookahead heads automatically receive default loss weights when
  missing;
- `sparse_prior_stage > 0` now requires `sparse_policy = true`;
- Phase 3 candidate recall scorecard typos were corrected;
- root and leaf sparse-prior candidates now receive crop-visible tactical
  overrides;
- `graph_hybrid_0` is recognized in dashboard summaries;
- dashboard inference can inspect sparse-policy logits;
- compact self-play records preserve `value_weight`;
- stale resignation keys were removed from active configs;
- stale regret helper now uses `selected_action_value` with fallback to
  `root_value`.

Those fixes are necessary, but they are not the whole completion story. The
sections below define the remaining implementation work and the hard gates that
should prove it.

## Source-Method Adaptation Audit

This audit checks the named methods against what is actually implemented today.
The rule for this section is simple: an implementation may be useful as a scout
or diagnostic, but it is not complete unless it implements the full method
contract and passes the listed acceptance gates.

| Method | Current state | Completion verdict |
|---|---|---|
| ResTNet | Attention blocks inside the 33x33 crop exist. Relative-bias control, hex-masked local trunk, D6-aware coordinate encoding, and paper-faithful gated FFN are not complete. | Partial crop scout |
| Regret / RGSC | Regret heads, selected-action-value recording, and regret-biased replay sampling exist. Full PRB restart search-control loop is not wired. | Partial regret auxiliary system |
| ASHA | Rung-style top-fraction pruning exists. Replayable ASHA metadata and strict same-resource promotion semantics need hardening. | Partial ASHA-style scheduler |
| BOHB | No Hyperband bracket plus KDE/TPE density-model sampler exists. | Not implemented |
| PB2 | Current code is PBT-style exploit/explore with random mutations. No continuous response model or uncertainty-aware acquisition exists. | Not implemented |

No-compromise requirement:

```text
Documentation and code must use honest names.
If a method is partial, configs/reports must call it a scout, proxy, or
baseline, not a completed implementation of the source method.
```

Docs updated by this audit:

```text
Docs/PHASE1_RESTNET_ACTION_CONTRACT_SCOUT_20260427.md
Docs/RGSC_IMPLEMENTATION.md
Docs/AUTOTUNING_METHODS_AND_48H_PLAN_20260427.md
```

## Milestone 0: Hex-Masked Convolution Contract

Status: not implemented in the active Python model path as of this audit.

Problem:

`Docs/game.md` explains that a square `3x3` convolution includes two axial
corner offsets, `(-1,-1)` and `(1,1)`, that are hex-distance 2 and are not real
hex neighbors. The current `Python/src/hexorl/model/network.py` path uses
ordinary `torch.nn.Conv2d` in `GatedResBlock` and `conv_in`; there is no
`HexConv2d` module and no optimizer post-step hook that zeros invalid kernel
entries.

Required behavior:

```text
CNN crop baseline:
  local 3x3 trunk convolutions are hex-masked

ResTNet crop scout:
  local convolutional trunk is hex-masked before/around attention blocks

graph_hybrid_0:
  any retained crop-local CNN trunk is hex-masked

global_graph_option1:
  no dense crop conv is required, but any optional conv/distillation side path
  must obey the same mask if it uses 3x3 axial-grid convolutions
```

Implementation requirements:

- Add a `HexConv2d` wrapper or a reusable masking hook for `3x3` axial-grid
  convolutions.
- Invalid kernel entries for offsets `(-1,-1)` and `(1,1)` must be zero after
  initialization and after every optimizer step.
- Decide explicitly whether `1x1` heads are exempt; they normally are, because
  they do not mix neighboring cells.
- Configs/reports must state whether a model is `hex_masked_conv = true`.

Acceptance tests:

```text
test_hexconv2d_invalid_corner_weights_start_zero
test_hexconv2d_invalid_corner_weights_remain_zero_after_optimizer_step
test_cnn_trunk_uses_hexconv2d_for_3x3_layers
test_restnet_trunk_uses_hexconv2d_for_3x3_layers
test_graph_hybrid_0_local_trunk_uses_hexconv2d
```

## Milestone A: Target And Head Correctness

### A1. Lookahead Turn Boundaries

Status: fixed in the active path, needs stronger property coverage.

Problem:

Hexo has a one-stone opening and then two-placement turns. Any target generator
that assumes alternating single placements trains future heads against the
wrong state.

Current required behavior:

```text
turn 0: one opening placement
turn 1+: two placements by the same player
lookahead horizon N: N future turn starts, not N future placements
```

Implementation requirements:

- Keep `_turn_boundary_indices()` based on player-run starts, not parity.
- Add a helper with an explicit name such as
  `hexo_turn_start_indices(records)` so every future target uses the same turn
  contract.
- Add property tests that build random legal histories and verify:
  - opening turn has length one;
  - later turns have length one or two depending on truncation/game end;
  - every lookahead target points at a future turn start;
  - mid-turn positions skip to the next opponent turn, not the same player's
    second placement.

Files:

```text
Python/src/hexorl/buffer/targets.py
Python/tests/test_training_data_pipeline.py
```

Acceptance tests:

```text
test_hexo_turn_boundaries_follow_player_runs
test_mid_turn_lookahead_targets_next_turn_start
test_random_histories_have_stable_hexo_turn_starts
```

### A2. Lookahead Perspective Safety

Status: fixed in the active path, needs a shared perspective helper.

Problem:

`root_value` is from the current player's perspective at that future position.
If a future position belongs to the opponent, the value must be negated before
it becomes a target for the source position.

Required implementation:

- Centralize perspective conversion:

```text
value_from_source_perspective(source_player, future_player, future_value)
```

- Use this helper for lookahead, regret diagnostics, dashboard replay labels,
  and any future value-lookahead variants.
- Add tests with hand-authored alternating player records where the expected
  sign is obvious.

Acceptance tests:

```text
test_lookahead_flips_future_player_perspective
test_lookahead_keeps_same_player_perspective
test_ema_lookahead_uses_source_perspective_for_every_future_term
```

### A3. Opponent Policy Target

Status: fixed in the active path, needs full-PCR edge tests and dashboard
visibility.

Problem:

The opponent-policy head must learn the opponent's next full-search turn
policy. Copying the next placement is wrong because Hexo frequently has two
placements by the same player.

Required behavior:

```text
source record i
find next record j where:
  player[j] != player[i]
  record[j] is the start of that opponent turn
  record[j].policy target is full PCR / full search
use policy[j] as opp_policy[i]
set opp_policy_weight[i] = policy_weight[j]
```

The source position's own PCR quality must not block training. If source turn
`x` is low-PCR and opponent turn `x+1` is full-PCR, the source record can still
train the opponent-policy head.

Implementation requirements:

- Preserve the current next-full-search opponent-turn lookup.
- Add tests for:
  - source low-PCR, next opponent full-PCR trains;
  - source full-PCR, next opponent low-PCR does not train;
  - same player's second placement is skipped;
  - end-of-game without a later opponent turn produces zero weight.
- Add metrics:

```text
opp_policy_valid_frac
opp_policy_mean_weight
opp_policy_target_turn_gap
loss_opp_policy_weighted
```

Files:

```text
Python/src/hexorl/buffer/targets.py
Python/src/hexorl/buffer/sampler.py
Python/src/hexorl/train/losses.py
Python/src/hexorl/dashboard/app.py
```

Acceptance tests:

```text
test_opponent_policy_uses_next_full_search_opponent_turn_start
test_opponent_policy_source_low_pcr_can_train_from_later_full_pcr
test_opponent_policy_ignores_low_pcr_opponent_turn
test_opp_policy_loss_uses_opp_policy_weight
```

### A4. Regret Rank And Regret Value

Status: must be completed against the paper-level target contract and verified
against the full training loop.

Problem:

The original review found the regret targets were a suffix average of
root-value error and the rank loss batch-minmax normalized. That is not
paper-exact. The active path now uses selected-action value and raw regret
scale, but the complete RGSC/PRB training loop has not been proven.

Required target contract:

```text
selected_action_value: value of the action actually played at the search root
perspective_outcome_i: final result from position i's current-player perspective
per_step_error_i: (selected_action_value_i - perspective_outcome_i)^2
regret_value_target_t: mean_{i=t..T}(per_step_error_i)
regret_rank_target_t: the same raw nonnegative trajectory regret scale, used
                      as the RGSC ranking objective bias without batch-local
                      minmax normalization
```

Implementation requirements:

- Confirm Rust MCTS always emits the selected child's value, not just root
  value.
- Store `selected_action_value` in every `PositionRecord` and compact record.
- Make missing selected-action value invalid for new production records. Legacy
  imports may read old records, but the active training path must not silently
  fall back to `root_value`.
- Keep the suffix-average squared-error target from the RGSC paper. Do not
  replace it with an unsquared residual such as
  `final_outcome - selected_action_value`.
- Add `regret_weight` and set it to zero for truncated/non-terminal records.
  Hexo has no draw, so max-move cutoffs contain useful policy/structure data
  but no valid final-outcome regret label.
- Add full RGSC candidate-state handling: played trajectory states are not
  enough. MCTS tree-node states that meet the visit/quality threshold must also
  be scored by the regret network and be eligible for PRB insertion.
- Add a target-debug endpoint in the dashboard to inspect:

```text
root_value
selected_action_value
final_outcome
per_step_error
regret_value_target
regret_rank_target
regret_weight
```

Acceptance tests:

```text
test_regret_value_uses_selected_action_value
test_regret_target_missing_selected_action_has_zero_weight
test_regret_targets_zero_weight_for_truncated_games
test_regret_suffix_average_matches_paper_equation_2
test_regret_rank_no_batch_minmax_normalization
test_compact_record_preserves_selected_action_value
test_rgsc_tree_node_states_can_enter_prb
```

Completion gate:

The docs and code must agree on whether the regret heads are:

- paper-aligned active training heads; or
- experimental auxiliary heads.

They cannot be described as exact while using heuristic targets.

Production config rule:

```text
if buffer.regret_fraction > 0:
  either regret heads are enabled and weighted
  or the config must explicitly name the behavior as replay-priority-only
```

Silent regret replay with no trained regret heads is not a completed RGSC
configuration.

### A5. Draws, Truncation, And Value Weighting

Status: mostly fixed, needs dashboard and trainer visibility.

Problem:

Draws/truncated games should still enter the buffer because they contain useful
policy and structure data. They should not train the terminal value head as if
they had a true result.

Required behavior:

```text
terminal decisive game:
  value_weight = 1
  policy_weight = PCR/search quality

draw or max-move truncation:
  value_weight = 0
  policy_weight = PCR/search quality
```

Implementation requirements:

- Keep compact `value_weight` serialization.
- Add dashboard columns for `result`, `terminal_reason`, `value_weight`, and
  `policy_weight`.
- Add trainer metrics:

```text
value_weight_mean
value_weight_zero_frac
policy_weight_mean
truncation_rate
draw_rate
```

Acceptance tests:

```text
test_truncated_games_keep_policy_targets_but_zero_value_weight
test_draw_games_keep_policy_targets_but_zero_value_weight
test_value_loss_ignores_zero_weight_records
```

## Milestone B: D6 Augmentation Must Be Exhaustive

Status: fixed for the current crop/sparse/`graph_hybrid_0` route, but the
finished spec requires exhaustive coverage for every active and spec-matching
action contract.

Principle:

D6 augmentation should transform the compact move history and global target
coordinates, then re-encode. This is safer than trying to rotate tensors after
the fact because legal moves, crop origin, sparse candidates, and pair
identities all stay tied to one transformed board.

Current model families that must pass:

```text
standard CNN dense policy
ResTNet crop model
candidate_policy_33
graph_hybrid_0
sparse_policy head
pair_policy head
axis head
opponent policy head
lookahead heads
regret heads
```

Required implementation:

- Add one shared D6 test harness that:
  - generates random legal histories;
  - applies all six symmetries;
  - replays transformed histories through Rust;
  - verifies legal masks transform bijectively;
  - verifies dense targets, sparse targets, opponent targets, and pair targets
    transform to the same global coordinates.
- Add pair-policy canonicalization tests:
  - ordered pair targets remain ordered if the model contract is ordered;
  - unordered pair targets use a single canonical representation if the model
    contract is unordered;
  - both forms reject duplicate coordinates unless explicitly allowed by the
    game rule.
- Add candidate-row tests:
  - candidate features do not contain target labels;
  - source bits, critical bits, win/block bits, and global coordinates transform
    correctly;
  - D6 is not disabled when sparse policy is enabled.
- Add axis-map tests:
  - perspective-indexed planes rotate/reflection-transform correctly;
  - own/opponent planes remain own/opponent, not color-specific red/blue, in
    training tensors.

Files:

```text
Python/src/hexorl/buffer/sampler.py
Python/src/hexorl/action_contract/candidates.py
Python/src/hexorl/axis_policy/
Python/tests/test_training_data_pipeline.py
Python/tests/test_axis_policy.py
```

Acceptance tests:

```text
test_d6_dense_policy_target_bijection
test_d6_sparse_policy_target_bijection
test_d6_pair_policy_target_bijection
test_d6_candidate_features_transform_without_label_leakage
test_d6_axis_planes_transform_perspective_safely
test_sparse_policy_does_not_disable_d6
```

Future graph-native D6 for `WINDOW6`, `COVER_SET`, relation bias, and true
global graph tokens belongs to `SPEC_FIX_MATCH_GLOBAL_GRAPH_MODEL_20260428.md`.

## Milestone C: Sparse Candidate Contract

### C1. Candidate Feature Schema

Status: label leakage fixed, schema needs to be frozen and documented.

Required schema:

Candidate features must only use data available at live inference time. They
may include:

```text
q, r, s
distance summaries
inside-crop flag
crop-local coordinates
legal rank
source bits
winning-cell bit
forced-block bit
outside-crop bit
critical-cell bit
```

They must not include:

```text
target probability
target-present bit
future outcome
root visit count not available for a fresh leaf
anything derived from policy_target_v2 except through target-only loss tensors
```

Implementation requirements:

- Add a versioned candidate feature spec constant:

```text
CANDIDATE_FEATURE_VERSION = 2
CANDIDATE_FEATURE_NAMES = (...)
```

- Store the feature version in checkpoints and run metadata.
- Dashboard should display feature names when inspecting sparse logits.

Acceptance tests:

```text
test_candidate_feature_names_match_tensor_width
test_candidate_features_do_not_include_policy_target_labels
test_dashboard_reports_candidate_feature_version
```

### C2. Critical Actions Must Not Be Silently Dropped

Status: missing mass accounting is fixed, but completion requires an explicit
critical-action overflow policy.

Problem:

The Phase 1 spec says critical actions override the normal budget. If the
implementation stores fixed-width candidate tensors, it cannot silently truncate
protected wins, forced blocks, or cover cells.

Required behavior:

```text
candidate_budget = normal budget for heuristic candidates
candidate_width = storage/inference width
critical actions are inserted before heuristic actions
if critical actions exceed candidate_width:
  hard diagnostic failure, not silent truncation
```

Implementation requirements:

- Add `critical_count`, `critical_overflow_count`, and
  `critical_overflow_examples` metrics.
- For training, if critical overflow occurs:
  - set sparse/pair weights to zero for that record;
  - keep dense policy/value training;
  - emit a high-priority event with the move history.
- For Phase 3 gates, any critical overflow is a failing condition.

Acceptance tests:

```text
test_critical_actions_are_inserted_before_heuristic_candidates
test_critical_overflow_zeroes_sparse_weight_and_logs_event
test_missing_mass_reflects_truncated_represented_target_mass
```

### C3. Candidate Recall Must Measure Discovery And Protection Separately

Status: required for completion.

Problem:

If diagnostics build the candidate set with target actions protected, recall can
be self-fulfilling. Training inclusion and live MCTS discovery are different
questions.

Required metrics:

```text
candidate_recall_top1_protected
candidate_recall_top4_protected
candidate_recall_top8_protected
candidate_recall_winning_move_protected
candidate_recall_forced_block_protected
candidate_recall_two_placement_cover_protected

candidate_discovery_top1
candidate_discovery_top4
candidate_discovery_top8
candidate_discovery_winning_move
candidate_discovery_forced_block
candidate_discovery_two_placement_cover
candidate_discovery_open_four
candidate_discovery_open_five
```

Protected recall answers:

```text
Did training tensors preserve the target after mandatory target inclusion?
```

Discovery recall answers:

```text
Would the live inference candidate builder have found this action without the
training target forcing it in?
```

Implementation requirements:

- Candidate recall evaluator must build both candidate sets:
  - protected mode with policy target inclusion;
  - discovery mode with no target inclusion.
- Phase 3 scorecards must use discovery metrics for sparse-prior safety gates.
- Training data QA should use protected metrics to catch tensor truncation.

Acceptance tests:

```text
test_candidate_recall_reports_protected_and_discovery_modes
test_discovery_recall_does_not_include_target_only_actions
test_phase3_candidate_gate_uses_discovery_metrics
```

### C4. Full-Board Tactical Oracle

Status: unresolved for outside-window cases.

Problem:

Current sparse-prior tactical overrides can use crop-visible hot planes. That
does not prove the candidate builder can find decisive actions outside the
current 33x33 crop.

Required implementation:

- Add a Rust or Python engine-backed tactical oracle that scans the full legal
  set, not just encoded crop planes.
- It must identify:

```text
win-now cells
forced-block cells
open-four cells
open-five cells
two-placement cover sets
outside-window tactical cells
```

- The candidate builder should accept oracle outputs as explicit protected or
  source-tagged cells.
- MCTS sparse-prior root and leaf candidate builders should use this oracle, not
  only crop hot planes.

Files:

```text
Rust engine / PyO3 bindings for tactical scan
Python/src/hexorl/action_contract/candidates.py
Python/src/hexorl/selfplay/worker.py
Python/tests/test_tactical_oracle.py
```

Acceptance tests:

```text
test_full_board_oracle_finds_win_outside_crop
test_full_board_oracle_finds_forced_block_outside_crop
test_candidate_builder_includes_oracle_critical_cells_outside_crop
test_sparse_prior_leaf_candidates_include_outside_crop_tactical_cells
```

## Milestone D: Sparse-Prior MCTS Telemetry

Status: stage validation fixed, runtime fallback telemetry missing.

Required stage semantics:

```text
stage 0:
  train sparse policy only
  MCTS consumes dense policy/default priors

stage 1:
  root sparse-prior mix
  leaf expansion still dense/default

stage 2:
  root and leaf sparse-prior path
  sparse action prior is preferred when candidate exists
  dense crop prior is fallback for crop-visible actions
  default prior is final fallback
```

Required telemetry:

```text
sparse_prior_stage
sparse_prior_root_candidate_count
sparse_prior_leaf_candidate_count
sparse_prior_root_hit_frac
sparse_prior_leaf_hit_frac
fallback_prior_use
fallback_prior_use_on_mcts_top1
fallback_prior_use_on_mcts_top4
fallback_prior_use_on_mcts_top8
sparse_vs_dense_disagreement
sparse_prior_forward_ms
sparse_prior_candidate_build_ms
```

Implementation requirements:

- Rust/Python MCTS prior gather must count whether each selected/root top-k
  action used:
  - sparse prior;
  - dense prior;
  - default fallback.
- PyO3 must expose these counters in search summaries.
- Self-play worker must aggregate them into position records and run metrics.
- Dashboard must chart fallback use by epoch and inspect it per game.
- Phase 3 scorecard must penalize fallback use on MCTS top-k.

Acceptance tests:

```text
test_sparse_prior_stage_1_only_uses_sparse_at_root
test_sparse_prior_stage_2_uses_sparse_at_root_and_leaf
test_mcts_reports_sparse_dense_default_prior_sources
test_phase3_scorecard_penalizes_fallback_prior_use_on_topk
```

## Milestone E: Pair Policy Contract

Status: current implementation is auxiliary only; the completed P1/P2/P3
system requires active MCTS integration.

Current reality:

`PairPolicyHead` scores selected candidate-row pairs and trains an auxiliary
loss. It is not consumed by MCTS and it is not the full
`policy_pair_first` / `policy_pair_second` / `policy_pair_joint` design.

No-half-measure decision:

Pair policy must become a real search contract, not just a dashboard or loss
diagnostic. Shadow logging is useful for validation, but a finished
implementation must make MCTS consume the pair policy during two-placement
turns.

Required target and debug implementation:

- Pair target generation uses full `policy_v2`, not a truncated local subset.
- Pair tensors support D6 augmentation with coordinate-identity tests.
- Pair loss is weighted by target quality.
- Dashboard can inspect:

```text
top pair logits
pair target mass
pair missing mass
first/second coordinate
pair_policy_top1_acc
pair_policy_topk_recall
```

- MCTS shadow logging computes what the pair policy would have preferred at
  root and is kept as a diagnostic after active integration.

Required active implementation:

- Implement the full three-head contract:
  - `policy_pair_first`: prior over legal first placements;
  - `policy_pair_second`: conditional prior over second placements given the
    selected first placement;
  - `policy_pair_joint`: prior over legal ordered two-placement macro actions.
- MCTS must use the pair policy on two-placement turns:
  - first-placement node priors combine `policy_place` and
    `policy_pair_first`;
  - after a first placement is selected, second-placement priors use
    `policy_pair_second`;
  - root and tactical diagnostics also evaluate `policy_pair_joint` over legal
    ordered pairs.
- The Rust/Python search contract must preserve global `(q,r)` identity for
  both placements.
- Add illegal-pair and duplicate-action guards.
- Add pair-policy fallback telemetry separate from single-placement fallback
  telemetry.

Acceptance tests:

```text
test_pair_policy_targets_use_full_policy_v2
test_pair_policy_d6_bijection
test_pair_policy_loss_uses_policy_weight
test_dashboard_pair_policy_inspection_endpoint
test_pair_policy_shadow_logging_does_not_change_mcts_choice
test_mcts_consumes_pair_policy_on_two_placement_turns
test_pair_policy_rejects_duplicate_and_illegal_pairs
test_pair_policy_prior_sources_are_reported
```

## Milestone F: `graph_hybrid_0` Honesty And Guardrails

Status: naming fixed, still needs clearer reporting.

Current reality:

`graph_hybrid_0` is:

```text
33x33 crop
-> CNN trunk
-> selected crop-cell sparse attention
-> scatter back into crop map
-> dense/sparse/pair heads
```

It is not:

```text
global sparse token input
WINDOW6 / LINE / COVER_SET token graph
relation-biased global Transformer
legal-token-only policy contract
graph-native D6
```

Implementation requirements:

- Keep `architecture = "graph_hybrid_0"` as the only current name.
- Accept old `architecture = "graph"` only as a config alias with a warning or
  normalization event.
- Dashboard labels must say `Graph Hybrid 0`, not `Global Graph`.
- Phase 3 trial names such as `graph384_windows`, `graph512_cover`, and
  `graph512_turn_pair_prior` should be renamed or described as selection
  presets, not real token-family ablations.
- Any true global graph implementation must use
  `architecture = "global_graph_option1"` and follow the separate spec.

Acceptance tests:

```text
test_graph_alias_normalizes_to_graph_hybrid_0
test_dashboard_labels_graph_hybrid_0_honestly
test_phase3_trial_metadata_marks_graph_hybrid_0_as_crop_compatible
```

## Milestone F2: Global Graph Alternatives

Status: required by the updated global graph spec.

Problem:

The maximal `global_graph768_champion` design is intentionally rich. If it is
the only tested design, a failure will be hard to diagnose and a success may be
hard to attribute. Phase 3 should compare several architectures while holding
the graph/action data contract fixed.

Required alternatives:

| Family | What It Tests |
|---|---|
| `global_xattn_0` | Whether all-legal global action identity plus context cross-attention is enough. |
| `global_line_window_0` | Whether explicit `WINDOW6`, `LINE`, and `COVER_SET` structure is the main win. |
| `global_pair_twostage_0` | Whether pair planning can be learned cheaply through first/second-stage scoring. |
| `global_graph_full_0` | Whether the full relation-biased graph design is justified. |
| `global_hybrid_action_0` | Whether a hex-masked crop local trunk plus global legal action head is a useful bridge/control. |
| `global_graph768_champion` | Scaled final candidate based on the best validated design. |

Shared contract:

```text
all legal Rust action rows are preserved
all 12 D6 transforms are tested
exact engine tactical labels are used
opening and two-placement masks are correct
future opponent policy has its own global target table
pair targets are legal and turn-phase aware
MCTS consumes global keyed priors
dashboard and replay inspect the same graph data
```

Implementation requirements:

- Add config names for every graph alternative.
- Use the same replay/graph builder for every alternative.
- Use the same legality, D6, tactical, pair, opponent-policy, and MCTS prior
  contract tests for every alternative.
- Compare alternatives on offline target reconstruction before allowing
  self-play.
- Promote only variants that pass the automated bug-isolation milestone below.

Acceptance tests:

```text
test_graph_alternatives_share_graph_data_contract
test_global_xattn_preserves_all_legal_rows
test_global_line_window_uses_exact_window_and_cover_labels
test_global_pair_twostage_masks_opening_and_second_placement_correctly
test_global_graph_full_uses_relation_bias_contract
test_global_hybrid_action_keeps_dense_policy_diagnostic_only
```

## Milestone G: Dashboard And Replay Debug Completeness

Status: required for completion. Current dashboard improvements are not enough
to satisfy the finished P1/P2/P3 contract.

Required dashboard visibility:

### Runs and Charts

Add charts/tables for:

```text
loss_policy
loss_sparse_policy
loss_pair_policy
loss_opp_policy
loss_value
loss_lookahead_*
loss_axis
loss_regret_rank
loss_regret_value
policy_top1/top4/top8
sparse_policy_top1/top4/top8
pair_policy_top1/top4/top8
value_weight_mean
truncation_rate
candidate_missing_mass
candidate_discovery_recall_*
fallback_prior_use_*
```

### Replay

Every replay position should expose:

```text
policy_target_v2
opp_policy_target_v2
pair_policy_target_v2
candidate rows
sparse logits if checkpoint loaded
pair logits if checkpoint loaded
root_value
selected_action_value
value_weight
policy_weight
opp_policy_weight
MCTS prior source counters
```

### Checkpoints

Checkpoint inspection must show:

```text
architecture
candidate feature version
enabled heads
loss weights
sparse prior stage compatibility
D6 compatibility
loadability
checkpoint config hash
```

### Tactical And D6 Debug

Add a debug panel where one position can be transformed through all six D6
symmetries and compared:

```text
legal mask
dense target
sparse target
pair target
axis target
candidate rows
top logits from loaded checkpoint
```

Acceptance tests:

```text
test_dashboard_metrics_endpoint_includes_all_enabled_losses
test_replay_endpoint_returns_policy_weights_and_regret_debug
test_dashboard_sparse_policy_inference_returns_candidate_logits
test_dashboard_pair_policy_inference_returns_pair_logits
test_dashboard_d6_debug_endpoint_returns_six_symmetries
```

## Milestone H: Phase 3 Scorecard And Autotune Completion

Status: operational but not yet fully spec-matching.

Required Phase 3 evaluators:

```text
checkpoint league evaluator
classical survival evaluator
tactical suite evaluator
outside-window evaluator
candidate recall evaluator
throughput/memory reporter
value calibration evaluator
policy target quality evaluator
```

### H1. Checkpoint League

Required behavior:

- Maintain a league with:

```text
top 6 active checkpoints
best previous checkpoint
best current baseline
best ResTNet baseline
best graph_hybrid_0 baseline
best global_graph_option1 checkpoint
best EMA checkpoint
best pair-policy checkpoint
```

- Evaluate both colors.
- Store rating mean, rating std, and lower confidence bound.
- Champion selection uses league LCB, not tiny noisy mini-match winrate.

Acceptance tests:

```text
test_checkpoint_league_persists_ratings
test_checkpoint_league_evaluates_both_colors
test_final_score_uses_league_lcb
```

### H2. Tactical And Outside-Window Suites

Required suites:

```text
win-now
forced-block
open-four
open-five
two-placement cover
outside-window win
outside-window block
separated-cluster long-span
late-game high-legal-count
```

Each suite position should save:

```text
move history
expected action set
expected pair/cover set when relevant
legal count
board span
source label
```

Acceptance tests:

```text
test_tactical_suite_positions_are_replayable
test_tactical_suite_expected_actions_are_legal
test_outside_window_suite_contains_actions_outside_33_crop
```

### H3. ASHA/BOHB And PB2

Required behavior:

- ASHA/BOHB narrows discrete model/static choices:

```text
model family
sim count
candidate budget
sparse prior stage
head bundle
batch size
```

BOHB is complete only if it implements both sides of BOHB:

```text
Hyperband-style brackets over resource budgets
model-based sampling from good/bad density estimates after warmup
```

Random or hand-built static candidate generation followed by successive
halving is useful, but it is not BOHB.

ASHA resource policy:

```text
default asha_resources = 8,12,14
default max_active_trials = 6
```

Rung semantics:

```text
epoch < 8:
  health checks only
  do not use classical survival or checkpoint-league strength for pruning

8 <= epoch < 12:
  pre-classical strategy scoring
  use target quality, tactical fixtures, outside-window diagnostics, value
  calibration, throughput, and hard bug sentinels

epoch >= 12:
  classical survival can enter the scorecard
  12-14 epochs is the first meaningful classical-survival ASHA comparison band
```

Acceptance tests:

```text
test_asha_default_resources_start_at_epoch_8
test_scorecard_ignores_classical_survival_before_epoch_12
test_short_health_rung_prunes_only_hard_failures
```

- PB2 mutates dynamic schedules:

```text
LR
weight decay
c_puct
c_puct_init
Dirichlet fraction
Dirichlet alpha mode
PCR low-sim probability
buffer recency decay
value loss weight
auxiliary loss multiplier
sparse policy loss
pair policy loss
regret replay fraction
```

No-half-measure rule:

The completed Phase 3 system must implement real PB2, not merely PBT with the
PB2 name. PBT can remain as a baseline/debug scheduler, but the spec-matching
path must include:

- a continuous hyperparameter response model over completed trial observations;
- uncertainty-aware proposals for dynamic knobs;
- compatibility checks before exploitation or checkpoint transfer;
- no shared mutable replay-buffer ownership after exploit unless explicitly
  designed and logged as shared lineage;
- explicit logging of model fit inputs, proposed mutations, accepted
  mutations, rejected mutations, clone failures, and any baseline PBT
  comparison events;
- deterministic replay of the scheduler decision from persisted metadata.

Acceptance tests:

```text
test_pbt_exploit_only_between_compatible_architectures
test_pbt_clone_does_not_share_mutable_replay_unless_declared
test_pbt_mutations_are_clamped_and_logged
test_bohb_creates_hyperband_brackets
test_bohb_samples_from_good_bad_density_models
test_pb2_fits_continuous_response_model
test_pb2_proposals_are_uncertainty_aware_and_clamped
test_mutation_events_include_source_method
test_scheduler_decisions_replay_from_persisted_metadata
```

### H4. Scorecard Formula

The Phase 3 scorecard should use the documented structure:

```text
health_warmup_score, epoch < 8 =
    0.45 * z(policy_target_quality)
  + 0.35 * z(value_calibration_score)
  + 0.20 * z(outside_window_robustness)

pre_classical_strategy_score, 8 <= epoch < 12 =
    0.30 * z(tactical_suite_score)
  + 0.25 * z(outside_window_robustness)
  + 0.25 * z(policy_target_quality)
  + 0.20 * z(value_calibration_score)

strength_score =
    0.40 * z(league_lcb)
  + 0.20 * z(outside_window_robustness)
  + 0.15 * z(tactical_suite_score)
  + 0.10 * z(classical_survival_score)
  + 0.10 * z(value_calibration_score)
  + 0.05 * z(policy_target_quality)

scheduler_score =
    strength_score
  - 0.10 * z(epoch_seconds)
  - 0.10 * z(truncation_rate)
  - 0.20 * z(illegal_or_crash_rate)
```

Use `strength_score` with classical survival and league components only at
`epoch >= 12` unless a run explicitly raises the threshold. This prevents ASHA
from pruning on noisy sub-strategy checkpoints.

Graph/candidate models add hard gates:

```text
candidate_discovery_winning_move >= 99.5%
candidate_discovery_forced_block >= 99.5%
candidate_discovery_two_placement_cover >= 99.0%
missing_target_policy_mass <= 1.0%
critical_overflow_count == 0
fallback_prior_use_on_mcts_topk decreases materially
```

Acceptance tests:

```text
test_scorecard_uses_documented_weights
test_scorecard_uses_health_mode_before_epoch_8
test_scorecard_uses_pre_classical_mode_before_epoch_12
test_scorecard_applies_candidate_hard_gates
test_scorecard_penalizes_illegal_crash_and_truncation_rates
```

## Milestone I: Config Strictness And Dead-Code Cleanup

Status: active configs cleaned for resignation, but strictness is still needed.

Required behavior:

- Stale or misspelled config keys must fail validation.
- Enabled heads must have corresponding targets and losses.
- Nonfunctional options must be removed, rejected, or implemented.

Implementation requirements:

- Use strict config models for active TOML:

```text
extra = "forbid"
```

- Keep legacy/import adapters separate if old run metadata needs to be read.
- Implement CNN dropout in residual blocks when `model.dropout > 0`; do not
  allow a silent no-op or a config that claims regularization without applying
  it.
- Reject enabled axis head without an `axis` loss.
- Reject sparse/pair config combinations that cannot produce the required
  tensors.
- Remove remaining active-doc references that imply resignation is part of the
  training path.

Acceptance tests:

```text
test_unknown_config_key_fails_validation
test_dropout_is_applied_for_each_architecture_that_accepts_dropout
test_enabled_head_requires_loss
test_resign_keys_are_rejected
test_sparse_pair_invalid_config_combinations_are_rejected
```

## Milestone J: Production Integration Test

Status: needed.

Add a small but complete production-pipeline test that exercises the current
best model family without relying on long training.

Test recipe:

```text
1. create temporary run directory
2. start a tiny graph_hybrid_0 or candidate-policy config
3. generate self-play games with D6 enabled
4. save compact records and dashboard DB rows
5. train for one tiny epoch
6. save checkpoint
7. import checkpoint through dashboard indexer
8. run noisy eval
9. open replay payload for one game
10. verify candidate/discovery/fallback metrics exist
```

Acceptance tests:

```text
test_tiny_production_pipeline_records_games_metrics_and_checkpoint
test_dashboard_can_replay_game_from_tiny_production_pipeline
test_tiny_pipeline_sparse_metrics_are_present_when_sparse_enabled
```

This test should be fast enough for CI but complete enough to catch broken
contracts before another overnight run.

## Milestone K: Automated Bug-Isolation And Engine-Sanity Gates

Status: required before trusting model strength, architecture comparisons, or
overnight/autotune conclusions.

Problem:

Self-play systems can appear to improve because a model, target generator, MCTS
path, dashboard replay path, or engine binding exploits a bug. A stronger
checkpoint is only meaningful if independent legality, replay, D6, tactical,
and evaluation checks agree that the games are valid and the labels are clean.

Required principle:

```text
Elo, loss curves, game length, or classical survival are never sufficient alone.
Promotion requires zero hard bug-sentinel failures.
```

### K1. Engine Invariant Suite

Required checks:

```text
opening is exactly player 0 at origin
turn order is P0, P1/P1, P0/P0, ...
legal moves are empty cells within distance 8 of an existing stone
legal move set is unchanged by dashboard/debug overlays
winner detection matches six-in-a-row on all three axes
terminal states reject further moves
compact replay reconstructs final board, current player, winner, and terminal reason
```

Acceptance tests:

```text
test_engine_opening_and_two_placement_turn_order
test_engine_legal_moves_match_radius_rule
test_engine_winner_matches_reference_axis_scan
test_engine_rejects_post_terminal_moves
test_compact_replay_reconstructs_terminal_state
```

### K2. Differential Reference Oracles

Maintain slow, simple reference checks for small and medium states:

```text
Python legal-move generator
Python six-in-row winner scanner
Python 6-window/threat scanner
Rust engine result
dashboard replay reconstruction
```

The reference implementation does not need to be fast. It exists to catch Rust
FFI, graph builder, target, and replay bugs.

Acceptance tests:

```text
test_rust_and_python_legal_moves_agree_on_random_histories
test_rust_and_python_winner_scans_agree_on_random_histories
test_rust_and_python_window_oracles_agree_on_tactical_fixtures
test_dashboard_replay_matches_engine_replay
```

### K3. Metamorphic D6 And Serialization Tests

Required checks for dense, sparse, pair, graph, and dashboard paths:

```text
all 12 D6 transforms preserve legality and winner
legal masks transform bijectively
policy, opponent-policy, and pair targets transform by global key
graph tokens, relation ids, relation float biases, and pair rows transform correctly
move-history serialize/deserialize/replay is identity
MCTS with uniform priors is invariant up to transformed action keys
```

Acceptance tests:

```text
test_d6_all_12_preserve_engine_state
test_d6_legal_mask_bijection_random_histories
test_d6_graph_token_relation_pair_equivariance
test_move_history_round_trip_identity
test_uniform_prior_mcts_d6_invariance
```

### K4. Tactical Holdout Suites

Holdout suites must be independent from model training labels and scheduler
training signals.

Required suites:

```text
win-now
forced block
two-placement cover
unblockable fork
block plus counterattack
outside-window win
outside-window block
separated-cluster defense
late-game high-legal-count state
```

Each fixture stores:

```text
move history
expected legal action set
expected pair/cover set when relevant
expected terminal result after action when relevant
legal count
board span
source/generator metadata
```

Acceptance tests:

```text
test_tactical_holdout_expected_actions_are_legal
test_tactical_holdout_oracle_labels_are_stable
test_outside_window_holdout_has_expected_action_outside_crop
test_late_game_holdout_exercises_high_legal_count
```

### K5. Shadow Evaluation Matrix

Promoted checkpoints must be evaluated under multiple modes:

```text
direct policy sampling
PUCT/MCTS with model priors
PUCT/MCTS with uniform priors for engine sanity
classical opponent
checkpoint league
fixed tactical holdout suite
outside-window suite
```

Quarantine rule:

```text
If strength appears only in one evaluator and bug sentinels or tactical suites
disagree, quarantine the checkpoint and inspect before promotion.
```

Acceptance tests:

```text
test_shadow_eval_runs_all_required_modes
test_shadow_eval_quarantines_bug_sentinel_failures
test_checkpoint_promotion_requires_tactical_and_replay_cleanliness
```

### K6. Anti-Leak And Label-Quality Tests

Fail promotion if:

```text
candidate/action features contain target probability or target-present bits
dashboard/debug labels enter live model features
future opponent policy target leaks into current-state features
low-PCR policy rows train policy as full-search rows
truncated games train terminal value or regret heads
graph alternatives use different target contracts
```

Acceptance tests:

```text
test_candidate_features_have_no_target_leakage
test_graph_features_have_no_future_or_dashboard_leakage
test_low_pcr_policy_weight_zero_for_policy_heads
test_truncation_zero_weights_value_and_regret
test_graph_alternatives_share_targets_and_masks
```

### K7. Bug Sentinel Metrics

Persist these metrics to JSONL/SQLite/dashboard and hard-gate promotion:

```text
illegal_move_rate
post_terminal_move_attempts
replay_mismatch_rate
d6_mismatch_rate
legal_mask_mismatch_rate
oracle_threat_mismatch_rate
fallback_prior_use_on_mcts_topk
missing_legal_action_rows
pair_mask_violation_rate
terminal_reason_distribution
truncation_rate
target_leakage_check_status
```

Hard gate:

```text
illegal_move_rate == 0
post_terminal_move_attempts == 0
replay_mismatch_rate == 0
d6_mismatch_rate == 0
legal_mask_mismatch_rate == 0
oracle_threat_mismatch_rate == 0
missing_legal_action_rows == 0
pair_mask_violation_rate == 0
target_leakage_check_status == pass
```

The scorecard may include softer penalties for truncation and fallback-prior
usage, but legality/replay/D6/oracle mismatches are hard failures.

## P1/P2/P3 Vision Coverage Matrix

Implementing this document plus
`Docs/SPEC_FIX_MATCH_GLOBAL_GRAPH_MODEL_20260428.md` satisfies the original
phase intent as follows.

### Phase 1: ResTNet And Action-Contract Scout

Finished coverage:

```text
best crop baseline remains available as a strict control
ResTNet/crop-attention variants are valid comparable baselines
policy_target_v2 is global-coordinate keyed
candidate/action-keyed sparse policy has no target leakage
candidate discovery and protected recall are measured separately
critical wins/blocks/cover cells cannot be silently capped away
full-board tactical oracle covers outside-window failures
sparse-prior MCTS stages have prior-source telemetry
D6 works for dense, sparse, pair, axis, and graph_hybrid_0 paths
fallback prior use is visible in training, replay, dashboard, and scorecards
```

This exceeds Phase 1 by making the action contract production-observable rather
than merely an offline scout.

### Phase 2: Graph/Transformer Architecture

Finished coverage:

```text
graph_hybrid_0 remains an honest crop-compatible baseline
global_graph_option1 replaces the crop as the primary state/policy contract
STATE/TURN/PLAYER/STONE/LEGAL/HOT_CELL/WINDOW6/LINE/COVER_SET/COMPONENT/PAIR_ACTION tokens exist
relation-biased global attention exists
legal-token policy replaces dense 1089 policy for the true graph path
pair policy is graph-native and active during two-placement turns
graph-native D6 is tested across tokens, relations, legal actions, and pairs
dashboard can inspect tokens, relations, logits, targets, and prior sources
global_xattn_0, global_line_window_0, global_pair_twostage_0, global_graph_full_0,
and optional global_hybrid_action_0 are comparable under one shared contract
```

This exceeds Phase 2 by requiring the champion graph model, not only a smaller
proof-of-concept graph.

### Phase 3: Autotune And Champion Selection

Finished coverage:

```text
ASHA/BOHB narrows static choices
real PB2 tunes dynamic schedules
checkpoint league uses both colors and lower confidence bounds
tactical and outside-window suites are first-class evaluators
candidate safety gates use discovery recall, critical overflow, and fallback-prior metrics
bug sentinels hard-gate promotion before Elo or loss curves can matter
scorecard formula matches the Phase 3 plan
EMA, pair-policy, graph_hybrid_0, baseline, and global_graph_option1 checkpoints are league-visible
production smoke test proves games, metrics, checkpoints, dashboard replay, eval, and sparse metrics
```

This exceeds Phase 3 by making scheduler decisions replayable from persisted
metadata and by forcing dashboard/debug parity with the training contracts.

## Implementation Order

### Step 1: Lock Contracts

- Add version constants for candidate features, compact records, and target
  bundles.
- Add strict config validation for stale keys and impossible feature combos.
- Add dashboard/config display of contract versions.

Exit gate:

```text
pytest Python/tests/test_config_and_guardrails.py -q
```

### Step 2: Exhaustive D6 And Target Tests

- Build the shared random-history D6 harness.
- Add dense, sparse, pair, axis, opponent-policy, and regret target tests.
- Confirm D6 remains enabled for sparse and `graph_hybrid_0`.

Exit gate:

```text
pytest Python/tests/test_training_data_pipeline.py Python/tests/test_axis_policy.py -q
```

### Step 3: Full-Board Tactical Oracle

- Implement full-board tactical scan.
- Wire root/leaf sparse candidates to oracle outputs.
- Add outside-window tactical suite.

Exit gate:

```text
pytest Python/tests/test_tactical_oracle.py Python/tests/test_training_data_pipeline.py -q
```

### Step 4: Sparse-Prior Telemetry

- Expose prior source counters from MCTS.
- Aggregate fallback/sparse/dense usage into records and dashboard metrics.
- Update Phase 3 scorecard gates.

Exit gate:

```text
pytest Python/tests/test_inference_server.py Python/tests/test_dashboard_foundation.py -q
```

### Step 5: Pair Policy Debug Completion

- Complete pair D6 tests.
- Add dashboard pair inspection.
- Add MCTS shadow logging and active MCTS pair-policy consumption.
- Validate pair-policy prior sources and legal-pair guards in self-play.

Exit gate:

```text
pytest Python/tests/test_training_data_pipeline.py Python/tests/test_dashboard_foundation.py -q
```

### Step 6: Phase 3 Evaluators

- Implement or verify checkpoint league, tactical suite, outside-window suite,
  candidate recall evaluator, value calibration, and scorecard formula.
- Implement real PB2 for the spec-matching scheduler path and keep PBT only as
  a named baseline/debug scheduler.

Exit gate:

```text
pytest Python/tests/test_phase3_autotune.py -q
```

### Step 7: End-To-End Production Smoke

- Add tiny production-pipeline integration test.
- Run all Python and Rust tests.

Exit gate:

```text
.venv/bin/python -m pytest Python/tests -q
cargo test
```

### Step 8: Bug-Isolation Promotion Gates

- Add engine invariant tests.
- Add independent Python reference oracles.
- Add all-12-D6 metamorphic tests for graph/action/pair/relation paths.
- Add tactical holdout suites.
- Add shadow evaluation matrix.
- Add anti-leak tests.
- Wire bug sentinel metrics into Phase 3 promotion gates and dashboard.

Exit gate:

```text
pytest Python/tests/test_engine_invariants.py Python/tests/test_tactical_oracle.py Python/tests/test_phase3_autotune.py -q
```

## Acceptance Definition

This plan is complete only when:

```text
all active configs validate strictly
all enabled heads train with matching targets and loss weights
D6 tests pass for dense, sparse, pair, axis, and graph_hybrid_0 paths
candidate discovery metrics are separate from protected training recall
critical candidate overflow cannot be hidden
full-board tactical oracle covers outside-window wins/blocks
MCTS exposes sparse/dense/default prior-source telemetry
dashboard can inspect sparse and pair policy data from real replay positions
Phase 3 scorecards use league/tactical/outside-window/candidate safety metrics
global graph alternatives are tested under one shared graph/action contract
engine/reference/replay/D6/tactical oracles agree on promoted checkpoints
bug sentinel hard gates are zero before promotion
real PB2 scheduler path exists, is persisted, and can be replayed deterministically
tiny production integration test saves games, metrics, checkpoint, and replay
```

Anything less than this is a partial implementation and should not be treated as
spec-matching.
