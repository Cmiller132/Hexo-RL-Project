# Implementation Spec Audit - 2026-04-28

This audit compares the current implementation against:

- `Docs/PHASE1_RESTNET_ACTION_CONTRACT_SCOUT_20260427.md`
- `Docs/TRANSFORMER_ARCHITECTURE_ABLATIONS_FOR_HEXO_20260427.md`
- `Docs/AUTOTUNING_METHODS_AND_48H_PLAN_20260427.md`
- `Docs/SPEC_FIX_MATCH_GLOBAL_GRAPH_MODEL_20260428.md`
- the seven model-head/data-path review findings from 2026-04-28

The goal is to identify implementation details that were only partially wired,
misleadingly named, or skipped.

## Executive Verdict

The project is much stronger than it was before the recent fixes, but it still
has two categories of incompleteness:

1. **Fixed during this audit:** concrete bugs where code claimed or implied a
   behavior but did not fully implement it.
2. **Intentional target work:** large spec items that are now documented as not
   implemented, especially the true global sparse graph model.

The seven original review findings are mostly fixed in the active training
path. The biggest remaining architectural gap is not hidden anymore:
`graph_hybrid_0` is a crop-compatible sparse-token scout, not
`global_graph_option1`.

## Fixed During This Audit

### Sparse Candidate Features Leaked Training Labels

Status: fixed.

`build_candidate_batch()` previously included target probability and a
target-present bit as candidate features. During live MCTS sparse inference,
candidate targets are empty, so sparse policy training saw information that
inference never gets.

Fix:

- removed target-probability and target-present features from candidate inputs;
- replaced them with inference-available tactical/source features:
  legal rank, winning-cell bit, forced-block bit, outside-crop bit, and
  critical-cell bit.

Code:

- `Python/src/hexorl/action_contract/candidates.py`
- regression: `test_candidate_features_do_not_include_policy_target_labels`

### Sparse Candidate Overflow Could Hide Dropped Supervision

Status: fixed.

The candidate builder can intentionally exceed the nominal budget for protected
actions. The sampler copied into fixed-width arrays and could silently truncate
rows.

Fix:

- batch candidate width now respects the effective configured sparse width;
- copied sparse target mass is checked, and `candidate_missing_mass` reflects
  dropped represented mass if truncation ever occurs;
- regression test covers preserved missing mass.

Code:

- `Python/src/hexorl/buffer/sampler.py`
- regression: `test_sparse_sampler_reports_missing_mass_if_protected_candidates_overflow_width`

### `lookahead_4` Could Validate But Not Train

Status: fixed.

The default buffer horizons are `4, 12, 36`, but default loss weights included
`lookahead_6`. A config with `lookahead_4` could validate and forward while
silently skipping the `lookahead_4` loss.

Fix:

- cross-section config validation now adds a default loss weight for any
  configured lookahead head missing from `train.loss_weights`.

Code:

- `Python/src/hexorl/config/schema.py`
- regression: `test_config_adds_default_loss_for_matching_lookahead_head`

### Sparse Prior Stage Could Be A No-Op

Status: fixed.

`sparse_prior_stage=1/2` validated even when `model.sparse_policy=false`.
Self-play only uses sparse priors when both are enabled, so this could silently
run dense priors.

Fix:

- validation now rejects `sparse_prior_stage > 0` unless sparse policy is
  enabled.

Code:

- `Python/src/hexorl/config/schema.py`
- regression: `test_sparse_prior_stage_requires_sparse_policy_contract`

### Candidate Recall Scorecard Had A Typo

Status: fixed.

`EvaluationServices.candidate_recall()` used `self.s.args` instead of
`self.args` in the gate expression, and another scoring path used the same typo
for target epoch seconds.

Fix:

- corrected both references to `self.args`.

Code:

- `scripts/run_phase3_48h_autotune.py`

### Live Sparse-Prior Candidates Skipped Tactical Overrides

Status: fixed for current crop-visible tactical planes.

Root and leaf sparse-prior inference called `build_candidate_batch()` with empty
targets and without winning/forced/cover overrides.

Fix:

- root sparse candidates now pass crop-visible win/block/cover cells extracted
  from encoded hot-cell planes;
- leaf sparse candidates do the same using pending-leaf metadata and the leaf
  tensor;
- candidate probe now uses the full `policy_v2` target rather than a sliced
  subset.

Code:

- `Python/src/hexorl/selfplay/worker.py`

Remaining limitation:

- tactical extraction still depends on the current encoded crop. True
  outside-window tactical inclusion belongs to the future global graph/token
  builder.

### Dashboard Misidentified `graph_hybrid_0`

Status: fixed.

Dashboard summary helpers only recognized `architecture == "graph"`, so new
`graph_hybrid_0` runs could display as a generic CNN.

Fix:

- dashboard recognizes both legacy `graph` and current `graph_hybrid_0`, and
  labels the current path as Graph Hybrid 0.

Code:

- `Python/src/hexorl/dashboard/app.py`

### Dashboard Dense Inference Could Not Inspect Sparse Policy

Status: improved.

The dashboard model cache called `model(x)` with no candidate tensors, so
`sparse_policy` outputs were absent.

Fix:

- dashboard inference now builds candidate rows from legal moves when the model
  has sparse policy enabled;
- the response includes top sparse-policy candidate logits.

Code:

- `Python/src/hexorl/dashboard/model_cache.py`

Remaining limitation:

- pair-policy dashboard inference is still not exposed because there is not yet
  a dashboard-facing pair candidate/debug contract.

### Compact Records Dropped `value_weight`

Status: fixed.

`PositionRecord.value_weight` masks value loss for truncated/non-terminal games,
but compact serialization did not preserve it.

Fix:

- compact record version advanced to 5;
- `value_weight` is serialized/deserialized;
- older records default to `1.0`.

Code:

- `Python/src/hexorl/selfplay/records.py`
- regression: `test_compact_record_v2_roundtrip_preserves_global_targets`

### Stale Resignation Config Keys

Status: fixed in active TOML configs.

Resignation had already been removed from the schema/path, but stale
`resign_threshold` and `resign_disable_prob` keys remained in TOML files.

Fix:

- removed stale resign keys from active configs.

Files:

- `Configs/default.toml`
- `Configs/production.toml`
- `Configs/reproducible.toml`
- `Configs/small_test.toml`
- `Configs/wsl_speed_probe.toml`

### Stale Regret Helper Used Root Value

Status: fixed.

The active training path uses `selected_action_value`, but
`buffer/regret_buffer.py::compute_regret()` still documented and used
`root_value`.

Fix:

- helper now uses `selected_action_value` when present, with `root_value` as
  fallback.

Code:

- `Python/src/hexorl/buffer/regret_buffer.py`

## Original Seven Findings

### 1. Lookahead Turn Boundaries

Status: fixed.

`_turn_boundary_indices()` now uses player-run starts, matching Hexo's one-stone
opening and later two-placement turns.

Evidence:

- `Python/src/hexorl/buffer/targets.py`
- `test_hexo_turn_boundaries_follow_player_runs`

### 2. Lookahead Perspective Safety

Status: fixed.

Future lookahead values are flipped when the future position's player differs
from the source position's player.

Evidence:

- `Python/src/hexorl/buffer/targets.py`
- `test_lookahead_flips_future_player_perspective`
- `test_mid_turn_lookahead_targets_next_turn_start`

### 3. Opponent Policy Target

Status: fixed.

Opponent policy now uses the next full-search opponent turn start. The current
turn's PCR quality does not block training from a later full-search opponent
turn.

Evidence:

- `Python/src/hexorl/buffer/targets.py`
- `test_opponent_policy_uses_next_full_search_opponent_turn_start`

### 4. Regret Heads Paper Exactness

Status: mostly fixed for the active path.

Active targets now use selected-action value, raw regret scale, and the ranking
loss no longer batch-minmax normalizes. The stale helper was fixed during this
audit.

Remaining caution:

- This is still an adaptation unless validated against the exact paper's full
  restart/PRB training loop. The head targets are much closer, but the full RGSC
  training system is not fully implemented.

### 5. Sparse Policy D6

Status: fixed for current crop/sparse/graph_hybrid_0 models.

The sampler transforms compact history and global targets, then re-encodes.
This is the right safe approach for current models.

Remaining caution:

- true graph-native D6 for `WINDOW6`, `COVER_SET`, relation bias, and
  `PAIR_ACTION` tokens is future work because those token batches do not exist.

### 6. Opponent Policy Weighting

Status: fixed.

`opp_policy_weight` is stored, sampled, and used in loss computation.

### 7. Axis Head Untrained In Production

Status: fixed.

Production has both the `axis` head and matching loss weight.

## Major Spec Gaps That Remain Intentional

### `global_graph_option1` Is Not Implemented

Status: not implemented; documented future work.

The true spec requires:

- global sparse token input;
- `STONE`, `LEGAL`, `HOT_CELL`, `WINDOW6`, `LINE`, `COVER_SET`,
  `COMPONENT`, and `PAIR_ACTION` token builders;
- relation/edge bias;
- policy logits over global legal tokens;
- graph-native pair policy;
- graph-native D6.

Current `graph_hybrid_0` does not do this. It remains:

```text
33x33 crop -> CNN trunk -> selected crop-cell sparse attention -> dense/sparse heads
```

This is now named honestly.

### Graph Token-Set Ablations Are Not Real Token-Family Ablations

Status: known limitation.

`graph384_windows`, `graph512_cover`, and `graph512_turn_pair_prior` currently
modify crop-cell selection weights and type hints. They do not create true
`WINDOW6`, `COVER_SET`, or `PAIR_ACTION` tokens.

### Pair Policy Is Auxiliary Only

Status: partially implemented.

Current `PairPolicyHead` scores selected candidate-row pairs for an auxiliary
loss. It is not consumed by MCTS and is not the spec's
`policy_pair_first` / `policy_pair_second` / `policy_pair_joint` design.

### Candidate Recall Metrics Are Still Limited

Status: partially implemented.

Current metrics include:

- top1/top4/top8 target recall;
- winning move recall;
- forced block recall;
- two-placement cover recall;
- missing target mass;
- outside-window target mass.

Missing from the original Phase 1 diagnostic plan:

- open-four/open-five recall;
- fallback-prior use;
- fallback-prior use on MCTS top-k;
- sparse-vs-dense disagreement;
- separated-cluster and board-span diagnostics.

### Outside-Window Tactical Overrides Are Not Fully Solved

Status: unresolved until global token builder.

Current tactical extraction uses crop hot-cell planes. That is enough for the
current crop-compatible models, but it cannot prove outside-window forced wins
or blocks are always included.

### Inference Still Requires Dense Policy And Value

Status: explicit now.

The inference server now fails early if `policy` or `value` heads are absent.
This matches the current self-play path, but not the future global-graph policy
contract.

## Validation Run

Focused tests after fixes:

```text
.venv/bin/python -m pytest Python/tests/test_training_data_pipeline.py \
  Python/tests/test_config_and_guardrails.py \
  Python/tests/test_inference_server.py \
  Python/tests/test_dashboard_foundation.py -q
```

Result during audit:

```text
82 passed
```

Additional focused regressions:

```text
test_candidate_features_do_not_include_policy_target_labels
test_sparse_sampler_reports_missing_mass_if_protected_candidates_overflow_width
test_config_adds_default_loss_for_matching_lookahead_head
test_sparse_prior_stage_requires_sparse_policy_contract
```

Result:

```text
4 passed
```

## Recommended Next Fixes

1. Add real fallback-prior telemetry from Rust/Python sparse prior gather.
2. Add pair-policy dashboard inspection and MCTS shadow logging.
3. Add outside-window tactical suites that cannot be passed by crop hot planes.
4. Keep `graph_hybrid_0` frozen except bug fixes.
5. Implement `global_graph384_windows` as the first true spec-matching global
   graph target.
