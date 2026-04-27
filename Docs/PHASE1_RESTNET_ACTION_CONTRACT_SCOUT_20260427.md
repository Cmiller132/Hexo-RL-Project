# Phase 1 Plan: ResTNet And Action-Contract Scout - 2026-04-27

This is the first document in the consolidated three-phase Hexo improvement
plan. The docs should be implemented as one connected program:

1. `Docs/PHASE1_RESTNET_ACTION_CONTRACT_SCOUT_20260427.md`
2. `Docs/TRANSFORMER_ARCHITECTURE_ABLATIONS_FOR_HEXO_20260427.md`
3. `Docs/AUTOTUNING_METHODS_AND_48H_PLAN_20260427.md`

The implementation order is:

```text
Phase 1: ResTNet + action-contract scout
Phase 2: Transformer input / 33x33 window replacement
Phase 3: autotuning and champion selection
```

Phase 1 keeps the current `33x33` crop alive long enough to answer two
questions cleanly:

```text
1. What is the strongest fixed-window baseline?
2. Can the existing system learn global action identity before we replace the
   crop with a token/graph Transformer?
```

The main architecture upgrade is ResTNet. The one other high-value experiment is
not a second crop layout and not a broad auxiliary-head sweep. It is:

```text
candidate-only sparse/action-keyed policy scout
```

This is the best companion because it attacks the same failure Phase 2 must
solve: the model and MCTS need to reason about legal `(q,r)` actions on an
infinite board, not only `0..1088` crop indices.

## Why This Phase Comes First

Hexo differs from Go-like fixed-board games in ways that make the current
`33x33` contract fragile:

- the board is infinite and has no edge/corner anchors;
- legal moves are all empty cells within distance 8 of any stone;
- turns have two placements, so policy quality affects first and second move
  planning inside one turn;
- 4-windows and 5-windows are both immediate win threats because a player has
  two placements;
- defensive quality depends on whether two block cells can cover all hot
  windows;
- a centroid crop can miss separated clusters, distant forced blocks, and
  outside-window winning actions.

ResTNet tests whether better long-range mixing inside `33x33` is enough. The
action-contract scout tests whether the pipeline can represent and train global
move identities before Phase 2 removes the crop bottleneck.

## Phase 1 Deliverables

Phase 1 must produce:

```text
best_current_33:
  strongest tuned current crop model

best_restnet_33:
  strongest attention-inside-crop model

candidate_policy_scout:
  proof that action-keyed targets/candidate priors can be learned safely

failure_map:
  outside-window, long-span, separated-cluster, forced-block, and fallback-prior
  diagnostics for all promoted models
```

These outputs are consumed directly by Phase 2. Phase 3 should not tune the
transformer until these baselines and diagnostics exist.

## System A: Best Current `33x33` Baseline

The current model remains the first control. Do not let Phase 2 compare a new
transformer against a stale baseline.

### Baseline Grid

| ID | Shape | Purpose |
|---|---|---|
| `base_96x12` | current trunk, `96x12` | fast lower baseline and throughput anchor |
| `base_128x16` | current trunk, `128x16` | main baseline |
| `base_160x20` | current trunk, `160x20` | capacity control |

Keep the search sweep narrow:

| Parameter | Values |
|---|---|
| full sims | `800`, `1200` |
| PCR low sims | `192`, `256`, `384` |
| policy top-k | `64`, `96`, `128` |
| temperature family | fast cool, slow cool |
| subtree reuse | `True` |

Expected default:

```text
base_128x16
full_sims = 800 or 1200
pcr_low_sims = 256
policy_top_k = 96
subtree_reuse = True
```

## System B: ResTNet Inside The Existing Crop

ResTNet is a conservative architecture improvement: keep the same input and
output contracts, but interleave residual CNN blocks with spatial Transformer
blocks over the `33x33` crop.

The Phase 1 test is not "is ResTNet modern?" The test is:

```text
Does attention inside the crop improve strength enough to become the Phase 2
teacher and the Phase 3 fixed-window finalist?
```

### ResTNet Block

Use a PreNorm spatial attention block:

```text
input: (B,C,33,33)
flatten: (B,1089,C)
add coordinate features
self-attention
MLP/SwiGLU
reshape: (B,C,33,33)
residual connection
```

Default settings:

| Parameter | Default |
|---|---|
| attention heads | `8` for width `128`, `8-10` for width `160` |
| MLP ratio | `2.0` |
| dropout | `0.0` |
| attention dropout | `0.0` |
| position encoding | coordinate MLP first |
| relative bias | ablation only |
| mode | replace residual blocks first |

### ResTNet Grid

| ID | Base | Attention blocks | Placement | Purpose |
|---|---|---:|---|---|
| `rest_128_2` | `128x16` | `2` | `5,10` | low-risk attention scout |
| `rest_128_3` | `128x16` | `3` | `5,10,14` | main ResTNet candidate |
| `rest_128_3_rel` | `128x16` | `3` | `5,10,14` | relative-bias control |
| `rest_160_3` | `160x20` | `3` | `6,12,18` | capacity-matched candidate |

Decision rules:

```text
If ResTNet improves Elo and value calibration but not outside-window buckets:
  use it as a stronger crop baseline, but proceed to Phase 2.

If ResTNet improves outside-window buckets only weakly:
  it is inferring missing context indirectly; the action/input contract is still
  the likely ceiling.

If ResTNet loses at equal wall-clock:
  do not tune it heavily in Phase 3 unless it helps as a Phase 2 teacher.
```

## System C: Candidate-Only Sparse/Action-Keyed Policy Scout

This is the one non-ResTNet experimental option in Phase 1.

It keeps the current `33x33` trunk and dense policy head, but adds a second
policy path keyed by global legal actions:

```text
current:
  crop tensor -> dense logits over 1089 crop cells

scout:
  crop tensor + candidate/action features -> sparse logits over selected global
  legal actions `(q,r)`
```

This is not the final transformer. It is a bridge that proves the training
records, target format, MCTS prior gather, and evaluation metrics can support an
infinite-board action dictionary.

### Replay Target V2

Add a new target while retaining legacy compatibility:

```text
policy_target_v2:
  [(q, r, prob), ...]

opp_policy_target_v2:
  [(q, r, prob), ...]

pair_policy_target_v2:
  [((q1, r1), (q2, r2), prob), ...]  # optional in Phase 1, useful for Phase 2

legacy_policy_target:
  {flat_idx: prob} derived only when action is inside the crop
```

Do not discard outside-window policy mass. The entire point is to measure it.

### Candidate Set Builder

The scout should build a candidate set per position. Critical actions override
caps.

Include:

- all legal actions if the count is below the cap;
- MCTS target top-k actions;
- actions with nonzero target policy mass;
- immediate winning moves;
- forced-block cells;
- hot-window cells;
- cover-set cells for two-placement defense;
- recent-neighborhood candidates around the last few moves;
- high-prior in-crop actions from the dense policy.

Recommended caps:

| Candidate budget | Role |
|---:|---|
| `128` | very fast recall scout |
| `256` | default Phase 1 budget |
| `384` | high-recall bridge to Phase 2 |

Hard rule:

```text
Never cap away immediate wins, forced blocks, MCTS target actions, or cells
needed to cover hot-window threat sets.
```

### Sparse Policy Head

The Phase 1 version can be simple:

```text
candidate features:
  global relative coordinates
  in-crop flag
  dense-policy logit if inside crop
  local trunk feature sampled if inside crop
  distance to centroid
  distance to last moves
  legal-neighborhood counts
  own/opponent window severity
  hot-window/cover-set flags

candidate encoder:
  small MLP or 2-layer attention over candidates

output:
  sparse logit per candidate `(q,r)`
```

Loss:

```text
dense_policy_loss:
  unchanged legacy loss on in-crop mass

sparse_policy_loss:
  CE/KL against policy_target_v2 restricted to candidate set

missing_mass_penalty:
  report only at first, optional soft penalty after recall is stable
```

Default weights:

```text
dense_policy = 1.0
sparse_policy = 0.25 early, then 0.5 if stable
candidate_recall_penalty = 0.0 until diagnostics are trusted
```

### MCTS Prior Gather

Stage the integration:

```text
Stage 0:
  train sparse policy offline only; MCTS still uses dense crop priors.

Stage 1:
  if sparse logit exists for a legal action, mix it into the prior:
    prior = 0.75 * dense_or_fallback + 0.25 * sparse

Stage 2:
  use sparse/action-keyed prior first:
    if sparse action logit exists:
      use sparse
    elif action inside 33x33:
      use dense crop logit
    else:
      use fallback
```

Phase 1 does not need to prove the sparse scout is stronger than ResTNet. It
needs to prove that decisive actions can be represented, trained, and served to
MCTS without destabilizing self-play.

## Required Instrumentation

Track for every Phase 1 model:

```text
train examples/sec
self-play positions/sec
inference latency p50/p95
GPU memory
policy CE/KL
value NLL/calibration
arena Elo / winrate
classical survival score
classical winrate
mcts_top1_outside_window_frac
mcts_top8_outside_window_frac
target_policy_mass_outside_window
winning_move_outside_window
forced_block_outside_window
hot_cell_outside_window
fallback_prior_use
fallback_prior_use_on_mcts_topk
value_error_by_board_span
policy_error_by_board_span
winrate_by_game_length
cluster_count
max_cluster_distance
```

For the action-contract scout, also track:

```text
candidate_recall_mcts_top1
candidate_recall_mcts_top4
candidate_recall_mcts_top8
candidate_recall_winning_move
candidate_recall_forced_block
candidate_recall_open_four
candidate_recall_open_five
candidate_recall_two_placement_cover
missing_target_policy_mass
sparse_policy_ce
sparse_policy_kl
sparse_vs_dense_disagreement
fallback_prior_use_after_sparse_mix
```

## Promotion Gates

The action-contract scout can feed Phase 2 only if it clears:

```text
candidate_recall_mcts_top1 >= 98%
candidate_recall_mcts_top8 >= 95%
candidate_recall_winning_move >= 99.5%
candidate_recall_forced_block >= 99.5%
candidate_recall_two_placement_cover >= 99%
missing_target_policy_mass <= 1%
fallback_prior_use_on_mcts_topk decreases materially
no illegal/crash regressions
```

ResTNet promotes if:

```text
positive score vs matched current model
or better value calibration with acceptable throughput
or useful teacher performance for Phase 2 offline training
```

The fixed-window baseline promotes regardless, because Phase 2 and Phase 3 need
a fair control.

## Phase 1 Run Order

1. Add the window/action instrumentation to the existing baseline.
2. Run `base_128x16` and `base_160x20` enough to choose `best_current_33`.
3. Run `rest_128_2`, `rest_128_3`, `rest_128_3_rel`, and `rest_160_3`.
4. Add `policy_target_v2` and candidate recall reporting.
5. Train `candidate_policy_33` offline from the same replay pool.
6. If recall gates pass, test sparse-prior mixing in MCTS.
7. Run a final Phase 1 comparison:

```text
best_current_33
best_restnet_33
candidate_policy_33
best_restnet_33 + candidate_policy, only if the individual parts were stable
```

## Outputs To Phase 2

Phase 2 requires:

- best current crop checkpoint and config;
- best ResTNet crop checkpoint and config;
- `policy_target_v2` schema;
- candidate-set builder and recall report;
- MCTS action-keyed prior gather path;
- outside-window failure map;
- fallback-prior report;
- value/policy error by board span;
- throughput numbers for current, ResTNet, and sparse-prior paths.

## Outputs To Phase 3

Phase 3 tunes only survivors. It consumes:

- finalist model families from Phase 1;
- safe ranges for sparse-policy loss and candidate budget;
- ResTNet depth/placement decision;
- MCTS sims/exploration settings that were stable;
- scorecard baselines for classical survival, checkpoint league, and
  outside-window robustness.

## Bottom Line

Phase 1 is not trying to solve the infinite board yet. It is building the runway:

```text
best crop baseline
best attention-inside-crop baseline
action-keyed targets and priors that can survive contact with MCTS
```

That gives Phase 2 a clean target: replace the crop input while keeping the new
global action contract. It gives Phase 3 a clean search space: tune the best
fixed-window, ResTNet, and transformer survivors instead of tuning everything.
