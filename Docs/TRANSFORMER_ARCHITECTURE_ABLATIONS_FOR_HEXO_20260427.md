# Phase 2 Plan: Transformer Input And `33x33` Window Replacement - 2026-04-27

This is the second document in the consolidated three-phase Hexo improvement
plan:

1. `Docs/PHASE1_RESTNET_ACTION_CONTRACT_SCOUT_20260427.md`
2. `Docs/TRANSFORMER_ARCHITECTURE_ABLATIONS_FOR_HEXO_20260427.md`
3. `Docs/AUTOTUNING_METHODS_AND_48H_PLAN_20260427.md`

Phase 2 should start only after Phase 1 has produced:

- `best_current_33`;
- `best_restnet_33`;
- global `policy_target_v2`;
- candidate recall reports;
- action-keyed MCTS prior gather;
- outside-window failure metrics.

The goal is to replace the `33x33` centroid crop with an infinite-board
representation while preserving a fair comparison against the best crop model
and the best ResTNet crop model.

## Executive Decision

The preferred replacement is:

```text
Option 1: global sparse Hex graph Transformer
```

It represents stones, legal moves, hot windows, cover constraints, and pair
actions as tokens on the infinite axial-coordinate board. Its policy is keyed by
global legal `(q,r)` actions, not by crop-relative indices.

The final Phase 2 comparison is intentionally narrow:

| System | Source | Input contract | Policy contract | Purpose |
|---|---|---|---|---|
| A. `best_current_33` | Phase 1 | `33x33` crop | flat `0..1088` logits | crop baseline |
| B. `best_restnet_33` | Phase 1 | `33x33` crop + attention | flat `0..1088` logits | attention-inside-crop control |
| C. `global_graph_option1` | Phase 2 target | sparse global tokens | legal `(q,r)` logits | true window replacement |

This answers the question cleanly:

```text
Does a true infinite-board action/token model beat both the best current crop
and the best attention-enhanced crop?
```

## Why A Transformer Can Beat The Window

The current crop has two separate limitations:

```text
input limitation:
  evidence outside the crop is missing

policy limitation:
  outside-crop legal actions fall back to weak/default priors
```

ResTNet only addresses the first limitation inside the crop. It can mix distant
cells within `33x33`, but it cannot directly represent a forced block outside
the crop or assign a real prior to an outside-crop legal move.

The graph Transformer addresses both:

```text
state:
  every important global object can become a token

policy:
  every selected legal action has its own global action identity
```

For Hexo, this matters because the decisive situations are often structural:

- 4-windows and 5-windows are both one-turn wins;
- defense is a two-placement set-cover problem;
- separated clusters can become coupled by hot-window threats;
- a legal action far from the centroid can be the only forced block;
- pair quality matters because each turn is two placements.

## Required Phase 1 Dependencies

Phase 2 should not implement its own disconnected target system. It should reuse
the Phase 1 action-contract work:

```text
policy_target_v2:
  [(q, r, prob), ...]

opp_policy_target_v2:
  [(q, r, prob), ...]

pair_policy_target_v2:
  [((q1, r1), (q2, r2), prob), ...]

candidate builder:
  critical moves override caps

MCTS prior gather:
  legal action `(q,r)` -> action token logit
```

Promotion gates inherited from Phase 1:

```text
candidate_recall_mcts_top1 >= 98%
candidate_recall_mcts_top8 >= 95%
candidate_recall_winning_move >= 99.5%
candidate_recall_forced_block >= 99.5%
missing_target_policy_mass <= 1%
fallback_prior_use_on_decisive_actions near zero
```

If these fail, fix Phase 1 before training the graph model. A transformer cannot
recover from a candidate builder that silently drops wins or forced blocks.

## Primary Architecture: Sparse Hex Graph Transformer

### Data Flow

```text
Hexo state
-> deterministic global token builder
-> relation/edge builder
-> typed graph Transformer
-> action-keyed policy heads
-> value and auxiliary heads
-> MCTS consumes priors by global `(q,r)`
```

The architecture should be translation-aware rather than absolute-board aware.
There are no corners or edges. Coordinates should be relative to meaningful
anchors:

- centroid;
- last move;
- current candidate;
- local component center;
- current hot-window center.

### Token Families

| Token | Required | Purpose |
|---|---|---|
| `STATE` | yes | global value aggregation |
| `TURN` | yes | opening, first placement, second placement |
| `PLAYER` | yes | side-to-move/color context |
| `STONE` | yes | full board memory |
| `LEGAL` | yes | action dictionary |
| `HOT_CELL` | yes | wins, blocks, forcing cells |
| `WINDOW6` | yes | explicit 6-in-row structure |
| `LINE` | yes | axis pressure and run/gap structure |
| `COVER_SET` | yes | two-placement defensive constraints |
| `COMPONENT` | optional early, useful later | separated cluster summaries |
| `PAIR_ACTION` | optional early, main final candidate | first/second placement planning |

The first viable model can omit `COMPONENT` and `PAIR_ACTION`. The expected
best model should include both once the basic graph path is stable.

### Token Budget Ladder

Use a ladder, not one giant jump:

| ID | Budget | Tokens | Pair policy | Purpose |
|---|---:|---|---|---|
| `graph256_cells` | `256` | `STATE`, `TURN`, `STONE`, `LEGAL`, `HOT_CELL` | no | minimal action-keyed model |
| `graph384_windows` | `384` | + `WINDOW6`, `LINE` | no | tests explicit win-condition tokens |
| `graph512_cover` | `512` | + `COVER_SET` | no | tests defensive set-cover modeling |
| `graph512_turn` | `512` | + `PAIR_ACTION` | yes | main candidate |
| `graph768_champion` | `768` | all above, higher caps | yes | final if throughput allows |

Expected winner:

```text
graph512_turn
```

The `graph768_champion` is useful only if `graph512_turn` is clearly capacity
limited and batching remains acceptable.

## Token Details

### `STONE`

Represent every stone unless the game is extremely long and memory proves
impossible. Hexo has no captures; old stones remain strategically relevant as
line blockers and threat anchors.

Features:

```text
owner
age / move index bucket
relative q,r,s to centroid
relative q,r,s to last move
component id bucket
axis-line participation counts
near-hot-window flag
```

### `LEGAL`

These are the policy actions. Each selected legal action must keep its global
`(q,r)` identity through batching, loss computation, and MCTS prior gather.

Features:

```text
relative coordinates
distance to nearest own stone
distance to nearest opponent stone
distance to last move
inside old 33x33 crop flag
old dense-policy prior if available
creates 5-window / 4-window / 3-window
blocks opponent 5-window / 4-window
cover-set membership
candidate source bits
```

Critical rule:

```text
Every legal action that is an immediate win, forced block, target-policy action,
or required cover-set action must be included even if the candidate budget is
full.
```

### `WINDOW6`

`WINDOW6` tokens explicitly encode Hexo's win condition. They reduce the burden
on attention to rediscover all 6-cell patterns from raw stones.

Features:

```text
axis
center coordinate
own_count
opp_count
empty_count
is_dead
is_hot_for_current
is_hot_for_opponent
empty cell references
```

These tokens should connect to:

- stones inside the window;
- legal cells inside the window;
- cover-set tokens created from the window;
- line tokens on the same axis.

### `COVER_SET`

Defense in Hexo is not only "block this cell." Because a player has two
placements, the model must learn whether two cells can cover all opponent hot
windows.

Features:

```text
threat_count
minimal_cover_size
number_of_valid_block_pairs
is_unblockable_by_two
candidate cells in cover
axis diversity
```

This should be one of the main reasons the graph model can beat the crop model:
the crop sees local patterns, while `COVER_SET` tokens expose the real defensive
constraint graph.

### `PAIR_ACTION`

Pair tokens are a controlled way to represent full-turn planning without making
MCTS branch over all legal pairs immediately.

Features:

```text
first action id
second action id
same_axis / different_axis
creates immediate win
blocks all opponent threats
block_plus_counterattack
pair distance bucket
joint target probability if available
```

Start with pair policy as an auxiliary head and prior-shaping signal. Only test
pair macro expansion after place-level MCTS is stable.

## Relation Biases

Use Graphormer-style relation bias or equivalent typed attention bias. The
model should know which tokens are geometrically or tactically related.

Recommended relations:

```text
hex_distance_bucket
direction_bucket
same_axis_q / same_axis_r / same_axis_s
same_window6
stone_in_window
legal_in_window
legal_completes_window
legal_blocks_window
legal_in_cover_set
pair_covers_threat_set
stone_in_component
legal_near_component
recent_move_relation
first_second_pair_relation
mutual_exclusion
D6_orbit_relation
```

D6 symmetry matters because the hex grid has 12 orientation/reflection
symmetries. Axis-related features must rotate/reflect with the board.

## Model Heads

### Policy Heads

Use action-keyed heads:

```text
policy_place:
  logits over LEGAL tokens

policy_pair_first:
  first placement logits for full-turn planning

policy_pair_second:
  conditional second placement logits after first placement

policy_pair_joint:
  logits over selected PAIR_ACTION tokens
```

Initial policy loss:

```text
policy_place = 1.0
pair_first = 0.10
pair_second = 0.10
pair_joint = 0.05
```

Raise pair-policy weight only if it improves MCTS move ordering without search
collapse.

### Value And Auxiliary Heads

Keep the heads aligned with Hexo's actual structure:

```text
value_binned
lookahead_4 / lookahead_12 / lookahead_36
opp_policy over LEGAL tokens
moves_left normalized or binned
win_now
opp_win_next
threat_count_delta
cover_set_status
axis pressure
regret_rank / regret_value
```

Do not tune every auxiliary independently in Phase 2. Use fixed low weights,
then let Phase 3 tune multipliers for survivors.

## ResTNet Integration

Phase 2 should compare against ResTNet, not necessarily embed ResTNet inside
the graph model on day one.

Use ResTNet in three ways:

1. `best_restnet_33` is the fixed-window control.
2. Its dense policy/value outputs can be teacher features for `LEGAL` tokens
   that fall inside the old crop.
3. Its checkpoint can initialize or distill the graph model's early trunk if a
   crop-to-token adapter is easy.

Recommended first graph runs:

```text
graph model trained from replay targets
old dense-policy prior used only as optional LEGAL-token feature
no hard dependence on ResTNet outputs at inference
```

This avoids building a graph model that only works when the crop model already
knows the answer.

## Training Plan

### Stage 1: Offline Replay Training

Train graph variants from the same replay pool used by Phase 1 finalists.

Purpose:

```text
validate batching
validate policy_target_v2
validate candidate recall
compare policy/value quality without self-play feedback loops
```

Promote only if:

```text
candidate recall gates pass
policy CE/KL is competitive with crop models on in-crop targets
outside-window policy mass is learned instead of dropped
value calibration is not worse by a large margin
throughput is within a plausible tuning range
```

### Stage 2: Shadow MCTS Priors

Run self-play where the graph model computes priors, but decisions are audited
against the crop model and classical tactical checks.

Track:

```text
graph_prior_rank_for_best_mcts_action
graph_prior_rank_for_forced_block
graph_prior_rank_for_win_now
fallback_prior_use
candidate_missing_decisive_action
search_entropy
illegal/crash rate
```

Do not promote if forced blocks are missing or pair priors create unstable
first-placement choices.

### Stage 3: Place-Level Self-Play

Enable `policy_place` as the MCTS prior. Keep pair policy auxiliary at first.

Run:

```text
graph256_cells_place
graph384_windows_place
graph512_cover_place
graph512_turn_place
```

Promote based on:

- checkpoint league score;
- outside-window robustness;
- tactical motif score;
- forced-block recall;
- throughput-adjusted self-play score.

### Stage 4: Pair-Aware Priors

Only after place-level MCTS is stable:

```text
graph512_turn_pair_prior:
  use pair policy to shift first-placement priors and second-placement priors

graph512_turn_pair_macro_topk:
  optionally add a small macro branch over top pair tokens
```

Expected safest winner:

```text
graph512_turn_pair_prior
```

Pair macro expansion is a research bet. It can help full-turn planning, but it
can also over-prune if the pair head is still immature.

## Evaluation Scorecard

Use the same scorecard for all three finalists:

```text
0.35 * checkpoint league LCB
0.20 * outside-window robustness
0.15 * tactical motif / forced-block score
0.10 * value calibration
0.10 * policy CE/KL on full-search targets
0.10 * throughput-adjusted self-play score
```

Outside-window robustness:

```text
winning_move_outside_window recall
forced_block_outside_window recall
target_policy_mass_outside_window CE/KL
mcts_topk_outside_window prior quality
fallback_prior_use_on_decisive_actions
```

Tactical motif score:

```text
4-window completion
5-window completion
single forced block
two-placement cover
block plus counterattack
unblockable-threat recognition
axis fork creation
separated-cluster defense
```

Vetoes:

```text
illegal/crash regression
candidate recall below gate
winning-move recall regression
forced-block recall regression
pair policy search collapse
wall-clock loss too large without outside-window gain
```

## Ablation Matrix

Do not run a broad architecture search before the action contract is stable.
Run the ladder below.

### Phase 2A: Minimal Graph

```text
graph256_cells
```

Goal:

```text
prove action-keyed transformer priors can train and batch
```

Pass if:

```text
candidate recall gates pass
policy CE is not wildly worse than crop
MCTS can consume priors without instability
```

### Phase 2B: Win-Condition Tokens

```text
graph384_windows
```

Goal:

```text
prove WINDOW6/LINE tokens improve immediate tactics and axis pressure
```

Pass if it improves:

```text
win_now
opp_win_next
4-window and 5-window motifs
axis-fork motifs
```

### Phase 2C: Defensive Cover Tokens

```text
graph512_cover
```

Goal:

```text
prove COVER_SET tokens improve two-placement defense
```

Pass if it improves:

```text
forced_block
two-placement cover
unblockable-threat recognition
classical survival without delaying losses artificially
```

### Phase 2D: Full-Turn Pair Planning

```text
graph512_turn
graph512_turn_pair_prior
```

Goal:

```text
prove pair-aware priors improve first/second placement coordination
```

Pass if it improves:

```text
checkpoint league LCB
block_plus_counterattack motifs
pair-policy top-k recall
second-placement value after first move
```

### Phase 2E: Champion Capacity Check

```text
graph768_champion
```

Run only if:

```text
graph512_turn passes gates
throughput is not the bottleneck
the loss/arena curves suggest capacity limitation
```

## Final Phase 2 Match

Finalists:

```text
best_current_33
best_restnet_33
global_graph_option1
```

Run three views:

```text
sample-normalized:
  same training examples

wall-clock-normalized:
  same compute time

search-normalized:
  same MCTS sims or same approximate move-time budget
```

The graph model should replace the window only if it clears:

```text
positive score vs best_current_33
positive score vs best_restnet_33 or clear outside-window advantage
candidate recall gates met
fallback prior use on decisive actions near zero
better long-span/separated-cluster behavior
no forced-block or winning-move regression
throughput acceptable enough for Phase 3 tuning
```

If the graph model wins only in outside-window buckets but loses short tactics,
do not discard it immediately. Add/strengthen `WINDOW6`, `COVER_SET`, and
distillation from `best_restnet_33`, then retry the `graph384_windows` and
`graph512_cover` steps.

If it loses everywhere, keep the Phase 1 candidate-policy path and proceed to
Phase 3 with fixed-window finalists.

## Outputs To Phase 3

Phase 3 consumes:

- the promoted graph config, if any;
- candidate budget and recall limits;
- pair-policy stability report;
- graph throughput and memory numbers;
- scorecard for graph vs current vs ResTNet;
- safe loss-weight ranges for graph auxiliary heads;
- MCTS priors/exploration settings that remained stable.

## Preferred Options

The top three options, in order:

1. `graph512_turn_pair_prior`: best chance to beat `33x33` because it combines
   global actions, win-condition tokens, cover constraints, and two-placement
   prior shaping.
2. `graph512_cover`: best lower-risk transformer if pair policy is unstable.
3. `best_restnet_33 + candidate_policy_33`: best fallback if the graph model is
   not ready; it still improves the action contract while preserving crop
   stability.

The favorite is `graph512_turn_pair_prior`, but it should earn that spot by
beating the crop and ResTNet controls on Hexo-specific failures, not by having a
more expressive architecture on paper.
