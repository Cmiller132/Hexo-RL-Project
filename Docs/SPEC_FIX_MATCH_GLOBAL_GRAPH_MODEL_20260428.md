# Spec-Fix Match Plan: True Global Sparse Graph Model

Date: 2026-04-28

This document defines what is required to make the implemented model match the
Phase 2 architecture spec in
`Docs/TRANSFORMER_ARCHITECTURE_ABLATIONS_FOR_HEXO_20260427.md`.

## Naming Correction

The current implementation is now named:

```text
graph_hybrid_0
```

That name is intentional. The model is a useful scout, but it is not the true
global sparse graph model. It keeps the old `(B,13,33,33)` dense crop contract,
selects important crop cells, runs sparse Transformer attention over those
selected cells, scatters the updated token features back into the crop map, and
uses the existing dense and candidate/action-keyed heads.

The target model from the Phase 2 spec should be named separately:

```text
global_graph_option1
```

`global_graph_option1` should replace the crop as the primary state and policy
contract. It should consume global sparse tokens and emit legal global `(q,r)`
policy logits.

Implementation order below is sequencing only. The finished Phase 2/3 graph
plan is not satisfied by `graph_hybrid_0`, a place-policy-only graph, a
diagnostic pair head, token names that are only score modifiers, or dense/crop
priors used as the primary search path. The finished target is
`global_graph768_champion` or a strictly stronger `global_graph_option1`
successor with the same complete contracts.

## Current Implementation Snapshot

Current `graph_hybrid_0`:

```text
encoded crop:          (B, 13, 33, 33)
local CNN trunk:       yes
sparse attention:      selected cells from inside the 33x33 crop
token families:        implicit cell type hints only
relation bias:         none
policy contract:       dense 1089 logits plus optional sparse candidate rows
pair policy:           auxiliary candidate-pair scorer
MCTS prior contract:   dense priors by default, optional sparse prior mix
D6 path:               transformed-history re-encode, valid for sparse rows
```

What it proves:

- Sparse token attention can run inside the training/inference loop.
- Action-keyed `(q,r)` sparse policy plumbing can be batched and trained.
- D6 augmentation can be made safe for candidate/action-keyed models.
- Candidate recall and sparse-prior telemetry can be measured.

What it does not prove:

- That an infinite-board representation beats the crop.
- That explicit `WINDOW6`, `LINE`, `COVER_SET`, or `PAIR_ACTION` tokens help.
- That relation-biased graph attention improves Hexo tactics.
- That policy over global legal tokens is stable enough to replace dense policy.

## Target Contract

`global_graph_option1` should have this contract:

```text
Hexo state / compact move history
-> deterministic global token builder
-> relation/edge bias builder
-> typed sparse graph Transformer
-> legal-token policy heads
-> value and auxiliary heads
-> MCTS consumes priors by global (q,r)
```

The dense crop may remain as a debug/distillation side input during transition,
but it must not be the primary state representation for the target model.

## Required Data Objects

Add a graph batch data contract under `Python/src/hexorl/action_contract/` or a
new `Python/src/hexorl/graph/` package.

Required tensors:

```text
token_features:       (B, T, F)
token_type:           (B, T)
token_qr:             (B, T, 2)
token_mask:           (B, T)
legal_token_indices:  (B, A)
legal_qr:             (B, A, 2)
legal_mask:           (B, A)
pair_token_indices:   (B, P)
pair_first_indices:   (B, P)
pair_second_indices:  (B, P)
relation_bias:        (B, H or 1, T, T)
policy_target:        (B, A)
opp_policy_target:    (B, A)
pair_policy_target:   (B, P)
```

Keep padding explicit. Every padded action or token must have a mask. No loss or
prior gather should rely on sentinel logits alone.

## Token Families

Implement token builders as separate, testable modules. The builder should be
deterministic for a given state, config, and D6 transform.

### `STATE`

One or a few global aggregation tokens.

Features:

- current player;
- placement phase: opening, first placement, second placement;
- move count;
- legal count bucket;
- current threat summary counts;
- board centroid and active-span summary.

### `TURN`

Separate from `STATE` so attention can learn phase-specific behavior.

Features:

- opening move flag;
- first placement flag;
- second placement flag;
- placements remaining this turn;
- whether threat filtering would force a subset for training/search.

### `PLAYER`

Player/perspective context tokens.

Features:

- current player;
- opponent player;
- color/perspective bit;
- own/opponent stone counts;
- own/opponent threat summaries.

### `STONE`

Represent every stone unless token budget is exceeded by extremely long games.
Because Hexo has no captures, old stones remain strategically relevant as
blockers, anchors, and line extenders.

Features:

- owner relative to current player;
- age bucket;
- move index bucket;
- axial `(q,r,s)` relative to centroid;
- axial `(q,r,s)` relative to last move;
- axis-line participation counts;
- component id bucket;
- near-hot-window flag.

### `LEGAL`

These are the primary policy actions. Every selected legal token must preserve
global `(q,r)` identity through batching, loss, D6, checkpoint inference, and
MCTS prior gather.

Features:

- global axial coordinate relative to centroid and last move;
- distance to nearest own stone;
- distance to nearest opponent stone;
- inside old `33x33` crop flag for diagnostics only;
- old dense prior or teacher prior for diagnostics/distillation only;
- creates current-player 3/4/5/6-window indicators;
- blocks opponent 3/4/5/6-window indicators;
- cover-set membership bits/counts;
- candidate-source bits;
- target-policy mass if this token is in a replay target.

Critical inclusion rule:

```text
immediate wins, forced blocks, target-policy actions, and required cover-set
actions must be included even if the normal candidate budget is full.
```

### `HOT_CELL`

Cells that participate in immediate wins, forced blocks, or high-pressure
windows. These may overlap with `LEGAL`, but keeping an explicit token type
helps the relation builder and diagnostics.

### `WINDOW6`

Explicit 6-cell win-condition tokens.

Features:

- axis;
- center coordinate;
- current-player stone count;
- opponent stone count;
- empty count;
- dead/alive flag;
- hot-for-current flag;
- hot-for-opponent flag;
- references to empty legal cells in the window.

These tokens should connect to stones and legal cells inside the same window and
to line tokens on the same axis.

### `LINE`

Axis-level line summaries that make long-range structure easier than forcing
attention to rediscover it from individual stones.

Features:

- axis id;
- line coordinate;
- current-player run/gap statistics;
- opponent run/gap statistics;
- maximum live window strength for each player;
- number of active windows on this line;
- number of legal cells that improve multi-axis strength.

### `COVER_SET`

Tokens for two-placement defensive constraints.

Features:

- player being defended against;
- threat window id(s);
- required legal cells;
- minimum stones needed to cover all immediate threats;
- whether one placement, two placements, or impossible defense is required;
- legal-token references included in the cover.

`COVER_SET` is important because Hexo defense is not just one best block. With
two-placement turns, the correct defensive policy often needs a set-cover view.

### `COMPONENT`

Required for the finished model. Summarize separated clusters so the model can
reason about long-span threats without needing all pairwise stone relations.

### `PAIR_ACTION`

The final graph candidate should include pair-action tokens.

Features:

- first legal token reference;
- second legal token reference;
- unordered-pair canonical id;
- same-axis flag;
- intersects active windows count;
- covers one threat set flag;
- covers multiple threat sets flag;
- own strength delta;
- opponent-block delta;
- pair distance bucket;
- duplicate/illegal/self-overlap mask.

## Relation Bias Builder

The current hybrid has no relation/edge bias. The true model needs typed
attention bias, preferably Graphormer-style additive bias per attention head.

Required relations:

```text
distance_bucket
direction_bucket
same_axis
same_line
same_window6
stone_in_window6
legal_in_window6
legal_in_cover_set
window6_to_cover_set
line_to_window6
legal_to_pair_action
pair_covers_threat_set
same_component
age_order_bucket
recent_move_relation
first_second_pair_relation
D6_orbit_relation
```

Implementation notes:

- Start with a compact relation id tensor `(B,T,T)` and an embedding table
  projected to `(heads,T,T)`.
- Keep a required float bias channel for clipped distance and strength
  quantities.
- Build relations on CPU in the replay/data-loader path first; only optimize
  after correctness tests pass.
- Mask padded tokens at attention time and mask padded action tokens in all
  losses.

## Network Architecture

Add a new class rather than extending the crop hybrid until it is tangled:

```text
GlobalHexGraphNet
```

Required finished version:

```text
token input projection
+ type embedding
+ coordinate/anchor embedding
+ phase/player embeddings
+ relation-biased Transformer blocks
+ state-token pooling
+ legal-token policy heads
+ pair-token policy heads
+ value/aux heads from pooled state and selected tactical tokens
```

Keep `HexNet` and `graph_hybrid_0` intact for baselines. The new class should
have a separate `build_global_graph_model_from_config()` path or a clean branch
inside `build_model_from_config()` keyed by:

```toml
[model]
architecture = "global_graph_option1"
```

Do not overload `graph_hybrid_0`.

## Policy Heads

Implement the policy heads from the architecture spec.

### `policy_place`

Logits over `LEGAL` tokens:

```text
(B, A)
```

This is the minimum viable true graph policy. It replaces flat `0..1088`
policy as the primary policy contract.

### `policy_pair_first`

Logits over legal first placements:

```text
(B, A)
```

Useful for shaping first-placement choices on two-placement turns.

### `policy_pair_second`

Conditioned second-placement scorer over legal second placements given the
selected first placement. It may be implemented with selected `PAIR_ACTION`
tokens, but the contract must be equivalent to a masked legal conditional
distribution.

### `policy_pair_joint`

Logits over `PAIR_ACTION` tokens:

```text
(B, P)
```

This is the main pair-turn prior. The current crop-compatible `PairPolicyHead`
is only an auxiliary scorer over candidate pairs and should not be treated as
this final head.

## Value And Auxiliary Heads

Keep the heads already fixed in the replay target work, but adapt their input
source to pooled graph state:

- binned `value`;
- binned EMA lookahead values;
- `regret_rank`;
- binned `regret_value`;
- `moves_left`;
- perspective-indexed dual axis strength and delta-norm axis targets.

Add graph-native tactical heads with clean engine-derived labels:

- `win_now`;
- `opp_win_next`;
- `threat_count_delta`;
- `cover_set_status`;
- `legal_token_quality` as a diagnostic head whose label source is explicitly
  documented and tested.

Do not add heads that duplicate noisy dashboard experiments unless the target is
stable and documented.

## Replay And D6

D6 augmentation is critical and must be graph-native.

Rules:

- Store compact move history and global `(q,r)` targets.
- Apply the sampled D6 transform to the move history and all global target keys.
- Rebuild graph tokens from the transformed state.
- Rebuild relation bias from transformed tokens.
- Rebuild legal/action/pair target rows by matching transformed global keys.

Never rotate padded token tensors directly as the source of truth. Re-encoding
from transformed history is slower but much safer.

Required D6 tests:

- every token type transforms equivariantly;
- every legal target key survives the transform;
- every immediate win and forced block remains included after transform;
- `WINDOW6` axes remap correctly under all six symmetries;
- `COVER_SET` contents remap correctly under all six symmetries;
- pair-action canonicalization is stable after transform;
- sparse policy loss is finite for every architecture using graph batches.

## MCTS Prior Integration

MCTS should consume graph priors by global action key:

```text
legal (q,r) -> logit/prob
```

Implementation steps:

1. Add inference output metadata containing `legal_qr` and `policy_place` logits.
2. Convert logits to priors only over Rust legal actions.
3. If a legal action is missing from graph candidates, use an explicit measured
   fallback prior and log it.
4. Fail tactical tests if missing candidates include immediate wins, forced
   blocks, target-policy actions, or required cover-set actions.
5. Keep dense crop prior mixing disabled by default for `global_graph`.

## Configuration Plan

Use separate architecture names:

```toml
architecture = "graph_hybrid_0"  # current crop-compatible scout
architecture = "global_graph_option1"  # true spec-match model
```

Suggested global graph configs:

```text
global_graph256_cells
global_graph384_windows
global_graph512_cover
global_graph512_turn
global_graph768_champion
```

Avoid reusing `graph512_turn_pair_prior` for both current and future models.
That name currently means a token-selection preset in the hybrid, not a real
pair-action graph schema.

## Migration Plan

### Step 1: Freeze Hybrid Semantics

- Keep `graph_hybrid_0` as a baseline.
- Keep old `architecture="graph"` accepted as an alias only.
- Update dashboards/docs so new trials use `graph_hybrid_0`.
- Do not add more target-spec behavior to the hybrid except bug fixes.

### Step 2: Build Complete Graph State Extractor

- Add a deterministic graph state builder from compact move history.
- Include `STATE`, `TURN`, `PLAYER`, `STONE`, `LEGAL`, `HOT_CELL`, `WINDOW6`,
  `LINE`, `COVER_SET`, `COMPONENT`, and `PAIR_ACTION`.
- Build candidate recall tests before model training.
- Save token debug payloads for dashboard inspection.

### Step 3: Add Relation Bias

- Implement relation id/bias construction.
- Add tests for window membership, axis relations, cover membership, distance
  buckets, and D6 remapping.
- Add a tiny `GlobalHexGraphNet` smoke test with masked attention.

### Step 4: Train Full Graph Heads Offline

- Train the full `global_graph_option1` head bundle on replay:
  - `policy_place`;
  - `policy_pair_first`;
  - `policy_pair_second`;
  - `policy_pair_joint`;
  - value, lookahead, moves-left, opponent policy, axis, tactical, and regret
    heads.
- Compare against dense crop and `graph_hybrid_0` on target reconstruction,
  legal recall, tactical recall, pair recall, value calibration, and throughput
  before self-play.

### Step 5: Validate Cover And Tactical Tokens

- Validate `COVER_SET` tokens and cover-status aux labels.
- Run forced-block and multi-threat suites.
- Require no missing decisive candidates.

### Step 6: Validate Pair Policy

- Validate `PAIR_ACTION` tokens.
- Validate `policy_pair_first`, `policy_pair_second`, and
  `policy_pair_joint`.
- Require pair candidate recall, finite pair loss, D6 pair equivariance, and
  legal-pair guards before self-play.

### Step 7: Self-Play Integration

- Add a graph inference path that returns legal keyed priors.
- Use graph priors as the active MCTS prior path for `global_graph_option1`.
- Use pair policy as the active two-placement prior path.
- Keep dense/crop priors only as measured diagnostics or emergency fallback
  counters, not as the primary graph policy.
- Run sample-normalized, wall-clock-normalized, and search-normalized matches
  against `best_current_33`, `best_restnet_33`, and `graph_hybrid_0`.

## Acceptance Tests

The true implementation is not complete until all of these pass:

- graph token builder reconstructs all stones and legal actions from compact
  history;
- every immediate win and forced block is included under token-budget pressure;
- D6 transform/re-encode preserves legal targets and pair targets;
- relation bias is correct for same-axis, same-window, cover-set, and pair
  relations;
- model forward works with padded token/action batches;
- policy loss masks padded actions and only trains valid legal tokens;
- MCTS prior gather consumes global `(q,r)` logits with measured fallback use;
- dashboard can inspect token families, relations, and policy logits;
- tactical suites pass for win-now, forced-block, cover-set, separated-cluster,
  and outside-window cases;
- benchmarks report throughput, memory, and fallback-prior use separately from
  `graph_hybrid_0`.
- pair policy actively shapes two-placement MCTS decisions and reports pair
  prior-source telemetry;
- PB2/Phase 3 can tune `global_graph_option1` as a first-class family using the
  same league, tactical, outside-window, candidate, and throughput evaluators as
  every other survivor.

## Non-Goals

- Do not add legacy Hexagon checkpoint compatibility to the graph path.
- Do not keep dense `1089` policy as the primary `global_graph_option1` policy.
- Do not make `WINDOW6`/`COVER_SET` names mere score modifiers.
- Do not rely on dashboard-only axis experiments as training targets until they
  are separately validated.
- Do not hide missing legal actions behind silent fallback priors.

## Finished Implementation Target

The spec-matching target is:

```text
global_graph768_champion
```

Required properties:

- removes the crop as the primary state and policy contract;
- includes all required token families;
- includes relation-biased global attention;
- trains place, pair, value, lookahead, opponent-policy, axis, tactical,
  moves-left, and regret heads;
- consumes global keyed priors in MCTS;
- consumes pair priors on two-placement turns;
- passes D6, tactical, outside-window, pair, dashboard, and Phase 3 evaluator
  gates.

Smaller configs are allowed only as implementation tests:

```text
global_graph256_cells
global_graph384_windows
global_graph512_cover
global_graph512_turn
```

They are not acceptance targets for the finished P2/P3 plan. The finished plan
requires `global_graph768_champion` or a strictly stronger successor with the
same complete contracts.
