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

## 2026-04-28 No-Compromise Audit Update

This spec has been tightened after auditing it against `Docs/game.md` and the
current implementation.

New hard rules:

```text
legal policy rows are all Rust-legal moves, not a capped candidate subset
D6 means all 12 transforms, not only six rotations
opening and two-placement turn semantics are explicit policy-mask contracts
pair policy must handle legal first moves, conditional second moves, and joint
  turn-pair priors without synthetic single-policy products as the final target
tactical tokens and labels must come from the exact engine threat/block oracle
opponent policy uses its own future global target table, not the current legal
  row table unless the rows are explicitly matched by global key
graph inference has a first-class token/relation/action IPC contract
```

Current `graph_hybrid_0` is still useful as a scout, but none of these rules
are satisfied by a crop-selected token model or by sparse candidates that can
drop legal actions.

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
opp_legal_qr:         (B, A_opp, 2)
opp_legal_mask:       (B, A_opp)
opp_policy_target:    (B, A_opp)
pair_policy_target:   (B, P)
```

Keep padding explicit. Every padded action or token must have a mask. No loss or
prior gather should rely on sentinel logits alone.

Action-row rule:

```text
A is the number of all legal actions in the state, padded per batch.
It is not a candidate budget.
Every Rust-legal action must appear exactly once in legal_qr/legal_mask.
```

The future opponent turn can have a different legal set from the source
position, so opponent-policy targets must carry their own global-key table or a
sparse key/value target that is matched by `(q,r)`. Do not train
`opp_policy_target` over the source position's `A` rows unless those rows have
been explicitly rebuilt for the opponent target state.

## Token Budget Policy

The true global graph model should not have a semantic token cap that drops
legal policy actions. Letting the model run slower as games get longer is
acceptable for correctness; silent action loss is not.

Required policy:

```text
all legal action rows are preserved
all stones are preserved for normal configured max-game lengths
all exact win/block/cover tactical structures are preserved
context summarization may reduce redundant WINDOW6/LINE/COMPONENT tokens only
  if exact legal-action policy rows and exact tactical labels remain intact
```

Why a token budget is still a performance concern:

```text
full self-attention is O(T^2)
pair-action tables can be O(A^2)
long games increase legal count and tactical-token count
```

Use engineering optimizations rather than semantic truncation:

- microbatch long positions;
- bucket batches by token/action count;
- score legal actions with cross-attention from state/context tokens if full
  legal-action self-attention is too slow;
- use sparse/block/local relation attention for stone/window context tokens;
- chunk pair-action scoring;
- record throughput by token count and legal count.

If a configured max-game length would exceed memory, fail the run with a clear
capacity error or lower the training game-length config. Do not silently drop
legal actions, stones, or decisive tactical rows to fit a fixed tensor width.

## Architecture Alternatives To Test

The strict graph/action data contract above should be shared by multiple model
families. That lets Phase 3 compare architectural ideas without confounding the
result with different target semantics, legal-action coverage, D6 transforms,
or tactical-oracle quality.

Every alternative below must obey:

```text
all legal action rows are preserved
all 12 D6 transforms are supported
opening and two-placement masks are correct
exact engine tactical labels are used
opponent policy uses its own future global target table
pair targets are legal and turn-phase aware
MCTS consumes global keyed priors
dashboard/replay can inspect the same graph data contract
```

### `global_xattn_0`

Purpose:

```text
Test whether all-legal global action identity is the main win.
```

Design:

- Build context tokens from `STATE`, `TURN`, `PLAYER`, `STONE`, `WINDOW6`,
  `LINE`, `COVER_SET`, and `COMPONENT`.
- Keep all `LEGAL` action rows.
- Score legal actions by cross-attention from legal-action queries into context
  tokens.
- Legal actions may use light self-attention or no full legal-to-legal
  self-attention in the first version.

Why test it:

- It preserves the correct infinite-board action contract.
- It avoids the worst `O(T^2)` pressure of full all-token attention.
- If it works, the most important missing ingredient was likely global legal
  action identity rather than a very rich relation graph.

Risks:

- May under-model legal-action interactions, especially fork geometry and
  pair-turn set-cover interactions.

### `global_line_window_0`

Purpose:

```text
Test whether explicit Hexo tactical structure beats generic graph attention.
```

Design:

- Emphasize `WINDOW6`, `LINE`, `COVER_SET`, and `HOT_CELL` tokens.
- Legal action embeddings are built from the tactical structures each action
  touches.
- Relation bias prioritizes same-window, same-line, cover-set, and multi-axis
  intersections.

Why test it:

- Hexo is won through window pressure, line strength, and unblockable
  multi-threat structure.
- This model should be easier to debug tactically than a fully generic graph.

Risks:

- If labels or tactical-token construction are wrong, the model can become
  brittle or overfit to hand-designed structure.

### `global_graph_full_0`

Purpose:

```text
Test the full relation-biased global graph design before champion scaling.
```

Design:

- Use all required token families.
- Use relation-biased Transformer blocks over the full graph context.
- Include legal-token, pair-token, value, lookahead, opponent-policy, axis,
  tactical, moves-left, and regret heads.

Why test it:

- This is the closest direct implementation of the maximal spec.
- If it wins after correctness gates, the rich graph design is justified.

Risks:

- Hardest to stabilize and profile.
- Failures can be harder to attribute because many mechanisms change together.

### `global_pair_twostage_0`

Purpose:

```text
Test whether two-placement strength needs an explicit pair planner without
materializing every pair through full graph attention.
```

Design:

- First stage scores every legal first placement.
- Second stage scores conditional legal second placements after a chosen or
  sampled first placement.
- A chunked joint-pair scorer evaluates high-priority first moves plus all
  legal conditional seconds.

Why test it:

- Hexo's turn is two placements, but full `A^2` pair tables can be expensive.
- This tests a cheaper pair-aware route.

Risks:

- If the first-stage filter is too narrow, it can hide the best pair. The
  filter must be auditable and tactical-critical first moves must be protected.

### `global_hybrid_action_0`

Purpose:

```text
Bridge from the current crop models to the global action contract.
```

Design:

- Use a hex-masked CNN/ResTNet crop trunk only as a local feature extractor or
  distillation helper.
- Use global legal `(q,r)` action rows as the primary policy output.
- Keep dense `1089` logits diagnostic-only.

Why test it:

- Easier to train and compare against current crop baselines.
- Helps isolate whether global action identity alone fixes major failures.

Risks:

- Still carries crop-context limitations if the global action scorer depends
  too heavily on local crop features.

### Recommended Test Order

```text
1. global_xattn_0
2. global_line_window_0
3. global_pair_twostage_0
4. global_graph_full_0
5. global_graph768_champion, scaled from the best validated design
```

`global_hybrid_action_0` can run in parallel as a bridge/control if it is cheap
enough. It should not replace the true graph path unless it satisfies the same
all-legal global action and bug-isolation gates.

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

Represent every stone for every position admitted by the configured training
game length. Because Hexo has no captures, old stones remain strategically
relevant as blockers, anchors, and line extenders. For pathological debug
positions beyond the configured capacity, the builder should fail loudly or use
an explicitly named summarization/debug mode that is not accepted as the
production `global_graph_option1` path.

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

These are the primary policy actions. Every Rust-legal action must preserve
global `(q,r)` identity through batching, loss, D6, checkpoint inference, and
MCTS prior gather. `LEGAL` rows are an all-legal action table, not a heuristic
candidate set.

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
- optional teacher/debug prior values only if they are available at inference
  time or explicitly marked as distillation-only inputs.

Target-policy mass belongs in `policy_target`, not in `token_features`. It must
not leak into live inference features.

Critical inclusion rule:

```text
immediate wins, forced blocks, target-policy actions, required cover-set
actions, and every legal Rust action must be present in the legal table.
```

For context token families such as `WINDOW6`, `HOT_CELL`, and `COVER_SET`,
critical tactical rows also override any optional context-token compression.

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

The source of truth must be the Rust/engine threat oracle, including exact
winning-turn status, must-block status, unblockable status, blocking cells, and
blocking pairs. Crop hot planes are not sufficient for the finished model.

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

Turn-state contract:

```text
opening turn:
  no pair-action loss or pair prior; player 0 has exactly one placement at
  origin

first placement of a normal turn:
  policy_pair_first scores all currently legal first placements
  policy_pair_joint covers every distinct two-stone turn pair that can be
  legally realized by at least one sequential order

second placement of a normal turn:
  policy_pair_second scores the legal second-placement table after the first
  placement has been applied
```

Pair identity:

- final board-state pair identity is canonicalized as an unordered pair of
  distinct cells;
- reachability/order metadata records whether only one order is legal or both
  orders are legal;
- duplicate cells are always illegal;
- synthetic products of single-placement policies are acceptable only as a
  bootstrap/debug target, not the finished pair-policy target.

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
- `WINDOW6` axes remap correctly under all 12 D6 transforms;
- `COVER_SET` contents remap correctly under all 12 D6 transforms;
- pair-action canonicalization is stable after transform;
- relation ids and float relation biases remap correctly under all 12 D6
  transforms;
- sparse policy loss is finite for every architecture using graph batches.

## MCTS Prior Integration

MCTS should consume graph priors by global action key:

```text
legal (q,r) -> logit/prob
```

Implementation steps:

1. Add inference output metadata containing all `legal_qr` rows and
   `policy_place` logits.
2. Convert logits to priors only over Rust legal actions.
3. If a Rust legal action is missing from `legal_qr`, fail the graph inference
   contract for that position. This is a model/data bug, not a normal fallback.
4. Emergency fallback priors may exist only for crash containment and must be
   logged as failures. They are not part of the accepted
   `global_graph_option1` policy path.
5. Fail tactical tests if missing legal/context rows include immediate wins, forced
   blocks, target-policy actions, or required cover-set actions.
6. Keep dense crop prior mixing disabled by default for `global_graph`.

## Graph Inference Contract

The inference server needs a first-class graph path. The existing dense tensor
IPC plus optional candidate arrays is not enough.

Required request options:

```text
compact move history, letting the server build graph tokens
or prebuilt token/action/relation tensors with versioned feature metadata
```

Required response:

```text
legal_qr
policy_place_logits
policy_pair_first_logits when applicable
policy_pair_second_logits or pair-conditioned scorer metadata when applicable
policy_pair_joint logits over pair rows when applicable
value and enabled aux heads
token/action masks and graph schema version
```

Every graph batch must carry a schema version so checkpoints, replay, dashboard,
and MCTS agree on feature order, relation ids, token families, and pair
semantics.

## Configuration Plan

Use separate architecture names:

```toml
architecture = "graph_hybrid_0"  # current crop-compatible scout
architecture = "global_graph_option1"  # true spec-match model
```

Suggested global graph configs:

```text
global_xattn_0
global_line_window_0
global_pair_twostage_0
global_graph_full_0
global_hybrid_action_0
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

- Train each graph alternative on the same replay/data contract:
  - `global_xattn_0`;
  - `global_line_window_0`;
  - `global_pair_twostage_0`;
  - `global_graph_full_0`;
  - `global_hybrid_action_0`, if used as a bridge/control.
- Keep the head bundle comparable across variants where possible:
  - `policy_place`;
  - `policy_pair_first`;
  - `policy_pair_second`;
  - `policy_pair_joint`;
  - value, lookahead, moves-left, opponent policy, axis, tactical, and regret
    heads.
- Compare against dense crop and `graph_hybrid_0` on target reconstruction,
  legal recall, tactical recall, pair recall, value calibration, and throughput
  before self-play.
- Promote to self-play only after the offline contract tests and bug-isolation
  sentinels pass.

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
- no legal action is dropped under token-budget pressure;
- every immediate win and forced block is included under context-token pressure;
- D6 transform/re-encode preserves legal targets and pair targets under all 12
  transforms;
- relation bias is correct for same-axis, same-window, cover-set, and pair
  relations;
- model forward works with padded token/action batches;
- policy loss masks padded actions and only trains valid legal tokens;
- MCTS prior gather consumes all global `(q,r)` logits with zero normal fallback
  use;
- dashboard can inspect token families, relations, and policy logits;
- tactical suites pass for win-now, forced-block, cover-set, separated-cluster,
  and outside-window cases;
- benchmarks report throughput, memory, and fallback-prior use separately from
  `graph_hybrid_0`.
- pair policy actively shapes two-placement MCTS decisions and reports pair
  prior-source telemetry;
- opening positions have no pair loss and no pair prior;
- second-placement positions use the legal table after the first placement,
  not the turn-start table;
- opponent policy targets are matched by their own future global legal table;
- graph inference IPC/server tests cover token/relation/action request and
  response paths;
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
global_xattn_0
global_line_window_0
global_pair_twostage_0
global_graph_full_0
global_hybrid_action_0
global_graph256_cells
global_graph384_windows
global_graph512_cover
global_graph512_turn
```

They are not acceptance targets for the finished P2/P3 plan. The finished plan
requires `global_graph768_champion` or a strictly stronger successor with the
same complete contracts.
