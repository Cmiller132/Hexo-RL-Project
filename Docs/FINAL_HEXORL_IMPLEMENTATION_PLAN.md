# Final Hexo-RL Implementation Plan

## Purpose

This document chooses one implementation path from the two deep research
reports and turns it into an opinionated, testable plan for Hexo-RL.

The decision is:

Build Hexo-RL around exact-simulator AlphaZero-style self-play where the
post-opening searched action is an unordered two-placement pair. Use
autoregressive and tactical structure to propose candidate pairs, but make the
MCTS edge, backup, policy target, and runtime action contract the completed
pair. Use a named `gumbel_sequential_halving_v1` root operator, proposal-aware
PUCT with progressive widening below the root, a minimal global graph encoder
with capped active six-cell window objects, and a mathematically symmetric
biaffine pair reranker trained with candidate-aware targets and dense auxiliary
views.

Implementation staging is explicit:

```text
V1 mainline:
  sampled joint-pair MCTS
  terminal-only tactical source
  pair_candidate_selector_v1 first
  no overload, fork, shared-pivot, or bounded hot-window race oracle

V2 expansion:
  richer tactical proposal sources
  overloads, shared pivots, fork labels, bounded tactical races
```

The first implementation priority is not a rich tactical oracle. The first
implementation priority is a debuggable, source-tagged, pair-native candidate
selector. V1 tactical logic exists only to protect exact terminal facts:

```text
win this turn
prevent opponent win next turn
impossible to prevent opponent win next turn
```

This is not a plan for keeping every proposed path alive. The final system
should have one mainline pair-action runtime. Sequential afterstate search,
fixed-crop models, materialized pair tokens, raw uniform-over-all-pairs
exploration, and MuZero learned dynamics are ablation or diagnostic tools, not
the flagship design.

## Source Reports Weighed

Primary local inputs:

- `Docs/deep-research-report.md`
- `Docs/deep-research-report(1).md`
- `Docs/JOINT_PAIR_MCTS_RESEARCH_NOTE.md`
- `Docs/GLOBAL_GRAPH_PAIR_HEADS.md`
- `Docs/GLOBAL_GRAPH_MINIMAL_TOKEN_PLAN.md`
- `Docs/MODEL_ARCHITECTURE_MODULARIZATION_PLAN.md`

External research backing used by the reports and adopted here:

- Danihelka et al., "Policy improvement by planning with Gumbel"
  https://openreview.net/forum?id=bERaNdoegnO
- Hubert et al., "Learning and Planning in Complex Action Spaces"
  https://arxiv.org/abs/2104.06303
- Wu, "Accelerating Self-Play Learning in Go"
  https://arxiv.org/abs/1902.10565
- Zaheer et al., "Deep Sets"
  https://arxiv.org/abs/1703.06114
- Lee et al., "Set Transformer"
  https://arxiv.org/abs/1810.00825
- Vinyals et al., "Pointer Networks"
  https://arxiv.org/abs/1506.03134
- Dozat and Manning, "Deep Biaffine Attention for Neural Dependency Parsing"
  https://arxiv.org/abs/1611.01734

## Assignment Contract

### Goal

Implement a superhuman-strength research path for Hexo-RL whose runtime,
training data, model outputs, and evaluation all agree that the true
post-opening action is an unordered two-stone pair.

### Success Criteria

- Pair macro-actions are the main MCTS action after the opening move.
- The search never relies on exhaustive legal-pair enumeration.
- Full Rust-legal placement rows remain the semantic legal table for self-play
  and training; tactical logic may propose, boost, prune, solve, or label, but
  it must not silently delete legal rows.
- Candidate generation combines model proposal, conditional proposal, tactical
  proposal, and structured exploration quotas.
- Root candidate admission uses `gumbel_sequential_halving_v1`, including
  completed-Q target construction. Any replacement must be named, documented,
  tested against the same target-construction contract, and ablated before use.
- Interior nodes use proposal-aware PUCT over admitted pair children with
  progressive widening.
- Pair rows are canonical, unordered, hashable, phase-aware, and D6-validated.
- Pair outputs never affect MCTS by head-name presence alone; an explicit pair
  strategy owns runtime behavior.
- The model uses legal-cell and window objects, not materialized all-pair input
  tokens.
- The pair scorer contains a true pair interaction term, preferably low-rank
  biaffine or bilinear, plus explicit tactical pair features.
- Training targets include pruned pair posteriors, cell marginals, conditional
  second-cell targets, softened pair targets, tactical auxiliaries, and
  proposal metadata.
- Unsampled legal pairs are treated as unknown, not as negative examples.
- The final architecture is accepted only after it demonstrates a statistically
  meaningful equal-wall-clock win over a fair sequential afterstate DAG baseline
  on mature checkpoints.

### Constraints

- Keep Rust as the exact rules boundary for legality, transitions, terminal
  state, and tactical oracle labels.
- Do not use hard threat filtering as the default legal-row source for flagship
  neural self-play or training.
- Do not introduce learned dynamics as the mainline.
- Do not keep old and new runtime pair semantics alive together after cutover.
- Do not materialize all pair actions as graph input tokens.
- Do not claim pair-action training complete from raw sparse visit-count
  cross-entropy alone.
- Do not let direct pair-head checks in self-play decide search behavior.
- Do not declare a phase complete without code-backed, artifact-backed, or
  command-backed evidence.

### Required Evidence

Every implementation phase must produce:

- Runtime consumers changed.
- Files changed.
- Legacy paths deleted, disabled, or quarantined.
- Tests and commands run with exit status.
- Artifacts produced under the appropriate docs or run artifact directory.
- Performance and utilization evidence for search or model hot paths.
- Contract examples or schema docs when boundaries change.
- Schema-version and cross-language boundary evidence for Rust/Python replay,
  inference, and search contracts.
- An explicit statement that no skipped, deferred, or manual-only requirement is
  being claimed complete.

### Stop Rules

Stop before claiming completion if:

- Pair row identity cannot be validated across replay, inference, and search.
- Full legal-row identity cannot be recovered after tactical processing.
- A component cannot distinguish full legal rows from admitted candidate rows
  and tactical proposal/label rows.
- The engine cannot apply both placements atomically for a pair MCTS edge.
- Pair strategy behavior cannot be separated from raw head-name checks.
- Candidate metadata cannot store source, proposal weight, and search outcome.
- Candidate metadata cannot store all source contributions and a combined
  `beta` or inclusion estimate.
- D6 transforms cannot preserve legal-row and pair-row identity.
- Terminal pair semantics cannot be stated consistently between Rust rules,
  search expansion, and training targets.
- The equal-wall-clock sequential DAG baseline cannot be built or fairly
  measured.
- Performance regressions make pair search non-viable without a documented
  mitigation.

## Final Architecture Decision

### Chosen Mainline

The mainline is sampled joint-pair MCTS with a minimal global graph model and a
biaffine pair-action head.

Use this stack:

```text
Rust exact simulator
  -> full legal rows
  -> minimal terminal tactical facts: win-in-1 / lose-in-1
  -> graph/set encoder over STATE, TURN, STONE, LEGAL, WINDOW6
  -> legal-cell embeddings h_i
  -> cell_marginal_proposal head
  -> anchor_conditioned_completion head
  -> cheap symmetric pair-retrieval score
  -> pair_candidate_selector_v1
  -> rich symmetric biaffine joint-pair scorer
  -> source-tagged sampled pair candidate set
  -> gumbel_sequential_halving_v1 root admission
  -> proposal-aware PUCT plus progressive widening over pair children
  -> candidate-aware pair targets, marginals, conditionals, and completed-Q targets
```

The action selected by self-play and evaluation after the opening is always the
completed pair. Cell-marginal and completion outputs exist to make candidate
generation and learning efficient. They are new-model proposal and auxiliary
views, not legacy single-placement policy heads, and they are not allowed to
redefine the game as two independent search plies.

### Why This Wins

Hexo's decisive moves are often pair interactions: double blocks, paired
extensions, overload creation, shared pivots, and two-cell tactical covers. A
sequential afterstate tree can find some of those moves, but it assigns search
statistics to an artificial ordered decomposition. That creates two problems:
the first placement receives too much semantic weight, and `(a, b)` aliases with
`(b, a)` unless the implementation does extra transposition work.

A full exhaustive pair tree has the right semantics but the wrong complexity.
With `m` legal cells, there are `m * (m - 1) / 2` unordered pair actions.
Exhaustive expansion is a wall-clock trap.

The hybrid design is the middle that actually fits the game:

- Pair-level MCTS preserves the real move object.
- Sampling and widening avoid the quadratic branch explosion.
- Autoregressive proposal recovers the speed advantage of factorization.
- The biaffine reranker models pair synergy directly.
- Candidate-aware training prevents the sampled action set from being mistaken
  for the whole action space.

The Gumbel paper is directly relevant because it addresses policy improvement
when the search does not visit all root actions. The implementation must use a
real root policy-improvement operator, not just add Gumbel noise to a top-k
list. Sampled MuZero is directly relevant because it formalizes planning over
sampled action subsets. The useful lesson from Sampled MuZero is not learned
dynamics; it is proposal-aware planning and target construction in large action
spaces.

## Decisions Adopted From Each Report

### Adopt From `deep-research-report.md`

- Pair actions are first-class and unordered.
- Sequential afterstate search is not the main semantics.
- Candidate generation should be sampled and source-aware.
- Raw sparse pair visit counts are not enough.
- Policy target pruning is mandatory for forced or exploratory candidates.
- Unsampled legal pairs are unknown, not hard negatives.
- Persistent raw uniform-over-all-pairs exploration is a poor long-run design.
- A fair sequential baseline must use DAG/transposition handling.
- Near-terminal tactical oracle support is allowed only for exact tactical
  states.

### Adopt From `deep-research-report(1).md`

- The final plan should include explicit first experiments, success criteria,
  and failure criteria.
- D6 row-table validation and policy/value consistency tests must be part of
  acceptance.
- Candidate admission must protect against support collapse during bootstrap.
- Replay should be stratified by tactical density, game phase, search surprise,
  pair-space size, and uncertainty.
- A tactical subsearch/proof-style trigger is valuable for sharp hot-window
  races, but it belongs in the V2 expansion path and must not block
  `sampled_joint_pair_v1`.

### Deliberate Synthesis

The reports disagree most on uniform exploration. This plan chooses the
structured-exploration position from `deep-research-report.md`, with one narrow
addition:

Do not keep a large persistent uniform-over-all-pairs floor. Instead, keep a
small structured exploration floor that samples from meaningful sources:

- uniform first-cell anchors,
- diverse second-cell completions,
- uniform active windows,
- cover sets,
- distance buckets,
- terminal tactical facts,
- model-surprise states,
- and rare blind canary samples.

Blind canary samples are allowed only as a capped diagnostic source, not as a
major training distribution. They must be source-tagged and can be pruned from
policy targets unless search proves them useful. This keeps the support-collapse
protection from `deep-research-report(1).md` without wasting most of the
candidate budget on void pairs.

## Rejected Mainlines

### Rejected: Full Exhaustive Joint-Pair MCTS

This has the right semantics but expands too many actions. It is useful only as
a tiny-position oracle or an offline benchmark. It should not be the runtime
mainline.

### Rejected: Sequential Afterstate MCTS As Main Planner

Sequential search is a necessary baseline, but it is not the primary design.
It introduces order into an unordered move and can hide pair synergy behind
single-cell saliency. If implemented, it must be a DAG baseline with
transposition merging; otherwise the comparison is unfair.

### Rejected: MuZero Learned Dynamics

Hexo already has exact and cheap rules in Rust. Learned dynamics add model error
and engineering load while failing to address the actual bottleneck, which is
candidate control over a combinatorial action space.

### Rejected: Materialized Pair Tokens In The Main Encoder

The current minimal graph token plan is right: pair rows should reference LEGAL
tokens. Materializing pair actions as input tokens reintroduces an `O(A^2)`
attention path and should stay out of the main runtime.

### Rejected: Pair-Only Sparse Visit Training

Pair visit targets are primary but not sufficient. Without marginals,
conditionals, softened targets, tactical supervision, and proposal correction,
the model can collapse onto the small support it already sees.

### Rejected: Simple Symmetric MLP As The Final Pair Scorer

The current symmetric MLP pair scorer is a valid baseline. It is not the final
architecture. The final pair head should include an explicit biaffine or
low-rank bilinear interaction term because Hexo's policy quality depends on
relations between two cells, not just the quality of each cell independently.

### Rejected: Hard Threat Filtering As Main Legal Semantics

Threat filtering is valuable as tactical knowledge, but it must not be the
default legal-row source for flagship neural self-play or training. A row table
called `LEGAL` must mean full Rust-legal placement rows, not a threat-censored
subset. Hard threat-constrained rows can remain only as diagnostic baselines,
classical-search helpers, or explicitly recorded exact solver overrides.

The final row split is:

```text
LEGAL rows = full Rust-legal placement rows
ADMITTED_SINGLE rows = optional sampled single-cell candidate rows
ADMITTED_PAIR rows = sampled canonical pair candidate rows
TACTICAL rows = labels, proposals, solver facts, and pruning metadata
```

No model, target builder, inference adapter, or search component may have to
guess whether a `LEGAL` row table was threat-filtered.

### Rejected: Legal-Order First-N Pair Scoring As Candidate Selection

A pair selector must not generate normal runtime candidates by walking legal
rows in nested-loop order and stopping at `max_pairs`. That method is simple
and deterministic, but it is biased toward early legal-row anchors and can miss
important pairs whose cells occur later in the legal table.

Legal-order first-N pair scoring is allowed only as a unit-test baseline or
diagnostic smoke test. The final pair scorer may score a supplied pair row set.
It must not define that set by first-N legal-order enumeration in normal
self-play or evaluation.

## Runtime Design

### Pair Action Contract

After the opening single move, every search action is:

```text
PairAction = unordered canonical pair of two distinct legal cells
```

Required invariants:

- The two cells are distinct.
- Both cells are legal in the current full-turn state.
- The pair is stored in canonical order under a stable ordering rule.
- `(a, b)` and `(b, a)` map to the same key.
- The pair row records its legal-row references.
- The pair row records its phase: first-placement pair, known-first second
  placement, or diagnostic-only.
- The pair row carries row-table identity and hash metadata.
- D6 transformations preserve pair identity after canonicalization.
- The pair row carries a schema version shared across Rust/Python boundaries.
- Terminal-equivalence metadata is available when one cell wins regardless of
  the second cell.

Atomic placement semantics are explicit: the MCTS edge applies both stones as a
single macro-action, and terminal evaluation occurs after the full pair unless
the Rust rules contract explicitly says the turn stops after the first winning
placement. The chosen rule must be reflected identically in Rust transition
code, search backpropagation, replay targets, and tactical fixtures.

### Legal Row Contract

Full Rust-legal rows are the semantic legal table for self-play, replay,
training, inference, and search. Tactical logic cannot silently replace them
with a threat-constrained subset.

Required row families:

```text
LEGAL
ADMITTED_SINGLE
ADMITTED_PAIR
TACTICAL
```

`LEGAL` contains all full Rust-legal placement rows for the current state.
`ADMITTED_SINGLE` and `ADMITTED_PAIR` contain sampled candidate rows used by
proposal, search, and target construction. `TACTICAL` contains oracle facts,
proposal rows, solver facts, and target-pruning metadata. Tactical rows may
reference legal rows, but they are not themselves proof that other legal rows do
not exist.

Main self-play and training should use:

```text
selfplay.legal_row_mode = "full_rust_legal"
selfplay.tactical_mode = "proposal_and_solver"
```

Any existing hard threat filter must be renamed or quarantined as:

```text
diagnostic_threat_filter_v0
```

Exact solver overrides are allowed only when recorded explicitly:

```text
solver_override = true
solver_reason = "winning_turn" | "unblockable" | "unique_must_block" | ...
full_legal_count = ...
admitted_candidate_count = ...
policy_target_pruned = true
```

They must not masquerade as ordinary MCTS over a smaller legal table.

### Candidate Proposal Contract

Every candidate pair is an object, not just a pair of coordinates. It must
carry enough information to audit how it entered search and how it may affect
training:

```text
candidate_id
pair_key
first_legal_row_id
second_legal_row_id
row_table_schema_version
source_contributions[]
combined_beta_or_inclusion_estimate
forced_exploration_flag
terminal_exact_flag
terminal_equivalence_flag
target_prune_flag
admission_generation
root_or_interior
```

`source_contributions[]` records every source that produced the candidate, not
only the source that won deduplication:

```text
source_type
source_rank
source_weight
local_probability_or_score
quota_id
```

Deduplication combines source contributions into one canonical candidate and
computes a combined `beta` or inclusion estimate. That estimate is required for
proposal-aware priors and sampled training losses. A candidate with missing
source metadata cannot enter search.

### Pair Strategy Authority

Create or promote two explicit runtime components:

```text
pair_candidate_selector_v1
sampled_joint_pair_v1
```

Existing modes such as `root_pair_mcts` or `full_pair_mcts` must not be treated
as final pair-action MCTS unless they truly expand, apply, and back up unordered
pair macro-actions. If they only convert pair logits back into single-placement
policy mass, quarantine or rename them as:

```text
root_pair_prior_blend_v0
leaf_pair_prior_blend_v0
```

`PairCandidateSelectorV1` owns:

- candidate generation,
- source quotas,
- canonicalization,
- deduplication,
- protected terminal candidate handling,
- source metadata,
- pre-rerank pools,
- and candidate selector telemetry.

`SampledJointPairStrategyV1` consumes selected candidates and owns:

- model output requests,
- pair prior construction,
- root Gumbel admission,
- interior progressive widening,
- pair expansion,
- pair backpropagation,
- pair-prior correction from proposal probabilities,
- pair-search telemetry,
- offline audit-oracle hooks for tiny or tactical positions,
- and target metadata emission.

The model architecture may declare pair-capable outputs. It may not decide that
MCTS consumes them. The pair strategy decides that.

### Search Algorithm

At each full-turn node:

1. Query Rust for legal cells and tactical state.
2. Build or retrieve legal-row identity.
3. Use `pair_candidate_selector_v1` to generate candidate pair proposals from
   source quotas.
4. Canonicalize pairs and deduplicate by canonical pair key.
5. Store all source contributions, combined `beta`, forced flags, tactical
   flags, terminal-equivalence flags, and target-prune flags.
6. At the root, use `gumbel_sequential_halving_v1`.
7. At non-root nodes, use proposal-aware PUCT over admitted children.
8. Widen a node according to:

```text
allowed_children = min(total_legal_pairs, ceil(c_pw * visits ** alpha_pw))
```

9. Expand a pair by applying both stones atomically through the exact engine.
10. Back up value on the pair edge.

`gumbel_sequential_halving_v1` has a mandatory algorithm contract:

```text
inputs:
  candidate logits
  candidate source contributions
  combined beta or inclusion estimates
  root simulation budget
  legal-pair count
  value estimate

outputs:
  admitted set
  gumbel values or admission order
  simulation allocation
  visit counts
  Q values
  completed-Q values
  improved policy target over admitted candidates
  target-prune flags
```

The root target must be derived from the search operator's improved posterior,
not from raw visit counts alone. If a different root operator is proposed, it
must define the same output fields and pass the same target-construction tests
before it can replace `gumbel_sequential_halving_v1`.

Proposal-aware PUCT priors are required for sampled candidate sets. Raw model
logits over a sampled set are forbidden unless the sampler is intentionally the
model prior and that assumption is recorded in the target metadata. The default
candidate prior should use a clipped proposal correction:

```text
prior_logit(a) = model_logit(a) - clip(log(beta(a)), min_beta_log, max_beta_log)
P_C(a | s) = softmax(prior_logit(a) / prior_temperature)
```

The exact clipping and temperature parameters are tunable, but the correction
cannot be omitted silently.

Initial tuning range:

```text
alpha_pw: 0.4 to 0.6
c_pw: tune by root candidate recall and equal-wall-clock strength
```

These are not fixed constants. They are required tuning dimensions for the
candidate-source and search-family ablations.

### Candidate Sources

The final candidate generator should use source quotas rather than one blended
black-box sampler.

Required sources:

- `terminal_exact_v1`: exact own win-in-one pairs, exact
  terminal-equivalent winning pairs, exact opponent loss-in-one block pairs,
  and exact impossible-to-block flag.
- `direct_pair_retrieval`: candidate pairs retrieved from a cheap symmetric
  low-rank pair proposal score over legal-cell embeddings.
- `policy_pair_rerank`: pairs reranked by the current joint-pair scorer after
  non-exhaustive pre-candidate retrieval.
- `anchor_conditioned_completion`: anchor-cell samples plus conditional
  second-cell
  completions.
- `cell_marginal_cross`: capped unordered cross-product of top cell-marginal
  cells.
- `structured_explore`: uniform anchors, distance buckets, active windows,
  cover sets, axis diversity, and novelty sampling.
- `blind_canary`: very small capped diagnostic sampling from legal pairs.

V1 tactical logic deliberately does not generate overload pairs, shared-pivot
forks, multi-axis pressure pairs, bounded hot-window races, or non-terminal
attack motifs. Those are V2 tactical proposal features. V1 tactical logic exists
only to prevent candidate selection from missing forced terminal actions. All
non-terminal pair quality should be discovered through model proposal,
conditional proposal, joint reranking, structured diversity, and search.

Every candidate stores:

```text
pair_key
first_legal_row_id
second_legal_row_id
source_contributions[]
combined_beta_or_inclusion_estimate
was_forced_exploration
was_terminal_exact
target_prune_flag
admission_generation
root_or_interior
```

The source quotas should anneal by training phase. `terminal_exact_v1` is
non-optional; protected terminal candidates cannot be evicted. The blind canary
quota can shrink to near-zero but should remain available in diagnostic runs.

`policy_pair_rerank` must not mean "score every unordered legal pair and take
top-k" during normal runtime. It works in two steps:

```text
1. Retrieve a bounded pre-candidate pool from direct pair retrieval,
   anchor-conditioned completions, terminal rows, structured rows, and canary
   rows.
2. Score only that bounded pool with the pair reranker, then admit top-k or
   diverse-k after canonical deduplication.
```

Dense all-pair scoring is allowed only for tiny diagnostic states, offline
audit oracle runs, or explicitly capped tactical fixtures. It is not the normal
self-play or evaluation path.

### Pair Candidate Selection

`pair_candidate_selector_v1` is the first real pair candidate generator. It
replaces legal-order first-N pair scoring as the main pair proposal method.

Inputs:

```text
full legal rows
legal-cell embeddings h_i
cell_marginal_logits over legal rows
pair_completion_logits(anchor, candidate)
cheap symmetric pair_proposal_score(i, j), if available
rich pair_joint_logits over candidate pair rows, if available
terminal_tactical_set_v1
root_or_interior budget
deterministic RNG seed
```

Outputs:

```text
source-tagged canonical pair candidates
per-candidate source contributions
protected terminal flags
initial proposal scores
combined beta or inclusion estimates
candidate row-table hash
```

Candidate sources:

1. `terminal_exact_v1`
   Always include own winning pairs. If no own winning pair exists, include
   exact block-loss pairs. Protected terminal candidates cannot be evicted.
2. `direct_pair_retrieval`
   Retrieve high-scoring unordered pairs from the cheap symmetric pair proposal
   score. This is the primary model-led source.
3. `anchor_conditioned_completion`
   Select anchor cells from `cell_marginal_logits` or pair proposal summaries.
   For each anchor, use `pair_completion_logits` to propose completions. Add
   both unordered orientations or use `logsumexp` over both orientations.
4. `cell_marginal_cross`
   Add a capped cross-product of top marginal cells. This source is secondary
   and exists for stability and debugging, not as the main policy.
5. `structured_diversity`
   Add non-tactical coverage from distance buckets, coordinate-sector diversity,
   legal-row bucket diversity, and top unary cells paired with diverse seconds.
6. `blind_canary`
   Add a tiny deterministic-seeded sample for diagnostics and support-collapse
   checks.

Final selection:

```text
canonicalize every pair
deduplicate by pair key
merge all source contributions
keep protected terminal pairs
fill remaining budget by source quotas plus joint rerank score
record why every candidate entered
```

#### Direct Pair Retrieval

`direct_pair_retrieval` is the pair-native candidate source. It lets the model
retrieve pairs whose relationship looks strong even when neither cell is a top
standalone marginal.

The graph encoder produces one embedding per legal cell:

```text
h_i = legal_cell_embedding(i)
```

A cheap retrieval head projects those embeddings:

```text
u_i = W_u h_i
v_i = W_v h_i
```

and scores unordered pairs with a symmetric low-rank proposal score:

```text
s_prop(i, j) =
  0.5 * (u_i dot v_j + u_j dot v_i)
  + cheap_pair_features(i, j)
```

This score is a retrieval signal, not the final policy. It is used to form a
bounded candidate set. The richer `pair_joint_logits` scorer and MCTS still
decide among admitted pair candidates.

Direct pair retrieval does not materialize pair tokens in the graph encoder and
does not make MCTS exhaustive over all pairs. It may use batched matrix
operations, approximate top-k retrieval, row buckets, or chunked scoring to
retrieve a bounded set. Normal runtime must still respect candidate budgets and
record source/proposal metadata.

## Model Design

### Final Model Family

Use the existing global graph direction as the base, but make the final
pair-specific architecture explicit:

```text
global_pair_biaffine_0
```

If the codebase prefers not to add a new architecture id, this can supersede
`global_pair_twostage_0` after the ablation. Do not keep both as permanent
mainline runtimes.

### Input Objects

Use the minimal object set:

- `STATE`
- `TURN`
- `STONE`
- `LEGAL`
- `WINDOW6`

Do not restore `PAIR_ACTION` tokens to the main runtime. Pair rows should
reference `LEGAL` tokens or known-first placement tokens.

`WINDOW6` is required because Hexo's tactics are naturally expressed through
six-cell windows: four-with-two-empties, five-with-one-empty, overlapping cover
sets, and overload pressure. This is the smallest exact tactical object that
matches the win condition.

Active window selection must be exact but capped. The engine or graph builder
must include:

- all hot own and opponent 4/5 windows,
- all immediate one-placement and two-placement win/block windows,
- windows touching current legal cells with enough occupied or legal-empty
  support to matter tactically,
- windows referenced by `terminal_exact_v1`,
- capped near-hot windows needed for legal-cell context,
- and deterministic priority overflow telemetry when the cap is exceeded.

The cap is part of the schema contract, not an incidental batch parameter. A
state that overflows the active-window budget must report how many windows were
omitted by priority class so tactical recall failures can be diagnosed.

### Relations

Required relations:

- legal cell to active window membership,
- stone to active window membership,
- legal cell to nearby legal cells by axial offset buckets,
- legal cell to stone proximity buckets,
- window to overlapping window,
- window to shared cover cell,
- same-axis and same-line relations,
- pair row references to first and second legal rows.

### Output Heads

The final model has these trainable outputs:

```text
cell_marginal_logits
pair_completion_logits
pair_proposal_score
pair_joint_logits
value
terminal_tactical_v1
pair_q_or_completed_q
uncertainty_or_regret
```

These are new-model output contracts. They are not legacy runtime heads.
`cell_marginal_logits` and `pair_completion_logits` are proposal and auxiliary
learning views; they do not define the searched action. `pair_joint_logits`
over admitted unordered pair rows is the main model policy prior consumed by
sampled joint-pair MCTS.

`policy_place` and `policy_pair_first` are retired as semantic names for the
final model. If an existing implementation temporarily keeps those tensor
names, output contracts must map them to the new concepts above, and the old
names must not appear as permanent runtime semantic authority.

The final pair scorer should have the shape:

```text
score(i, j) =
  symmetric_biaffine(h_i, h_j)
  + symmetric_mlp(state, h_i + h_j, abs(h_i - h_j), h_i * h_j, pair_features)
```

The biaffine term must be mathematically order-invariant. Use one of:

```text
symmetric_biaffine(i, j) =
  0.5 * (h_i^T U h_j + h_j^T U h_i)
```

or enforce `U = U^T` by parameterization. A naive `h_i^T U h_j` term is not
acceptable for unordered joint pairs.

Required V1 pair features:

- axial distance,
- same-axis indicator,
- same-line indicator,
- same-window indicator,
- terminal exact win flag,
- terminal-equivalent win flag,
- terminal exact block flag,
- blocks-all-opponent-win-requirements flag,
- impossible-to-block state flag,
- source-contribution bitset,
- legal-row phase flags.

V2 pair feature expansion:

- overload count,
- shared-pivot count,
- fork count,
- axis-diversity pressure score,
- bounded tactical-race score,
- non-terminal hot-window creation score,
- opponent hot-window cover count,
- overlap cover count.

The joint pair score must be invariant to pair order. The known-first second
head remains ordered and conditional.

## Training Design

### Replay Metadata

Self-play records must store enough data to reconstruct why a pair was searched
and how it became a target:

```text
candidate_pairs
candidate_selector_version
terminal_tactical_v1
candidate_source_contributions
combined_beta_or_inclusion_estimates
proposal_correction_parameters
legal_pair_count
legal_row_schema_version
pair_row_schema_version
root_gumbel_values_or_admission_order
root_simulation_allocation
visit_counts
q_values
completed_q_values
forced_exploration_flags
policy_target_prune_flags
terminal_equivalence_flags
candidate_selection_reason
selected_pair
search_surprise_metrics
```

Without this metadata, sampled-pair training becomes biased and hard to debug.

### Target Stack

The training target stack is:

1. Pruned candidate-set joint pair posterior.
2. Completed-Q or regularized pair posterior when visits are sparse.
3. Marginal single-cell target derived from the pair posterior.
4. Conditional second-cell target for meaningful first cells.
5. Softened pair target for lower-mass but plausible candidates.
6. Sampled ranking or sampled-softmax loss over admitted and sampled-negative
   candidates, corrected by proposal weight where applicable.
7. Exact tactical auxiliary labels.
8. Value and pair-regret/completed-Q heads.

The rule is absolute:

```text
unsampled legal pair != negative pair
```

Only admitted pairs judged weak by search, or explicitly sampled negatives with
known proposal metadata, can contribute negative ranking signal.

Terminal exact candidates receive protection in candidate admission, but raw
terminal protection must not pollute ordinary policy targets:

- If own win-in-one exists, the improved target may collapse to
  terminal-winning candidates. Terminal-equivalent pairs must be collapsed or
  tie-broken deterministically.
- If opponent lose-in-one exists and own win-in-one does not, non-blocking pairs
  may be target-pruned or marked negative only if the terminal proof says they
  fail to block all opponent immediate win requirements.
- If `impossible_to_block` is true, do not fabricate a winning defensive
  target. Record tactical status and let value/search handle the losing state.

Pair target completeness must be explicit. A target builder may set
`pair_policy_complete = true` only when one of these support contracts is true:

```text
support_type = "exhaustive_legal_pair_table"
support_type = "admitted_candidate_set_with_explicit_negatives"
support_type = "completed_q_candidate_posterior"
```

Otherwise, pair targets train only over admitted candidate rows, and every row
must carry source and proposal metadata. Sparse positive rows alone do not imply
that omitted legal pairs have zero target mass.

Terminal-equivalent pair targets require special handling. If a single legal
cell wins under the exact rules, then every legal pair containing that cell may
be equivalent with respect to the game outcome. In that case:

- train the cell/tactical heads strongly on the winning cell,
- mark pair rows containing that cell as terminal-equivalent,
- either collapse the pair posterior over equivalent winning pairs or choose
  deterministic second-cell tie-breaks from tactical criteria,
- and do not teach arbitrary preferences among filler second cells as if they
  were strategic pair differences.

Conditional second-cell targets must be safe for unordered pair data. If an
unordered pair `{i, j}` has target mass, training should create both conditional
views where legal and meaningful:

```text
pi(second = j | first = i)
pi(second = i | first = j)
```

The two views train `pair_completion_logits(anchor, candidate)` and are weighted
through the cell marginal or anchor target. Training only the canonical first
cell is forbidden because it leaks row-order artifacts into the conditional
head.

### Tactical Supervision

Rust should remain the production source for tactical labels. V1 tactical
supervision is terminal-only. V2 expands tactical supervision after V1
candidate recall, row identity, target correctness, and pair-search runtime are
stable.

### Tactical Supervision V1

Required V1 labels:

- immediate one-placement wins,
- immediate two-placement wins,
- terminal-equivalent winning pairs,
- opponent immediate win requirements,
- pair blocks all opponent immediate win requirements,
- impossible-to-block opponent immediate win.

The Rust tactical oracle should produce a legal-row-keyed and pair-row-keyed
proposal payload:

```text
TerminalTacticalSetV1 {
  status
  status = quiet | win_in_one | must_block_loss_in_one | loss_in_one_unblockable
  winning_single_cells
  winning_pairs
  terminal_equivalent_pairs
  opponent_win_requirements
  block_loss_pairs
  impossible_to_block
}
```

Tactical labels are not hand-authored legal filters. They are:

- auxiliary training targets,
- candidate proposal sources,
- tactical benchmark generators,
- target-pruning metadata,
- and near-terminal exact search triggers.

### Tactical Supervision V2

V2 tactical labels:

- own hot-window creation,
- overload creation,
- shared pivot,
- multi-axis fork,
- bounded hot-window race,
- non-terminal pressure score,
- axis-diversity count.

V2 labels are not required for `sampled_joint_pair_v1` completion. They may be
added after the V1 candidate selector and pair-search runtime are stable, and
they should be ablated before becoming part of the mainline.

### Self-Play Curriculum

Use a curriculum over search budget, candidate budget, replay mix, and tactical
state exposure. Do not use a one-stone-per-turn surrogate game as the main
curriculum because it changes the action semantics.

Bootstrap phase:

- high candidate diversity,
- large `terminal_exact_v1` protection,
- high structured exploration quota,
- high direct-pair-retrieval, anchor-completion, and cell-marginal-cross quota
  once the heads exist,
- small blind canary quota,
- larger root candidate set,
- aggressive widening,
- high visit temperature,
- synthetic fixtures for win-in-one and lose-in-one candidate recall.

Growth phase:

- increase model-led proposal share,
- keep `terminal_exact_v1` always active,
- reduce blind canary source,
- add V2 tactical labels only after V1 selector/search stability,
- introduce surprise-weighted replay,
- gradually raise simulation budget.

Mature phase:

- mostly model-led candidate admission,
- `terminal_exact_v1` remains non-optional,
- structured novelty remains small but present,
- lower action temperature,
- hard-state replay,
- equal-wall-clock arena evaluation,
- V2 tactical sources may be ablated before inclusion.

## Implementation Phases

### Phase 0: Decision Record And Baseline Inventory

Goal:

Write down the final architecture decision and inventory current pair-search,
pair-head, replay, and contract behavior.

Success criteria:

- This document exists and is linked from the relevant planning index.
- Current pair strategies, pair heads, row contracts, target builders, and
  tactical oracle paths are listed.
- Legacy paths that conflict with the final plan are named.
- Current hard threat-filter entry points are listed and classified as
  diagnostic, classical-search, exact-solver, or removal candidates.
- Current `root_pair_mcts` and `full_pair_mcts` behavior is classified as
  prior-blend baseline behavior unless it truly expands pair macro-actions.

Required evidence:

- `rg` audit for direct pair-head self-play consumption.
- `rg` audit for pair target builders and pair strategy modes.
- `rg` audit for `constrain_threats`, `threat_constrained_moves`,
  `root_pair_mcts`, `full_pair_mcts`, and `pair_logits_to_action_logits`.
- `git status --short`.

Stop rules:

- Stop if the current runtime pair authority cannot be identified.
- Stop if current hard threat-filter paths cannot be separated from full legal
  row construction.

### Phase 1: Legal Rows And Threat-Filter Cutover

Goal:

Make full Rust legal rows the unambiguous semantic legal table and move threat
logic out of hard legal filtering for main neural self-play and training.

Success criteria:

- Main neural self-play and training use full Rust-legal placement rows.
- Hard threat filtering does not delete `LEGAL` rows in flagship training or
  self-play.
- Threat-constrained rows, if retained, are renamed or quarantined as
  `diagnostic_threat_filter_v0`.
- Tactical oracle outputs become legal-row-keyed labels, candidate proposals,
  solver facts, benchmark facts, and target-pruning metadata.
- Full legal count, admitted candidate count, and solver override metadata are
  recorded separately.
- Tests prove tactical oracle labels do not delete legal rows.

Required evidence:

- Unit tests for `LEGAL`, `ADMITTED_SINGLE`, `ADMITTED_PAIR`, and `TACTICAL`
  row-family separation.
- Tests proving `constrain_threats` is disabled or unavailable in flagship
  neural self-play/training configs.
- Tests proving exact solver overrides are explicitly flagged and include
  `solver_reason`, `full_legal_count`, and `admitted_candidate_count`.
- Code-search audit showing hard threat filtering is absent from main training
  and self-play legal-row construction.

Stop rules:

- Stop if a training or self-play `LEGAL` row table can be threat-censored
  without explicit diagnostic or solver metadata.
- Stop if tactical candidate/proposal rows cannot be keyed back to full legal
  rows.

### Phase 2: Pair Contracts And Row Identity

Goal:

Make canonical unordered pair rows and known-first second-placement rows strict
contracts across replay, inference, and search.

Success criteria:

- Pair rows carry stable row-table identity.
- Duplicate, illegal, stale, same-count-reordered, and wrong-phase pair rows are
  hard errors.
- D6 transforms preserve pair row identity after canonicalization.
- Legacy `policy_pair_first` semantics are resolved and mapped to
  `cell_marginal_logits`, `pair_anchor` diagnostics, or deletion. It must not
  pretend unordered targets are ordered.

Required evidence:

- Contract tests for row identity and D6.
- Cross-language fuzz tests for Rust/Python legal rows, pair rows, tactical
  oracle payloads, and replay serialization schema versions.
- Negative tests for illegal pairs, duplicate pairs, and wrong phase.
- Import/search audit showing no consumer guesses pair semantics from tensor
  shape alone.

Stop rules:

- Stop if row identity cannot survive replay-to-training and inference-to-search
  boundaries.

### Phase 3: Pair Candidate Selector V1

Goal:

Implement a source-tagged, pair-native candidate selector that replaces
legal-order first-N pair scoring in normal runtime.

Success criteria:

- Full legal rows are preserved.
- `terminal_exact_v1` computes only win-in-one, block-loss-in-one,
  terminal-equivalent pairs, and impossible-to-block flags.
- Candidate generator supports `terminal_exact_v1`,
  `direct_pair_retrieval`, `anchor_conditioned_completion`,
  `cell_marginal_cross`, `joint_rerank_pool`,
  `structured_diversity`, and `blind_canary`.
- Protected terminal pairs cannot be evicted.
- Pair candidates are canonicalized and deduplicated.
- Every candidate stores all source contributions, combined `beta` or inclusion
  estimate, forced flags, tactical flags, and target-prune flags.
- Tactical oracle labels are filtered to explicit legal rows.
- Structured exploration replaces raw persistent uniform-over-all-pairs as the
  main exploration floor.
- `pair_joint_logits` scores only the bounded candidate pool in normal
  runtime.
- Legal-order first-N pair scoring is quarantined as diagnostic only.

Required evidence:

- Unit tests for source quotas, multi-source deduplication, legal filtering,
  combined-beta accounting, and metadata.
- Unit tests where a winning pair appears late in legal-row order and is still
  included.
- Unit tests where opponent has multiple immediate win requirements and only
  valid block pairs are protected.
- Unit tests where no two-cell block exists and `impossible_to_block` is true.
- D6 tests for `terminal_exact_v1` and candidate selection.
- Source quota tests showing non-model sources survive early bad policies.
- Tests proving `direct_pair_retrieval` can admit a high-scoring pair whose
  cells are not top-ranked by `cell_marginal_logits`.
- Regression test proving normal runtime does not use first-N legal-pair
  enumeration.
- Tactical benchmark fixtures for wins and blocks.
- Artifact showing candidate-source mix on a fixed position set.
- Offline audit-oracle artifact on tiny or tactical positions where exhaustive
  pair scoring is allowed.

Stop rules:

- Stop if a candidate can enter search without complete source, proposal,
  forced/prune, and beta metadata.
- Stop if a terminal win/block pair can be evicted by model ranking.
- Stop if the selector depends on legal-row order except for deterministic
  tie-breaks.
- Stop if the selector cannot prove which candidates were terminal, model-led,
  conditional, diversity, or canary.

### Phase 4: Sampled Joint-Pair MCTS

Goal:

Make sampled pair macro-actions the main search runtime.

Success criteria:

- Pair child expansion applies both placements atomically.
- Root admission uses `gumbel_sequential_halving_v1` with completed-Q target
  construction.
- Interior nodes use proposal-aware PUCT plus progressive widening.
- Pair strategy, not model head presence, controls all pair runtime behavior.
- Search telemetry includes candidate recall, source mix, visits, Q, widening,
  latency, and selected action.
- Terminal-equivalent pairs are marked and handled consistently in search
  targets.

Required evidence:

- MCTS unit tests for pair expansion, canonical keys, and no order aliasing.
- Fixed-position search traces.
- Throughput profile for node expansions, pair scoring, inference latency, and
  candidate generation.
- Proposal-correction artifact showing raw logits, beta-corrected priors, and
  clipped correction parameters.
- `rg` audit proving self-play no longer directly checks pair head names for
  behavior outside pair strategy code.

Stop rules:

- Stop if pair search only works by keeping a parallel sequential runtime path.

### Phase 5: Biaffine Pair Model

Goal:

Implement the final minimal global graph pair model.

Success criteria:

- Uses STATE, TURN, STONE, LEGAL, and WINDOW6 objects.
- Does not materialize PAIR_ACTION tokens in the main path.
- Scores pair rows through a symmetric biaffine or low-rank bilinear term plus
  symmetric MLP features.
- Enforces order-invariant biaffine math by symmetrized scoring or symmetric
  parameterization.
- Keeps known-first second head conditional and phase-gated.
- Exposes output contracts for every runtime-consumed head.
- Defines active `WINDOW6` inclusion priorities, caps, and overflow telemetry.

Required evidence:

- Model shape tests for all heads.
- Symmetry tests for unordered joint scores.
- Phase tests for known-first second rows.
- Active-window cap and overflow tests.
- Performance profile versus current pair MLP and graph controls.

Stop rules:

- Stop if pair rows inflate the attention token sequence in the main runtime.

### Phase 6: Candidate-Aware Training Targets

Goal:

Replace raw sparse pair visit training with the full target stack.

Success criteria:

- Replay stores candidate set, source, proposal weight, visits, Q, completed-Q,
  and prune flags.
- Pair target pruning removes forced exploration traffic where appropriate.
- Marginal and conditional targets are generated from the pruned joint target.
- Conditional second-cell targets include both unordered views where legal and
  meaningful.
- Terminal-equivalent one-cell wins do not train arbitrary filler-stone pair
  preferences.
- Unsampled legal pairs are never used as hard negatives.
- Sampled ranking or sampled-softmax loss is proposal-aware.

Required evidence:

- Unit tests for target projection and pruning.
- Unit tests for terminal-equivalent targets and unordered-safe conditional
  targets.
- Negative tests for unsampled-pair hard negatives.
- Training smoke test with all target heads active.
- Artifact comparing raw-count target entropy to pruned/completed-Q posterior.

Stop rules:

- Stop if replay cannot distinguish admitted, forced, sampled-negative, and
  unsampled pairs.

### Phase 7: Self-Play Curriculum And Replay Control

Goal:

Train the pair system without support collapse.

Success criteria:

- Bootstrap, growth, and mature schedules are configurable and artifacted.
- Replay sampling can stratify by phase, tactical density, pair-space size,
  search surprise, and uncertainty.
- Hard-state replay is available for archived tactical states.
- Two-budget self-play is available: many cheap searches for value throughput,
  fewer deeper searches for stronger policy targets.

Required evidence:

- Self-play smoke run with source telemetry.
- Replay distribution artifact.
- Search-surprise and entropy dashboard output.
- Fixed tactical suite results over checkpoints.
- Distributed self-play/inference performance artifact covering batching, queue
  backpressure, CPU candidate-generation time, GPU utilization, and
  pair-scores/sec.

Stop rules:

- Stop if policy entropy collapses before candidate recall and tactical recall
  are healthy.
- Stop if batching, queue backpressure, CPU candidate generation, GPU
  utilization, or pair scoring throughput fails the phase budget.

### Phase 8: Fair Ablations And Cutover

Goal:

Prove the final mainline is better than the alternatives and delete or
quarantine obsolete paths.

Success criteria:

- Search-family shootout:
  - sampled joint-pair MCTS,
  - sequential afterstate DAG baseline,
  - autoregressive proposal with pair-level search.
- Pair-head ablation:
  - autoregressive-only,
  - symmetric MLP,
  - biaffine reranker,
  - final proposal plus reranker.
- Candidate-source ablation:
  - policy only,
  - policy plus terminal exact,
  - policy plus structured diversity,
  - policy plus terminal exact plus structured diversity.
- Target-stack ablation:
  - raw visit counts,
  - pruned posterior,
  - completed-Q posterior,
  - full target stack.
- Tactical-object ablation:
  - no window tokens,
  - window tokens only,
  - window tokens plus V1 terminal auxiliaries,
  - V2 tactical proposal path.

Required evidence:

- Equal-simulation arena results.
- Equal-wall-clock arena results.
- Equal-neural-evaluation results.
- Candidate recall of selected and tactical best pairs.
- D6 consistency report.
- Performance/utilization profile.
- Deletion or quarantine proof for superseded runtime paths.

Stop rules:

- Stop if pair macro-search only wins at equal simulations but loses at equal
  wall-clock after proposal quality is mature.

## Metrics And Dashboards

Minimum metrics:

```text
legal_pair_count_per_root
full_legal_row_count_per_root
admitted_pair_count_per_root
admitted_single_count_per_root
candidate_source_mix
candidate_selector_version
terminal_tactical_v1_status_mix
solver_override_rate
diagnostic_threat_filter_usage
candidate_recall_best_search_pair
candidate_recall_exact_win_pair
candidate_recall_exact_block_pair
normalized_pair_target_entropy
visit_count_gini
search_surprise_kl
policy_entropy_by_phase
pair_prior_calibration
root_q_variance
value_calibration_by_phase
replay_distribution_by_phase
replay_distribution_by_tactical_density
fixed_simulation_elo
fixed_wall_clock_elo
fixed_neural_eval_elo
node_expansions_per_second
pair_scores_per_second
inference_latency_p50_p95
queue_backpressure
batch_fill_rate
gpu_utilization
cpu_candidate_generation_time
candidate_generation_latency_p95
tactical_suite_accuracy
terminal_candidate_recall
d6_policy_consistency
d6_value_consistency
```

Success is not "the pair model wins at equal simulations." Success is "the pair
model wins at equal wall-clock after the proposal model is mature, while
candidate recall and tactical recall remain high."

## Acceptance Criteria For The Final Mainline

The mainline is accepted only when all of these are true:

- Sampled joint-pair MCTS has a statistically meaningful equal-wall-clock Elo
  win over the sequential DAG baseline on mature checkpoints.
- Flagship self-play and training preserve full Rust-legal row identity and do
  not use hard threat-filtered `LEGAL` tables.
- Candidate recall contains the final best-search pair on at least 95% of roots
  in the deep-search audit suite.
- Exact immediate winning pairs have at least 99.5% candidate recall.
- Mandatory block pairs have at least 99.0% candidate recall.
- Pair target entropy does not collapse prematurely.
- Search surprise decreases over training without eliminating tactical or
  structured source diversity.
- D6 transforms preserve legal-row identity, pair-row identity, policy
  consistency, and value consistency within documented numeric tolerances.
- Runtime pair behavior is controlled only by explicit pair strategy.
- The model does not materialize all pair actions as attention tokens.
- Unsampled legal pairs are never trained as negatives.
- Performance evidence includes fixed-wall-clock strength, not only
  fixed-simulation strength.
- Batching, queue backpressure, CPU candidate-generation time, GPU utilization,
  and pair-scores/sec meet the phase performance budgets.
- Obsolete paths are removed or quarantined with import and code-search proof.

## Failure Criteria

The final path should be reconsidered if:

- Pair macro-search loses to the sequential DAG baseline at equal wall-clock
  after proposal quality matures.
- The best tactical pair is frequently outside the admitted candidate set.
- The biaffine reranker does not improve pair recall over the simpler MLP after
  fair tuning. V2 overload-state recall is a required ablation before adding
  overload features to the mainline.
- `pair_candidate_selector_v1` cannot protect exact terminal win/block pairs.
- `direct_pair_retrieval` cannot improve pair candidate recall over unary and
  conditional sources after fair tuning.
- Tactical performance depends entirely on the tactical proposer and the model
  does not internalize window/cover structure.
- D6 consistency fails under pair-row transforms.
- Queue backpressure or pair scoring latency prevents useful self-play
  throughput.
- Policy entropy collapses while value calibration is still poor.
- Tactical filtering silently censors `LEGAL` rows in main self-play or
  training.

## Practical First Build

The first build should not try to implement every advanced idea at once. It
should create a thin but real vertical slice:

1. Full-legal-row and threat-filter cutover for main neural self-play/training.
2. Canonical pair contracts and D6 tests.
3. `pair_candidate_selector_v1` with `terminal_exact_v1`,
   `direct_pair_retrieval`, `anchor_conditioned_completion`,
   `cell_marginal_cross`, `joint_rerank_pool`, `structured_diversity`, and
   `blind_canary`.
4. Candidate proposal contract with multi-source deduplication, combined beta,
   prune flags, and terminal protection.
5. `gumbel_sequential_halving_v1` root admission over sampled pair rows.
6. Proposal-aware PUCT plus progressive widening.
7. Existing graph trunk with non-materialized pair rows.
8. Symmetric biaffine pair scorer behind one explicit architecture or strategy
   flag.
9. Active `WINDOW6` selection caps and overflow telemetry.
10. Replay metadata for source contributions, beta, visits, Q, completed-Q, and
   terminal-equivalence flags.
11. Pruned joint target plus marginal, unordered-safe conditional, and
   terminal-equivalent targets.
12. Fixed terminal tactical benchmark suite plus offline tiny-state audit
    oracle.
13. Equal-wall-clock comparison against the current best baseline.

That vertical slice gives fast evidence on the only question that matters:
whether pair semantics can be made strong per wall-clock, not just elegant per
simulation.

## Final Recommendation

Proceed with sampled joint-pair MCTS as the mainline. Keep pair actions as the
semantic unit of search and training, but make pair discovery cheap through
direct pair retrieval, anchor-conditioned completion, terminal exact proposal,
cell-marginal support, joint reranking, and structured exploration. V1 does not
attempt a rich tactical oracle; it builds the strongest possible minimal
pair-native candidate selector. V2 tactical sources can expand into overloads,
pivots, forks, and bounded hot-window races after V1 candidate recall, row
identity, target correctness, and pair-search runtime are stable. Use the
minimal global graph token schema, do not materialize all pair actions as input
tokens, and upgrade the pair head from a symmetric MLP baseline to a biaffine
reranker with explicit pair features.

The strongest combined plan is not a compromise that averages the two reports.
It is a sharper version of both:

- take `deep-research-report(1).md`'s operational criteria and D6 discipline,
- take `deep-research-report.md`'s target hygiene and critique of raw uniform
  pair exploration,
- keep the current project contract principle that pair behavior must live in
  explicit pair strategies,
- and require equal-wall-clock evidence before the architecture is declared the
  winner.
