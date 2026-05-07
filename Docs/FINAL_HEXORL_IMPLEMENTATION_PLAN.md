# Final Hexo-RL Implementation Plan

## Purpose

This document chooses one implementation path from the two deep research
reports and turns it into an opinionated, testable plan for Hexo-RL.

The decision is:

Build Hexo-RL around exact-simulator AlphaZero-style self-play where the
post-opening searched action is an unordered two-placement pair. Use
autoregressive and tactical structure to propose candidate pairs, but make the
MCTS edge, backup, policy target, and runtime action contract the completed
pair. Use Gumbel-style root admission, PUCT with progressive widening below the
root, a minimal global graph encoder with active six-cell window objects, and a
symmetric biaffine pair reranker trained with candidate-aware targets and dense
auxiliary views.

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
- Candidate generation combines model proposal, conditional proposal, tactical
  proposal, and structured exploration quotas.
- Root candidate admission uses Gumbel-style sampling or an equivalent
  simple-regret policy-improvement operator.
- Interior nodes use PUCT over admitted pair children with progressive widening.
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
- The final architecture beats a fair sequential afterstate DAG baseline at
  equal wall-clock on mature checkpoints.

### Constraints

- Keep Rust as the exact rules boundary for legality, transitions, terminal
  state, and tactical oracle labels.
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
- An explicit statement that no skipped, deferred, or manual-only requirement is
  being claimed complete.

### Stop Rules

Stop before claiming completion if:

- Pair row identity cannot be validated across replay, inference, and search.
- The engine cannot apply both placements atomically for a pair MCTS edge.
- Pair strategy behavior cannot be separated from raw head-name checks.
- Candidate metadata cannot store source, proposal weight, and search outcome.
- D6 transforms cannot preserve legal-row and pair-row identity.
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
  -> legal rows and active tactical windows
  -> graph/set encoder over STATE, TURN, STONE, LEGAL, WINDOW6
  -> single-cell unary head
  -> conditional second-cell pointer-style head
  -> symmetric biaffine joint-pair reranker
  -> sampled pair candidate set
  -> Gumbel root admission
  -> PUCT plus progressive widening below root
  -> candidate-aware training targets
```

The action selected by self-play and evaluation after the opening is always the
completed pair. The single-cell and conditional heads exist to make candidate
generation and learning efficient. They are not allowed to redefine the game as
two independent search plies.

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
when the search does not visit all root actions. Sampled MuZero is directly
relevant because it formalizes planning over sampled action subsets. The useful
lesson from Sampled MuZero is not learned dynamics; it is proposal-aware
planning and target construction in large action spaces.

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
  races, but it must stay bounded and exact.

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
- tactical motifs,
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

### Pair Strategy Authority

Create or promote one explicit runtime strategy:

```text
sampled_joint_pair_v1
```

This strategy owns:

- pair row generation,
- candidate source quotas,
- model output requests,
- pair prior construction,
- root Gumbel admission,
- interior progressive widening,
- pair-prior correction from proposal probabilities,
- pair-search telemetry,
- and target metadata emission.

The model architecture may declare pair-capable outputs. It may not decide that
MCTS consumes them. The pair strategy decides that.

### Search Algorithm

At each full-turn node:

1. Query Rust for legal cells and tactical state.
2. Build or retrieve legal-row identity.
3. Generate candidate pair proposals from source quotas.
4. Canonicalize and deduplicate pairs.
5. Store source and proposal weight for every admitted candidate.
6. At the root, use Gumbel-Top-k or equivalent without-replacement admission.
7. At non-root nodes, use PUCT over admitted children.
8. Widen a node according to:

```text
allowed_children = min(total_legal_pairs, ceil(c_pw * visits ** alpha_pw))
```

9. Expand a pair by applying both stones atomically through the exact engine.
10. Back up value on the pair edge.

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

- `policy_pair`: pairs from the current joint-pair prior or reranker.
- `policy_anchor_conditional`: first-cell samples plus conditional second-cell
  completions.
- `tactical_exact`: immediate wins, immediate blocks, minimal covers,
  overload-creating pairs, shared-pivot forks, and hot-window races.
- `structured_explore`: uniform anchors, distance buckets, active windows,
  cover sets, axis diversity, and novelty sampling.
- `blind_canary`: very small capped diagnostic sampling from legal pairs.

Every candidate stores:

```text
pair_key
first_legal_row_id
second_legal_row_id
source
source_rank
proposal_probability_or_weight
was_forced_exploration
was_tactical_exact
admission_generation
root_or_interior
```

The source quotas should anneal by training phase. The tactical quota should not
vanish; the blind canary quota can shrink to near-zero but should remain
available in diagnostic runs.

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
policy_place
policy_pair_first
policy_pair_second
policy_pair_joint
value
tactical_auxiliary_heads
pair_regret_or_completed_q
```

The final pair scorer should have the shape:

```text
score(i, j) =
  biaffine(h_i, h_j)
  + symmetric_mlp(state, h_i + h_j, abs(h_i - h_j), h_i * h_j, pair_features)
```

Required pair features:

- axial distance,
- same-axis indicator,
- same-window indicator,
- number of self hot windows completed,
- number of opponent hot windows covered,
- overlap cover count,
- fork or overload count,
- axis diversity,
- tactical source flags,
- legal-row phase flags.

The joint pair score must be invariant to pair order. The known-first second
head remains ordered and conditional.

## Training Design

### Replay Metadata

Self-play records must store enough data to reconstruct why a pair was searched
and how it became a target:

```text
candidate_pairs
candidate_sources
proposal_weights
legal_pair_count
root_gumbel_values_or_admission_order
visit_counts
q_values
completed_q_values
forced_exploration_flags
policy_target_prune_flags
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

### Tactical Supervision

Rust should remain the production source for tactical labels. Required labels:

- immediate one-placement wins,
- immediate two-placement wins,
- opponent immediate threats,
- legal cells covering hot opponent windows,
- pair covers of multiple threats,
- own hot-window creation,
- impossible-to-cover opponent positions,
- bounded hot-window threat races,
- overload and axis-diversity counts.

Tactical labels are not hand-authored policy filters. They are:

- auxiliary training targets,
- candidate proposal sources,
- tactical benchmark generators,
- and near-terminal exact search triggers.

### Self-Play Curriculum

Use a curriculum over search budget, candidate budget, replay mix, and tactical
state exposure. Do not use a one-stone-per-turn surrogate game as the main
curriculum because it changes the action semantics.

Bootstrap phase:

- high candidate diversity,
- large tactical quota,
- high structured exploration quota,
- larger root candidate set,
- aggressive widening,
- high visit temperature,
- tactical label pretraining or mixed supervised bootstrap.

Growth phase:

- increase model-led proposal share,
- keep tactical source alive,
- reduce blind canary source,
- introduce surprise-weighted replay,
- gradually raise simulation budget.

Mature phase:

- mostly model-led candidate admission,
- persistent tactical and structured novelty quotas,
- lower action temperature,
- hard-state replay,
- equal-wall-clock arena evaluation,
- periodic candidate-source ablations.

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

Required evidence:

- `rg` audit for direct pair-head self-play consumption.
- `rg` audit for pair target builders and pair strategy modes.
- `git status --short`.

Stop rules:

- Stop if the current runtime pair authority cannot be identified.

### Phase 1: Pair Contracts And Row Identity

Goal:

Make canonical unordered pair rows and known-first second-placement rows strict
contracts across replay, inference, and search.

Success criteria:

- Pair rows carry stable row-table identity.
- Duplicate, illegal, stale, same-count-reordered, and wrong-phase pair rows are
  hard errors.
- D6 transforms preserve pair row identity after canonicalization.
- `policy_pair_first` semantics are resolved: unordered marginal target or
  diagnostic-only. It must not pretend unordered targets are ordered.

Required evidence:

- Contract tests for row identity and D6.
- Negative tests for illegal pairs, duplicate pairs, and wrong phase.
- Import/search audit showing no consumer guesses pair semantics from tensor
  shape alone.

Stop rules:

- Stop if row identity cannot survive replay-to-training and inference-to-search
  boundaries.

### Phase 2: Candidate Sources And Tactical Oracle Integration

Goal:

Implement source-tagged candidate generation and exact tactical proposal.

Success criteria:

- Candidate generator supports policy, conditional, tactical, structured
  exploration, and blind canary sources.
- Every candidate stores source and proposal metadata.
- Tactical oracle labels are filtered to explicit legal rows.
- Structured exploration replaces raw persistent uniform-over-all-pairs as the
  main exploration floor.

Required evidence:

- Unit tests for source quotas, deduplication, legal filtering, and metadata.
- Tactical benchmark fixtures for wins, blocks, covers, and overloads.
- Artifact showing candidate-source mix on a fixed position set.

Stop rules:

- Stop if a candidate can enter search without source and proposal metadata.

### Phase 3: Sampled Joint-Pair MCTS

Goal:

Make sampled pair macro-actions the main search runtime.

Success criteria:

- Pair child expansion applies both placements atomically.
- Root admission uses Gumbel-style without-replacement candidate admission or an
  equivalent tested policy-improvement operator.
- Interior nodes use PUCT plus progressive widening.
- Pair strategy, not model head presence, controls all pair runtime behavior.
- Search telemetry includes candidate recall, source mix, visits, Q, widening,
  latency, and selected action.

Required evidence:

- MCTS unit tests for pair expansion, canonical keys, and no order aliasing.
- Fixed-position search traces.
- Throughput profile for node expansions, pair scoring, inference latency, and
  candidate generation.
- `rg` audit proving self-play no longer directly checks pair head names for
  behavior outside pair strategy code.

Stop rules:

- Stop if pair search only works by keeping a parallel sequential runtime path.

### Phase 4: Biaffine Pair Model

Goal:

Implement the final minimal global graph pair model.

Success criteria:

- Uses STATE, TURN, STONE, LEGAL, and WINDOW6 objects.
- Does not materialize PAIR_ACTION tokens in the main path.
- Scores pair rows through a symmetric biaffine or low-rank bilinear term plus
  symmetric MLP features.
- Keeps known-first second head conditional and phase-gated.
- Exposes output contracts for every runtime-consumed head.

Required evidence:

- Model shape tests for all heads.
- Symmetry tests for unordered joint scores.
- Phase tests for known-first second rows.
- Performance profile versus current pair MLP and graph controls.

Stop rules:

- Stop if pair rows inflate the attention token sequence in the main runtime.

### Phase 5: Candidate-Aware Training Targets

Goal:

Replace raw sparse pair visit training with the full target stack.

Success criteria:

- Replay stores candidate set, source, proposal weight, visits, Q, completed-Q,
  and prune flags.
- Pair target pruning removes forced exploration traffic where appropriate.
- Marginal and conditional targets are generated from the pruned joint target.
- Unsampled legal pairs are never used as hard negatives.
- Sampled ranking or sampled-softmax loss is proposal-aware.

Required evidence:

- Unit tests for target projection and pruning.
- Negative tests for unsampled-pair hard negatives.
- Training smoke test with all target heads active.
- Artifact comparing raw-count target entropy to pruned/completed-Q posterior.

Stop rules:

- Stop if replay cannot distinguish admitted, forced, sampled-negative, and
  unsampled pairs.

### Phase 6: Self-Play Curriculum And Replay Control

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

Stop rules:

- Stop if policy entropy collapses before candidate recall and tactical recall
  are healthy.

### Phase 7: Fair Ablations And Cutover

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
  - policy plus raw uniform,
  - policy plus tactical,
  - policy plus tactical plus structured exploration.
- Target-stack ablation:
  - raw visit counts,
  - pruned posterior,
  - completed-Q posterior,
  - full target stack.
- Tactical-object ablation:
  - no window tokens,
  - window tokens only,
  - window tokens plus auxiliaries,
  - full tactical proposal path.

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
admitted_pair_count_per_root
candidate_source_mix
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
gpu_utilization
cpu_candidate_generation_time
tactical_suite_accuracy
d6_policy_consistency
d6_value_consistency
```

Success is not "the pair model wins at equal simulations." Success is "the pair
model wins at equal wall-clock after the proposal model is mature, while
candidate recall and tactical recall remain high."

## Acceptance Criteria For The Final Mainline

The mainline is accepted only when all of these are true:

- Sampled joint-pair MCTS beats the sequential DAG baseline at equal wall-clock
  on mature checkpoints.
- Candidate recall contains the final best-search pair on the overwhelming
  majority of roots.
- Exact immediate winning pairs and mandatory block pairs have near-perfect
  candidate recall.
- Pair target entropy does not collapse prematurely.
- Search surprise decreases over training without eliminating tactical or
  structured source diversity.
- D6 transforms preserve legal-row identity, pair-row identity, policy
  consistency, and value consistency.
- Runtime pair behavior is controlled only by explicit pair strategy.
- The model does not materialize all pair actions as attention tokens.
- Unsampled legal pairs are never trained as negatives.
- Performance evidence includes fixed-wall-clock strength, not only
  fixed-simulation strength.
- Obsolete paths are removed or quarantined with import and code-search proof.

## Failure Criteria

The final path should be reconsidered if:

- Pair macro-search loses to the sequential DAG baseline at equal wall-clock
  after proposal quality matures.
- The best tactical pair is frequently outside the admitted candidate set.
- The biaffine reranker does not improve overloaded-state pair recall over the
  simpler MLP after fair tuning.
- Tactical performance depends entirely on the tactical proposer and the model
  does not internalize window/cover structure.
- D6 consistency fails under pair-row transforms.
- Queue backpressure or pair scoring latency prevents useful self-play
  throughput.
- Policy entropy collapses while value calibration is still poor.

## Practical First Build

The first build should not try to implement every advanced idea at once. It
should create a thin but real vertical slice:

1. Canonical pair contracts and D6 tests.
2. Source-tagged candidate generator with tactical and structured sources.
3. Root Gumbel candidate admission over sampled pair rows.
4. Interior PUCT plus progressive widening.
5. Existing graph trunk with non-materialized pair rows.
6. Biaffine pair scorer behind one explicit architecture or strategy flag.
7. Replay metadata for candidate source, proposal weight, visits, and Q.
8. Pruned joint target plus marginal and conditional targets.
9. Fixed tactical benchmark suite.
10. Equal-wall-clock comparison against the current best baseline.

That vertical slice gives fast evidence on the only question that matters:
whether pair semantics can be made strong per wall-clock, not just elegant per
simulation.

## Final Recommendation

Proceed with sampled joint-pair MCTS as the mainline. Keep pair actions as the
semantic unit of search and training, but make pair discovery cheap through
autoregressive proposal, exact tactical proposal, and structured exploration.
Use the minimal global graph token schema, do not materialize all pair actions
as input tokens, and upgrade the pair head from a symmetric MLP baseline to a
biaffine reranker with explicit tactical features.

The strongest combined plan is not a compromise that averages the two reports.
It is a sharper version of both:

- take `deep-research-report(1).md`'s operational criteria and D6 discipline,
- take `deep-research-report.md`'s target hygiene and critique of raw uniform
  pair exploration,
- keep the current project contract principle that pair behavior must live in
  explicit pair strategies,
- and require equal-wall-clock evidence before the architecture is declared the
  winner.

