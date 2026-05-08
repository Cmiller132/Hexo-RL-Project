# Hexo-RL V1 Architecture Proposal

## Decision Summary

V1 is an architecture-complete but risk-contained pair-action system for
Hexo-RL. After the opening move, the runtime action, MCTS edge, replay target,
and model prior all operate on an unordered two-placement pair.

V1 includes the real pair-native runtime:

- full Rust-legal start-of-turn row tables,
- source-tagged pair candidate selection,
- exact hot-window tactical candidate support,
- blockwise-exact direct pair retrieval,
- anchor-conditioned completion,
- structured exploration and diagnostic canaries,
- symmetric low-rank biaffine `pair_joint_logits` over bounded candidate rows,
- `gumbel_sequential_halving_v1` root admission,
- proposal-aware PUCT with cached progressive widening,
- candidate-aware replay and training targets,
- fixed wall-clock evaluation against a fair sequential DAG baseline.

V1 deliberately excludes speculative complexity that would make failures hard
to diagnose: V2 tactical oracles, ANN retrieval before exact retrieval proof,
learned dynamics, materialized all-pair input tokens, hardcoded two-placement
solvers, and trainable uncertainty/completed-Q heads before their targets are
stable.

## Non-Negotiable Contracts

### Pair Action

```text
PairAction = unordered canonical pair of two distinct start-of-turn legal cells
```

Both cells reference the same original full Rust-legal table for the current
two-placement turn. The first placement does not expand, shrink, or reorder the
second-placement candidate table inside that turn.

Required invariants:

- `(a, b)` and `(b, a)` have one canonical key.
- Both cells are distinct and legal in the start-of-turn row table.
- Pair rows carry legal-row references, phase, row-table identity, hash
  metadata, schema version, and D6-stable identity.
- Normal pair edges apply both stones atomically through the exact Rust engine.
- Opening `(0, 0)` and rare engine-declared one-placement terminal phases are
  explicit single-action exceptions.

### Legal And Tactical Rows

Runtime row families are:

```text
LEGAL rows = full Rust-legal placement rows
ADMITTED_SINGLE rows = optional sampled single-cell rows
ADMITTED_PAIR rows = sampled canonical pair rows
TACTICAL rows = labels, proposals, candidate-support facts, target-support metadata
```

Main neural self-play and training use:

```text
selfplay.legal_row_mode = "full_rust_legal"
selfplay.tactical_mode = "proposal_and_label"
```

Hard threat filtering is not a legal-row source for flagship neural self-play
or training. Any retained threat-filtered path must be quarantined as
`diagnostic_threat_filter_v0` or classical-search tooling.

### Tactical Support, Not Solver

Exact hot-window facts improve candidate support and observability. They do not
force normal two-placement moves, collapse candidate sets, or filter
non-tactical candidates.

On normal two-placement turns:

```text
tactical_candidate_source = true
tactical_reason = "own_hot_completion" | "opponent_hot_cover" | "impossible_to_cover_flag"
search_required = true
legal_table_filtered = false
```

Hardcoded non-search actions are allowed only for true one-placement phases:

```text
hardcoded_action = true
hardcoded_reason = "opening_center" | "single_placement_terminal_win"
search_performed = false
```

### Candidate Metadata

Every admitted pair candidate is an auditable object:

```text
candidate_id
pair_key
first_legal_row_id
second_legal_row_id
row_table_schema_version
source_contributions[]
proposal_propensity_metadata
forced_exploration_flag
terminal_exact_flag
terminal_equivalence_flag
target_support_flags
admission_generation
root_or_interior
```

Each source contribution records:

```text
source_type
source_rank
source_weight
local_probability_or_score
quota_id
inclusion_kind
exact_inclusion_probability
heuristic_propensity
correction_mode
```

`inclusion_kind`:

```text
stochastic_sample
deterministic_top_k
tactical_protected
structured_quota
diagnostic_canary
unknown
```

`correction_mode`:

```text
exact_importance
clipped_propensity
uncorrected_logged
training_forbidden
```

Deterministic top-k, tactical-protected, and quota candidates must not be
treated as if they have clean stochastic inclusion probabilities. Missing
source or proposal-propensity metadata is a search-admission error.

## V1 Runtime Architecture

```text
Rust exact simulator
  -> start-of-turn full legal rows
  -> exact hot-window tactical payload
  -> graph/set encoder over STATE, TURN, STONE, LEGAL, WINDOW6
  -> legal-cell embeddings h_i
  -> cell_marginal_logits
  -> pair_completion_logits
  -> pair_proposal_score
  -> pair_candidate_selector_v1
  -> bounded candidate reservoir
  -> symmetric low-rank biaffine pair_joint_logits
  -> gumbel_sequential_halving_v1 root admission
  -> proposal-aware PUCT and progressive widening
  -> candidate-aware replay targets
```

Runtime authority is split across three components:

| Component | Owns | Must Not Own |
|---|---|---|
| `pair_candidate_selector_v1` | source quotas, canonicalization, deduplication, tactical support, source metadata, selector telemetry | final move choice |
| `pair_scorer_v1` | cheap retrieval scores, blockwise-exact top-k, bounded rich rerank, score batching, scoring telemetry | candidate-source semantics, exhaustive all-pair scoring |
| `sampled_joint_pair_v1` | pair prior construction, root Gumbel admission, progressive widening, pair expansion/backup, proposal correction, search telemetry, target metadata | behavior triggered merely by model head names |

The final root decision is always a canonical unordered `PairAction`. Cell
marginals, pair completions, retrieval scores, and pair-to-single projections
may be logged or trained as auxiliaries, but they cannot choose final
placements.

## Candidate Sources

Candidate generation uses source quotas rather than one blended sampler.

| Source | Purpose | V1 Contract |
|---|---|---|
| `terminal_exact_v1` | exact hot-window completion/cover support | non-optional; protected from eviction; never forces normal two-placement choice |
| `direct_pair_retrieval` | retrieve strong relational pairs | blockwise-exact before ANN/MIPS; D6, diagonal, dedup, row-order, chunking proof required |
| `anchor_conditioned_completion` | recover useful autoregressive proposal structure | train and infer with unordered-safe orientations |
| `cell_marginal_cross` | stable dense-policy support and debug baseline | capped cross-product of top marginal cells |
| `structured_diversity` | prevent support collapse | distance buckets, windows, cover sets, axis diversity, novelty |
| `blind_canary` | diagnostic support-collapse probe | tiny deterministic-seeded quota, may shrink in mature runs |
| `rich_pair_rerank` | rank bounded source pool | scoring stage, not a candidate source; cannot evict protected tactical rows or define the pool alone |

`direct_pair_retrieval` starts as blockwise-exact retrieval over a cheap
symmetric score:

```text
u_i = W_u h_i
v_i = W_v h_i
s_prop(i, j) = 0.5 * (u_i dot v_j + u_j dot v_i) + cheap_pair_features(i, j)
```

Approximate ANN/MIPS retrieval is a later performance optimization only after
exact retrieval passes recall, D6, diagonal-mask, deduplication, row-order, and
chunking-stability tests.

## Search Contract

At each expanded full-turn node:

1. Query Rust for start-of-turn legal rows and tactical payload.
2. Build or validate legal-row and pair-row identity.
3. Run one model evaluation to produce value, legal-cell embeddings, and
   proposal heads.
4. Build one bounded candidate reservoir from all candidate sources.
5. Canonicalize and deduplicate pair rows.
6. Store source contributions and typed proposal-propensity metadata.
7. Score the bounded reservoir with `pair_joint_logits`.
8. Admit initial children with tactical protection, source quotas, diversity,
   and rich rerank score.
9. At root, apply `gumbel_sequential_halving_v1`.
10. At interior nodes, use proposal-aware PUCT.
11. Widen by revealing cached reservoir rows.
12. Expand selected pair edges atomically through Rust and back up value.

Interior lifecycle is a hard performance contract: one neural evaluation, one
candidate-reservoir build, and one bounded scoring pass per expanded full-turn
node. Widening must not call the model once per pair, once per second
placement, or once per widening event. One higher-budget reservoir refill is
allowed only under explicit configuration and telemetry.

`gumbel_sequential_halving_v1` must emit:

```text
admitted set
gumbel values or admission order
simulation allocation
visit counts
Q values
completed-Q values
improved policy target over admitted candidates
target-support flags
```

Search priors use typed proposal correction only when assumptions are valid:

```text
prior_logit(a) =
  model_logit(a) - clip(log(proposal_correction_weight(a)), min_log, max_log)
P_C(a | s) = softmax(prior_logit(a) / prior_temperature)
```

Rows marked `uncorrected_logged` may enter search for recall but retain that
status in replay. Rows marked `training_forbidden` are diagnostic only.

Initial widening tuning range:

```text
alpha_pw = 0.4 to 0.6
c_pw = tune by root candidate recall and equal-wall-clock strength
```

## Model Contract

V1 model family:

```text
global_pair_biaffine_0
```

Input objects:

```text
STATE
TURN
STONE
LEGAL
WINDOW6
```

The model must not materialize all pair actions as graph input tokens. Pair
rows reference legal tokens or known-first placement tokens.

Required relations:

- legal cell to active window,
- stone to active window,
- legal cell to nearby legal cell offset buckets,
- legal cell to stone proximity buckets,
- window to overlapping window,
- window to shared cover cell,
- same-axis and same-line,
- pair row to first and second legal rows.

Required trainable V1 outputs:

```text
cell_marginal_logits
pair_completion_logits
pair_proposal_score
pair_joint_logits
value
terminal_tactical_v1
```

Logged or target-builder artifacts first, not required V1 trainable heads:

```text
pair_q_or_completed_q
uncertainty_or_regret
```

`cell_marginal_logits` and `pair_completion_logits` are proposal and auxiliary
views. `pair_joint_logits` over admitted unordered pair rows is the main model
prior consumed by sampled pair MCTS.

V1 pair scorer:

```text
pair_joint_logit(i, j) =
  cell_marginal_logit(i)
  + cell_marginal_logit(j)
  + symmetric_biaffine(h_i, h_j)
  + symmetric_mlp(state, h_i + h_j, abs(h_i - h_j), h_i * h_j, pair_features)
```

Order-invariant biaffine term:

```text
symmetric_biaffine(i, j) =
  0.5 * (h_i^T U h_j + h_j^T U h_i)
```

or enforce `U = U^T` by parameterization. A naive `h_i^T U h_j` term is not
acceptable for unordered pairs.

Required V1 pair features:

- axial distance,
- same-axis indicator,
- same-line indicator,
- same-window indicator,
- terminal exact win flag,
- terminal-equivalent win flag,
- terminal exact cover flag,
- covers-all-opponent-win-requirements flag,
- impossible-to-cover state flag,
- legal-row phase flags.

Source-contribution features are an ablation group only; the default pair
scorer should learn board geometry and pair interaction rather than shortcutting
on source labels.

V2-only pair feature expansion:

- overload count,
- shared-pivot count,
- fork count,
- axis-diversity pressure score,
- bounded tactical-race score,
- non-terminal hot-window creation score,
- opponent hot-window cover count,
- overlap cover count.

## Training And Replay Contract

Replay must store:

```text
candidate_pairs
candidate_selector_version
terminal_tactical_v1
candidate_source_contributions
proposal_propensity_metadata
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
target_support_flags
terminal_equivalence_flags
candidate_selection_reason
selected_pair
search_surprise_metrics
neural_calls_per_expanded_full_turn_node
reservoir_refill_events
```

Training target stack:

1. Pruned candidate-set joint pair posterior.
2. Completed-Q or regularized pair posterior when visits are sparse.
3. Marginal single-cell target derived from the pair posterior.
4. Conditional second-cell target for meaningful first cells.
5. Softened pair target for lower-mass plausible candidates.
6. Sampled ranking or sampled-softmax over admitted and sampled-negative
   candidates, corrected by typed proposal metadata where valid.
7. Exact tactical auxiliary labels.
8. Value targets and optional pair-regret/completed-Q distillation after the
   completed-Q target contract is stable.

Absolute rule:

```text
unsampled legal pair != negative pair
```

Only admitted pairs judged weak by search or explicit sampled negatives with
known proposal metadata may contribute negative ranking signal.

Pair target completeness must be explicit:

```text
support_type = "exhaustive_legal_pair_table"
support_type = "admitted_candidate_set_without_explicit_negatives"
support_type = "admitted_candidate_set_with_explicit_negatives"
support_type = "completed_q_candidate_posterior"
```

Terminal-equivalent targets do not train arbitrary filler-stone preferences. If
a true one-placement phase or exact transition says one cell ends the game,
train the cell/tactical heads strongly and either omit arbitrary pair-policy
training or collapse equivalent pair mass with documented deterministic
tie-breaks.

Conditional second-cell targets must emit both unordered-safe views when legal
and meaningful:

```text
pi(second = j | first = i)
pi(second = i | first = j)
```

Training only the canonical first cell is forbidden.

## Tactical Contract

Rust remains the production source for tactical labels.

V1 tactical payload:

```text
TerminalTacticalSetV1 {
  status
  status = quiet | hot_completion_available | hot_cover_required | hot_cover_impossible
  winning_single_cells
  hot_completion_pairs
  terminal_equivalent_pairs
  opponent_win_requirements
  hot_cover_pairs
  impossible_to_cover
}
```

V1 labels:

- immediate one-placement wins for one-placement phases,
- immediate two-placement 4/5-window completion pairs,
- terminal-equivalent winning pairs,
- opponent immediate win requirements,
- pair covers all opponent immediate win requirements,
- impossible-to-cover opponent immediate win requirements.

V2 tactical labels are deferred until V1 is stable:

- own hot-window creation,
- overload creation,
- shared pivot,
- multi-axis fork,
- bounded hot-window race,
- non-terminal pressure score,
- axis-diversity count.

## Evaluation Gates

V1 cannot be accepted by equal-simulation strength alone. Final acceptance
requires equal-wall-clock proof under documented hardware, batching,
candidate-budget, opponent-checkpoint, confidence-interval, and arena stopping
protocols.

Minimum acceptance gates:

- sampled joint-pair MCTS beats a fair sequential afterstate DAG baseline at
  equal wall-clock on mature checkpoints;
- flagship self-play and training preserve full Rust-legal row identity;
- hard threat-filtered `LEGAL` tables are absent from main runtime;
- root move selection returns canonical unordered `PairAction`;
- pair-to-single projection is never final move choice;
- normal two-placement hot-window states are searched, not force-played;
- exact deterministic tactical fixtures have 100% inclusion for own
  hot-window completion pairs, opponent cover pairs, and impossible-to-cover
  classification unless a deliberately excluded class is named;
- candidate recall includes best-search pair, audit top-k mass,
  equivalence-class recall, and regret-weighted recall;
- direct pair retrieval improves final candidate-pool recall over unary and
  conditional-only sources;
- blockwise-exact retrieval passes D6, diagonal-mask, deduplication,
  legal-row-order, and chunking-stability tests before ANN/MIPS enters normal
  runtime;
- unsampled legal pairs are never trained as negatives;
- pair target entropy does not collapse prematurely;
- D6 preserves legal-row identity, pair-row identity, policy consistency, and
  value consistency within documented tolerances;
- batching, queue backpressure, CPU candidate-generation time, GPU utilization,
  candidate-generation p95, neural calls per expanded full-turn node, reservoir
  refill rate, and pair-scores/sec meet phase budgets;
- obsolete runtime paths are removed or quarantined with import and code-search
  proof.

Required metric groups:

| Group | Examples |
|---|---|
| Legal/action support | `full_legal_row_count_per_root`, `legal_pair_count_per_root`, `admitted_pair_count_per_root` |
| Candidate recall | `candidate_recall_best_search_pair`, `candidate_recall_audit_topk_mass`, `regret_weighted_candidate_recall`, `candidate_recall_equivalence_class` |
| Tactical support | `terminal_tactical_v1_status_mix`, `tactical_candidate_support_recall`, `tactical_suite_accuracy` |
| Retrieval | `direct_retrieval_recall_at_k_vs_exhaustive_small_state`, `retrieval_d6_consistency`, `retrieval_chunking_stability` |
| Search quality | `visit_count_gini`, `search_surprise_kl`, `root_q_variance`, `pair_prior_calibration` |
| Training health | `normalized_pair_target_entropy`, `policy_entropy_by_phase`, `value_calibration_by_phase` |
| Performance | `node_expansions_per_second`, `pair_scores_per_second`, `inference_latency_p50_p95`, `candidate_generation_latency_p95`, `gpu_utilization` |

## Implementation Phase Matrix

| Phase | Goal | Runtime Change | Required Evidence | Stop Gate |
|---|---|---|---|---|
| 0. Inventory | Identify current pair heads, search modes, legal filters, replay targets | none | `rg` audits for pair-head consumption, pair strategies, `constrain_threats`, `root_pair_mcts`, `full_pair_mcts`, `pair_logits_to_action_logits`; `git status --short` | runtime pair authority cannot be identified |
| 1. Legal rows | Make full Rust legal rows authoritative | quarantine threat filters; tactical rows become labels/proposals | row-family tests; config tests disabling threat filters; one-placement hardcoded-action metadata tests; code-search audit | `LEGAL` can be silently threat-censored or normal two-placement tactics can auto-play |
| 2. Pair identity | Make pair rows strict cross-boundary contracts | start-of-turn pair table, canonical keys, D6 identity | row identity/D6 tests; no intra-turn legal-table expansion tests; Rust/Python fuzz; illegal/duplicate/wrong-phase negatives | pair identity cannot survive replay, inference, and search |
| 3. Candidate selector | Replace legal-order first-N generation | implement all V1 sources, metadata, dedup, exact retrieval | source quota tests; tactical support tests; direct retrieval tests; D6/chunking/order audits; recall artifacts | candidate lacks metadata; tactical candidates can be evicted; selector depends on legal-row order |
| 4. Pair MCTS | Make sampled pair macro-actions the main search runtime | atomic pair expansion, root Gumbel, proposal-aware PUCT, cached widening | MCTS unit tests; hot-window search traces; throughput profiles; proposal-correction artifacts; head-name audit | pair search needs sequential fallback or pair-to-single final projection |
| 5. Biaffine model | Implement V1 graph pair model | `global_pair_biaffine_0`, bounded symmetric pair scorer | shape tests, symmetry tests, phase tests, window overflow tests, performance profile | pair rows inflate main attention sequence or symmetry fails |
| 6. Targets | Replace raw sparse pair training | replay metadata and candidate-aware target builder | projection/pruning tests; unordered conditional tests; terminal-equivalence tests; unsampled-negative negatives; training smoke | replay cannot distinguish admitted, explicit-negative, forced, and unsampled rows |
| 7. Curriculum | Train without support collapse | bootstrap/growth/mature schedules, replay stratification, hard-state replay | self-play smoke, source telemetry, replay distribution, tactical suite, distributed performance artifact | entropy collapses before candidate/tactical recall and throughput are healthy |
| 8. Cutover | Prove superiority and delete obsolete paths | fair ablations and runtime cleanup | equal-simulation, equal-wall-clock, equal-neural-eval arenas; D6 report; performance profile; deletion/import proof | pair macro-search loses fair equal-wall-clock comparison after proposal maturity |

## Non-Goals And Deferred Work

| Not V1 Mainline | Reason |
|---|---|
| full exhaustive pair MCTS | correct semantics, unacceptable branching |
| sequential afterstate MCTS as main planner | useful baseline only; imposes ordered half-move statistics |
| MuZero learned dynamics | Rust rules are exact; learned dynamics add error and do not solve candidate control |
| materialized all-pair input tokens | reintroduces `O(A^2)` attention path |
| pair-only sparse visit training | starves dense supervision and mislabels sampled support |
| symmetric MLP as final scorer | ablation only; V1 mainline uses explicit biaffine/bilinear interaction |
| hard threat-filtered `LEGAL` rows | tactical knowledge must not censor semantic legal rows |
| legal-order first-N pair generation | biased by row order; diagnostic only |
| ANN/MIPS retrieval before exact proof | performance optimization after exact retrieval correctness/recall evidence |
| trainable `pair_q_or_completed_q` and `uncertainty_or_regret` heads | logged artifacts first; promote only with stable targets and winning consumers |
| V2 tactical oracles | overloads, pivots, forks, and races wait until V1 selector/search/targets are stable |
| learned set decoders or generative widening samplers | experiments only after V1; harder to debug and calibrate |

## Practical Build Order

1. Cut over start-of-turn full legal rows and threat-filter quarantine.
2. Implement canonical pair row identity, D6 tests, and one-placement exception
   metadata.
3. Implement `pair_candidate_selector_v1` with all V1 sources.
4. Implement typed proposal-propensity metadata and candidate-support flags.
5. Implement `gumbel_sequential_halving_v1`.
6. Implement proposal-aware PUCT and cached reservoir widening.
7. Use existing graph trunk with `STATE`, `TURN`, `STONE`, `LEGAL`, `WINDOW6`.
8. Implement bounded symmetric low-rank biaffine pair scorer.
9. Add active `WINDOW6` caps and overflow telemetry.
10. Add replay metadata for candidates, source contributions, proposal
    metadata, Q/completed-Q artifacts, target-support flags, and selected pair.
11. Implement pruned joint, marginal, unordered conditional, softened pair, and
    terminal-equivalence targets.
12. Build fixed tactical candidate-support fixtures and tiny-state audit
    oracle.
13. Run equal-wall-clock comparison against the current best baseline and fair
    sequential DAG baseline.

## Source Of Truth

This proposal is the implementer-facing V1 architecture summary. The long-form
decision record remains:

```text
Docs/FINAL_HEXORL_IMPLEMENTATION_PLAN.md
```

If the two documents disagree, prefer this proposal for V1 scope and the
long-form plan for detailed rationale unless a later phase document explicitly
supersedes both.
