# Phase 3 Plan: Autotuning And Champion Selection - 2026-04-27

This is the third document in the consolidated three-phase Hexo improvement
plan:

1. `Docs/PHASE1_RESTNET_ACTION_CONTRACT_SCOUT_20260427.md`
2. `Docs/TRANSFORMER_ARCHITECTURE_ABLATIONS_FOR_HEXO_20260427.md`
3. `Docs/AUTOTUNING_METHODS_AND_48H_PLAN_20260427.md`

Phase 3 tunes the survivors. It is not a broad research sweep. By the time this
phase starts, Phase 1 and Phase 2 should have already answered:

```text
best_current_33:
  strongest current crop model

best_restnet_33:
  strongest attention-inside-crop model

candidate_policy_33:
  whether action-keyed candidate priors are safe

graph_hybrid_0:
  whether the crop-compatible sparse-token hybrid is viable as a scout
```

The purpose of Phase 3 is to turn the best survivor into the strongest model
possible in a fixed compute budget, while still keeping enough tournament
coverage to avoid choosing a noisy checkpoint.

## Executive Plan

Use:

```text
ASHA/BOHB target:
  static architecture/search choices

PB2 target, with PBT fallback only until PB2 exists:
  dynamic schedules for LR, exploration, replay freshness, and loss weights

checkpoint league:
  primary strength signal

classical survival:
  early external anchor, never a substitute for wins

Hexo tactical suites:
  forced-block, win-now, cover-set, outside-window, and pair-turn checks
```

Default 48-hour compute shape:

```text
0-4h:
  calibration and finalist import

4-16h:
  static ASHA/BOHB finalist narrowing target

16-32h:
  PB2 schedule search target, or clearly labeled PBT fallback

32-44h:
  champion training

44-48h:
  final arena and checkpoint selection
```

Protect the champion training block. If earlier phases run long, cut scout
trials first.

## 2026-04-28 Implementation Completeness Audit

Current status:

```text
ASHA-style rung pruning: partially implemented
BOHB: not implemented
PBT-style exploit/explore: partially implemented
PB2: not implemented
checkpoint league / tactical suites: partially implemented with proxies
```

This matters because Phase 3 is supposed to select a champion, not merely run a
long script. A scheduler can only be called complete when the method named in
the docs is actually the method being used.

### ASHA Completion Requirements

The current supervisor has rung resources and top-fraction promotion. That is a
useful ASHA-style loop, but it is not complete until it has:

- explicit rung tables persisted as first-class artifacts;
- asynchronous promotion semantics or a documented synchronous-successive-
  halving mode with no ASHA overclaim;
- per-trial resource accounting based on completed epochs, wall-clock, and
  self-play positions;
- deterministic replay of each prune/promote decision from persisted metrics;
- separate hard-failure quarantine vs normal statistical pruning;
- score normalization only within the correct rung and comparable resource
  budget.

Acceptance tests:

```text
test_asha_rungs_are_persisted_and_replayable
test_asha_promotion_uses_same_resource_level_only
test_asha_hard_failures_are_quarantined_not_ranked
test_asha_scheduler_replay_reproduces_promotions
```

### BOHB Completion Requirements

BOHB is currently a plan word, not an implementation. A no-compromise BOHB path
must include Hyperband brackets plus model-based configuration sampling.

Required behavior:

```text
Hyperband brackets over resource budgets
KDE/TPE-style density model over good vs bad completed configs
random exploration fraction for robustness
conditional search space handling for model family / sparse / graph settings
separate treatment of invalid/crashed configs
persisted bracket, budget, sample source, and density-model state
```

BOHB must choose static/discrete or mixed static settings such as:

```text
model family
sim count
candidate budget
sparse prior stage
head bundle
batch size
graph token budget
```

Acceptance tests:

```text
test_bohb_creates_hyperband_brackets
test_bohb_fits_good_bad_density_models
test_bohb_samples_from_model_after_warmup
test_bohb_handles_conditional_graph_space
test_bohb_replay_reconstructs_sample_source
```

Until these exist, use the phrase "ASHA/static narrowing" rather than
"ASHA/BOHB".

### PB2 Completion Requirements

The current Phase 3C implementation is PBT-like: bottom trials clone compatible
top trials and mutate parameters by random multiplicative exploration. That is
not PB2.

A real PB2 implementation must add:

- a continuous response model over recent trial observations;
- uncertainty-aware proposals for dynamic hyperparameters;
- acquisition logic that balances predicted score improvement and uncertainty;
- clamped proposals inside the documented ranges;
- compatibility-safe checkpoint/optimizer/replay transfer;
- no shared mutable replay-buffer object between donor and cloned trial unless
  the replay ownership model explicitly supports shared lineage;
- persisted model-fit inputs, normalized observations, proposed mutations,
  accepted mutations, rejected mutations, and final chosen values;
- explicit rejection events when cloning, mutation, or compatibility checks
  fail;
- deterministic replay of the PB2 decision from stored metadata.

Dynamic knobs controlled by PB2:

```text
learning rate
weight decay
c_puct
c_puct_init
Dirichlet fraction
Dirichlet alpha mode and magnitude
PCR low-sim probability
buffer recency decay
value loss weight
auxiliary loss multiplier
sparse policy loss
pair policy loss
regret replay fraction
```

Acceptance tests:

```text
test_pb2_fits_continuous_response_model
test_pb2_acquisition_uses_uncertainty
test_pb2_proposals_are_clamped_and_logged
test_pb2_respects_conditional_parameters
test_pb2_scheduler_replay_reproduces_mutations
test_pbt_remains_named_baseline_not_spec_path
```

Until these pass, Phase 3C should be called a PBT fallback, not PB2.

### Current Supervisor Search-Space Reality

The current implementation uses narrower, host-guarded ladders than the broad
ideal search space above. In particular, graph/candidate trials may be scaled
down to lighter simulation counts, smaller worker counts, and selected
candidate budgets to survive local CPU/RAM/inference limits.

That is acceptable as a runtime safety adaptation, but reports must separate:

```text
planned no-compromise search space
actual host-limited search space used in a run
families blocked by hard safety gates
families pruned statistically within a fair rung
```

Acceptance tests:

```text
test_autotune_manifest_records_planned_and_actual_search_space
test_host_guards_are_reported_separately_from_scheduler_pruning
test_blocked_family_is_not_counted_as_method_failure
```

### Scorecard Completion Requirements

The current scorecard uses useful live diagnostics, but some tactical scores are
proxy values rather than authored tactical-suite results. A champion-selection
scorecard is complete only when:

- checkpoint league ratings are persistent and evaluated on both colors;
- tactical suites contain replayable positions with expected legal action sets;
- outside-window suites actually include actions outside the 33x33 crop;
- candidate recall is split into protected training recall and live discovery
  recall;
- illegal/crash/truncation penalties cannot be hidden by good loss curves;
- final selection uses league lower-confidence bound, not raw mini-match
  winrate.

Acceptance tests:

```text
test_checkpoint_league_persists_lcb
test_tactical_suite_positions_are_replayable
test_outside_window_suite_has_outside_crop_expected_actions
test_scorecard_uses_discovery_recall_for_sparse_gates
test_final_score_penalizes_illegal_crash_and_truncation
```

Completion rule:

```text
Do not mark Phase 3 complete because the supervisor ran for 48 hours.
Mark it complete only when ASHA/BOHB, PB2, scorecards, and final arena all
match the contracts above.
```

## Finalist Pool

The pool depends on Phase 2 results.

### If The Graph Transformer Passes

Tune:

```text
best_current_33
best_restnet_33
candidate_policy_33 or restnet_candidate_policy_33
graph_hybrid_0
```

Expected winner prior:

```text
graph_hybrid_0 = crop-compatible sparse token Transformer + action-keyed priors
```

But only keep it as favorite if it preserves tactical reliability and throughput
well enough for self-play volume. It should not be treated as the true global
window replacement until the Phase 2 spec-match work is implemented.

### If The Graph Transformer Does Not Pass

Tune:

```text
best_current_33
best_restnet_33
candidate_policy_33
best_restnet_33 + candidate_policy_33, if stable
```

In this fallback, Phase 3 still benefits from Phase 2 because the action-keyed
policy path can reduce fallback priors even if the full graph model is not yet
ready.

## Graph Architecture Alternatives

Phase 3 should not assume the maximal graph design is the only useful
Transformer-family candidate. It should tune and compare a small set of graph
alternatives that share the same strict data contract from
`Docs/SPEC_FIX_MATCH_GLOBAL_GRAPH_MODEL_20260428.md`.

All graph alternatives must preserve:

```text
all legal Rust action rows
all 12 D6 transforms
opening and two-placement turn masks
exact engine-derived tactical labels
future global opponent-policy target table
legal pair-policy masks and targets
global keyed MCTS priors
dashboard/replay graph inspection
```

Candidate families:

| Family | Purpose | Phase 3 role |
|---|---|---|
| `global_xattn_0` | Legal-action queries cross-attend to global context tokens. | First serious all-legal global action baseline. |
| `global_line_window_0` | Emphasizes `WINDOW6`, `LINE`, `COVER_SET`, and multi-axis tactical structure. | Tests whether explicit Hexo structure is the main win. |
| `global_pair_twostage_0` | Scores legal first moves, then conditional legal second moves and chunked joint pairs. | Tests pair-turn planning without full `A^2` graph attention everywhere. |
| `global_graph_full_0` | Full relation-biased token graph with all required token families. | Tests the maximal architecture before champion scale. |
| `global_hybrid_action_0` | Hex-masked crop trunk as local feature/helper, but global legal action policy is primary. | Bridge/control against current crop models. |
| `global_graph768_champion` | Scaled version of the best validated global design. | Final champion candidate, not the first debugging target. |

Comparison rule:

```text
The shared graph/action contract must be identical across graph alternatives.
Only the model architecture should change.
```

This prevents a weaker contract from winning by accidentally dropping hard
positions, leaking target features, using different legal masks, or hiding
fallback priors.

## What The Tuner Should Not Do

Avoid these traps:

```text
Do not tune every head independently.
Do not tune many architecture families that failed Phase 1/2 gates.
Do not treat longer losses against the classical opponent as a real win.
Do not let low-sim PCR train the policy head as if it were full-search data.
Do not choose the highest-sim setting without a wall-clock penalty.
Do not let candidate budgets cap away wins, forced blocks, or cover cells.
Do not compare graph alternatives that use different legal-action or tactical
label contracts.
```

For PCR samples:

```text
policy_weight = 0 for low-search PCR policy targets
value/lookahead/aux weights may remain active
```

This preserves value and representation learning without poisoning the policy
head with low-quality visit distributions.

## Static Search Space

ASHA/BOHB should choose discrete settings that are expensive or awkward to
mutate mid-run.

| Parameter | Values |
|---|---|
| model family | Phase 1/2 finalists only |
| model size | inherited finalist size, optional `96x12`/`128x16` scout mirrors |
| full sims | `800`, `1200`, optional `1600` finalist-only |
| PCR low sims | `192`, `256`, `384` |
| policy top-k | `64`, `96`, `128` |
| candidate budget | `256`, `384`, `512`, graph only |
| graph family | `global_xattn_0`, `global_line_window_0`, `global_pair_twostage_0`, `global_graph_full_0`, optional `global_hybrid_action_0`, then `global_graph768_champion` |
| graph token set | finalist token set, optional one lower-risk variant |
| subtree reuse | `True` |
| temperature schedule family | fast cool, slow cool |
| head bundle | structural, prediction, regret, full_aux_light, graph_tactical |

Recommended defaults:

```text
fixed-window finalist:
  full_sims = 1200
  pcr_low_sims = 256
  policy_top_k = 96

graph finalist:
  full_sims = 800 or 1200
  all legal action rows preserved
  context token budget only if exact legal/tactical rows remain intact
  pair policy = prior-shaping first, macro expansion only if proven stable
```

Use `1600` sims only for a late finalist check. It is too expensive for broad
scouting unless throughput is much better than expected.

## Dynamic PB2 Search Space

The spec-matching scheduler is PB2. A PBT fallback can tune the same values for
debugging and comparison, but it does not make Phase 3 complete.

| Parameter | Range |
|---|---:|
| current LR | `5e-3` to `1e-2` |
| weight decay | `1e-5` to `5e-4` |
| `c_puct` | `1.1` to `2.2` |
| `c_puct_init` | `1000` to `20000` |
| `dirichlet_fraction` | `0.10` to `0.35` |
| fixed `dirichlet_alpha` | `0.01` to `0.05` |
| scaled alpha total | `5.0` to `12.0` |
| `pcr_low_sim_prob` | `0.50` to `0.85` |
| `buffer.recency_decay` | `0.95` to `0.995` |
| value loss weight | `1.0` to `2.0` |
| auxiliary loss multiplier | `0.5` to `1.5` |
| sparse/candidate policy loss | `0.10` to `0.75`, candidate models only |
| pair policy loss | `0.02` to `0.25`, graph only |
| graph aux multiplier | `0.5` to `1.5`, graph only |
| regret replay fraction | `0.00` to `0.12`, if regret heads are enabled |

Recommended initial center:

```text
LR = 7e-3
weight_decay = 1e-4
c_puct = 1.6
c_puct_init = 8000
dirichlet_fraction = 0.22
fixed dirichlet_alpha = 0.02
scaled alpha total = 8.0
pcr_low_sim_prob = 0.70
buffer.recency_decay = 0.98
value loss weight = 1.3
auxiliary loss multiplier = 1.0
sparse/candidate policy loss = 0.35
pair policy loss = 0.08
```

Use either fixed alpha or scaled-total alpha in a trial, not both as active
noise at once. Scaled-total alpha is often easier to compare across positions
with different plausible root widths:

```text
effective_alpha = scaled_alpha_total / max(root_candidate_count, 1)
```

## Model Heads

Use bundles, not independent head toggles.

| Bundle | Heads | Role |
|---|---|---|
| structural | policy, value, lookahead_4/12/36, axis | strong baseline |
| prediction | structural + opp_policy + moves_left | opponent and horizon modeling |
| regret | structural + regret_rank + regret_value | high-learning-potential states |
| full_aux_light | structural + opp_policy + moves_left + regret heads | default if stable |
| graph_tactical | graph policy/value + win/block/cover/pair auxiliaries | transformer finalist |

Moves-left target should be normalized or binned:

```text
moves_left_norm = log1p(moves_left) / log1p(max_game_turns)
```

Set:

```text
moves_left_weight = 0 for truncated/non-terminal records
```

For the transformer, prioritize heads that match Hexo's structure:

```text
win_now
opp_win_next
cover_set_status
threat_count_delta
pair_policy
regret_rank / regret_value
```

## Scoring Method

The tuner needs one scalar score, but the score must be assembled from signals
that match Hexo's failure modes.

## Bug-Resistant Automated Evaluation

The scorecard must distinguish real model strength from engine bugs, target
bugs, masking bugs, and evaluation leaks. A model should not be promoted simply
because it discovers a broken edge case.

Required bug-isolation layers:

### Engine Invariant Suite

Run before every serious tuning batch:

```text
legal move set contains only empty cells within radius 8 of an existing stone
opening is exactly player 0 at origin
turn order is P0, P1/P1, P0/P0, ...
winner detection matches six-in-a-row oracle on every axis
no post-terminal moves are accepted
replay move history reconstructs the same terminal state and winner
threat filtering never changes human/game legal moves
```

### Differential Oracles

Maintain at least one slow, simple reference implementation for small/medium
positions:

```text
Python reference legal-move generator
Python reference win detector
Python reference 6-window scanner
Rust engine result
dashboard replay reconstruction
```

Promotion requires agreement between independent paths on fixed and randomized
states.

### Metamorphic Tests

For every model family and replay target:

```text
all 12 D6 transforms preserve legality, winner, threats, targets, and pair rows
move-history serialize/deserialize/replay is identity
policy target probability mass is unchanged by transform
legal masks are bijective under transform
MCTS with uniform priors is invariant up to transformed action keys
```

### Tactical Holdout Suites

Use hand-authored and generated positions that are never used for scheduler
training labels:

```text
win-now
forced block
two-placement cover
unblockable fork
block plus counterattack
outside-window win/block
separated-cluster defense
late-game high-legal-count state
```

Each suite stores expected legal action sets or pair sets, not just expected
win/loss.

### Shadow Evaluation

Run each promoted model under multiple evaluators:

```text
direct policy sampling
PUCT/MCTS with model priors
MCTS with uniform priors for engine sanity
classical opponent
checkpoint league
fixed tactical suite
```

If a model only improves under one evaluator and fails invariant/tactical
suites, quarantine it as a probable implementation artifact.

### Anti-Leak Tests

Fail promotion if:

```text
candidate/action features contain target probability or target-present bits
dashboard/debug-only labels enter live model features
future opponent target rows leak into current-state features
low-PCR policy rows train policy as full-search rows
truncated games train terminal value or regret heads
```

### Bug Sentinel Metrics

Persist and gate on:

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
```

Any nonzero legal/replay/D6/oracle mismatch is a hard failure for architecture
promotion, regardless of apparent Elo.

### Checkpoint League

Maintain a league containing:

```text
top 6 active checkpoints
best previous checkpoint
best_current_33 reference
best_restnet_33 reference
graph_hybrid_0 reference, if available
one conservative classical-reference model
```

Use both colors and fixed opening seeds when possible. Select by lower
confidence, not raw mean:

```text
league_lcb = rating_mean - 1.0 * rating_std
```

This avoids promoting a checkpoint that won a tiny noisy mini-match.

### Classical Survival Score

Classical survival is useful because weak models may rarely win. It should be
secondary because a model can learn to delay losses.

Do not use classical survival for very early pruning. In the first few epochs,
models are often essentially random; a move-20 loss versus a move-30 loss is
mostly noise, not strategy. Treat early ASHA rungs as health checks until the
model has enough training to express a policy.

Recommended thresholds:

```text
epoch < 8:
  health/scaffold scoring only
  ignore classical survival and checkpoint-league strength

8 <= epoch < 12:
  strategy-prep scoring
  use target quality, tactical fixtures, outside-window diagnostics, value
  calibration, throughput, and hard bug gates
  keep classical survival visible but do not let it drive pruning

epoch >= 12, preferably 12-14:
  classical survival becomes meaningful enough to enter ASHA/PB2 scorecards
```

Default ASHA resource ladder:

```text
8, 12, 14
```

Use fewer active trials rather than returning to 2/5-epoch rungs if compute is
tight. A 2/5-epoch rung is useful for crash/NaN detection only, not for
classical strength.

Per-game score:

```text
survival_ratio = clamp(moves / baseline_loss_p75[color], 0.0, 1.25)

if model wins:
  css_game = 1.00 + 0.05 * min(survival_ratio, 1.25)
elif model loses legally:
  css_game = 0.15 + 0.55 * min(survival_ratio, 1.00)
elif game truncates:
  css_game = 0.25 + 0.20 * min(survival_ratio, 1.00)
elif illegal move, crash, or invalid state:
  css_game = -0.50
```

Properties:

```text
a long loss can beat a short loss
a loss cannot beat a win
truncation is not treated as a hidden win
baseline normalization adapts by color
```

### Hexo Tactical Suite

Evaluate fixed tactical sets:

```text
4-window completion
5-window completion
single forced block
two-placement cover
block plus counterattack
unblockable-threat recognition
axis fork creation
separated-cluster defense
outside-window win
outside-window forced block
```

For graph/candidate-policy models, add:

```text
candidate_recall_winning_move
candidate_recall_forced_block
candidate_recall_two_placement_cover
fallback_prior_use_on_mcts_topk
missing_target_policy_mass
```

### Scheduler Score

Use z-scores within the current tuning generation:

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

Use the full `strength_score` only at `epoch >= 12`. Before that, use the
health/pre-classical formulas above and keep classical survival visible as a
diagnostic only.

If the league is not populated yet after the classical-survival threshold,
temporarily raise the weights on classical survival, tactical suite, and
policy/value diagnostics. Before that threshold, use the warmup or
pre-classical formulas above.

### Final Selection Score

Final selection should emphasize actual play strength:

```text
final_score =
    0.55 * z(final_league_lcb)
  + 0.20 * z(outside_window_robustness)
  + 0.15 * z(final_tactical_suite_score)
  + 0.05 * z(final_classical_survival_score)
  + 0.05 * z(final_classical_winrate)
  - 0.10 * z(illegal_or_crash_rate)
```

Throughput should break ties unless a model is too slow to generate enough
self-play for future training.

## 48-Hour Run Plan

Assumption:

```text
1 production epoch ~= 600 seconds
48 hours = 172800 seconds ~= 288 production epoch-equivalents
```

### Phase 3A: Import And Calibration, 4 Hours

Load the Phase 1/2 finalists. Measure true throughput under the same hardware
and data path.

Run:

```text
best_current_33 throughput + arena probe
best_restnet_33 throughput + arena probe
candidate_policy_33 throughput + recall probe, if available
graph_hybrid_0 throughput + recall probe, if available
```

Outputs:

```text
baseline_loss_p75 by color
initial league ratings
candidate recall sanity check
epoch-time ratios
memory feasibility
starting scorecard
```

Budget:

```text
~24 production epoch-equivalents
```

### Phase 3B: Static Finalist Narrowing, 12 Hours

Use ASHA/BOHB to narrow static choices:

```text
full_sims: 800 vs 1200
pcr_low_sims: 192 vs 256 vs 384
policy_top_k: 64 vs 96 vs 128
candidate_budget: 256 vs 384 vs 512, if applicable
head bundle: finalist-compatible bundles only
```

Keep the pool small:

```text
4 to 6 active trials by default
health_resource = optional 2-5 epochs for crash/NaN checks only
min_strategy_resource = 8 epochs
classical_survival_resource = 12-14 epochs
promotion = top half after each strategy rung
```

Default:

```text
asha_resources = 8,12,14
asha_promote_fraction = 0.5
```

Do not compare classical survival from sub-8-epoch checkpoints as a strength
signal. If a short health rung is used, prune only hard failures:

```text
non-finite losses
illegal/crash rate
zero self-play positions
broken replay/D6/oracle sentinels
runaway throughput/memory failures
```

Output:

```text
top 2 model families
top 2 search recipes
best head bundle per family
go/no-go for 1200 sims
go/no-go for graph/candidate policy in champion block
```

Budget:

```text
~72 production epoch-equivalents
```

### Phase 3C: PB2 Schedule Search, 16 Hours

Until real PB2 exists, this stage may run a clearly labeled PBT fallback. PBT
results are useful for screening, but they are not the completed Phase 3C
method.

Population:

```text
population = 6 to 8
perturb_interval = 2 epochs
generations = 4 to 6
```

Dynamic knobs:

```text
LR
weight_decay
c_puct
c_puct_init
dirichlet_fraction
dirichlet_alpha mode
pcr_low_sim_prob
buffer.recency_decay
value loss weight
auxiliary loss multiplier
sparse/candidate policy loss, if applicable
pair policy loss, if applicable
regret replay fraction, if applicable
```

PB2 is required for the spec-matching path because it models continuous
hyperparameter changes. PBT is acceptable only as a fallback/baseline:

```text
exploit:
  copy checkpoint/optimizer/replay pointer from a stronger compatible trial

explore:
  multiply continuous params by 0.8 or 1.2
  resample with probability 0.20
  clamp to valid range
```

Do not exploit across incompatible architectures unless a checkpoint-conversion
path is explicitly implemented. Copying a graph checkpoint into a crop model is
not a meaningful mutation.

Output:

```text
best schedule path
best exploration schedule
best replay freshness
best loss multipliers
champion recipe
shadow recipe
```

Budget:

```text
~96 production epoch-equivalents
```

### Phase 3D: Champion Training, 12 Hours

Train the best recipe continuously.

Primary:

```text
model = best Phase 3C family
search = best static recipe
schedule = best PB2 path, or explicitly labeled PBT fallback path
heads = best compatible bundle
checkpoint every epoch
evaluate every 2 epochs
```

Shadow, if compute allows:

```text
model = strongest alternative family
search = conservative 800/192 or 1200/256 recipe
```

Use EMA checkpoint as a candidate. Try model soup only for same architecture and
same heads.

Budget:

```text
~72 production epoch-equivalents
```

### Phase 3E: Final Arena, 4 Hours

Candidates:

```text
last 6 primary checkpoints
best EMA checkpoint
best shadow checkpoint
best_current_33
best_restnet_33
graph_hybrid_0, if not primary
```

Run:

```text
checkpoint league, both colors
classical opponent arena, both colors
tactical suite
outside-window suite
fixed eval sims
Dirichlet noise off
near-greedy temperature
```

Select by `final_score`.

Budget:

```text
~24 production epoch-equivalents
```

## Budget Summary

| Stage | Purpose | Time | Epoch-equivalents |
|---|---|---:|---:|
| 3A | Import and calibration | `4h` | `24` |
| 3B | Static finalist narrowing | `12h` | `72` |
| 3C | PB2 schedule search, PBT fallback only until PB2 exists | `16h` | `96` |
| 3D | Champion training | `12h` | `72` |
| 3E | Final arena | `4h` | `24` |
| Total | | `48h` | `288` |

If actual epoch time is slower than 600 seconds, cut Phase 3B trial count and
Phase 3C population. Protect Phase 3D and Phase 3E.

## Implementation Rough-In

### Trial State

Add or adapt a trial abstraction:

```python
TrialState(
    trial_id,
    family,
    cfg,
    model,
    optimizer,
    replay_handle,
    checkpoint_path,
    epoch,
    wall_time_s,
    metrics_history,
    score_history,
    mutation_history,
)
```

Required operations:

- create from config;
- train one epoch;
- generate self-play;
- evaluate checkpoint;
- save/load checkpoint;
- clone compatible trial;
- mutate dynamic config;
- log exploit/explore events.

### Evaluation Services

Implement evaluators as separate services so every family uses the same scoring
path:

```text
checkpoint league evaluator
classical survival evaluator
tactical suite evaluator
outside-window evaluator
candidate recall evaluator
throughput/memory reporter
```

### PB2 And PBT Mutation Hooks

Mutation must update:

- optimizer LR;
- optimizer weight decay;
- MCTS config for new self-play;
- Dirichlet config;
- replay sampler recency/regret settings;
- loss weights;
- sparse/candidate policy weights;
- pair-policy weights;
- graph auxiliary multipliers.

### Reporting

Every trial writes:

```text
config
family
static recipe
dynamic hyperparameters by epoch
score components
arena records
promotion/exploit events
checkpoint paths
candidate recall reports
replay stats
throughput stats
```

Final report writes:

```text
why the champion won
whether graph beat crop/ResTNet
whether 1200 beat 800 per wall-clock
whether candidate-policy priors helped
whether pair policy helped
whether regret replay helped
final league and classical scores
remaining failure modes
```

## Pruning And Promotion Gates

Prune immediately:

```text
illegal_or_crash_rate > 0
NaN/Inf loss
candidate recall below hard gates
policy target mass silently dropped
forced-block recall regression is severe
epoch time > 2.5x reference without matching score gain
```

Soft-prune:

```text
classical survival below baseline by > 0.5 sigma after 2 evals
league LCB below median by > 0.5 sigma after enough games
policy target quality collapses while value improves
truncation rate rises without winrate or league improvement
pair policy destabilizes first-placement search
```

Promote:

```text
top-quartile scheduler_score
zero illegal/crash rate
candidate recall gates satisfied if applicable
improving league or tactical score
throughput still plausible for champion training
```

## Expected Winning Recipes

If Phase 2 passes:

```text
model = graph512_turn_pair_prior
candidate_budget = 512
full_sims = 800 or 1200
pcr_low_sims = 256
policy_top_k = not applicable for global policy, legacy only if hybrid
pair policy loss = 0.05 to 0.15
dirichlet_fraction = 0.15 to 0.28
c_puct = 1.4 to 1.9
recency_decay = 0.975 to 0.99
```

If Phase 2 does not pass:

```text
model = best_restnet_33 + candidate_policy_33
full_sims = 1200
pcr_low_sims = 256
policy_top_k = 96
sparse policy loss = 0.25 to 0.50
dirichlet_fraction = 0.18 to 0.30
c_puct = 1.5 to 2.0
```

Conservative fallback:

```text
model = best_current_33 or best_restnet_33
full_sims = 800
pcr_low_sims = 192
policy_top_k = 96
```

## Bottom Line

Phase 3 is where the three plans become one system:

```text
Phase 1 gives strong crop baselines and action-keyed targets.
Phase 2 gives the transformer replacement and its safety metrics.
Phase 3 tunes only the survivors with a scorecard that understands Hexo.
```

The best possible 48-hour run is not the one with the most knobs. It is the one
that spends most of its compute on the few model families that passed the right
gates: tactical reliability, outside-window robustness, action-keyed prior
quality, and real checkpoint strength.
