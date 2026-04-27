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

best_graph_option1:
  whether the transformer window replacement is viable
```

The purpose of Phase 3 is to turn the best survivor into the strongest model
possible in a fixed compute budget, while still keeping enough tournament
coverage to avoid choosing a noisy checkpoint.

## Executive Plan

Use:

```text
ASHA/BOHB:
  static architecture/search choices

PB2/PBT:
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
  static ASHA/BOHB finalist narrowing

16-32h:
  PB2/PBT schedule search

32-44h:
  champion training

44-48h:
  final arena and checkpoint selection
```

Protect the champion training block. If earlier phases run long, cut scout
trials first.

## Finalist Pool

The pool depends on Phase 2 results.

### If The Graph Transformer Passes

Tune:

```text
best_current_33
best_restnet_33
candidate_policy_33 or restnet_candidate_policy_33
best_graph_option1
```

Expected winner prior:

```text
best_graph_option1 = graph512_turn_pair_prior
```

But only keep it as favorite if it preserves tactical reliability and throughput
well enough for self-play volume.

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

## What The Tuner Should Not Do

Avoid these traps:

```text
Do not tune every head independently.
Do not tune many architecture families that failed Phase 1/2 gates.
Do not treat longer losses against the classical opponent as a real win.
Do not let low-sim PCR train the policy head as if it were full-search data.
Do not choose the highest-sim setting without a wall-clock penalty.
Do not let candidate budgets cap away wins, forced blocks, or cover cells.
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
  candidate_budget = 512
  pair policy = prior-shaping first, macro expansion only if proven stable
```

Use `1600` sims only for a late finalist check. It is too expensive for broad
scouting unless throughput is much better than expected.

## Dynamic PB2/PBT Search Space

PB2/PBT should tune values that plausibly change over training.

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

### Checkpoint League

Maintain a league containing:

```text
top 6 active checkpoints
best previous checkpoint
best_current_33 reference
best_restnet_33 reference
best_graph_option1 reference, if available
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

If the league is not populated yet, temporarily raise the weights on classical
survival, tactical suite, and policy/value diagnostics.

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
best_graph_option1 throughput + recall probe, if available
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
4 to 8 active trials
min_resource = 1 epoch
promotion = 2x
max_resource = 4 epochs
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

### Phase 3C: PB2/PBT Schedule Search, 16 Hours

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

PB2 is preferred if available because it models continuous hyperparameter
changes. PBT is acceptable and simpler:

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
schedule = best PB2/PBT path
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
best_graph_option1, if not primary
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
| 3C | PB2/PBT schedule search | `16h` | `96` |
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

### PBT/PB2 Mutation Hooks

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
