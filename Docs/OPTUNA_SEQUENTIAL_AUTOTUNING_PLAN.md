# Optuna Architecture Scout And Sequential Autotuning Plan - 2026-05-06

This document defines the next autotuning direction for Hexo-RL. The project is
still in an early validation stage: the first priority is to prove that the
global graph architectures, pair strategies, runtime orchestration, replay
contracts, and scorecards work end to end. After that, the same infrastructure
should support longer hyperparameter tuning and champion training.

The plan intentionally separates architecture scouting from local schedule
tuning. A single uncontrolled search space that mixes architecture, pair
strategy, runtime settings, training schedule, and search knobs would be hard to
interpret and would hide bugs behind noisy HPO scores.

## Recommendation

Use Optuna as the tested HPO control plane, not as the Hexo domain authority.

```text
Optuna owns:
  trial identity
  parameter suggestions
  durable study storage and resume
  intermediate metric history
  conservative prune/continue decisions after Hexo reports safe scores
  study inspection

Hexo owns:
  typed recipe construction
  full Config materialization and validation
  runtime probes, performance calibration, and speed quarantine
  self-play and training epochs
  checkpoint branching and lineage
  replay, legal-row, pair-row, D6, target, and candidate-safety checks
  tactical, classical, arena, and scorecard evaluation
  champion selection
  quarantine, retest, and debug bundles
```

There are two modes:

```text
architecture scout mode:
  train global graph candidates from scratch to a fair epoch floor
  compare architecture and pair-strategy fit for Hexo
  quarantine unhealthy or too-slow candidates with debug evidence

production tuning mode:
  after one or more families prove viable, run narrower per-family studies
  tune local schedule/search knobs and train champion lineages
```

### Optuna Setup Summary

At the expected production budget of about `600s` per epoch, one fair
epoch-12 candidate costs roughly two GPU-hours before evaluation overhead. This
is too expensive for broad optimizer exploration during the initial scout.
Optuna should therefore start as a durable study ledger around a fixed queued
candidate plan, then become an active optimizer only after one or more families
survive the scout.

Recommended setup:

```text
Phase 1 architecture scout:
  sampler: fixed/enqueued candidate plan
  Optuna sampler object: TPESampler is acceptable as the study shell
  pruner: NopPruner
  minimum resource: 12 epochs
  strength pruning before epoch 12: disabled
  hard failures: Hexo quarantine, not Optuna score pruning

Phase 3 per-family tuning:
  sampler: TPESampler(multivariate=true, group=true)
  startup trials: at least 8 completed trials per study when possible
  pruner: conservative SuccessiveHalvingPruner or custom floor pruner
  min_resource: 12 epochs
  search scope: narrow schedule/search knobs inside one family and pair mode
```

The best general-purpose Optuna optimizer for this workload is TPE, not a
Gaussian-process or evolutionary sampler. The search space mixes categorical,
integer, float, and conditional parameters; TPE handles that shape well and can
learn from completed expensive trials. It will not be statistically useful for
choosing among only eight architecture candidates, so those candidates should be
explicitly enqueued.

## Project Intent

The long-term goal is a superhuman Hexo model. The near-term goal is a
multi-day, resumable, diagnostic architecture scout focused on global graph
models.

Important game/model assumptions:

- Hexo is effectively played on an unbounded board.
- Placement radius `n=8` defines the active legal/action neighborhood.
- `33x33` windows are crop-model implementation details, not the board itself.
- Global graph models should be judged as legal-row/action-table models.
- Pair reasoning is diagnostic today but important enough to test as part of
  architecture search.

## Success Criteria

- A run can resume from durable Optuna storage and Hexo artifacts.
- Every candidate records its full materialized config in the database and in
  artifacts.
- Every trial has typed recipes, scorecards, runtime evidence, checkpoints, and
  debug evidence.
- Phase 1 trains each non-quarantined candidate from scratch to epoch 12.
- Phase 1 uses at least a tunable minimum of generated self-play positions per
  epoch, defaulting to 3000.
- Metric-based pruning is disabled before the epoch-12 scout floor.
- Health and speed failures quarantine candidates with useful diagnostics.
- Quarantined candidates can be marked ready for config-based retest.
- Pair strategy is tested in multiple modes, including pair-influenced MCTS.
- Final champion selection uses Hexo gates and evaluation evidence, not Optuna
  value alone.

## Constraints

- Tuning is sequential on one production GPU.
- Phase 1 should finish when all non-quarantined scout candidates reach epoch
  12, ideally within 48 hours.
- The initial scout should include at most 8 candidates.
- The initial scout is global-graph-only; dense CNN is not part of Phase 1.
- `global_graph768_champion` should be included initially at full intended
  configuration and quarantined if it is too slow.
- If runtime probing cannot find any safe candidate above 2 generated self-play
  positions per second, the model is quarantined for speed.
- Runtime knobs and model/search semantics must be separated. Calibration may
  optimize hardware usage, but it must not change model behavior except for
  ordinary batching/waiting effects.
- Early failures usually indicate bugs or integration issues and must produce
  diagnostic artifacts.

## Non-Goals

- Do not build a broad generic HPO service.
- Do not make BOHB, PB2, or any custom scheduler the production centerpiece.
- Do not tune dense CNN as the Phase 1 focus.
- Do not ask Optuna to discover the initial architecture scout allocation from
  scratch.
- Do not prune models on noisy pre-epoch-12 strength metrics.
- Do not treat runtime probe failure as model weakness.
- Do not let Optuna decide whether a model is semantically valid.

## System Shape

### 1. Typed Recipes

The tuner should express tunable choices through typed recipe objects.

```text
ModelRecipe:
  architecture family
  architecture contract version
  channels, blocks, graph dimensions, and token budget
  head bundle and output contract
  graph row/table settings

PairStrategySpec:
  mode: none | root_pair_mcts | full_pair_mcts
  pair row budget
  pair prior mix
  pair batching/chunking settings
  root-only versus deeper expansion behavior

SearchRecipe:
  full MCTS simulations
  PCR low-simulation count and probability
  policy target top-k
  temperature family
  c_puct and c_puct_init
  Dirichlet fraction and alpha scale

ScheduleSpec:
  learning rate or LR multiplier
  weight decay
  recency decay
  value, auxiliary, graph, and pair loss weights

RuntimeSpec:
  self-play workers
  batch size per worker
  inference max batch size
  inference wait time
  memory safety envelope
```

Optuna should suggest recipe fields. Hexo should convert recipes into `Config`
objects through one validated construction path, then store the full config in
Optuna user attributes and in candidate artifacts.

During Phase 1, most recipe fields are fixed by the candidate plan. Optuna
should receive these through `enqueue_trial` or equivalent explicit queued trial
creation. During Phase 3, Optuna may suggest the schedule/search fields inside
one surviving family and pair mode.

### 2. Candidate Identity And Versioning

Human-readable identity should be candidate-first. Optuna trial identity should
remain available but should not be the primary way humans navigate artifacts.

Example candidate ids:

```text
global_xattn_0__none__v1
global_line_window_0__root_pair_mcts__v1
global_pair_twostage_0__full_pair_mcts__v1
global_graph768_champion__none__v1
```

The database and artifacts should record:

```text
candidate id
architecture id
pair strategy mode
recipe schema version
graph contract version
full materialized Config
config hash
git SHA
host profile
Optuna study name and trial number
```

Full config logging is sufficient for exact reproducibility. The readable
candidate id exists so a human can scan a run directory without decoding trial
numbers.

### 3. Runtime Probe, Performance Calibration, And Speed Quarantine

Before a candidate enters real Phase 1 training, run a runtime probe. This probe
also acts as performance calibration, following the current supervisor's runtime
sweep pattern: generate candidate worker/batch/wait settings, run self-play-only
probes, record throughput and resource telemetry, select a safe runtime spec,
and cache that selection by recipe/host identity.

The probe uses the candidate's real model/search/pair settings and measures
generated self-play positions per second without filtering.

Calibration may choose:

```text
self-play worker count
batch size per worker
inference max batch size
inference max wait time
safe CPU/GPU/memory envelope
```

Calibration must not choose:

```text
architecture
pair strategy mode
pair row cap or pair behavior
MCTS simulation count
search semantics
loss weights
training schedule
```

Candidate dimensions can include:

```text
self-play workers
batch size per worker
inference max batch size
inference max wait time
```

The selected runtime spec should score:

```text
generated positions/sec
positions/min
GPU utilization
GPU memory
system memory headroom
swap growth
queue stalls
zero-progress behavior
runtime errors
```

The calibration cache should include the selected runtime candidate, all
measured runtime candidate rows, memory safety summaries, GPU snapshots, and
enough identity to reject stale, unsafe, or suboptimal cached selections after
code or config changes.

The cache key should include fields that affect runtime compatibility or model
behavior, including architecture id, pair strategy mode, pair row budget,
full-search simulations, PCR simulations, head bundle, graph/token settings,
host profile, and relevant code/config hashes. It should not use Optuna trial
number as the identity because equivalent recipes should be able to reuse safe
runtime calibration.

Speed quarantine rule:

```text
if runtime probe cannot find any safe runtime candidate above
2 generated self-play positions/sec:
  quarantine that candidate for speed
  save a debug bundle
  continue tuning other candidates
```

The quarantine reason should classify the suspected bottleneck when possible:

```text
graph construction
inference latency
pair row explosion
MCTS simulation cost
CPU queue starvation
GPU underutilization
memory or swap pressure
checkpoint or model-load failure
unknown
```

### 4. Quarantine And Retest

Quarantine is not permanent. It is a compute-safety and diagnosis state.

Candidate lifecycle:

```text
pending
runtime_probe
running
healthy
promoted
champion_candidate
quarantined
ready_for_retest
retesting
```

The run should continue after a candidate is quarantined. A config file or run
manifest can mark quarantined candidates as `ready_for_retest`; the orchestrator
can then reinsert them in a later resumed run.

Retested candidates should keep old evidence intact and should record the new
full config, code SHA, and contract version. The old failed candidate should not
be overwritten.

### 5. Scorecards

Hexo should produce one scorecard per evaluated checkpoint. Optuna sees a scalar
only after Hexo has written the component metrics that explain it.

Scorecard components:

```text
training health:
  finite losses, output finite checks, target quality, value calibration,
  replay quality

self-play health:
  generated positions/sec, positions per minute, no-progress status,
  truncation rate, fallback prior use, strength per generated position,
  strength per wall-clock second

semantic safety:
  legal moves, legal-row identity, pair-row identity, replay reconstruction,
  D6 consistency, target leakage, legal-mask agreement

strategy signal:
  tactical suite, outside-window checks, pair-head quality, pair overhead,
  candidate recall where applicable

late strength:
  classical survival, classical win/draw rate, viewed games
```

The primary scalar for Phase 1 should be classical survival based. Models are
not expected to beat the fixed strong classical opponent early, so survival
length provides the useful strength gradient.

Use one scalar throughout Phase 1:

```text
classical_survival_lcb
```

`classical_survival_lcb` should be a conservative lower-confidence estimate of
survival against the fixed classical opponent. Per-game survival evidence should
reward:

```text
longer loss length against classical
draws or max-move games
rare wins against classical
zero illegal/crash behavior
```

The score should be computed with the same formula before and after epoch 12.
Before the epoch-12 scout floor, it is logged for ranking and trend visibility
but must not cause strength-based pruning.

## Phase Flow

### Phase 0: Contract Smoke And Runtime Probe

Run each candidate through a short diagnostic pass before full training.

Required checks:

```text
Config can be built from recipe
graph batches build for representative histories
legal rows align to Rust legal moves
pair rows align when pair mode is enabled
model outputs have expected shapes
outputs and losses are finite
checkpoint save/load works
runtime probe reaches safe speed and memory behavior
runtime probe calibrates worker/batch/wait settings
debug bundle is written on failure
```

### Phase 1: Balanced Global Graph Architecture Scout

Train each non-quarantined candidate from scratch to epoch 12.

With the current planning estimate of `600s` per epoch, the epoch-12 floor costs
about two GPU-hours per candidate. This makes Phase 1 a controlled experiment,
not a broad hyperparameter search. The optimizer must preserve the fair floor
and spend compute on comparable architecture evidence.

Defaults:

```text
max_candidates: 8
min_epochs: 12
min_generated_selfplay_positions_per_epoch: 3000
speed_quarantine_positions_per_sec: 2.0
schedule_quantum_epochs: 2
include_dense_control: false
metric_pruning_before_epoch_12: disabled
```

Initial architecture coverage:

```text
global_xattn_0
global_line_window_0
global_pair_twostage_0
global_graph_full_0
global_graph768_champion
```

Pair strategy modes:

```text
none:
  no pair priors consumed by MCTS

root_pair_mcts:
  pair heads influence root priors only

full_pair_mcts:
  pair heads may influence deeper MCTS expansion as well as root priors
```

The initial candidate planner should include one default candidate per
architecture and must include pair-strategy candidates within the 8-candidate
cap. The allocation must be config-editable. The initial recommended allocation
is:

```text
5 default architecture candidates
1 global_pair_twostage_0 root_pair_mcts candidate
1 global_pair_twostage_0 full_pair_mcts candidate
1 global_graph_full_0 root_pair_mcts candidate
```

This keeps the scout small while ensuring pair strategy is tested as an
architecture/search choice.

Phase 1 Optuna behavior:

```text
create one Optuna study for scout traceability
enqueue every candidate from candidate_plan before optimization starts
use NopPruner for metric pruning
report classical_survival_lcb and health metrics each evaluated checkpoint
do not call trial.should_prune for strength before epoch 12
return the epoch-12 classical_survival_lcb for completed candidates
mark Hexo quarantines as failed/pruned trials with structured user attributes
```

Phase 1 should use round-robin scheduling in two-epoch quanta. For example,
train candidate A for epochs 1-2, candidate B for epochs 1-2, and continue
cycling. This balances exposure to machine/run conditions while avoiding the
model-switching overhead of strict every-epoch alternation.

### Phase 2: Scout Review And Candidate Promotion

After all non-quarantined Phase 1 candidates reach epoch 12, rank candidates
for continuation. Phase 2 is review and promotion, not early pruning.

Promotion evidence:

```text
classical_survival_lcb as the primary scalar
classical win/draw rate as supporting evidence
self-play health and generated positions/sec
strength per generated position
strength per wall-clock second
pair overhead versus benefit
pair-head quality where applicable
tactical and outside-window signals
zero illegal/crash behavior
```

Model tournament Elo belongs in Phase 2 and later, not in the Phase 1 target
scalar. Phase 2 can use a small candidate tournament to compare survivors, but
the promotion scalar should remain primarily anchored to classical survival
until models can beat the classical opponent often enough for win rate to carry
the signal.

The best candidates may receive more compute after the Phase 1 floor, but Phase
1 itself is complete only when the epoch-12 scout floor is reached or candidates
are quarantined with evidence.

### Phase 3: Per-Family Optuna Tuning

Create separate studies per surviving architecture family and pair mode. This
prevents Optuna from confusing architecture identity with schedule quality.

Recommended study names:

```text
study_architecture_scout_v1
study_global_xattn_0__none__schedule_v1
study_global_pair_twostage_0__root_pair_mcts__schedule_v1
study_global_pair_twostage_0__full_pair_mcts__schedule_v1
study_global_graph_full_0__root_pair_mcts__schedule_v1
```

Good tuning knobs:

```text
lr_multiplier around the family-stable LR
weight_decay
c_puct and c_puct_init
dirichlet_fraction and scaled_alpha_total
pcr_low_sim_prob
recency_decay
value_loss_weight and auxiliary multiplier
pair_loss_weight for pair-head candidates
pair_prior_mix for pair-MCTS candidates
full_sims only when runtime headroom is proven
```

Recommended Optuna sampler:

```python
TPESampler(
    multivariate=True,
    group=True,
    n_startup_trials=8,
    seed=seed,
)
```

TPE should be preferred because these studies contain mixed and conditional
parameters. Examples include pair-only `pair_prior_mix`, pair-head-only
`pair_loss_weight`, runtime-gated `full_sims`, and categorical temperature
families. Gaussian-process samplers are less appropriate for this conditional
mixed search space and should not be the default production optimizer.

Recommended pruning after the scout:

```python
SuccessiveHalvingPruner(
    min_resource=12,
    reduction_factor=2,
)
```

If Optuna's built-in pruner cannot express the exact Hexo floor and hard-gate
semantics cleanly, implement a small Hexo floor pruner wrapper that refuses all
metric pruning before the configured signal epoch and delegates only after
scorecards are safely written.

Avoid broad architecture mutations inside these studies. If a model contract or
architecture implementation changes, create a new candidate identity and retest.

### Phase 4: Champion Lineages

Stop broad exploration. Train several protected lineages rather than choosing a
single checkpoint immediately.

Candidate champion slots:

```text
best non-pair global graph
best root-pair global graph
best full-pair global graph, if viable
best pair-trained diagnostic model if different from pair-MCTS winners
```

### Phase 5: Final Selection

Final selection should begin with 400 arena games against one fixed strong
classical opponent. The classical opponent details can be tuned later, but the
Phase 5 opponent should be fixed within a run.

Selection gates:

```text
zero illegal/crash rate
better arena/classical result than alternatives
strong classical survival
stable self-play health
no tactical regression
acceptable wall-clock throughput
inspectable saved games and replay evidence
```

Optuna's best trial value is evidence, not the final authority.

## Pair Strategy Details

### `none`

Pair heads may be absent or may be trained for diagnostics, but MCTS consumes
only base legal-row policy/value outputs.

### `root_pair_mcts`

At the root position, the model scores pair rows. Pair probability mass is
projected or blended into root action priors before MCTS simulations.

Benefits:

```text
simple runtime boundary
easier row-identity validation
directly tests whether pair reasoning improves the immediate move
lower latency than deeper pair use
```

Risks:

```text
pair reasoning affects only the current root
pair heads may look useful diagnostically without improving deeper search
```

### `full_pair_mcts`

Pair priors may be requested for non-root expansion states as MCTS descends the
tree. This means pair rows, pair logits, and pair-prior blending can affect
deeper search decisions, not just the root move.

When enabled, `full_pair_mcts` must be the complete intended implementation. It
should not be a partial substitute, root-only shortcut, or reduced-behavior
stand-in. If the full implementation is not ready, the candidate should remain
unavailable rather than being represented by an incomplete mode.

Benefits:

```text
pair reasoning can guide the whole search, not only the first action
two-placement interactions may be represented more consistently
deeper tactical pair traps may be discovered more often
```

Costs and risks:

```text
more inference and batching complexity
larger pair-row counts
higher latency per expanded node
more row-identity and stale-token failure modes
harder cache and queue behavior
possible lower total MCTS visits at equal wall-clock
```

`full_pair_mcts` should be config-gated but allowed in the initial Phase 1
scout. It must be measured against `none` and `root_pair_mcts` by strength per
wall-clock, not only raw win rate.

## Further Exploration

Deeper pair priors are worth exploring, but should remain a controlled
experiment until root pair strategy is healthy and measurable.

Open questions for the deeper search experiment:

```text
Should pair priors be used at every expanded node or only pair-turn nodes?
Should pair rows be capped differently at root and leaf expansion?
Can pair logits be cached safely across tree re-rooting?
Should pair priors affect only expansion priors or also target construction?
What pair_prior_mix preserves MCTS robustness when pair heads are immature?
How much wall-clock strength is lost from fewer simulations?
```

Required evidence for claiming full-pair search is beneficial:

```text
same or better classical survival per wall-clock
same or better arena result at fixed time budget
measured pair-row latency and queue impact
zero legal-row or pair-row identity failures
debuggable replay examples where pair priors changed the selected move
```

## Suggested Config Surface

```toml
[autotune.scout]
enabled = true
max_candidates = 8
min_epochs = 12
estimated_epoch_seconds = 600
estimated_candidate_hours = 2
min_generated_selfplay_positions_per_epoch = 3000
target_phase_hours = 48
schedule_quantum_epochs = 2
include_dense_control = false
candidate_plan = [
  "global_xattn_0:none",
  "global_line_window_0:none",
  "global_pair_twostage_0:none",
  "global_graph_full_0:none",
  "global_graph768_champion:none",
  "global_pair_twostage_0:root_pair_mcts",
  "global_pair_twostage_0:full_pair_mcts",
  "global_graph_full_0:root_pair_mcts",
]

[autotune.optuna]
storage = "sqlite:///runs/<run_id>/optuna.sqlite3"
phase1_sampler = "queued_tpe_shell"
phase1_pruner = "nop"
phase1_enqueue_candidate_plan = true
phase3_sampler = "tpe"
tpe_multivariate = true
tpe_group = true
tpe_startup_trials = 8
phase3_pruner = "successive_halving_after_floor"
pruner_min_resource_epochs = 12
pruner_reduction_factor = 2

[autotune.runtime_probe]
enabled = true
speed_quarantine_positions_per_sec = 2.0
measure = "generated_selfplay_positions_per_second"
mode = "calibrate_and_select"
behavior_invariant = true

[autotune.quarantine]
continue_after_quarantine = true
allow_retest = true
ready_for_retest = []

[autotune.pair_strategy]
modes = ["none", "root_pair_mcts", "full_pair_mcts"]
full_pair_mcts_enabled = true

[autotune.final_eval]
classical_arena_games = 400
classical_opponent = "fixed_strong"

[autotune.scoring]
target_scalar = "classical_survival_lcb"
phase1_uses_model_tournament = false
phase2_uses_model_tournament = true
```

## Artifact Layout

Use candidate-first layout for human readability, while keeping Optuna trial
metadata inside each candidate folder.

```text
runs/<run_id>/
  optuna.sqlite3
  study_manifest.json
  runtime_cache.json
  candidates/
    global_pair_twostage_0__full_pair_mcts__v1/
      candidate_manifest.json
      optuna_trial.json
      full_config.toml
      recipe.json
      runtime_spec.json
      quarantine.json
      events.jsonl
      scorecards.jsonl
      checkpoints/
      debug_bundles/
        <timestamp>_<reason>/
          repro_command.txt
          failing_history.json
          replay.json
          legal_rows.json
          pair_rows.json
          model_output_summary.json
          runtime_telemetry.json
          runtime_probe_results.jsonl
          dashboard_links.json
  scout/
    phase1_summary.md
    architecture_comparison.json
    pair_strategy_comparison.json
  champion/
    selection_report.md
    final_scorecards.jsonl
```

Debug bundles should be optimized for code diagnosis, not just experiment
tracking. When a model fails, the bundle should preserve enough state to rebuild
the failing batch, replay the position, inspect legal/pair rows, and reproduce
the probe or epoch.

## Pruning Policy

Use two pruning layers.

Hexo hard pruning/quarantine is immediate:

```text
illegal moves
post-terminal moves
replay mismatch
D6 mismatch
legal-mask mismatch
pair-row mismatch
target leakage
missing legal rows
non-finite loss or output
zero self-play positions
memory-unsafe runtime behavior
runtime probe cannot find a safe runtime candidate above speed threshold
checkpoint load failure
```

Metric pruning is delayed:

```text
before Phase 1 epoch 12:
  no strength-based pruning
  health and speed quarantine only
  Optuna pruner: NopPruner

after Phase 1:
  promote by scorecards and wall-clock evidence

per-family tuning studies:
  allow conservative pruning after the configured signal epoch
  preferred Optuna pruner: SuccessiveHalvingPruner with min_resource >= 12
```

Optuna trial states should distinguish compute-safety quarantine from ordinary
metric pruning through user attributes:

```text
hexo_status: completed | quarantined | metric_pruned | failed
quarantine_reason: null or structured reason
completed_epochs
final_scorecard_path
debug_bundle_path
```

## Migration Strategy

The safest migration is incremental.

1. Add typed scout/tuning recipe objects and a recipe-to-config builder.
2. Add candidate-first artifact layout and full config logging.
3. Add runtime probe speed quarantine and debug bundles.
4. Add global-graph-only Phase 1 scout to epoch 12.
5. Add pair strategy modes: `none`, `root_pair_mcts`, `full_pair_mcts`.
6. Add Optuna storage and resume around the scout controller.
7. Add per-family Optuna studies for survivors.
8. Retire or quarantine the old production ASHA/BOHB/PB2 supervisor path.

## Required Evidence For A Complete Implementation

```text
recipe validation:
  typed recipes create valid Config objects without raw family mutation

full config traceability:
  every candidate stores exact materialized Config in artifacts and database

Optuna persistence:
  interrupted studies resume without losing trial state

Optuna sampler/pruner setup:
  Phase 1 uses queued candidate trials with no metric pruner; Phase 3 uses
  per-family TPE studies with pruning disabled until the signal floor

Phase 1 floor:
  non-quarantined candidates reach epoch 12 with configured generated positions

speed quarantine:
  probe cannot find any safe >2 generated positions/sec runtime candidate and
  writes a diagnostic bundle

pair strategy:
  none, root_pair_mcts, and full_pair_mcts are config-selectable and tested

hard sentinels:
  semantic failures quarantine before Optuna ranking

runtime separation:
  runtime probe failures are reported separately from model score

scorecard traceability:
  classical_survival_lcb can be traced back to fixed-classical game lengths,
  outcomes, confidence calculation, and hard penalties

champion selection:
  final checkpoint selection is reproduced from saved artifacts
```

## Stop Rules

Stop or pause the run rather than continuing blindly when:

```text
scorecards cannot be written or resumed reliably
checkpoint lineage becomes ambiguous
the same hard sentinel fires across many unrelated candidates
runtime calibration cannot find any memory-safe candidate
all candidates fail before producing positions
evaluation fixtures fail in a way that hides model quality
```

Individual candidate failures should usually quarantine that candidate and let
the rest of the scout continue.
