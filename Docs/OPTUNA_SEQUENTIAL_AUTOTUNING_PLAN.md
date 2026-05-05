# Optuna Sequential Autotuning Plan - 2026-05-05

This document proposes a simpler production autotuning design for Hexo-RL. It
does not try to complete the older ASHA/BOHB/PB2 plan. The recommendation is
to replace most custom scheduler logic with an Optuna-backed sequential tuning
controller while keeping Hexo-specific training, validation, artifacts, and
champion selection under project-owned code.

## Recommendation

Use Optuna as the tested HPO control plane, not as the domain authority.

```text
Optuna owns:
  trial identity
  parameter suggestions
  persistence and resume
  intermediate metric history
  prune/continue decisions after Hexo reports safe scores
  study inspection

Hexo owns:
  typed recipe construction
  runtime calibration
  training epochs
  checkpoint branching
  replay, legal, D6, target, and candidate-safety checks
  tactical and arena evaluation
  scorecard construction
  final champion selection
```

The tuning method should be sequential checkpoint branching. A model needs
roughly 10-15 epochs before strength signals become meaningful, and epochs are
about 300 seconds. That makes broad low-fidelity HPO a poor default. The tuner
should spend most compute on a small number of well-chosen branches from a
competent checkpoint.

## Goal

Produce a production model on the 4070 Ti / 7950X / 32 GB RAM host using a
small, controlled number of sequential experiments. The system should improve
training/search schedules without hiding bugs, overfitting to noisy early
metrics, or relying on a custom unproven HPO stack.

## Success Criteria

- A run can resume from durable Optuna storage and Hexo artifacts.
- Every trial has typed recipes, checkpoint lineage, scorecards, and debug
  evidence.
- Hard Hexo sentinels can fail a trial immediately without depending on Optuna
  scalar scoring.
- Metric-based pruning is delayed until the model has enough epochs for
  meaningful signal.
- Tuning branches from competent checkpoints instead of repeatedly training
  many fresh configs from zero.
- The final champion is selected by Hexo evaluation gates, not by Optuna value
  alone.
- The implementation leaves the old custom ASHA/BOHB/PB2 paths optional,
  quarantined, or removed from the production tuner.

## Constraints

- Tuning is sequential on one production GPU.
- Epochs cost about 300 seconds.
- Meaningful strength signal usually starts around epoch 10-15.
- Low-epoch scores are useful for health, legality, target quality, and
  throughput, but not for final strength ranking.
- Runtime knobs and model/search semantics must be tuned separately.
- The search space must stay small enough that 5-20 meaningful observations can
  guide decisions.

## Non-Goals

- Do not build a broad generic HPO service.
- Do not make BOHB, PB2, or any custom scheduler the production centerpiece.
- Do not flatten model architecture, search settings, runtime settings, and
  schedule settings into one uncontrolled Optuna search space.
- Do not prune promising models on pre-competence arena noise.
- Do not use Optuna to decide whether a trial is semantically valid.

## System Shape

### 1. Typed Recipe Layer

The production tuner should express all tunable choices through typed recipe
objects. Exact class names are left to the implementer, but the boundaries
should stay clear.

```text
ModelRecipe:
  architecture family
  channels and blocks
  head bundle
  sparse or graph settings
  token budget and graph layers where relevant

SearchRecipe:
  full MCTS simulations
  PCR low-simulation count and probability
  policy target top-k
  temperature family
  exploration settings that affect search semantics

ScheduleSpec:
  learning rate or LR multiplier
  weight decay
  c_puct and c_puct_init
  Dirichlet fraction and alpha scale
  recency decay
  loss-weight multipliers

RuntimeSpec:
  self-play workers
  batch size per worker
  inference max batch size
  inference wait time
  memory safety envelope
```

The Optuna search space should suggest recipe fields, not mutate raw config
paths directly. Hexo should convert recipes into `Config` objects through one
validated construction path.

### 2. Runtime Calibration

Runtime tuning should remain separate from model tuning. Before training a new
model/search recipe, run a short self-play runtime probe and cache the result by
recipe identity and host profile.

Candidate dimensions can include:

```text
self-play workers
batch size per worker
inference max batch size
inference max wait time
```

The selected runtime spec should score throughput, GPU utilization, RAM
headroom, swap growth, stalls, and zero-progress failures. Runtime probe
failures should never be interpreted as model weakness.

### 3. Optuna Study Controller

The controller should use Optuna through an ask/tell style loop or a thin
objective wrapper. The important boundary is that Hexo runs the trial and then
reports structured results back to Optuna.

Recommended responsibilities:

```text
create or load study
enqueue known-good seed recipes
ask for a new branch recipe
attach Hexo trial metadata as user attributes
receive intermediate epoch scores
mark trial complete, pruned, or failed
persist study state in SQLite or another durable storage backend
```

The first production implementation can use a conservative sampler such as TPE
or random/QMC with a narrow search space. More advanced samplers are optional
only after enough clean observations exist.

### 4. Trial Runner

The trial runner should be deliberately plain. It receives a typed trial plan,
runs epochs, writes artifacts, and returns scorecards.

Each trial should record:

```text
trial id
Optuna study and trial number
parent checkpoint, if branched
recipe identities and hashes
runtime spec
epoch metrics
checkpoint paths
scorecards
hard sentinel status
prune or failure reason
```

The runner should not contain optimizer policy beyond obeying explicit
continue, branch, prune, and champion commands.

### 5. Evaluation And Scorecards

Hexo should produce one scorecard per evaluated checkpoint. Optuna should see a
single scalar for optimization, but the scalar must be traceable to component
metrics.

Scorecard components should include:

```text
training health:
  finite losses, target quality, value calibration, replay quality

self-play health:
  positions per minute, truncation rate, fallback prior use, no-progress status

semantic safety:
  legal moves, replay reconstruction, D6, target leakage, legal-mask agreement

strategy signal:
  tactical suite, outside-window checks, candidate recall where applicable

late strength:
  classical survival, checkpoint arena lower confidence bound, viewed games
```

Before epoch 10, the scalar should mostly represent health and learnability.
After epoch 10-15, tactical and arena metrics can carry more weight.

## Sequential Tuning Flow

### Phase 0: Seed And Runtime Probe

Start with one known-good production recipe. Run runtime calibration and cache
the selected runtime spec. This phase verifies the host can execute the recipe
without memory pressure or zero-progress behavior.

Expected compute: 10-20 minutes.

### Phase 1: Baseline Warmup

Train the known-good baseline until it reaches the first meaningful signal
window.

Default budget:

```text
10-12 epochs
about 50-60 minutes
```

Optuna should receive intermediate health scores, but metric-based pruning
should be disabled except for hard failures.

### Phase 2: First Branch Round

Create a small number of branches from the baseline checkpoint. These branches
should test local schedule/search variants, not broad architecture changes.

Example branch families:

```text
baseline continuation
lower LR and more stable value learning
higher exploration
higher MCTS simulations if runtime allows
stronger auxiliary or tactical weighting
```

Default budget:

```text
3-5 branches
3-5 epochs per branch
paired evaluation seeds and fixtures
```

This is the main tuning loop. Optuna can suggest variants, but the search space
should remain close to the baseline.

### Phase 3: Local Refinement

Keep the best one or two branches and run one smaller refinement round. This
round should adjust only parameters that had plausible signal in the first
branch round.

Default budget:

```text
1-2 branches
4-6 epochs per branch
```

The controller should favor continuing the best branch unless the evidence for
a new branch is clear.

### Phase 4: Champion Lock

Stop searching. Continue the winning branch for a protected training block and
evaluate checkpoints. Optuna can still record observations, but it should no
longer create new variants.

Default budget:

```text
10+ additional epochs, or a fixed wall-clock champion block
```

### Phase 5: Final Selection

Hexo selects the final checkpoint from the champion lineage. Selection should
use hard sentinels, tactical holdouts, classical survival, arena lower
confidence bound, and inspectable saved games.

Optuna's best trial value is evidence, not the final authority.

## Search Space Guidance

The first production search space should be intentionally narrow.

Good early candidates:

```text
lr_multiplier:
  local multiplier around the known-good LR

weight_decay:
  small bounded range around current value

c_puct and c_puct_init:
  modest exploration changes

dirichlet_fraction and scaled_alpha_total:
  exploration noise, especially opening behavior

pcr_low_sim_prob:
  balance cheap exploration and full-search targets

recency_decay:
  replay freshness pressure

value_loss_weight and auxiliary multiplier:
  target-learning balance

full_sims:
  small categorical set only when runtime headroom is known
```

Riskier knobs should be gated behind separate experiments:

```text
architecture family
head bundle
graph token budget
graph layers
sparse prior stage
major replay schema changes
large runtime worker changes
```

## Pruning Policy

Use two pruning layers.

Hexo hard pruning is immediate:

```text
illegal moves
post-terminal moves
replay mismatch
D6 mismatch
legal-mask mismatch
target leakage
missing legal rows
non-finite loss or output
zero self-play positions
memory-unsafe runtime behavior
checkpoint load failure
```

Optuna pruning is delayed and conservative:

```text
before epoch 10:
  no metric pruning except hard Hexo failures

epoch 10-15:
  allow pruning only for clearly dominated branches under paired evaluation

after epoch 15:
  allow normal study pruning based on scorecard scalar and confidence
```

This avoids discarding models during the noisy pre-competence window.

## Artifact Layout

A run should keep Optuna storage and Hexo artifacts side by side.

```text
runs/<run_id>/
  optuna.sqlite3
  study_manifest.json
  runtime_cache.json
  trials/
    <hexo_trial_id>/
      trial_plan.json
      recipe.json
      lineage.json
      runtime_spec.json
      events.jsonl
      scorecards.jsonl
      checkpoints/
      debug_bundles/
  champion/
    selection_report.md
    final_scorecards.jsonl
```

The exact filenames can change, but the design should preserve enough
information to reconstruct why every branch was created, continued, pruned, or
selected.

## Migration Strategy

The safest migration is incremental.

1. Add typed recipe objects and a recipe-to-config builder.
2. Add an Optuna study wrapper that can run one known-good recipe end to end.
3. Add checkpoint branching from a baseline checkpoint.
4. Add narrow Optuna suggestions for schedule/search variants.
5. Add delayed pruning and Hexo hard-prune integration.
6. Add report generation and dashboard links.
7. Retire or quarantine the old production ASHA/BOHB/PB2 supervisor path.

The old stack can remain available for historical comparison, but production
model tuning should have one preferred path.

## Required Evidence For A Complete Implementation

An implementation should not be considered complete until it produces
reviewable evidence for:

```text
recipe validation:
  typed recipes create valid Config objects without raw family mutation

Optuna persistence:
  interrupted studies resume without losing trial state

checkpoint branching:
  child trials record parent checkpoint and compatible recipe lineage

hard sentinels:
  semantic failures prune trials before Optuna ranking

delayed pruning:
  metric pruning is inactive before the configured signal epoch

runtime separation:
  runtime sweep failures are reported separately from model score

scorecard traceability:
  Optuna scalar values can be traced back to component metrics

champion selection:
  final checkpoint selection is reproduced from saved artifacts
```

## Stop Rules

Stop the tuning run rather than continuing blindly when:

```text
hard sentinels fire across multiple related branches
runtime calibration cannot find a memory-safe candidate
all branches fail before producing positions
scorecards cannot be written or resumed reliably
checkpoint lineage becomes ambiguous
evaluation fixtures fail in a way that hides model quality
the champion block would be shortened enough to make final selection noisy
```

These stop rules are part of the design. The tuner should prefer a clean failed
run over a long run whose artifacts cannot explain what happened.

