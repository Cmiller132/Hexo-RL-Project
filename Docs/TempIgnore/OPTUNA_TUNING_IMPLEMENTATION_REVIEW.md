# Optuna Tuning Implementation Review

This document reviews the first-level Optuna/autotuning implementation added in `d0d27b2 Add autotune scout package and pair MCTS support`. It focuses on what changed, how the tuning flow now works, what is affected, and which parts should be treated as highest-risk during the next review pass.

Review inputs:

- `git diff --name-status 45dcdec..HEAD`
- `git show --stat d0d27b2`
- `Python/src/hexorl/config/schema.py`
- `Python/src/hexorl/autotune/*`
- `Python/src/hexorl/tuning/*`
- `Python/src/hexorl/eval/scorecard.py`
- `Python/src/hexorl/search/pair_strategy.py`
- `Python/src/hexorl/selfplay/worker.py`
- `scripts/run_phase1_optuna_scout.py`
- `scripts/run_fixed_classical_eval.py`
- `scripts/run_phase2_phase3_review.py`
- `scripts/run_phase3_optuna_tpe.py`
- `scripts/select_optuna_champion.py`
- `Python/tests/test_*optuna*.py` and related tuning tests

## Executive Summary

The implementation turns the earlier sequential autotuning plan into a concrete, artifact-backed pipeline. The core design is now candidate-first rather than trial-first: each architecture/search candidate gets a durable candidate directory, a manifest, a materialized config, scorecards, events, checkpoints, and debug bundles. Optuna is used as the orchestration and bookkeeping layer, but Hexo-owned scorecards and hard gates remain the source of truth for promotion and final champion selection.

The Phase 1 scout is intentionally conservative. It enqueues the fixed candidate plan into Optuna, uses a TPE sampler shell for study continuity, disables pruning with `NopPruner`, and runs candidates round-robin in epoch quanta until every surviving candidate reaches the minimum 12-epoch signal floor. Phase 3 is where Optuna becomes a true optimizer: promoted families receive separate TPE studies with multivariate/group TPE and a delayed Successive Halving pruner that cannot prune before scorecard evidence exists.

The implementation also promotes pair-search experiments into explicit modes: `none`, `root_pair_mcts`, and `full_pair_mcts`. These modes are integrated into config validation, candidate recipes, self-play worker setup, runtime identity, artifacts, scorecards, and tests. The current pair implementation is still a pair-prior integration over normal action-level MCTS rather than the research note's ideal first-class pair-action MCTS.

## New Dependency And Config Surface

`Python/pyproject.toml` now requires:

```text
optuna>=4.7,<5
```

The config schema now has a dedicated `autotune` section. The important pieces are:

- `autotune.scout`: controls Phase 1 candidate plan, 12-epoch floor, 600-second estimated epoch time, 2-hour candidate estimate, quantum size, and the 8-candidate maximum.
- `autotune.optuna`: controls Optuna storage and allowed sampler/pruner modes.
- `autotune.runtime_probe`: defines generated self-play position throughput checks and speed quarantine threshold.
- `autotune.quarantine`: controls candidate quarantine and retest behavior.
- `autotune.pair_strategy`: declares allowed pair modes and whether `full_pair_mcts` is enabled.
- `autotune.scoring`: makes `classical_survival_lcb` the target scalar.
- `autotune.final_eval`: records the intended final fixed-classical arena settings.

The default Phase 1 candidate plan is global-only and includes:

```text
global_xattn_0:none
global_line_window_0:none
global_pair_twostage_0:none
global_graph_full_0:none
global_graph768_champion:none
global_pair_twostage_0:root_pair_mcts
global_pair_twostage_0:full_pair_mcts
global_graph_full_0:root_pair_mcts
```

Validation rejects duplicate candidates, invalid pair modes, non-global candidates unless explicitly allowed, more than 8 candidates, and any plan that omits `global_graph768_champion:none`.

## Candidate Recipe System

The new `hexorl.autotune` package adds recipe and artifact primitives:

- `ModelRecipe` describes a global model family: architecture id, channels, blocks, heads, graph token budget, graph layers, and output heads.
- `PairStrategySpec` describes `none`, `root_pair_mcts`, or `full_pair_mcts`, including pair-row budget, pair prior mix, chunking, and whether scoring is root-only.
- `SearchRecipe`, `ScheduleSpec`, and `RuntimeSpec` capture MCTS/search, epoch schedule, and runtime knobs.
- `CandidateRecipe` combines those parts and materializes a validated Hexo `Config`.

Candidate ids are deterministic:

```text
<architecture_id>__<pair_mode>__v<schema_version>
```

That matters because artifacts, Optuna trials, runtime probe cache entries, scorecards, promotion reports, and champion reports all key off the same candidate identity.

## Artifact Layout

Each candidate now gets a durable directory under:

```text
runs/<run_id>/candidates/<candidate_id>/
```

The writer creates:

```text
candidate_manifest.json
optuna_trial.json
full_config.json
recipe.json
runtime_spec.json
events.jsonl
scorecards.jsonl
checkpoints/
debug_bundles/
```

This is one of the strongest parts of the implementation. It makes the tuning pipeline restartable and reviewable without requiring Optuna's database to be the only source of truth.

## Phase 1 Scout

Phase 1 is implemented by `Phase1OptunaScoutController`.

The controller:

- Creates or resumes `study_architecture_scout_v1`.
- Uses `TPESampler` only as a queued shell.
- Uses `NopPruner`; pruning is disabled before the 12-epoch floor.
- Enqueues exactly one trial per candidate plan entry.
- Stores study-level metadata: phase, candidate plan hash, plan entries, sampler, pruner, minimum epochs, quantum size, and the next candidate index.
- Runs candidates in round-robin quanta, defaulting to 2 epochs per scheduling quantum.
- Writes candidate manifests, full materialized configs, trial metadata, event logs, checkpoints, and scorecards.
- Supports resume by checking candidate-plan hash, candidate directories, existing scorecards, checkpoint lineage, and trial mapping.

The script entry point is:

```text
scripts/run_phase1_optuna_scout.py
```

It supports two runner modes:

- `--dry-run`: deterministic smoke/resume runner.
- `--production`: real self-play/training epoch quanta through `EpochScoutEpochRunner`.

This matches the earlier conclusion that 600 seconds per epoch and 12 epochs per candidate makes full candidate evaluation expensive. The implementation avoids early pruning in Phase 1 because an early random model is not trustworthy enough for model-family elimination.

## Runtime Probe And Quarantine

The runtime probe system adds a separate calibration layer for runtime knobs:

- self-play workers
- batch size per worker
- inference max batch size
- inference max wait

The runtime identity includes candidate id, architecture, heads, pair mode, pair cap, MCTS simulation counts, graph-token settings, host profile, config hash, code hash, contract versions, and extra metadata. The cache deliberately excludes Optuna trial number so equivalent candidate/runtime probes can be reused across restarts.

The runtime probe is designed to select only configurations that meet the generated self-play throughput threshold. If no safe configuration exists, the candidate is quarantined and receives a debug bundle instead of silently polluting the study.

Important safeguard: applying runtime knobs is checked with a semantic config hash that ignores runtime-only fields. This is meant to prove runtime calibration did not alter model/search semantics.

Quarantine records support:

- `quarantined`
- `ready_for_retest`
- `retesting`
- recovered/failed retest outcomes
- evidence paths
- bottleneck classification

This is a good fit for expensive tuning because a candidate that is too slow, memory-bound, or unstable can be removed from the active loop without being confused with a weak model.

## Scorecards And Target Metric

The new target scalar is:

```text
classical_survival_lcb
```

The scorecard system now includes fixed-classical game evidence and a lower-confidence-bound survival score. Per-game evidence records outcome, moves survived, max moves, penalties, seed, winner, and reason.

The scoring intuition is:

- A model win scores above full survival.
- A draw or full survival scores as survival.
- A loss receives partial credit based on how long the model survived.
- Illegal moves and crashes are penalized.
- The final scalar is a lower confidence bound, so uncertain or low-sample candidates are ranked conservatively.

The fixed-classical evaluation entry point is:

```text
scripts/run_fixed_classical_eval.py
```

The script default is 20 games per candidate. The config also records a 400-game final evaluation target under `autotune.final_eval`, so the intended final champion evaluation is heavier than the default scout/runner smoke setting.

## Phase 2 Review

Phase 2 review is implemented as a deterministic ranking/filtering pass over scorecards.

The review:

- Requires the 12-epoch floor.
- Requires the target scalar to be `classical_survival_lcb`.
- Excludes quarantined or failed candidates.
- Excludes candidates without fixed-classical evidence.
- Excludes candidates with illegal/crash indicators.
- Ranks remaining candidates by scorecard scalar, not by raw Optuna value alone.

The script entry point is:

```text
scripts/run_phase2_phase3_review.py
```

It writes:

```text
phase2_promotion_report.json
phase3_study_specs.json
review_manifest.json
```

This is the handoff from architecture scouting to true hyperparameter tuning.

## Phase 3 Optuna TPE

Phase 3 is implemented by `Phase3OptunaTpeRunner`.

Unlike Phase 1, Phase 3 actually asks Optuna to suggest hyperparameters. Each promoted architecture/pair family receives its own study, using:

```text
TPESampler(multivariate=True, group=True, n_startup_trials=8)
```

The delayed pruner wraps Successive Halving:

```text
SuccessiveHalvingPruner(min_resource=signal_floor_epoch, reduction_factor=2)
```

The wrapper refuses to prune until:

- the configured signal floor is reached
- a scorecard exists
- the trial has reported compatible evidence

Suggested Phase 3 parameters include:

- learning-rate multiplier
- weight decay
- `c_puct`
- `c_puct_init`
- Dirichlet noise fraction
- scaled alpha total
- PCR low-simulation probability
- recency decay
- value and auxiliary loss weights
- pair loss weight and pair prior mix for pair-mode candidates

The runner validates that Phase 3 suggestions do not change architecture identity, pair mode, or required output heads. This is important: Phase 3 tunes within a promoted family; it does not mutate the architecture search space.

The script entry point is:

```text
scripts/run_phase3_optuna_tpe.py
```

## Champion Selection

Champion selection is now scorecard-driven rather than Optuna-value-driven.

The selector:

- Reads candidate scorecards.
- Applies hard gates.
- Requires completed epoch floor.
- Requires fixed-classical evidence.
- Ranks by Hexo's `classical_survival_lcb`.
- Stores reproduction command and final report metadata.

The script entry point is:

```text
scripts/select_optuna_champion.py
```

This is the right separation of concerns. Optuna is useful for proposing and tracking trials, but the project-specific scorecard decides what actually wins.

## Pair MCTS Changes

Pair strategy is now explicit:

```text
none
root_pair_mcts
full_pair_mcts
```

`root_pair_mcts` applies pair-derived priors at the root. `full_pair_mcts` enables leaf pair scoring as well, and it requires a global graph architecture.

The self-play worker now builds a `PairStrategy` from config and uses:

- root pair priors when pair strategy is enabled
- leaf pair scoring only when `leaf_pair_scoring_enabled` is true
- pair-prior metrics only when pair priors are applicable
- finite-logit validation in pair scoring paths

This implementation is useful for testing whether pair heads improve search, but it is not yet the best research-backed pair-action architecture described in `JOINT_PAIR_MCTS_RESEARCH_NOTE.md`. MCTS still fundamentally searches normal actions, with pair logits blended back into action priors. It does not yet treat `(q1, r1, q2, r2)` as a first-class macro action with its own progressive widening, proposal policy, joint backup semantics, and tree statistics.

## Scripts Added Or Updated

New tuning-facing scripts:

```text
scripts/run_phase1_optuna_scout.py
scripts/run_fixed_classical_eval.py
scripts/run_phase2_phase3_review.py
scripts/run_phase3_optuna_tpe.py
scripts/select_optuna_champion.py
scripts/repair_phase1_scorecards_from_dashboard.py
```

The repair script exists because Phase 1 scorecards can be reconstructed from dashboard metrics. That is useful operationally, but it also highlights that production scoring/evaluation needs careful monitoring so Phase 1 candidates are not promoted using placeholder or incomplete evidence.

## Tests Added

The implementation adds focused tests for:

- config validation and default autotune surface
- candidate recipe materialization
- Phase 1 Optuna scout enqueue/resume/quarantine behavior
- runtime probe cache, semantic hash protection, and quarantine classification
- fixed-classical survival LCB scorecards
- Phase 2 review/ranking
- Phase 3 study specs, sampler/pruner behavior, dry-run optimization, and identity protection
- champion selection
- legacy scheduler export removal
- pair strategy validation and worker behavior

These tests are broad enough to catch many integration regressions. They mostly validate orchestration and artifact contracts; they do not prove that the resulting tuning strategy is sample-efficient or strategically strong.

## Main Behavioral Impact

The biggest runtime impact is that model/search development now has a staged lifecycle:

1. Define candidate families in config.
2. Materialize each candidate into durable artifacts.
3. Run Phase 1 round-robin scout to 12 epochs.
4. Evaluate candidates against fixed classical opponents.
5. Promote candidates using scorecard evidence.
6. Run per-family Phase 3 TPE tuning.
7. Select champion using Hexo scorecards.

The biggest model/search impact is that pair strategies are now tunable candidate dimensions. The project can compare no-pair, root-only pair priors, and full leaf pair scoring under the same artifact and scorecard framework.

## Weak Spots And Follow-Up Review Targets

The Phase 1 scalar path needs the closest review. In production, `EpochScoutEpochRunner` records the training/evaluation scalar available from epoch stats. If `classical_survival_lcb` has not been produced yet, the fallback is currently neutral/zero-like. That is acceptable for orchestration tests, but production promotion should depend on fixed-classical scorecards, not placeholder Phase 1 values.

The fixed-classical default is 20 games per candidate, while the final-eval config says 400 games. This is fine for fast scouting, but final champion selection should be run with enough games to make the lower confidence bound meaningful.

The pair MCTS implementation is still a first-level integration. `full_pair_mcts` means leaf pair prior scoring is enabled; it does not mean the tree treats a pair as a single macro move. If the research goal is best possible pair-level architecture, the next implementation should replace blended action priors with first-class pair-action tree statistics and progressive widening over candidate pairs.

Runtime probe support is well designed, but it only protects candidates when a real probe runner is wired into the production controller. The Phase 1 CLI currently constructs the controller with the epoch runner; review should confirm whether production launch scripts also pass a real runtime probe runner or whether this remains an injectable test/extension point.

The new Optuna layer appears intentionally constrained, but operational consistency matters. Study names, candidate-plan hashes, storage URLs, and candidate ids must remain stable across restarts. Manual edits to candidate plan order or recipe schema version will invalidate resume assumptions.

Legacy tuning modules are no longer exported from `hexorl.tuning`, but some old files/scripts remain in the tree. That may be acceptable for historical reference, but phase-completion cleanup should decide whether they are deprecated, quarantined, or removed.

## Bottom Line

This is a solid first-level implementation of the sequential Optuna plan. It correctly avoids over-trusting early noisy models, makes runtime feasibility a first-class gate, separates architecture scouting from hyperparameter optimization, and keeps final decisions grounded in Hexo scorecards instead of generic Optuna trial values.

The two most important next checks are:

1. Prove that production Phase 1 candidates always get real fixed-classical scorecards before promotion.
2. Decide whether `full_pair_mcts` is intended to remain pair-prior leaf scoring or evolve into true first-class pair-action MCTS.

## Verification Notes

Focused tests were identified for the review:

```text
Python/tests/test_optuna_config_surface.py
Python/tests/test_optuna_scout.py
Python/tests/test_runtime_probe.py
Python/tests/test_scorecard_lcb_and_review.py
Python/tests/test_phase3_optuna_runner.py
Python/tests/test_champion_selection_and_legacy.py
```

The local shell could not execute them because `pytest` is not installed in the active Python environment:

```text
python -m pytest ... -> No module named pytest
```

No source files were changed by this review document.
