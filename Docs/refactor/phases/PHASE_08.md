# Phase 08 - Evaluation + Dashboard + Autotune

## Purpose
Finish the user-facing convergence layer after replay, training, inference, search, and self-play have moved to contracts.

Phase 08 makes evaluation, dashboard/debugging, and autotuning consume the same registry, contracts, policy providers, read-only inspectors, typed recipes, and traces as the runtime. This phase must not preserve dense-only assumptions, dashboard-private reconstruction, raw config mutation, or architecture-string runtime sizing branches.

## Target Modules
- `eval/arena.py`, `players.py`, `policy_player.py`, `scorecard.py`, `league.py`
- `dashboard/contract_inspector.py`, `model_inspector.py`, `graph_inspector.py`, `replay_views.py`
- dashboard backend routes and read-only service adapters
- `tuning/recipes.py`, `family_spaces.py`, `scheduler.py`, `runtime_sweep.py`, `scoring.py`, `manifests.py`, `reporting.py`
- import-audit and artifact-generation tooling for eval/dashboard/tuning cutover

## Source Of Truth
- `Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_V2_20260429.md`
- Registered `ModelFamily` entries and their declared capabilities
- `ModelRecipe` and recipe manifests, not mutable raw config dictionaries
- `PositionContract`, `LegalActionTable`, `CandidateTable`, `PairActionTable`, graph contracts, telemetry contracts, and checkpoint manifests

## V2 Requirements
- Arena evaluation must use `PolicyProvider` for every registered model family.
- Evaluation must not assume dense policy output, dense action indices, dense-only model inputs, or architecture-name dispatch.
- Dashboard must use `ContractInspector` and read-only services only.
- Dashboard must display contract hash, source, schema/version, checkpoint manifest version, inference protocol version, model family, recipe identity, and trace id wherever those facts are available.
- Dashboard must not privately reconstruct legal rows, D6 transforms, candidates, pair rows, graph tokens, graph relations, model inputs, model outputs, replay projections, or checkpoint cleanup.
- Autotune must use typed `ModelRecipe`, `family_spaces`, `runtime_sweep`, `scoring`, `manifests`, and `reporting`.
- Autotune must mutate recipes through typed recipe transforms only. It must not patch raw config fields, architecture strings, or model-family internals.
- Recipe dry-run validation must happen before launch and must explain every rejection.
- Scheduler decisions must be explicit, logged, and tied to score components, validation results, runtime budgets, and progress signals.
- Every runtime sweep must have no-progress watchdogs with actionable logs.
- Old scripts and runtime sizing branches superseded by typed tuning entrypoints must be deleted or quarantined outside runtime imports.

## Evaluation Work
- Convert `PolicyPlayer` to call `ModelFamily.build_policy_provider(...)` or registry-equivalent construction for every registered family.
- Ensure dense CNN, RestNet, graph hybrid, global graph, and future families evaluate through the same `PolicyProvider` interface.
- Add registry enumeration in arena smoke tests so no registered family is skipped accidentally.
- Route pair behavior through `PairStrategy`; evaluation must not consume pair priors merely because a model has a pair-capable head.
- Report policy-source telemetry for each evaluated position, including model family, provider type, input contract, output contract, legal row count, pair rows scored, trace id, and warnings.
- Remove dense-only player code paths, dense action-index assumptions, model-class checks, and architecture string gates from eval.
- Preserve scorecard/league behavior while making player construction recipe/manifest based.

## Dashboard Work
- Make `ContractInspector` the single backend entry point for inspecting histories, legal tables, tactical reports, candidates, pairs, graph contracts, D6 transforms, model inputs, model outputs, replay positions, and traces.
- Allow dashboard routes to depend only on contracts, inspectors, checkpoint manifests, registry metadata, and read-only services.
- Required views:
  - `/history`
  - `/legal-table`
  - `/tactical`
  - `/candidates`
  - `/pairs`
  - `/graph`
  - `/d6`
  - `/model-input`
  - `/model-output`
  - `/trace`
  - `/replay`
  - `/checkpoint`
  - `/recipe`
  - `/autotune`
- Every relevant view must surface hash/source/version/trace facts:
  - history hash
  - legal table hash and source
  - candidate contract hash and source
  - pair action table hash and generation
  - graph schema/relation schema versions
  - model spec version and family
  - checkpoint schema version
  - inference protocol version
  - replay schema version
  - recipe id/config hash
  - trace id and span timings
- Delete or quarantine dashboard code that rebuilds model inputs, legal rows, candidates, pair rows, graph payloads, replay projection, or D6 facts outside canonical services.
- Dashboard errors must point to the failing owner: engine/legal, candidate builder, pair table builder, graph builder, inference adapter, train adapter, policy provider, replay projector, or checkpoint manifest.
- Dashboard tests must prove displayed model inputs match training inputs for golden positions.
- Dashboard must be able to load or generate a single-position debug bundle and compare engine state, contracts, D6 transforms, training targets, model inputs, raw model outputs, policy-provider priors, pair strategy output, MCTS result, and replay record identity.
- Dashboard mismatch views must show both sides of the comparison, the hashes/schema/source fields that disagree, and the likely owner subsystem. It must not hide mismatches behind generic "invalid input" messages.

## Autotune Work
- Introduce typed `ModelRecipe` as the only tuning unit.
- Implement family-specific search spaces in `tuning/family_spaces.py`; spaces come from registered families, not script-local architecture branches.
- Implement `runtime_sweep` as typed `RuntimeSpec` candidates with host-profile validation, token/row/batch/memory budgets, and no-progress watchdog settings.
- Implement scoring as named components, including quality, throughput, stability, resource use, validation failures, stall penalties, and budget penalties.
- Implement manifests for every trial, including recipe identity, model family, model spec version, input/output/action contracts, runtime spec, host profile, git SHA, command, seeds, validation results, scheduler decisions, trace ids, artifacts, and final score components.
- Implement reporting that explains selected, rejected, aborted, retried, promoted, and stopped trials with reasons.
- Add recipe dry-run validation for:
  - family capability compatibility
  - head/loss compatibility
  - input/output/action contract compatibility
  - pair strategy compatibility and caps
  - inference protocol compatibility
  - runtime budget compatibility
  - checkpoint/manifest compatibility
- Add scheduler logs for every decision, including promotion, early stop, retry, budget reduction, watchdog abort, and final recommendation.
- Add no-progress watchdogs for self-play progress, inference response progress, training batch progress, evaluation progress, and artifact/report writing.
- Logs must be actionable: each validation failure, stall, abort, or poor score should include the likely subsystem to inspect and the relevant trace ids.
- Poor-learning reports should include enough debug-bundle references to distinguish model underperformance from bad targets, bad legal rows, bad D6 transforms, policy mapping errors, MCTS misuse, replay corruption, or scheduler/runtime failures.
- Delete or quarantine old autotune scripts that mutate raw config, encode model-family internals, or contain runtime sizing architecture branches.

## Required Deletions And Quarantine
- Delete dense-only eval player assumptions after `PolicyProvider` parity is proven.
- Delete dashboard private model-input reconstruction.
- Delete dashboard private legal/candidate/pair/graph reconstruction.
- Delete autotune family internals inside old scripts.
- Delete runtime sizing architecture branches that duplicate typed `RuntimeSpec` or family-space logic.
- Quarantine one-off migration or comparison scripts under a non-runtime tool path if they are still needed temporarily.
- Ensure quarantined scripts are not imported by runtime, training, dashboard, evaluation, or tuning packages.

## Import Audits
- `eval/` must not import model classes directly for behavior dispatch.
- `dashboard/` must not import sampler-private builders, trainer-private adapters, worker internals, or model-input reconstruction helpers.
- `tuning/` must not import self-play worker internals except through public orchestration/runtime interfaces.
- No Phase 08 runtime path may dispatch on architecture string prefixes.
- No Phase 08 runtime path may use deprecated aliases or compatibility facades.
- Import-boundary tests should fail on banned paths once deletion is complete.

## Artifacts
- Evaluation registry coverage report listing every registered family and provider used.
- Dashboard fixture parity report for required views.
- Autotune dry-run validation report for at least one valid and one rejected recipe per family.
- Runtime sweep report with watchdog configuration and simulated no-progress outcome.
- Import audit report for eval/dashboard/tuning banned dependencies.
- Deletion/quarantine report naming removed scripts, quarantined scripts, and remaining approved one-off tools.

## Parallel Subagent Work
- S1: eval `PolicyProvider` conversion and registry coverage.
- S2: dashboard `ContractInspector` routes, required views, and read-only service boundaries.
- S3: dashboard hash/source/version/trace display and parity with training/replay projections.
- S4: typed autotune recipes, family spaces, dry-run validation, and manifests.
- S5: runtime sweep, scheduler decisions, scoring, watchdogs, reporting, actionable logs, import audits, artifacts, and hard gates.
- Orchestrator: verify deletion/quarantine scope for old eval, dashboard, autotune, and runtime-sizing paths.

## Mandatory Tests
- Arena evaluates every registered model family through `PolicyProvider`.
- Arena fails fast if a registered family lacks a valid policy provider.
- Eval has no dense-only action assumptions and no architecture-string dispatch.
- No pair scoring occurs during evaluation unless `PairStrategy` explicitly enables it.
- Dashboard fixture parity: dashboard/training/replay inputs match on golden positions.
- Dashboard required views render from `ContractInspector` or read-only services only.
- Dashboard display assertions cover hash/source/version/trace fields.
- Dashboard debug-bundle view displays engine, contracts, D6, targets, model outputs, policy priors, MCTS, and replay comparisons for golden positions.
- Dashboard mismatch tests corrupt one subsystem at a time and assert the displayed owner subsystem is correct.
- Dashboard has no private reconstruction imports for legal, D6, candidates, pairs, graph, model inputs, or replay projection.
- Autotune accepts valid recipes and rejects incompatible recipes in dry-run before launch.
- Autotune mutates typed `ModelRecipe` values only; raw config mutation tests fail.
- Family spaces exist for every registered family.
- Runtime sweep validates host limits, row/token budgets, memory budgets, and watchdog configuration.
- Scheduler decision tests assert reason codes and score components.
- No-progress watchdog tests cover self-play, inference, training, evaluation, and artifact writing.
- Autotune logs include actionable validation, lifecycle, scheduler, abort, and next-debugging-action messages.
- Autotune poor-learning reports link to trace/debug bundles or summarize the likely failure class across model, training targets, engine, D6, policy mapping, MCTS, replay, and runtime scheduling.
- Import audits reject old scripts, runtime sizing architecture branches, deprecated aliases, and dashboard reconstruction helpers.

## Hard Gates
- `arena can evaluate every registered family through PolicyProvider`
- `dashboard/training inputs match on golden positions`
- `dashboard uses ContractInspector/read-only services only`
- `dashboard displays hash/source/version/trace facts`
- `autotune rejects incompatible recipes during dry-run`
- `autotune mutates ModelRecipe, not raw config fields`
- `runtime sweeps have no-progress watchdogs`
- `scheduler decisions are logged with reasons`
- `old scripts and runtime sizing architecture branches are deleted or quarantined`
- `import audits find no banned eval/dashboard/tuning dependencies`
- `artifacts are generated for eval coverage, dashboard parity, autotune dry-run, runtime sweep, and deletion/quarantine`

## Exit Criteria
- Evaluation works for every registered family through `PolicyProvider`, with no dense-only assumptions.
- Dashboard is a contract inspector, not a sampler/trainer/model-input reconstructor.
- Dashboard required views expose the hashes, sources, versions, manifests, and traces needed to debug mismatches.
- Dashboard can localize single-position behavior mismatches across engine, contracts, D6, targets, model outputs, policy priors, MCTS, and replay.
- Autotune runs through typed recipes, family spaces, runtime sweeps, scoring, manifests, reporting, scheduler decisions, watchdogs, and actionable logs.
- Old scripts, private reconstruction paths, and architecture-branch runtime sizing code are deleted or quarantined away from runtime imports.
- Mandatory tests, import audits, artifacts, and hard gates pass.
