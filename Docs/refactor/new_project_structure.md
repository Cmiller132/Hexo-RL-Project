# New Project Structure After The V2 Modular Refactor

Date: 2026-04-30

This document explains the current repository layout after the major V2 modular
refactor, how the main runtime paths are intended to fit together, what each
refactor phase changed, and whether the refactor appears to have been completed
with enough care.

The short answer is: the project now has a real, coherent modular shape and the
old runtime paths were replaced deliberately. The strongest evidence supports
the architecture cutover, import/deletion discipline, and boundary testing. The
care level was high enough to treat the refactor as architecturally complete,
but not high enough to treat every phase artifact as perfect proof. Some
evidence is local, synthetic, proxy-based, missing, or inconsistently recorded.

This is therefore a structure map and a conservative audit, not a celebratory
rubber stamp.

## Review Scope

The review was framed from the user request as:

```text
Goal
Break down the current project structure after the major refactor under
Docs/refactor and create new_project_structure documentation.

Success criteria
- Explain how the repository is laid out now.
- Explain how the runtime architecture flows.
- Review phase evidence and work completed across Docs/refactor.
- Use subagents for independent phase, Python/runtime, and Rust/CI review.
- Say whether the refactor was done properly and with enough care.
- Be extremely thorough and evidence-focused.

Constraints
- Preserve unrelated changes.
- Use current repository files as source of truth.
- Do not claim completion from narrative alone.
- Distinguish architectural completion from evidence quality.

Required evidence
- Current filesystem and import layout.
- Refactor phase docs, manifests, completion packets, audits, tests, telemetry,
  performance, deletion manifests, and final CI artifacts.
- Current architecture policy audit.

Stop rules
- If a requirement or claim is not backed by current files or artifacts, call it
  out as a residual risk instead of smoothing it over.
```

## Top-Level Repository Layout

```text
.
|-- AGENTS.md
|-- Cargo.toml
|-- Cargo.lock
|-- Configs/
|-- Docs/
|-- Python/
|-- benches/
|-- crates/
|-- scripts/
|-- tools/
|-- .github/workflows/
```

### `Cargo.toml` And `crates/`

The Rust workspace is the canonical game/rules/search engine layer.

```text
crates/
|-- hexgame-core/
|-- hexgame-py/
|-- hexgame-bench/
|-- hexgame-cli/
```

- `crates/hexgame-core/`
  - Owns Hexo rules, board state, legal move generation, tensor encoding,
    tactical classification, classical search, and neural MCTS.
  - Public stable facades are documented as `rules`, `encoding`, `tactics`,
    and `classical`.
  - `mcts` remains public as a documented FFI exception for the PyO3 crate.
- `crates/hexgame-py/`
  - Builds the Python `_engine` extension.
  - Owns the PyO3 API surface and byte protocols exposed to Python.
  - Important files:
    - `src/lib.rs`
    - `src/engine.rs`
    - `src/encode.rs`
    - `src/protocol.rs`
- `crates/hexgame-bench/`
  - Criterion benches for encoding, MCTS, and tactics/threats.
- `crates/hexgame-cli/`
  - Rust CLI and executable caller.

Rust should be read as the production rules boundary, not as a self-validating
oracle. Python accepts Rust as authoritative for core game semantics, but still
validates Rust-derived rows, byte payloads, root/batch tokens, hashes, and model
outputs before runtime consumers act on them.

### `Python/`

The Python package is the main ML, self-play, replay, training, evaluation,
dashboard, and autotune runtime.

```text
Python/
|-- pyproject.toml
|-- src/hexorl/
|-- tests/
|-- dashboard_frontend/
```

- `Python/pyproject.toml`
  - Defines package `hexorl` version `0.2.0`.
  - CLI entry point: `hexorl = hexorl.cli:main`.
- `Python/src/hexorl/`
  - The refactored runtime package.
- `Python/tests/`
  - Phase-shaped and subsystem tests.
- `Python/dashboard_frontend/`
  - Vite/React dashboard frontend.

### `Docs/`

`Docs/` contains both older project docs and the V2 refactor control plane.

Important post-refactor documents:

- `Docs/RUST_API.md`
- `Docs/refactor/README.md`
- `Docs/refactor/PHASED_IMPLEMENTATION_PLAN.md`
- `Docs/refactor/V2_REQUIREMENTS_MATRIX.md`
- `Docs/refactor/CI_STRATEGY.md`
- `Docs/refactor/PERFORMANCE_STRATEGY.md`
- `Docs/refactor/phases/PHASE_00.md` through `PHASE_09.md`
- `Docs/refactor/artifacts/phase_00/` through `phase_09/`
- `Docs/refactor/rust_review/`
- this file, `Docs/refactor/new_project_structure.md`

### `Configs/`

Current configuration files live under `Configs/`.

```text
Configs/
|-- default.toml
|-- default_config.toml
|-- production.toml
|-- reproducible.toml
|-- small_test.toml
|-- wsl_speed_probe.toml
```

Configuration loading and validation are owned by `hexorl.config`.

### `scripts/`, `tools/`, And `benches/`

- `scripts/`
  - Runtime helpers, dashboard launch helpers, ablation runners, baseline
    capture scripts, and rendering utilities.
- `tools/refactor/`
  - Final V2 audit/smoke tooling:
    - `phase09_policy_audit.py`
    - `phase09_final_smoke.py`
    - `phase09_performance_probe.py`
    - `phase09_artifact_validator.py`
- `benches/`
  - Python benchmark/profiling scripts for inference, model layers, self-play,
    stability, and training.

### `.github/workflows/`

The current CI workflow is `.github/workflows/ci.yml`.

It defines:

- `rust-fast`
- `deep-oracle`
- `python-fast`
- `architecture-policy`
- `dashboard-build`
- `final-v2-smoke`

This is consistent with the refactor's tiered CI model: fast PR checks plus
scheduled/manual deep and final smoke gates.

## Current Python Runtime Layout

Current source package:

```text
Python/src/hexorl/
|-- __init__.py
|-- cli.py
|-- runtime.py
|-- axis_policy/
|-- config/
|-- contracts/
|-- dashboard/
|-- engine/
|-- epoch/
|-- eval/
|-- graph/
|-- inference/
|-- models/
|-- replay/
|-- search/
|-- selfplay/
|-- train/
|-- tuning/
```

The old runtime source packages are no longer present as tracked runtime code:

- `hexorl.action_contract`
- `hexorl.buffer`
- `hexorl.model`

During this review, cache-only local directories for those old names were found
under `Python/src/hexorl/` from prior bytecode and `.DS_Store` files. They had
no tracked source files and were removed locally so the current absent-path
policy audit reflects the intended final state.

## Package Responsibilities

### `hexorl.cli`, `hexorl.runtime`, `hexorl.config`

Entry/config/runtime ownership.

- `cli.py`
  - CLI entry point.
- `runtime.py`
  - Runtime/host coordination helpers.
- `config/loader.py`
  - Config file loading.
- `config/schema.py`
  - Typed config schema and validation.

These modules sit near the top of the runtime stack. They should configure and
wire the system, not recreate game semantics, replay formats, model-family
logic, or inference protocols.

### `hexorl.contracts`

Canonical Python data contracts.

```text
contracts/
|-- actions.py
|-- candidates.py
|-- coordinates.py
|-- debug.py
|-- graph.py
|-- history.py
|-- identity.py
|-- legal.py
|-- pairs.py
|-- replay.py
|-- symmetry.py
|-- tactical.py
|-- targets.py
|-- telemetry.py
|-- validation.py
```

This package is the Python data-contract authority. It owns typed, validated,
hashable, source/version-aware value objects for:

- compact move history
- legal action rows
- candidate rows and features
- pair action rows
- graph/replay/target payload identity
- D6 symmetry utilities
- tactical payload shape
- debug and telemetry payloads
- contract validation errors

Architectural rule: `contracts/` should stay pure. It should not import model
runtime, inference servers, search orchestration, training loops, dashboard
rendering, or tuning orchestration.

### `hexorl.engine`

Python-facing Rust boundary.

```text
engine/
|-- __init__.py
|-- encoding.py
|-- history.py
|-- legal.py
|-- parity.py
|-- rust.py
|-- tactical.py
```

Responsibilities:

- load `_engine` through `engine/rust.py`
- wrap Rust history encode/decode into Python contracts
- validate Rust legal rows into `LegalActionTable`
- expose encoding helpers
- provide D6 parity hooks
- expose tactical scanning through the current Rust-facing tactical model

`hexorl.engine.rust` is the central loader. Runtime code should not import the
extension module directly.

### `hexorl.graph`

Graph semantic construction and tensor projection.

```text
graph/
|-- __init__.py
|-- batch.py
|-- collate.py
|-- semantic_builder.py
|-- tensorize.py
```

Responsibilities:

- build graph semantic contracts from canonical history/legal/candidate/pair
  contracts
- tensorize graph semantic contracts for model input
- collate graph batches

Important note: `graph/batch.py` remains as compatibility exports over the
split graph modules. Runtime source currently imports the split modules
directly; tests still import `hexorl.graph.batch`. That file is not an old
runtime semantic owner, but it should be watched so future runtime code does not
slide back into monolithic graph imports.

### `hexorl.models`

Model registry, specs, capabilities, checkpointing, model families, trunks, and
heads.

```text
models/
|-- capabilities.py
|-- checkpoint.py
|-- factory.py
|-- global_graph.py
|-- network.py
|-- registry.py
|-- specs.py
|-- families/
|-- heads/
|-- trunks/
```

Responsibilities:

- model family registry and descriptors
- `ModelSpec` normalization and capability detection
- model construction
- train adapter lookup
- inference manifest lookup
- loss plan and default recipe/tune-space lookup
- checkpoint save/load/inspect through `CheckpointManager`

Important nuance: `models/factory.py` is not a purely passive model builder.
It is the facet registry owner and can resolve train adapters and inference
metadata. That cross-link is intentional in the current architecture.

The old singular package `hexorl.model` has been replaced by `hexorl.models`.

### `hexorl.inference`

Typed inference protocol, adapters, shared-memory transport, batching, client,
server, and telemetry.

```text
inference/
|-- __init__.py
|-- batching.py
|-- client.py
|-- protocol.py
|-- server.py
|-- shm_queue.py
|-- shm_transport.py
|-- telemetry.py
|-- adapters/
```

Responsibilities:

- require `InferenceProtocolManifest` before request submission
- dispatch by request kind/protocol, not architecture string
- pack/wait/decode/reset through the transport owner
- fail fast on protocol mismatch or timeout
- validate inference responses before search/policy use
- preserve batching and backpressure telemetry
- adapt dense, sparse, global graph, and pair-scoring requests

### `hexorl.search`

Policy/search boundary.

```text
search/
|-- __init__.py
|-- context.py
|-- engine_adapter.py
|-- expansion.py
|-- mcts_runner.py
|-- pair_strategy.py
|-- policy_provider.py
|-- priors.py
```

Responsibilities:

- `SearchContext` as search-time contract context
- `PolicyProvider` as the only path from model/inference outputs to row-mapped
  priors
- `PairStrategy` as the only owner of pair selection/caps/scoring policy
- `EngineAdapter` as the only Python caller of Rust MCTS lifecycle APIs
- root and leaf generation token checks
- legal row identity, offset, finite-prior, and stale-batch validation

This package is the gate between model outputs and Rust MCTS. It is the correct
place to reject malformed priors, stale legal rows, stale generation tokens,
bad pair rows, and non-finite values before search mutates state.

### `hexorl.selfplay`

Self-play orchestration, game execution, worker lifecycle, record writing, and
telemetry.

```text
selfplay/
|-- __init__.py
|-- game_runner.py
|-- orchestrator.py
|-- record_writer.py
|-- records.py
|-- regret_buffer.py
|-- rgsc.py
|-- telemetry.py
|-- worker.py
```

Responsibilities:

- `GameRunner` owns game execution.
- `SelfPlayWorker` is lifecycle/IPC oriented.
- `orchestrator.py` wires worker execution.
- `record_writer.py` writes canonical replay records.
- `telemetry.py` owns heartbeats, no-progress payloads, contract traces, and
  behavior debug bundles.
- `rgsc.py` and `regret_buffer.py` retain restart/regret support outside the
  deleted old `hexorl.buffer` package.

Architectural rule: workers should not own game semantics, replay assembly,
pair strategy internals, graph construction internals, or direct MCTS wiring.

### `hexorl.replay`

Canonical replay runtime.

```text
replay/
|-- __init__.py
|-- codec.py
|-- fixtures.py
|-- projector.py
|-- sampler.py
|-- storage.py
```

Responsibilities:

- `ReplayGameRecord` and `ReplayPositionRecord` codec
- replay storage with bounded capacity/backpressure telemetry
- sampler over canonical replay records
- projector that turns replay records into training batches
- golden/corruption fixtures

Current runtime path:

```text
selfplay/record_writer.py
  -> replay/codec.py
  -> replay/storage.py
  -> replay/sampler.py
  -> replay/projector.py
  -> train/adapters.py
```

The old `hexorl.buffer` runtime path was replaced.

### `hexorl.train` And `hexorl.epoch`

Training adapters, losses, trainer, EMA, and epoch-level orchestration.

```text
train/
|-- __init__.py
|-- adapters.py
|-- ema.py
|-- losses.py
|-- trainer.py

epoch/
|-- __init__.py
|-- pipeline.py
```

Responsibilities:

- train through `TrainAdapter`, not model-class branching
- consume replay projector outputs
- validate targets, including pair target semantics
- own losses and trainer loop
- run epoch/pipeline orchestration using registry-backed model construction

### `hexorl.eval`

Evaluation and scoring surfaces.

```text
eval/
|-- __init__.py
|-- arena.py
|-- checkpoint_league.py
|-- classical.py
|-- elo.py
|-- players.py
|-- scorecard.py
|-- tactical_suite.py
```

Responsibilities:

- arena evaluation through registered policy/provider paths
- classical opponent support
- checkpoint league and Elo tracking
- tactical suite fixtures
- scorecard calculations

The refactor goal was to remove dense-only assumptions and route evaluation
through the same provider/registry contracts as self-play.

### `hexorl.dashboard`

Dashboard backend, inspection, replay/debug display, snapshots, and play
support.

```text
dashboard/
|-- __init__.py
|-- app.py
|-- arena_service.py
|-- checkpoints.py
|-- contract_inspector.py
|-- db.py
|-- fixtures.py
|-- model_cache.py
|-- play.py
|-- pseudocode.py
|-- recorder.py
|-- render.py
|-- replay.py
```

Responsibilities:

- FastAPI app and dashboard service routes
- contract inspection through read-only inspector services
- replay/debug payload display
- checkpoint indexing
- DB/recorder/play support
- match snapshot rendering

`contract_inspector.py` is the intended dashboard dispatch point for read-only
inspection. Dashboard routes should not privately rebuild candidate, pair,
graph, replay, or model semantics.

### `hexorl.tuning`

Typed autotune/runtime-sweep layer.

```text
tuning/
|-- __init__.py
|-- family_spaces.py
|-- manifests.py
|-- recipes.py
|-- reporting.py
|-- runtime_sweep.py
|-- scheduler.py
|-- scoring.py
|-- validation.py
```

Responsibilities:

- typed `ModelRecipe`
- family tune spaces
- scheduler/scoring logic
- runtime-sweep and watchdog reporting
- validation and manifests

The old ASHA/BOHB/PB2 module paths and Phase 3 autotune scripts were deleted
as runtime paths during Phase 08.

### `hexorl.axis_policy`

Axis-policy prototype/support package.

```text
axis_policy/
|-- DESIGN.md
|-- __init__.py
|-- core.py
|-- dual_strength.py
|-- experiments.py
|-- legacy_influence.py
|-- registry.py
```

This package remains outside the main refactor spine. It is currently best read
as dashboard/eval-facing tactical or policy-analysis support rather than a
central runtime authority.

## Dependency Direction

The intended dependency flow after the refactor is:

```text
Rust hexgame-core
  -> hexgame-py _engine
  -> hexorl.engine
  -> hexorl.contracts
  -> hexorl.graph / hexorl.replay
  -> hexorl.models / hexorl.inference / hexorl.search
  -> hexorl.selfplay
  -> hexorl.replay
  -> hexorl.train
  -> hexorl.eval / hexorl.dashboard / hexorl.tuning
```

More precisely:

- `contracts` and `engine` are low-level authorities.
- `graph` builds semantic/tensor projections from contracts and engine outputs.
- `models` owns registry/spec/checkpoint/family capabilities.
- `inference` owns request/response protocol and adapter collation/decode.
- `search` consumes engine, contracts, graph, models, and inference through
  explicit `PolicyProvider`, `PairStrategy`, and `EngineAdapter` boundaries.
- `selfplay` composes `GameRunner`, search, inference, replay writing, and
  telemetry.
- `replay` stores, samples, and projects canonical records for training.
- `train` consumes projected replay batches through adapters.
- `eval`, `dashboard`, and `tuning` are application/inspection/orchestration
  surfaces that should consume canonical contracts instead of recreating them.

## Runtime Data Path

### Self-Play To Training

```text
GameRunner
  -> SearchContext
  -> PolicyProvider
  -> PairStrategy
  -> EngineAdapter
  -> Rust MCTS via _engine
  -> ContractTrace / SelfPlayDebugBundle
  -> SelfPlayRecordWriter
  -> ReplayGameRecord
  -> ReplayStorage
  -> ReplayDataset
  -> replay.projector
  -> TrainAdapter
  -> Trainer
```

Key safety points:

- Rust MCTS calls are concentrated in `search/engine_adapter.py`.
- Pair scoring only happens through `PairStrategy`.
- Model priors are mapped to legal rows through `PolicyProvider`.
- Replay records carry history/legal identity and reject transient MCTS tokens
  as replay semantics.
- Training receives projected batches, not old buffer-owned decoded records.

### Inference

```text
InferenceClient
  -> InferenceProtocolManifest
  -> shared-memory queue / transport
  -> InferenceServer
  -> model family adapter
  -> response telemetry
  -> PolicyProvider/SearchEvaluation
```

Key safety points:

- request kind/protocol replaces architecture-string dispatch
- handshake mismatch fails fast
- stale slots and stale response ids are tested
- response telemetry carries protocol/model/count/timing/warning context

### Dashboard Inspection

```text
Dashboard route
  -> ContractInspector
  -> read-only inspector service
  -> canonical contracts / replay / checkpoint / debug bundle
```

Key safety points:

- dashboard should inspect, not rebuild private semantics
- displayed identity includes source/version/hash/protocol/family/recipe where
  available

## Refactor Phase Summary

### Phase 00 - Baseline Freeze, Guardrails, And Evidence

Owned rows: `V2-000` through `V2-006`.

What changed:

- Established baseline evidence and rollback/control-plane artifacts.
- Captured legacy runtime inventories.
- Added/validated pair-strategy guardrails.
- Captured HostProfile and baseline performance artifacts.
- Established structured logging and watchdog evidence.

Evidence inspected:

- `Docs/refactor/artifacts/phase_00/MANIFEST.md`
- `agent_completion_packet.md`
- command transcripts
- config hashes
- import audits
- performance baselines
- watchdog samples
- inventories
- exit reports

Care assessment:

- Strong baseline/control-plane intent.
- Some manifest-referenced artifacts are not present in the current checkout,
  including `logs/structured_events.jsonl` and
  `performance/training_smoke_run/epoch_0001.pt`.
- This weakens the archival completeness of the phase, but not the later source
  layout itself.

### Phase 01 - Engine + Contracts Foundation

Owned rows: `V2-010` through `V2-016`.

What changed:

- Added `hexorl.contracts`.
- Added `hexorl.engine`.
- Centralized Python-facing Rust boundary.
- Established compact history, legal table, D6, source/hash/schema policy.
- Removed direct runtime `_engine` usage outside the engine boundary.

Evidence inspected:

- focused Phase 01 pytest transcript: `169 passed`
- compile transcript
- import audits
- deletion manifest
- debug payload
- performance timing JSON
- contract examples

Care assessment:

- Good architectural foundation.
- Full Python pytest timed out and was explicitly marked non-closing.
- That is acceptable as a Phase 01 scoped gate, but should not be confused with
  full-project proof at that point.

### Phase 02 - Candidate/Pair/Graph Builder Convergence

Owned rows: `V2-020` through `V2-025`.

What changed:

- Moved candidate semantics to `contracts/candidates.py`.
- Moved pair action semantics to `contracts/pairs.py`.
- Split graph semantic construction from tensorization/collation.
- Removed production `PairCandidateBatch`.
- Demoted old graph batch ownership to split-module exports.

Evidence inspected:

- Phase 02 contracts tests
- global graph contract tests
- training data pipeline tests
- dashboard tactical smoke tests
- import audits
- builder debug bundle
- performance smoke JSON
- deletion manifest

Care assessment:

- Stronger than average phase evidence.
- No full-suite closure in this phase, but focused evidence maps well to the
  owned runtime areas.

### Phase 03 - Model Registry, TrainAdapter, And CheckpointManager

Owned rows: `V2-030` through `V2-035`.

What changed:

- Replaced `hexorl.model` with `hexorl.models`.
- Added registry/spec/capability system.
- Routed trainer through `TrainAdapter`.
- Added `CheckpointManager`.
- Added model family descriptors/facets.
- Corrective pass: moved built-in descriptor construction into
  `models/families/*`, made `models/heads/*` and `models/trunks/*` real
  component modules, and reduced `models/factory.py` to registry/public API
  routing.
- Corrective pass: `CheckpointManager.inspect()` now reads a lightweight
  `checkpoint_manifest.json` member from the checkpoint archive without
  loading model weights.

Evidence inspected:

- model registry tests
- train adapter/checkpoint tests
- import/deletion audit
- registry/checkpoint examples
- training debug bundle sample
- performance notes

Care assessment:

- Structure is substantially cleaner.
- After the corrective pass, the model package is no longer just a facade
  layout; family modules own descriptor construction and head/trunk modules
  participate in runtime builders.
- Exit report path is non-standard (`exit_gates/exit_gate_report.md` instead of
  a root-level `exit_gate_report.md`), which is an evidence organization issue.
- Earlier timeout/instability is documented as resolved or non-closing.

### Phase 04 - Inference Protocol And Adapters

Owned rows: `V2-040` through `V2-046`.

What changed:

- Added typed inference protocol manifest.
- Added request-kind dispatch.
- Added shared-memory transport lifecycle ownership.
- Added dense/sparse/global/pair adapters.
- Added response telemetry and backpressure/timeout evidence.

Evidence inspected:

- inference tests
- protocol handshake matrix
- timeout audit
- mutation/corruption report
- batching/backpressure profile
- response telemetry snapshot
- deletion/import proof

Care assessment:

- Runtime shape is correct and tested.
- Matrix status for Phase 04 rows is recorded as `done: ...` rather than the
  matrix's declared `complete` status. This is a bookkeeping defect.
- Artifacts are flatter than the master artifact template, which makes evidence
  harder to validate mechanically.

### Phase 05 - PolicyProvider, PairStrategy, EngineAdapter

Owned rows: `V2-050` through `V2-057`.

What changed:

- All self-play priors flow through `PolicyProvider`.
- Pair scoring is owned by `PairStrategy`.
- Rust MCTS lifecycle is owned by `EngineAdapter`.
- Stale root/batch token semantics are carried through Python search.
- Worker-owned MCTS and pair chunk logic was removed.
- Corrective pass: policy-provider creation now resolves via registry
  descriptor capabilities and extension registrations, graph-hybrid providers
  fail loudly without candidate contracts, and explicit pair strategies score
  through inference-backed pair-scoring providers rather than replay targets.
- Corrective pass: inference handshake uses negotiated server manifests and
  batching policy ownership moved into `inference/batching.py`.

Evidence inspected:

- `Python/tests/search`
- config/engine/production smoke tests
- Rust stale-token tests
- maturin build transcript
- import audit
- MCTS trace/error/debug samples
- pair strategy docs
- performance profile

Care assessment:

- Boundary discipline is a major improvement.
- Performance evidence is useful but partially mock/proxy-based, so it is not a
  production throughput proof.

### Phase 06 - GameRunner And SelfPlayWorker Cleanup

Owned rows: `V2-060` through `V2-065`.

What changed:

- `GameRunner` owns game execution.
- `SelfPlayWorker` became lifecycle/IPC-oriented.
- Self-play telemetry/debug bundle support was expanded.
- Record writing moved behind explicit writer boundaries.
- Worker direct semantic ownership was deleted.

Evidence inspected:

- self-play tests
- search/self-play integration tests
- replay boundary tests
- import audits
- self-play telemetry samples
- debug bundle
- self-play smoke profile

Care assessment:

- Good separation of runtime responsibilities.
- Deep real Rust plus live inference-server self-play sweep appears to be
  scheduled/deep evidence rather than strong local phase evidence.

### Phase 07 - Replay Cutover

Owned rows: `V2-070` through `V2-075`.

What changed:

- Added canonical `hexorl.replay` runtime.
- New self-play writes canonical replay records.
- Sampler/training consume `ReplayStorage` and `replay.projector`.
- Old `hexorl.buffer` imports were removed from Phase 07-owned runtime scopes.
- RGSC regret buffer moved out of `hexorl.buffer`.
- Corrective pass: `TrainAdapter` now accepts only `ProjectedReplayBatch` from
  `replay/projector.py`; raw tuple training batches and legacy tuple export
  were removed.
- Corrective pass: graph pair metadata now carries canonical pair rows, phase,
  known-first state, and masks through tensorization/collation, and train-time
  validation checks semantic identity rather than only shapes.

Evidence inspected:

- replay codec/storage/projector tests
- replay import audit
- Phase 06/07 record boundary tests
- production smoke updates
- debug bundle sample
- replay throughput profile
- replay contract examples

Care assessment:

- Replay runtime path is clearly mapped and testable.
- The old `buffer` package was not fully deleted until Phase 09, which was
  explicitly planned rather than accidental.

### Phase 08 - Evaluation, Dashboard, And Autotune

Owned rows: `V2-080` through `V2-086`.

What changed:

- Evaluation routes through policy providers and model registry.
- Dashboard routes through `ContractInspector`.
- Dashboard surfaces contract hash/source/version/protocol/model/recipe
  identity.
- Autotune uses typed recipes and family spaces.
- Legacy ASHA/BOHB/PB2 module/script paths were deleted.
- Corrective pass: `ContractInspector` is now dispatcher-only; focused
  read-only services live in `dashboard/inspection_services.py` and shared
  position contract builders live in `eval/position_services.py`.
- Corrective pass: dashboard model inference delegates through provider-backed
  `dashboard/model_inference.py`, and the remaining runtime scripts use typed
  recipes/section transforms instead of raw config mutation.

Evidence inspected:

- eval provider tests
- dashboard contract inspector tests
- tuning typed autotune tests
- dashboard fixture parity report
- autotune dry-run validation
- runtime sweep/watchdog report
- runtime utilization sweep

Care assessment:

- Application-level surfaces were brought into the new architecture.
- Performance/utilization evidence remains mostly synthetic/proxy.

### Phase 09 - Final Deletion And CI Enforcement

Owned rows: `V2-090` through `V2-100`.

What changed:

- Deleted final old runtime package paths:
  - `hexorl.action_contract`
  - `hexorl.buffer`
  - old singular `hexorl.model`
- Added final architecture policy audit tooling.
- Added final smoke tooling.
- Added artifact validator and performance probe.
- Updated CI workflow to enforce Rust/Python/dashboard/policy/final smoke tiers.
- Corrective pass: policy audit now rejects dashboard-private reconstruction
  helpers and raw script config mutation that the first audit allowed.
- Corrective pass: Phase 09 smoke config construction uses typed recipes rather
  than direct `cfg.model.architecture` mutation.

Evidence inspected:

- final conformance report
- final smoke summary/debug/autotune artifacts
- policy audit JSON
- mutation/corruption report
- rust suspicion report
- CI tier inventory
- flaky quarantine report
- artifact retention policy
- performance comparison JSON
- command transcript claims

Care assessment:

- Strongest phase for final source layout and deletion policy.
- Current local `phase09_policy_audit.py` passes with zero findings.
- Performance evidence is a synthetic final-V2 hot-path proxy. It contains
  `0.0` for several proxy throughput fields and should not be treated as a
  production-grade performance proof.
- `evidence_reconciliation.md` references
  `telemetry_samples/phase09_trace_samples.jsonl`, but that file is absent in
  the current checkout.
- The artifact validator reports no missing artifacts despite at least one
  referenced missing path, so validator coverage is incomplete.

## Current Test And Evidence Map

### Python Tests

Current test directories:

```text
Python/tests/
|-- contracts/
|-- dashboard/
|-- engine/
|-- eval/
|-- inference/
|-- models/
|-- phase09/
|-- replay/
|-- search/
|-- selfplay/
|-- train/
|-- tuning/
```

Coverage areas:

- contracts and engine boundary:
  - `Python/tests/contracts/*`
  - `Python/tests/engine/*`
  - `Python/tests/test_engine_smoke.py`
  - `Python/tests/test_engine_invariants.py`
- candidates/pairs/graph:
  - `Python/tests/contracts/test_phase02_builders.py`
  - `Python/tests/test_global_graph_contract.py`
- models/train/checkpoints:
  - `Python/tests/models/test_phase03_model_registry.py`
  - `Python/tests/train/test_phase03_train_adapter_checkpoint.py`
- inference:
  - `Python/tests/inference/*`
  - `Python/tests/test_inference_server.py`
- search:
  - `Python/tests/search/*`
- self-play:
  - `Python/tests/selfplay/*`
- replay:
  - `Python/tests/replay/*`
- eval/dashboard/tuning:
  - `Python/tests/eval/*`
  - `Python/tests/dashboard/*`
  - `Python/tests/tuning/test_phase08_typed_autotune.py`
- final policy:
  - `Python/tests/phase09/test_phase09_policy_audit.py`

### Rust Tests And Benches

Rust evidence comes from:

- `cargo test --workspace`
- `cargo test --workspace --release`
- `cargo clippy --workspace --release -- -D warnings`
- scheduled/manual ignored oracle tests
- Criterion benches in `crates/hexgame-bench`

The Rust review documents report substantial hardening:

- transactional history/state loading
- root/batch MCTS generation tokens
- fallible MCTS APIs
- stale token rejection
- complete tactical status model
- sparse full-board tactical scanning
- stable public facades

### CI Evidence

The current CI workflow includes:

- Rust fmt/tests/release/clippy
- scheduled/manual deep oracle
- Python V2 shard with maturin extension build
- architecture policy audit
- dashboard frontend build
- scheduled/manual final V2 smoke

This is appropriate for a refactor of this size. The caution is that local
artifact evidence and remote CI provenance are not the same thing. The phase
artifacts claim passing final commands, but this review did not verify remote
GitHub run IDs or scheduled final-SHA runs.

## Was The Refactor Done Properly?

### Architectural Answer

Yes, after the corrective pass, the source structure now reflects the refactor
goals:

- Runtime semantics are no longer concentrated in old ad hoc packages.
- `contracts/` and `engine/` establish low-level boundaries.
- Rust is treated as canonical for rules/search primitives but not blindly
  trusted across Python boundaries.
- Candidate, pair, graph, replay, model, inference, search, self-play,
  training, eval, dashboard, and tuning responsibilities are separated.
- Old runtime source packages are gone.
- The current integrated architecture policy audit passes with zero findings.
- The 14 code-review findings around model facades, legacy train tuples,
  dashboard reconstruction, checkpoint inspect, Python legal fallback,
  pair-scoring bypasses, policy-provider switches, mutable graph semantics,
  batching ownership, protocol handshake, stale pair helpers, provider bypasses,
  and raw script config mutation have been closed in source and tests.

The modular shape is real. This is not just a set of new folders placed beside
old code.

### Evidence Answer

Mostly, but with caveats.

The refactor was handled with more care than a typical large code shuffle:

- Every phase has a phase doc.
- Every phase has a manifest and completion packet.
- Matrix rows exist for every V2 requirement.
- Import/deletion audits were created.
- Focused tests were added for the new boundaries.
- Debug bundles and telemetry samples were produced.
- Performance evidence was at least considered in every hot-path phase.
- Final policy audit and final smoke tooling exist.

However, the evidence is not uniformly strong:

- Some artifacts referenced by manifests/reconciliation files are missing in the
  current checkout.
- Phase 04 matrix statuses use non-standard values.
- Several performance reports are local, smoke, mock, or synthetic proxy
  evidence rather than stable production-runner benchmark proof.
- Panic/public API inventory is more narrative than line-by-line.
- Some final Rust suspicion evidence maps concerns to tests/files rather than
  embedding command output.
- Remote final-SHA CI run provenance was not verified in this review.

The correct conclusion after the corrective pass is:

```text
The refactor is properly completed at the source architecture and runtime
cutover level for the reviewed V2 boundaries. The corrective pass closed the
remaining facade/bypass paths found in review. The remaining concerns are
evidence-quality, remote CI provenance, and production-grade performance
hardening follow-ups, not active old runtime architecture.
```

## Residual Risks

### R1 - Missing Referenced Artifacts

Observed missing paths in the current checkout:

- `Docs/refactor/artifacts/phase_09/telemetry_samples/phase09_trace_samples.jsonl`
- `Docs/refactor/artifacts/phase_00/logs/structured_events.jsonl`
- `Docs/refactor/artifacts/phase_00/performance/training_smoke_run/epoch_0001.pt`

Impact:

- Weakens archive completeness.
- Makes automated artifact validation less trustworthy.

Suggested fix:

- Either restore these artifacts, update manifests to mark them superseded or
  intentionally omitted, or extend the artifact validator so references from
  manifests/reconciliation files are checked mechanically.

### R2 - Phase 04 Matrix Status Values

Observed:

- `V2-040` through `V2-046` use `done: ...` instead of declared allowed status
  `complete`.

Impact:

- Low source-code risk, moderate evidence hygiene risk.

Suggested fix:

- Normalize Phase 04 rows to `complete` and keep the exit-gate path in the proof
  column or signoff notes.

### R3 - Performance Evidence Is Not Production-Grade

Observed:

- Phase 09 performance comparison is a synthetic hot-path proxy.
- Several throughput fields are `0.0`.
- Queue/backpressure evidence is partly textual references.
- Some earlier phase performance artifacts are local smoke/mock measurements.

Impact:

- The architecture may be correct, but final throughput/regression claims are
  weaker than the refactor standard asks for.

Suggested fix:

- Run stable-runner scheduled benchmarks for inference batching, MCTS,
  self-play, replay projection, and training.
- Store machine-readable JSON with real workload descriptions, host metadata,
  comparison baselines, and accepted regression decisions.

### R4 - Panic/Public API Inventory Is Too Narrative

Observed:

- Rust panic/assert inventory classifies remaining sites broadly.
- Source still contains production `expect`/`assert` patterns that may be valid
  invariants, but they are not line-by-line classified in the artifact reviewed.

Impact:

- Error-boundary confidence is lower than ideal for a Rust/Python runtime.

Suggested fix:

- Generate a line-level Rust panic/assert/unwrap inventory.
- Classify each site as test-only, impossible invariant, fallible API boundary,
  or required fix.

### R5 - `graph/batch.py` Is Compatibility-Looking

Observed:

- `Python/src/hexorl/graph/batch.py` remains as compatibility exports.
- Runtime source appears to use split modules directly; tests still import
  `hexorl.graph.batch`.

Impact:

- Low current risk, but future code may copy test imports and revive the old
  monolithic graph surface.

Suggested fix:

- Either document it as test/export-only compatibility with an expiry, or move
  tests to split-module imports and delete it if no longer needed.

### R6 - Remote CI Provenance Not Verified Here

Observed:

- Phase artifacts and CI workflow exist.
- This review did not verify remote GitHub run IDs or scheduled final-SHA runs.

Impact:

- Local evidence supports closure, but release-grade CI provenance still needs
  a final external check.

Suggested fix:

- Attach final GitHub Actions run URLs or run IDs to Phase 09 artifact metadata.

## Maintenance Rules Going Forward

### Do Not Reintroduce These Runtime Paths

Do not add runtime code under:

- `Python/src/hexorl/action_contract/`
- `Python/src/hexorl/buffer/`
- `Python/src/hexorl/model/`

Do not add imports of:

- `hexorl.action_contract`
- `hexorl.buffer`
- `hexorl.model`

### Keep Semantic Owners Clear

- Legal/history/tactical/encoding boundary: `hexorl.engine` plus Rust.
- Contract identity/source/hash/version: `hexorl.contracts`.
- Candidate and pair rows: `hexorl.contracts.candidates` and
  `hexorl.contracts.pairs`.
- Graph semantics: `hexorl.graph.semantic_builder`.
- Graph tensor projection/collation: `hexorl.graph.tensorize` and
  `hexorl.graph.collate`.
- Model family capabilities: `hexorl.models`.
- Inference protocol and transport: `hexorl.inference`.
- Policy output mapping: `hexorl.search.policy_provider`.
- Pair scoring/caps: `hexorl.search.pair_strategy`.
- Rust MCTS calls: `hexorl.search.engine_adapter`.
- Game execution: `hexorl.selfplay.game_runner`.
- Worker lifecycle: `hexorl.selfplay.worker`.
- Replay records/storage/sampling/projection: `hexorl.replay`.
- Training target/loss consumption: `hexorl.train`.
- Evaluation players/arena: `hexorl.eval`.
- Dashboard read-only inspection: `hexorl.dashboard.contract_inspector`.
- Typed autotune recipes/runtime sweeps: `hexorl.tuning`.

### Keep CI Policy Active

The architecture policy audit should remain a required gate:

```text
python3 tools/refactor/phase09_policy_audit.py --output <artifact path>
```

It currently checks:

- old runtime paths absent
- banned old imports absent
- architecture-string runtime gates limited to allowlisted files
- direct Rust MCTS calls limited to allowlisted files
- dashboard private reconstruction limited to `contract_inspector.py`
- duplicate FFI byte decoders
- skipped/xfail Phase 09 tests

### Treat Performance As A Living Gate

The refactor made performance observable, but the final proof should continue
to mature. For hot-path work, require:

- HostProfile
- workload description
- throughput
- p50/p95 latency
- queue/backpressure metrics
- GPU utilization or proxy timing
- baseline comparison
- accepted-regression owner and reason

## Review Evidence From This Documentation Pass

Subagents used:

- Phase evidence reviewer:
  - reviewed phases `00` through `09`
  - found coherent closure evidence plus missing/thin/inconsistent artifacts
- Python structure reviewer:
  - reviewed `Python/src/hexorl`, tests, scripts/tools/configs
  - found a cohesive runtime package and no active old runtime imports
- Rust/CI/performance reviewer:
  - reviewed Rust crate layout, Rust API docs, CI, Phase 09 verification
  - found strong API narrowing and boundary discipline, with performance and
    panic-inventory caveats

Local commands run by the primary reviewer included:

```text
git status --short
rg --files Docs/refactor | sort
find Python/src/hexorl -path '*/__pycache__' -prune -o -name '.DS_Store' -prune -o -type f -print | sort
rg -n "hexorl\.(action_contract|buffer|model)\b|from hexorl import (action_contract|buffer|model)\b|import hexorl\.(action_contract|buffer|model)\b" Python/src Python/tests Docs/refactor .github
git ls-files Python/src/hexorl/action_contract Python/src/hexorl/buffer Python/src/hexorl/model
python3 tools/refactor/phase09_policy_audit.py --output /tmp/phase09_policy_audit_current.json
test -f Docs/refactor/artifacts/phase_09/telemetry_samples/phase09_trace_samples.jsonl
test -f Docs/refactor/artifacts/phase_00/logs/structured_events.jsonl
test -f Docs/refactor/artifacts/phase_00/performance/training_smoke_run/epoch_0001.pt
rg -n "done:|partial|deferred|manual_only|implemented_but_unused|shim_remains|unit_only" Docs/refactor/V2_REQUIREMENTS_MATRIX.md
```

Current notable results:

- `git status --short` was clean before documentation edits.
- `git ls-files` shows no tracked files under old runtime packages.
- Current Phase 09 architecture policy audit passes with zero findings after
  local cache-only old package directories were removed.
- Missing artifact checks returned absent for the paths listed in residual risk
  `R1`.
- Matrix search found non-standard Phase 04 `done: ...` row statuses.

## Bottom Line

The refactor was done with serious architectural care. The codebase now has a
clear modular runtime spine, old source packages are gone, Rust/Python
boundaries are explicit, and the main data paths are governed by contracts,
adapters, providers, and inspectors rather than implicit reconstruction.

The remaining work is not "finish the refactor" so much as "harden the proof":
normalize a few phase records, repair missing artifact references, strengthen
production performance evidence, classify Rust panic/assert sites line by line,
and attach final CI provenance.
