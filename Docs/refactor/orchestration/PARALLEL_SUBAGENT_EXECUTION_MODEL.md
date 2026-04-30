# Parallel Subagent Execution Model

Source of truth: `Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_V2_20260429.md`

## Roles

- **Orchestrator Agent:** owns phase scope, interface freeze, merge sequencing, conformance review, artifact audit, and V2 requirement matrix closure.
- **Subagent 1 - Contracts/Engine/Schema:** contracts, engine boundary, parity schema, validation, source/version/hash policy.
- **Subagent 2 - Runtime/Self-Play/Inference:** worker, game runner, IPC, inference transport, runtime wiring, observability, Rust FFI/MCTS failure localization.
- **Subagent 3 - Models/Search/Tuning:** model registry, families, checkpoints, policy providers, pair strategies, recipe spaces.
- **Subagent 4 - Replay/Training/Eval/Dashboard:** replay storage/projection, train adapters, eval players, dashboard inspectors.
- **Subagent 5 - Quality/CI/Docs:** tests, fixtures, import audits, deletion manifests, telemetry artifacts, CI jobs, docs.

## Core Rule

Parallelism must never create parallel semantics.

Only one subagent owns each public contract or interface. Other subagents consume it. If a shared interface is unclear, work stops until the orchestrator freezes that interface.

## Phase Mini-Gates

Every phase runs these gates in order:

1. **Scope Freeze:** Orchestrator names exact V2 requirement matrix rows and anti-goals.
2. **Agent Execution Contract:** Each assignment states Goal, Success Criteria, Constraints, Required Evidence, and Stop Rules.
3. **Interface Freeze:** Public contracts, adapters, manifests, builders, or service APIs are reviewed before implementation.
4. **Fixture Freeze:** Golden fixtures, baseline artifacts, and expected trace/log samples are named.
5. **CI Routing:** S5 proposes CI tier, timeout, owner, and artifact path for every required check; the orchestrator approves before implementation.
6. **Parallel Implementation:** Subagents work only in non-overlapping write scopes.
7. **Integration Branch:** Orchestrator integrates and resolves cross-boundary assumptions.
8. **Deletion Sweep:** Phase-owned old paths are deleted or quarantined outside runtime.
9. **Audit And Test:** Unit, integration, parity, performance, telemetry, and import checks run.
10. **Adversarial Review:** A reviewer tries to prove an old path, stale input, malformed payload, silent fallback, or partial implementation can still affect runtime.
11. **Conformance Review:** Orchestrator updates matrix rows and signs off or rejects the phase.

## Strict Gate Protocol

A phase fails if any condition is unmet:

- a replacement exists but is not consumed by runtime
- old runtime path remains reachable
- compatibility facade remains under `Python/src/hexorl/`
- architecture string behavior gates remain outside registry/spec tests
- private legal/history/D6/candidate/pair/graph reconstruction remains in production
- pair scoring can happen without explicit `PairStrategy`
- inference protocol mismatch can hang
- checkpoint loading has silent partial behavior
- dashboard, trainer, eval, or autotune reconstructs private approximations
- required telemetry/log samples are missing
- tests do not cover the invariant that changed
- Rust-derived data is accepted without Python contract validation, stale-token checks, or structured error ownership
- docs do not match shipped behavior
- required checks are unclassified, manual-only, flaky without replacement coverage, or missing artifacts
- performance-sensitive changes lack host/utilization evidence
- new centralized owners lack extension-proof tests or executable examples

## Subagent Completion Packet

Each subagent final report for a phase must include:

```text
closed V2 rows
runtime consumers changed
files changed
legacy paths deleted or quarantined
tests and commands run with exit status
artifacts produced
performance/utilization evidence for hot paths
contract examples/docs added where relevant
known blockers, if any
explicit statement that no skipped/deferred/manual-only requirement is being claimed complete
```

The orchestrator reconciles all completion packets into:

```text
Docs/refactor/artifacts/phase_XX/agent_completion_packet.md
Docs/refactor/artifacts/phase_XX/evidence_reconciliation.md
```

Narrative status without command output, artifact paths, and deletion/import proof is not enough to close a matrix row.

## Phase-To-Subagent Split

### Phase 00 - Baseline Freeze, Guardrails, And Evidence
- S1: implicit data-shape and contract-risk inventory.
- S2: runtime baseline probes, pair guard smoke, self-play/no-progress logging samples.
- S3: architecture-string and pair-policy dependency inventory.
- S4: replay/training/eval baseline artifacts and config hashes.
- S5: CI timing, artifact templates, trace schema samples, exit-gate report.

### Phase 01 - Engine + Contracts Foundation
- S1: contracts, schema/version/hash/source policy, validation.
- S2: `engine/` Rust boundary, direct `_engine` import removal, FFI protocol validation, and production fallback removal.
- S3: model/search consumers updated to contract/engine outputs where phase-owned.
- S4: replay/training/eval/dashboard fixture adapters for contract parity.
- S5: Rust/Python parity tests, import audits, contract docs.

### Phase 02 - Candidate/Pair/Graph Builder Convergence
- S1: `CandidateContractBuilder` and diagnostics.
- S2: `PairActionTableBuilder`, phase semantics, known-first handling, caps.
- S3: `GraphSemanticBuilder` and graph relation/schema extraction.
- S4: `GraphTensorizer`/collator and training/eval/dashboard projections.
- S5: golden parity tests and banned private-builder import audits.

### Phase 03 - Model Registry, TrainAdapter, And CheckpointManager
- S1: `ModelSpec`, capabilities, family validation.
- S2: runtime factory cutover to `models/` and old `model/` import removal.
- S3: family implementations, policy provider declarations, recipe/tune spaces.
- S4: `TrainAdapter`, loss wiring, one-batch family tests.
- S5: checkpoint manifest tests, registry docs, deletion manifest.

### Phase 04 - Inference Protocol And Adapters
- S1: `InferenceProtocolManifest`, request/response schemas, Rust FFI protocol identities.
- S2: transport lifecycle, slot generation counters, handshake, timeout/fail-fast behavior.
- S3: dense/sparse/global/pair adapters and capability mapping.
- S4: training/eval/dashboard interoperability expectations for inference outputs.
- S5: protocol mismatch tests, response telemetry tests, latency/throughput artifacts.

### Phase 05 - PolicyProvider, PairStrategy, EngineAdapter
- S1: search context/evaluation contracts and pair strategy schema.
- S2: runtime integration, tokenized Rust MCTS lifecycle, structured `MCTSError` mapping, and direct Rust MCTS call removal.
- S3: policy providers, pair strategies, global graph pair-head consumption gates.
- S4: replay/training pair metadata validation hooks.
- S5: no-implicit-pair tests, cap tests, telemetry assertions, import audits.

### Phase 06 - GameRunner And SelfPlayWorker Cleanup
- S1: self-play handoff contracts and trace payloads.
- S2: `GameRunner`, worker lifecycle split, orchestrator wiring.
- S3: search/provider/engine adapter integration.
- S4: record writer and replay handoff consistency.
- S5: deterministic game tests, heartbeat/no-progress/game-summary log assertions.

### Phase 07 - Replay Cutover
- S1: replay record schema and codec validation.
- S2: runtime record writing and old runtime import removal.
- S3: model/train adapter projection requirements.
- S4: sampler/projector/sample-to-loss migration.
- S5: round-trip, corruption, projection, import-audit, and data-quality tests.

### Phase 08 - Evaluation, Dashboard, And Autotune
- S1: inspector/recipe metadata contracts.
- S2: dashboard services, read-only route adapters, runtime sizing branch removal.
- S3: autotune recipes, family spaces, scheduler/runtime sweep/scoring/reporting.
- S4: eval players, arena, scorecard, dashboard parity views.
- S5: dashboard build/tests, recipe dry-run, scheduler/no-progress logging tests.

### Phase 09 - Final Deletion And CI Enforcement
- S1: requirement matrix closure and schema/alias deletion proof.
- S2: runtime import graph, Rust API/protocol/MCTS suspicion gates, and engine/self-play/inference/search deletion proof.
- S3: model/checkpoint/train/eval/tuning deletion proof.
- S4: replay/dashboard/final smoke verification.
- S5: CI policy jobs, final conformance report, documentation cleanup.

## Orchestrator Review Checklist

- All phase acceptance tests pass with artifacts attached.
- All owned V2 matrix rows are closed.
- All phase-owned legacy imports are removed or test-only.
- Deletion manifest and banned-import checks agree.
- Contract version/source/hash fields are asserted in tests.
- Required telemetry/log samples exist and are actionable.
- Rust-facing artifacts include malformed FFI, stale MCTS token, invariant-probe, panic/unwrap inventory, and public API drift evidence.
- No unresolved TODO/FIXME remains in changed runtime code for phase-owned work.
- Rollback tag is created and recovery smoke is documented.
