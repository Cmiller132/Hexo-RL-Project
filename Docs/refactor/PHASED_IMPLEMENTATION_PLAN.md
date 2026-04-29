# Modular Refactor - Master Implementation Plan

Date: 2026-04-29

Source of truth: `Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_V2_20260429.md`

## Purpose

Translate the V2 redesign into a strict, test-gated, breaking-refactor execution program.

This program is not a compatibility migration. It is a controlled cutover to a cohesive architecture. A phase is not complete because a new path exists; it is complete only when the old runtime path it replaces is deleted, quarantined outside runtime, or proven unreachable by import/code-search gates.

## Phase Sequence

The program executes in strict order:

0. `phases/PHASE_00.md` - Baseline Freeze, Guardrails, And Evidence
1. `phases/PHASE_01.md` - Engine + Contracts Foundation
2. `phases/PHASE_02.md` - Candidate/Pair/Graph Builder Convergence
3. `phases/PHASE_03.md` - Model Registry, TrainAdapter, And CheckpointManager
4. `phases/PHASE_04.md` - Inference Protocol And Adapters
5. `phases/PHASE_05.md` - PolicyProvider, PairStrategy, EngineAdapter
6. `phases/PHASE_06.md` - GameRunner And SelfPlayWorker Cleanup
7. `phases/PHASE_07.md` - Replay Cutover
8. `phases/PHASE_08.md` - Evaluation, Dashboard, And Autotune
9. `phases/PHASE_09.md` - Final Deletion And CI Enforcement

No phase may defer its own core requirements to a later phase. Later phases can build on earlier phases, but they cannot be used as a hiding place for partial implementation, legacy runtime paths, or missing tests.

## Execution Model

Execution may use five subagents plus one orchestrator, but parallelism is subordinate to interface discipline.

Each phase must run through these mini-gates:

```text
1. Interface freeze
2. Fixture/artifact freeze
3. Implementation work on non-overlapping ownership slices
4. Integration branch
5. Deletion/import-audit sweep
6. Test/CI/telemetry evidence capture
7. Orchestrator conformance review
```

Shared contracts and public interfaces must be frozen before parallel implementation begins. A subagent may not invent a second shape for data that already has a contract owner.

## Promotion Rule

A phase may advance only when all are true:

- mandatory unit, integration, parity, policy, deletion, telemetry, and performance checks pass
- correctness checks do not assume the existing runtime is trusted; every phase-owned boundary has an independent or cross-validated oracle
- mutation-safety checks prove contracts, views, tensor projections, D6 transforms, replay records, and model targets cannot be silently changed after hashing/validation
- V2 requirement matrix rows owned by the phase are complete
- no implementation remains "present but unused"
- no runtime compatibility shim remains without an explicit non-runtime quarantine and expiry
- all phase-owned legacy imports are removed or test-only
- structured logs/traces required by the phase have sample artifacts
- CI commands and local command transcripts are attached
- rollback point is tagged and recovery smoke is documented
- orchestrator signs off with evidence

## Strictness Policy

The orchestrator must reject signoff if any implementation is feature-incomplete, partially wired, spec-divergent, or only unit-tested when V2 requires runtime consumption.

The following are phase blockers:

- a new module exists but no runtime consumer uses it
- a replacement path exists but the old path remains reachable
- a compatibility shim remains in `Python/src/hexorl/` after its owner phase
- pair scoring can happen without `PairStrategy`
- dashboard/training/eval reconstruct data that should come from contracts
- inference protocol mismatch can hang instead of fail fast
- autotune can mutate raw config fields instead of typed recipes
- logs cannot explain where a self-play/autotune stall occurred
- verification relies on the old implementation as the only oracle
- a contract or tensor can be mutated after validation without changing its hash, version, or trace identity
- a Rust/Python parity test checks only shape/count and not row identity, ordering, legality, and semantic meaning

## Verification Philosophy

The current project structure is not a trusted baseline. The refactor must verify behavior as if subtle bugs already exist in the engine boundary, encoders, D6 transforms, target builders, tensor projections, inference mapping, MCTS integration, and replay storage.

Every phase that introduces or cuts over a data boundary must include:

- golden position fixtures with known histories, legal rows, terminal status, D6 variants, targets, and expected failure cases
- independent or cross-implementation checks, such as Rust replay against Python contract validation, inverse D6 transforms, replay round-trips, and model-output-to-legal-row reconstruction
- identity checks for row ordering, row ids, hashes, schema versions, source labels, and trace ids
- immutability or mutation-detection tests for cached views, NumPy/Torch tensors, replay records, model targets, graph tensors, and transport buffers
- negative tests that deliberately corrupt histories, rows, masks, hashes, targets, tensor shapes, protocol versions, and pair semantics
- single-position debug bundles that can localize whether a failure belongs to engine, contracts, D6, candidates, pairs, graph tensorization, training targets, inference, policy mapping, MCTS, self-play, replay, dashboard, or autotune

Shape-only tests, smoke-only tests, and tests that merely compare old output to new output are not enough.

## Required Program Artifacts

Each phase writes artifacts under:

```text
Docs/refactor/artifacts/phase_XX/
```

Each phase must include:

```text
MANIFEST.md
commands/
test_output/
import_audits/
deletion_manifest/
telemetry_samples/
fixtures_or_references/
exit_gate_report.md
```

`MANIFEST.md` must identify the git SHA, command lines, relevant config hashes, generated files, owners, and any intentional non-runtime migration tooling.

## V2 Requirement Matrix

`Docs/refactor/V2_REQUIREMENTS_MATRIX.md` is the orchestrator's master gate. Every V2 invariant must map to:

```text
requirement id
source V2 section
owner phase
owner module/package
implementation proof
test proof
CI proof
deletion/import proof
telemetry/debug proof
signoff owner
status
```

No phase can close with an owned matrix row marked partial, deferred, or implemented-but-unconsumed.
