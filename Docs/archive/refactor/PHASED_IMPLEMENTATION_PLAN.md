# Modular Refactor - Master Implementation Plan

Date: 2026-04-29

Source of truth: `Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_V2_20260429.md`

Execution guardrails: `Docs/refactor/EXECUTION_QUALITY_GUARDRAILS.md`

## Purpose

Translate the V2 redesign into a strict, test-gated, breaking-refactor execution program.

This program is not a compatibility migration. It is a controlled cutover to a cohesive architecture. A phase is not complete because a new path exists; it is complete only when the old runtime path it replaces is deleted, quarantined outside runtime, or proven unreachable by import/code-search gates.

The Rust refactor has already completed its Phase 2 hardening slice before this Python/project execution plan begins. Treat that Rust work as the current baseline and intended production rules boundary, not as proof that the boundary is fully trusted. Phase 00 must capture the post-Rust baseline and every later phase must keep Rust outputs under semantic validation, structured error handling, and replayable debug evidence.

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

Phase docs define outcomes, invariants, forbidden paths, artifacts, and gates. They should not be treated as rigid internal recipes unless a name or field is explicitly part of a public contract or protocol. If a cleaner implementation structure is discovered, the orchestrator must update the affected phase doc and matrix row before implementation, then hold the implementation to the same runtime cutover, deletion, observability, test, and performance proof.

## Agent Execution Contract

Every phase assignment should be written in a compact outcome-first form:

```text
Goal
Success criteria
Constraints
Required evidence
Stop rules
```

Implementers may choose the internal design that best satisfies the frozen interfaces and matrix rows. They may not claim completion until required evidence is attached.

Stop and escalate before coding if:

- a required invariant conflicts with current docs or code
- a requested deletion would remove the only runtime path before its replacement is consumed
- a phase-closing test would need to be skipped, xfailed, or made manual-only
- performance evidence cannot be produced for a hot-path change
- the implementation requires adding a compatibility path not already approved as non-runtime migration tooling

## Promotion Rule

A phase may advance only when all are true:

- mandatory unit, integration, parity, policy, deletion, telemetry, and performance checks pass
- correctness checks do not assume the existing runtime is trusted; every phase-owned boundary has an independent or cross-validated oracle
- mutation-safety checks prove contracts, views, tensor projections, D6 transforms, replay records, and model targets cannot be silently changed after hashing/validation
- agent completion is backed by inspectable evidence packets rather than narrative claims
- contract examples and docs exist for new public contracts, adapters, protocol objects, debug bundles, and extension points
- hot-path changes include utilization evidence and do not silently reduce batching, parallelism, or GPU/CPU throughput
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

The completed Rust hardening slice improves the engine boundary but does not remove this suspicion requirement. Later Python phases should treat Rust as the canonical production rules implementation and still validate every Rust-derived payload before it influences training or search. In practice this means:

- Do not reintroduce Python legal/history/D6 fallbacks to compensate for Rust uncertainty.
- Do wrap Rust outputs in Python contracts with schema, source, hash, row identity, and mutation checks.
- Do keep stale root tokens, stale batch tokens, malformed FFI bytes, non-finite priors, invalid row lengths, and illegal move submissions as first-class negative tests.
- Do add logging and debug bundles that identify whether a failure belongs to Rust replay/legal/tactics/MCTS, Python contract validation, inference decode, policy mapping, replay projection, or training target construction.
- Do use Rust invariant hooks in debug, test, and probe paths where they narrow fault localization, while keeping hot paths performance-aware.

Every phase that introduces or cuts over a data boundary must include:

- golden position fixtures with known histories, legal rows, terminal status, D6 variants, targets, and expected failure cases
- independent or cross-implementation checks, such as Rust replay against Python contract validation, inverse D6 transforms, replay round-trips, and model-output-to-legal-row reconstruction
- identity checks for row ordering, row ids, hashes, schema versions, source labels, and trace ids
- immutability or mutation-detection tests for cached views, NumPy/Torch tensors, replay records, model targets, graph tensors, and transport buffers
- negative tests that deliberately corrupt histories, rows, masks, hashes, targets, tensor shapes, protocol versions, and pair semantics
- single-position debug bundles that can localize whether a failure belongs to engine, contracts, D6, candidates, pairs, graph tensorization, training targets, inference, policy mapping, MCTS, self-play, replay, dashboard, or autotune

Shape-only tests, smoke-only tests, and tests that merely compare old output to new output are not enough.

## Modularity Philosophy

The refactor centralizes semantic ownership without centralizing every operation. A concept should have one canonical authority and many cheap projections. Contracts should remain small, typed, immutable or mutation-guarded, and free of process orchestration, inference, search, dashboard rendering, replay storage, and training loops.

Future expansion should happen through capability registries, adapters, projections, contract versions, and request-kind payloads. It should not happen through subsystem-private reconstruction, hidden fallbacks, or adding a second semantic implementation for legal rows, histories, candidates, pairs, graph meaning, policy targets, replay records, or MCTS lifecycle state.

Every centralized owner must have an extension-proof test. The test should prove that a new model family, request kind, graph token family, candidate feature block, pair selector, inspector, or projection can be registered through the intended interface without editing unrelated runtime internals.

## Performance Philosophy

Correctness checks must be designed around runtime cost:

- construction, decode, replay, test, and debug/probe paths may run full validation
- hot paths should rely on cheap identity checks, generations, hashes, row ids, schema ids, shape/count checks, finite checks, and immutable views
- inference must preserve GPU batching through explicit request queues, adapter-owned collation, and bounded backpressure
- CPU-heavy Rust/search/replay work should remain parallelizable across workers, games, or batches
- debug bundles must be available on demand, but not allocated during every search leaf or inference request unless requested

Every phase that touches inference, self-play, MCTS, replay projection, or training must record host profile, throughput, latency, queue/backpressure behavior, and regression notes in its artifacts.

## CI Philosophy

CI is tiered. Fast PR gates enforce architecture and focused correctness; deep scheduled gates enforce expensive oracle, fuzz/property, benchmark, GPU batching, and long-run behavior. Final V2 closure requires current passing evidence from every required tier, even if some checks are not run on every PR.

No required final gate may remain manual-only. Flaky required checks must be fixed, given an owner and expiry, or removed from closure claims until they are reliable.

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
performance/
contract_examples/
agent_completion_packet.md
evidence_reconciliation.md
exit_gate_report.md
```

`MANIFEST.md` must identify the git SHA, command lines, relevant config hashes, generated files, owners, and any intentional non-runtime migration tooling.

Phase artifacts must be retained long enough to diagnose regressions across later phases. Deep oracle runs, performance baselines, malformed-input fixtures, telemetry samples, and behavior debug bundles must not be overwritten by later runs without a manifest entry that records the superseding artifact, runner profile, and git SHA.

## CI Evidence Model

Every phase-owned required check must be classified as one of:

```text
local
pr_required
deep
scheduled
final
artifact_only
```

The phase artifact must record the check owner, timeout, command, artifact path, runner requirements, and promotion rule. A phase-closing invariant may not be `artifact_only` unless it is documentation or generated evidence that is separately validated by CI.

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
