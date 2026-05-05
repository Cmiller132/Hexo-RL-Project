# Execution Quality Guardrails

Date: 2026-04-29

This document refines the V2 refactor plan around four program risks:

- agents may skip steps, claim completion without proof, or defer hard work
- centralized contracts may become rigid mega-objects instead of useful boundaries
- robustness work may accidentally reduce host utilization or GPU throughput
- CI may become either too shallow to enforce the plan or too slow to use

These guardrails are part of the refactor source of truth. Phase docs describe required outcomes, proof, and deletion gates. They are not intended to freeze every internal implementation detail unless a field, module, method, or protocol is explicitly named as a public contract.

## Agent-Proof Completion

No phase may be signed off from narrative status alone. Every closed requirement needs an evidence packet that another agent, reviewer, or CI job can inspect without trusting the implementer.

Each phase evidence packet must include:

- the V2 rows closed by the phase
- the exact runtime path that consumes the new implementation
- tests proving the old path is unreachable or quarantined outside runtime
- command transcripts with exit codes
- import/code-search audit output
- deletion manifest for replaced files, functions, aliases, shims, and fallbacks
- negative tests for the boundary being changed
- telemetry or debug-bundle samples for failures the phase is meant to localize
- performance smoke or benchmark deltas for hot paths touched by the phase
- contract examples and docs for new public contracts or adapters

Agent claims do not count as proof. A row is complete only when the proof is machine-checkable, artifact-backed, or independently reviewable from committed files.

## Review Model

Every phase has two review passes:

- **Conformance review:** checks the requirement matrix, deletion/import gates, tests, artifacts, and docs.
- **Adversarial review:** tries to find a way the old behavior, stale data, malformed data, silent fallback, or partial implementation can still influence runtime.

The adversarial review must include at least one deliberately corrupted or stale input for each new boundary. For Rust-facing work, this includes malformed FFI bytes, stale MCTS tokens, invalid policy lengths, non-finite priors, illegal move rows, and invariant-probe failures where applicable.

If a phase changes a hot path, conformance review must also check performance evidence. Robustness checks may be tiered, but they cannot disappear from the plan.

## Centralization Without Mega-Objects

The architecture should centralize semantic ownership, not every operation.

Preferred middle ground:

- one canonical semantic owner per concept
- many cheap projections from that owner
- narrow service interfaces for runtime consumers
- immutable or mutation-guarded contract payloads
- explicit capability registries for extension
- adapters that translate between subsystem needs and canonical contracts
- no subsystem-private reconstruction of canonical facts

Avoid:

- one giant context object that accumulates unrelated state
- contracts that perform search, inference, training, dashboard rendering, process management, or replay storage
- builders that own both semantic validation and tensor layout policy when those can be separated
- convenience facades that hide stale data, source identity, or fallback behavior

Extension should happen by adding a new capability, adapter, projection, or contract version. It should not happen by bypassing the canonical owner or adding a second semantic implementation.

## Contract Examples And Documentation

Every public contract introduced by the refactor must include executable examples or tests that show:

- construction from the canonical source
- validation failure on malformed input
- stable schema/source/hash identity
- mutation safety for cached, NumPy, Torch, or zero-copy views
- projection into at least one real runtime consumer
- a small debug payload that explains a failure

The examples should be short and practical. They are meant to teach future implementers the correct path so that strict architecture does not become a barrier to use.

Contract docs must explain:

- who owns the semantic truth
- which fields are identity-bearing
- which fields are derived projections
- when full validation runs
- what the hot-path validation budget is
- what error class is raised when the contract rejects data

## Performance And Host Utilization

The refactor must improve correctness without turning the system into a single-threaded validation pipeline.

Performance principles:

- Keep full semantic validation at construction, decode, replay, debug, and test boundaries.
- Keep hot-path checks cheap: hashes, generations, lengths, schema ids, row ids, finite checks, and immutable views.
- Batch GPU work aggressively through explicit inference queues and adapter-owned collation.
- Use bounded backpressure instead of unbounded waits.
- Keep CPU-heavy Rust/search/replay work parallelizable across games, workers, or batches.
- Avoid per-leaf Python object churn in MCTS and inference loops.
- Do not allocate debug bundles on hot paths unless debug/probe mode requests them.
- Measure throughput and latency before and after every phase that touches inference, self-play, MCTS, replay projection, or training.

Required utilization evidence for hot-path phases:

- host profile: CPU cores, GPU model when present, memory, OS, Python, Rust, PyTorch/CUDA versions
- batch size, queue depth, active worker count, GPU utilization or proxy timing, CPU utilization, and wait time
- p50/p95 latency for inference request, MCTS selection/backprop, replay projection, train step, and self-play move loop where relevant
- throughput counters for positions/sec, games/sec, samples/sec, and train batches/sec where relevant
- failure evidence showing backpressure, timeout, and cancellation paths fail loudly

Performance regressions are allowed only when the phase explicitly records the tradeoff, the owning V2 rows still close, and a follow-up row is added before signoff. Silent regressions fail the phase.

## CI Strategy

CI must be tiered, not weak.

Recommended tiers:

- **Local developer gate:** fast unit tests, fmt/lint, focused import audits, and changed-area policy checks.
- **PR required gate:** workspace Rust tests, focused Python tests for touched areas, architecture/import policy checks, smoke self-play/inference/replay tests, and fast malformed-input checks.
- **Merge/deep gate:** broader Python suite, dashboard build, Rust release tests, Maturin rebuild, behavior debug-bundle tests, corruption/mutation suites, and end-to-end smoke.
- **Scheduled deep gate:** slow oracle tests, fuzz/property tests, benchmark comparison, GPU batching/utilization runs, long self-play smoke, replay data-quality checks, and dashboard/autotune report checks.
- **Release/final V2 gate:** every matrix row closed, all deletion gates green, public API drift checked, performance baselines recorded, and final smoke archived.

No test may be hidden behind a manual-only process at final V2 closure. Expensive checks may move to scheduled/deep tiers during development, but Phase 09 cannot close until their latest passing artifacts are attached and the CI policy explains how they continue to run.

Flaky tests must be fixed, quarantined with an owner and expiry, or moved to a diagnostic suite that does not count for closure. A flaky required gate is not a passing gate.

## Phase Authoring Style

Phase docs should emphasize:

- desired runtime ownership
- invariants
- forbidden legacy paths
- observable behavior
- required tests and artifacts
- performance/CI evidence
- deletion gates

Phase docs should avoid locking in unnecessary internals. Names in phase docs have these meanings:

- **Public contract:** exact name and semantics are required.
- **Primary ownership area:** the named module/package is the intended owner, but internal file layout may change if the matrix and docs are updated before implementation.
- **Illustrative shape:** pseudo-code or sketches explain intent and may be replaced by a cleaner implementation that proves the same outcomes.

When implementers find a better structure, they should update the affected phase doc and matrix row before coding. The update must preserve the single semantic owner, deletion gates, observability, tests, performance evidence, and no-legacy-support rule.
