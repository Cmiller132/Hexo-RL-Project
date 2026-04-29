# Parallel Subagent Execution Model (5 Subagents + 1 Orchestrator)

## Roles

- **Orchestrator Agent (primary):** owns phase gating, review rigor, merge sequencing, and spec compliance.
- **Subagent-1 (Contracts/Schema):** contracts, typed dataclasses, validators, schema/version policy.
- **Subagent-2 (Engine/Runtime):** Rust boundary, runtime wiring, self-play/inference integration.
- **Subagent-3 (Models/Search):** model families, registry, search providers, pair strategy.
- **Subagent-4 (Data/Training/Eval):** replay pipeline, sampler/projector, train/eval adapters.
- **Subagent-5 (Quality/Observability/Docs):** tests, parity harnesses, telemetry, dashboards, docs.

## Coordination Rules

1. Orchestrator defines phase scope, acceptance criteria, and anti-goals.
2. Subagents work in parallel only on non-overlapping ownership slices.
3. Shared interfaces are contract-first and must be approved before implementation merges.
4. Every merged slice includes tests and telemetry updates; no deferred validation.
5. Orchestrator blocks phase close until strict checklist and artifact audit are complete.

## Strict Gate Protocol (Required)

A phase fails if any condition is unmet:

- incomplete feature paths (partial adapterization, hidden fallbacks, dead branches)
- missing tests for changed invariants
- parity mismatches not explicitly approved with incident note
- CI instability or flaky tests unaddressed
- docs not matching shipped behavior

## Phase-to-Subagent Split

### Phase 00
- S1: baseline schema inventory and contract risk map
- S2: runtime baseline probes and commands
- S3: architecture-string dependency audit
- S4: replay/training baseline metric capture
- S5: CI baseline timing + artifact template

### Phase 01
- S1 leads contract packages and validators
- S2 adapts runtime call sites to contract constructors (no cutover)
- S3 validates model input dependency constraints
- S4 adds sampler/trainer contract fixtures
- S5 authors contract test suites and docs

### Phase 02
- S1 defines legal/history parity contract outputs
- S2 implements `engine/` Rust boundary and disables production fallbacks
- S3 updates search/model callsites for engine-origin legal rows
- S4 updates replay decode/parity fixture flows
- S5 runs golden corpus parity and publishes mismatch report

### Phase 03
- S1 finalizes `ModelSpec`/capability schemas
- S2 wires runtime model creation through registry
- S3 implements family registry/adapters/checkpoint map
- S4 updates train/eval adapter consumption paths
- S5 runs capability gating tests and docs

### Phase 04
- S1 versioned inference protocol contracts
- S2 transport/batching/shm integration
- S3 inference adapters per family (dense/sparse/global/pair)
- S4 training compatibility for protocol changes
- S5 throughput/latency bench + protocol compatibility tests

### Phase 05
- S1 pair strategy schema and policy invariants
- S2 runtime integration with explicit strategy selection
- S3 pair scoring strategies and capped enumeration logic
- S4 replay/training handling for pair metadata
- S5 no-implicit-pair regression tests and telemetry assertions

### Phase 06
- S1 selfplay contract boundaries
- S2 split worker into game_runner + orchestration components
- S3 search hooks and policy-provider boundaries
- S4 record writing and downstream compatibility
- S5 deterministic seeded regressions and completeness checks

### Phase 07
- S1 replay contract/schema finalization
- S2 runtime writes canonical replay records
- S3 model/training adapters for unified projector
- S4 full sample->batch->loss migration and parity
- S5 data-quality checks and drift alarms

### Phase 08
- S1 dashboard inspection contracts
- S2 service endpoints backed by canonical contracts
- S3 model/graph inspectors alignment
- S4 eval/debug views consistency with replay/training
- S5 dashboard fixture tests and visual diff checks

### Phase 09
- S1 remove deprecated schemas and aliases
- S2 remove legacy runtime paths/imports
- S3 remove legacy model/search utilities
- S4 remove legacy replay/buffer pathways
- S5 CI policy hardening + final conformance report

## Review Checklist Used by Orchestrator

- All phase acceptance tests pass with artifacts attached.
- No unresolved TODO/FIXME in changed runtime code.
- All legacy imports targeted for the phase are removed.
- Contract version/source fields present and asserted in tests.
- Rollback tag created and recovery smoke tested.
