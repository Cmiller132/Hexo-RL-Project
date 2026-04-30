# V2 Requirements Matrix

Date: 2026-04-29

Source of truth: `Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_V2_20260429.md`

This matrix is the orchestrator's master signoff surface. A phase cannot close unless every row it owns is complete, consumed by the intended runtime path, tested, observable where relevant, and backed by deletion/import-audit proof.

Status values:

```text
planned
in_progress
blocked
complete
```

Forbidden status at phase close:

```text
partial
deferred
implemented_but_unused
unit_only
shim_remains
manual_only
```

## Matrix

| ID | Requirement | Owner Phase | Primary Modules | Required Proof | Status |
|---|---|---:|---|---|---|
| V2-000 | Baseline freeze has git tag, archive manifest, command transcripts, config hashes, and artifact manifest. | 00 | `Docs/refactor/artifacts/phase_00/` | Manifest, command logs, baseline report, rollback tag. | planned |
| V2-001 | `global_xattn` default pair strategy is `none` and emits zero pair rows unless explicitly opted in. | 00 | config/runtime/search smoke | Test output, trace sample, pair summary log. | planned |
| V2-002 | Accidental pair scoring is guarded before broad refactor work begins. | 00 | self-play/search guard tests | Failing/regression test when pair scoring occurs without strategy. | planned |
| V2-003 | Structured self-play/autotune/no-progress logging samples exist before deeper cutovers. | 00 | telemetry/log artifacts | Heartbeat, phase transition, no-progress, pair summary, autotune lifecycle samples. | planned |
| V2-004 | Legacy fallback, architecture-string, duplicate helper, stale runtime path inventory is complete and mapped to owner phases. | 00 | repo audit artifacts | Inventory report and owner phase mapping. | planned |
| V2-005 | Verification strategy treats the current runtime as untrusted and requires independent/cross-validated oracles, negative fixtures, and mutation checks. | 00 | refactor docs/artifacts | Verification plan, golden fixture inventory, corruption case inventory. | planned |
| V2-006 | Phase 00 captures HostProfile, CPU/GPU utilization where available, inference throughput, self-play phase profile, replay projection throughput, and training step baseline. | 00 | benches, runtime probes, artifacts | JSON benchmark output, host profile, config hashes, command logs. | planned |
| V2-010 | `contracts/` is pure data/value objects and imports no model/inference/search/train/dashboard/tuning orchestration. | 01 | `contracts/` | Import purity test, contract validation tests. | planned |
| V2-011 | `engine/` is the only Python-facing Rust rules boundary and runtime direct `_engine` imports are removed. | 01 | `engine/` | Runtime import audit, engine API tests, direct `_engine` deletion proof. | planned |
| V2-012 | Rust is production source for legal rows; Python legal fallback is fixture-only, and Rust legal rows are semantically validated before contract use. | 01 | `engine/legal.py`, `contracts/legal.py` | Parity tests, fallback guard, source telemetry sample, occupied/duplicate/terminal/current-player negative tests. | planned |
| V2-013 | Compact history has one Python contract and Rust parity, backed by the centralized PyO3 history byte protocol. | 01 | `contracts/history.py`, `engine/history.py`, `crates/hexgame-py/src/protocol.rs` | Encode/decode parity tests, malformed-byte tests, invalid history tests. | planned |
| V2-014 | D6 transforms live in one Python module and parity-test against Rust for coordinates/history/legal/dense tensors. | 01 | `contracts/symmetry.py`, `engine/parity.py` | Rust/Python D6 parity tests. | planned |
| V2-015 | Contract schema/version/source/hash policy is defined and tested for ndarray-backed contracts. | 01 | `contracts/*` | Stable hash tests, schema version tests, source label tests. | planned |
| V2-016 | Engine/legal/history/D6 verification checks semantic identity, row ordering, terminal/current-player state, source/hash/version, Rust invariant hooks, and mutation safety. | 01 | `engine/*`, `contracts/*` | Golden, negative, mutation, invariant-hook, and semantic identity tests. | planned |
| V2-020 | Candidate construction has one semantic authority for rows, identity, validation, and hash, while registered feature/selector extensions cannot bypass validation. | 02 | `contracts/candidates.py` | Golden parity, extension-proof tests, self-play/replay/training/dashboard consumption. | planned |
| V2-021 | Pair action construction has one semantic authority for row identity and phase semantics, while `PairStrategy` owns selection/caps/scoring. | 02 | `contracts/pairs.py`, `search/pair_strategy.py` | Phase-aware pair tests, known-first tests, D6 pair tests, extension-proof tests. | planned |
| V2-022 | `PairCandidateBatch` is deleted or demoted to semantics-free projection from `PairActionTable`. | 02 | pair projection code | Import/deletion audit, projection-only tests. | planned |
| V2-023 | Graph semantic construction is separate from graph tensorization/collation. | 02 | `graph/semantic_builder.py`, `graph/tensorize.py`, `graph/collate.py` | Projection tests, graph schema tests, banned private rebuild audit. | planned |
| V2-024 | Self-play, replay, training, eval, dashboard fixtures consume the same candidate/pair/graph builders. | 02 | cross-package consumers | Golden equality tests and import audits. | planned |
| V2-025 | Candidate, pair, and graph projections are pure, D6-verified, mutation-safe, and corruption-tested against canonical contracts. | 02 | contracts/graph projection code | Projection immutability tests, D6 inverse/composition tests, corruption failures. | planned |
| V2-030 | `models/` registry/spec/capability system is authoritative; runtime does not import `hexorl/model`. | 03 | `models/` | Build tests, import audit, deletion manifest. | planned |
| V2-031 | Every model family registers through facet/descriptor components for model build, train adapter, inference adapter/manifest, policy provider, loss plan, default recipe, and tune space. | 03 | `models/families/*`, `models/registry.py` | Registry matrix tests and fake-family extension proof without trainer/inference/search/dashboard edits. | planned |
| V2-032 | Trainer uses `TrainAdapter`, not model class or architecture checks. | 03 | `train/adapters.py`, `train/trainer.py` | One-batch every family, no branch import/code audit. | planned |
| V2-033 | Pair target training validation covers first, second known-first, joint pair, opening no-pair semantics. | 03 | `train/adapters.py`, losses | Turn/phase/provenance tests. | planned |
| V2-034 | `CheckpointManager` owns save/load/inspect, strict load, manifest, no duplicate cleanup. | 03 | `models/checkpoint.py` | Manifest round-trip, inspect without weights, duplicate cleanup audit. | planned |
| V2-035 | Training debug bundle proves replay -> contracts -> tensors -> targets -> outputs -> loss inputs, with mutation/corruption guards. | 03 | `train/adapters.py`, debug tooling | Single-position bundle, target alignment tests, mutation/corruption tests. | planned |
| V2-040 | `InferenceProtocolManifest` is required before inference request submission, uses a stable base envelope plus request-kind payload schemas, and records Rust FFI protocol identities for Rust-derived rows. | 04 | `inference/protocol.py`, `engine/`, `crates/hexgame-py/src/protocol.rs` | Handshake tests, manifest validation tests, fake request-kind extension proof, FFI protocol source/hash assertions. | planned |
| V2-041 | Inference dispatch uses request kind/protocol, not architecture string. | 04 | `inference/server.py` | Dispatch tests, architecture-string audit. | planned |
| V2-042 | Protocol mismatch fails fast with structured errors and no indefinite IPC wait, including Rust row/hash/protocol mismatches. | 04 | inference client/server/transport | Negative tests for version/kind/schema/caps/heads/timeouts/stale Rust hashes. | planned |
| V2-043 | Transport owns pack/ready/wait/timeout/decode/reset lifecycle with slot request/response sequence counters. | 04 | `inference/shm_transport.py` or documented single transport owner | Submit lifecycle deletion audit, stale-ready/stale-slot integration tests. | planned |
| V2-044 | Every inference response includes protocol/contract/model/count/timing/warning telemetry plus FFI protocol source and row hashes when Rust-derived data is present. | 04 | inference telemetry | Response telemetry assertions. | planned |
| V2-045 | Inference verification catches stale buffers, stale ids, stale slot generations, bad masks, wrong rows/shapes, malformed Rust row identities, non-finite outputs, and post-validation mutation before policy/search consumption. | 04 | inference transport/adapters | Inference debug bundle, mutation/corruption tests, response validation tests. | planned |
| V2-046 | Inference batching/backpressure keeps GPU batching effective under synthetic and self-play load without unbounded waits. | 04 | `inference/batching.py`, inference transport/adapters | Fill rate, queue depth, p50/p95 wait, GPU utilization or proxy timing, timeout/backpressure tests. | planned |
| V2-050 | All self-play model priors flow through `PolicyProvider`. | 05 | `search/policy_provider.py` | Dense/restnet/graph/global provider tests. | planned |
| V2-051 | `EngineAdapter` is the only Python caller of Rust MCTS APIs. | 05 | `search/engine_adapter.py` | Import/code audit, MCTS integration tests. | planned |
| V2-052 | `PairStrategySpec` independently validates root, leaf, full, diagnostic caps. | 05 | `search/pair_strategy.py` | Cap validation and rejection tests. | planned |
| V2-053 | No pair scoring happens from architecture/config/head presence. | 05 | search/self-play/inference | No-implicit-pair tests and import audit. | planned |
| V2-054 | Global graph policy heads have first-class row-mapped contracts and telemetry. | 05 | models/inference/search/train telemetry | Shape, known-first, joint row, MCTS telemetry tests. | planned |
| V2-055 | Policy/search verification proves raw model outputs map to intended legal rows before MCTS and MCTS cannot mutate validated inputs. | 05 | `search/*`, inference adapters | Policy/search debug bundle, stale row/hash tests, mutation guard tests. | planned |
| V2-056 | Python search uses the canonical fallible Rust MCTS API only, preserving root/batch tokens and structured `MCTSError` ownership. | 05 | `search/engine_adapter.py`, `crates/hexgame-py` | Stale root token tests, stale batch token tests, invalid prior tests, import audit proving no panic wrapper or string fallback use. | planned |
| V2-057 | MCTS leaf selection/backprop is batched through `EngineAdapter` and Rust hot paths, with no Python per-node hot loop. | 05 | `search/engine_adapter.py`, Rust MCTS | Split MCTS timings, leaf batch benchmarks, stale-token tests, mutation guards. | planned |
| V2-060 | `GameRunner` owns game execution and composes explicit provider/adapter/pipeline outputs without owning canonical builder internals. | 06 | `selfplay/game_runner.py` | Constructor/API tests, builder-internal import audit, integration smoke. | planned |
| V2-061 | `SelfPlayWorker` is lifecycle/IPC only. | 06 | `selfplay/worker.py` | Import/code audit for game loop, pair chunking, replay assembly, MCTS wiring. | planned |
| V2-062 | Self-play logs are actionable for stalls, slow phases, pair scoring, inference waits, contract mismatches. | 06 | `selfplay/telemetry.py` | Heartbeat, no-progress, game summary, policy timing, pair summary samples. | planned |
| V2-063 | `ContractTrace` includes legal/candidate/pair/token/relation counts and required timing spans. | 06 | contracts/selfplay telemetry | Trace schema tests and sample artifacts. | planned |
| V2-064 | Self-play behavior debug bundle localizes failures across engine, contracts, D6, targets, model outputs, policy mapping, MCTS, and replay. | 06 | `selfplay/telemetry.py`, `selfplay/game_runner.py` | Single-position/game bundle, mutation guard tests, replay identity tests. | planned |
| V2-065 | Self-play process/thread ownership uses HostProfile budgets and propagates backpressure across inference and replay writing. | 06 | `selfplay/orchestrator.py`, `selfplay/game_runner.py`, runtime spec | Worker sweep, CPU/GPU/queue telemetry, no-progress logs, queue high-watermark tests. | planned |
| V2-070 | New self-play writes only new replay records with Rust FFI protocol source, compact history hash, and reconstructed legal-row hash. | 07 | `selfplay/record_writer.py`, `replay/codec.py` | Runtime smoke, write validation tests, Rust replay/invariant proof. | planned |
| V2-071 | Sampler reads only new replay records and training batches flow through `replay/projector.py`; old `buffer` runtime ownership is moved or deleted for sampler/train paths. | 07 | `replay/sampler.py`, `replay/projector.py`, train | Sample-to-loss tests, buffer import audit. | planned |
| V2-072 | Old replay/buffer decode code is absent from Phase 07 runtime imports; dashboard/eval inspection removal is owned by Phase 08. | 07 | replay/train/selfplay/epoch runtime | Banned import audit, deletion manifest. | planned |
| V2-073 | Replay round-trip, corruption handling, projection equality, Rust replay validation, and data-quality tests pass. | 07 | `replay/*`, `engine/` | Test output and artifacts. | planned |
| V2-074 | Replay verification preserves trace-to-record-to-projector semantic identity and rejects mutation/corruption before training without persisting transient MCTS root/batch tokens as replay semantics. | 07 | replay/selfplay/train | Trace-to-record identity tests, D6 replay tests, Rust replay tests, mutation/corruption report. | planned |
| V2-075 | Replay storage, sampler, and projector have bounded queues, vectorized projection, prefetch policy, memory budget reporting, and throughput budgets. | 07 | `replay/*` | Write/read/project samples/sec, corruption tests, queue/backpressure tests, memory profile. | planned |
| V2-080 | Evaluation uses `PolicyProvider` for every registered family. | 08 | `eval/*` | Arena/provider tests, no dense-only audit. | planned |
| V2-081 | Dashboard uses `ContractInspector` as a dispatcher over read-only inspector services, not as a mega-object that reconstructs private semantics. | 08 | `dashboard/*`, inspector services | Route tests, inspector extension proof, banned sampler-private import audit. | planned |
| V2-082 | Dashboard displays contract hash/source/version, trace id, checkpoint manifest, protocol, model family, recipe identity. | 08 | dashboard routes/views | Display assertion tests and screenshots/artifacts where relevant. | planned |
| V2-083 | Autotune uses typed `ModelRecipe` and family spaces; no raw config mutation for family behavior. | 08 | `tuning/*`, scripts | Recipe dry-run tests, raw-config mutation audit. | planned |
| V2-084 | Autotune logs recipe validation, trial lifecycle, scheduler decisions, no-progress watchdogs, and likely subsystem cause. | 08 | tuning telemetry/reporting | Logging sample artifacts and tests. | planned |
| V2-085 | Dashboard and autotune reports can use behavior debug bundles to distinguish model, training-target, engine, D6, policy, MCTS, replay, and runtime failures. | 08 | dashboard/tuning reporting | Debug-bundle route tests, mismatch owner tests, poor-learning report samples. | planned |
| V2-086 | Autotune `RuntimeSpec` searches host-utilization knobs separately from model semantics and scores utilization, throughput, stability, and stalls. | 08 | `tuning/*`, runtime spec | Dry-run, watchdog, scheduler score-component artifacts, runtime sweep tests. | planned |
| V2-090 | Final CI enforces all architecture invariants automatically. | 09 | `.github`, tests, audits | CI jobs and policy checks. | planned |
| V2-091 | Final import graph has no banned runtime modules or compatibility facades. | 09 | whole repo | Import graph report. | planned |
| V2-092 | Final self-play -> replay -> train -> eval -> dashboard smoke is archived. | 09 | whole runtime | Final smoke artifact. | planned |
| V2-093 | V2 docs and refactor docs describe only the new architecture and all matrix rows are complete. | 09 | docs | Final conformance report. | planned |
| V2-094 | Final CI includes behavior-bundle, mutation-safety, and corruption tests so subtle correctness failures are caught automatically. | 09 | CI/tests/artifacts | CI policy checks, verification artifact bundle, final conformance report. | planned |
| V2-095 | Final CI keeps Rust suspicion gates active: malformed FFI bytes, stale MCTS tokens, Rust/Python parity, invariant probes, public API drift, panic/unwrap inventory, and structured engine error checks. | 09 | CI/tests/artifacts | Engine smoke/invariant tests, MCTS stale-token tests, malformed protocol tests, public API diff, panic inventory, debug-bundle sample, clippy/test commands. | planned |
| V2-096 | CI tier contract exists and every required check maps to a tier, owner, timeout, runner requirement, artifact path, and promotion rule. | 09 | `.github`, `Docs/refactor/CI_STRATEGY.md`, artifacts | CI policy file, generated check inventory, final SHA CI evidence. | planned |
| V2-097 | Artifact retention and phase evidence manifests are machine-validated. | 09 | CI/tests/artifacts | Manifest validator output, run ids, artifact paths, supersession records. | planned |
| V2-098 | Flaky/quarantine policy is enforced with owner, issue, expiry, affected V2 rows, and deterministic replacement coverage for PR gates. | 09 | CI/tests/artifacts | Quarantine report, replacement-test proof, scheduled continued execution evidence. | planned |
| V2-099 | Performance budgets have HostProfile/runner metadata, JSON comparison tooling, scheduled artifacts, and threshold ownership. | 09 | CI, benches, `Docs/refactor/PERFORMANCE_STRATEGY.md` | Benchmark JSON, comparison output, scheduled run artifacts, accepted-regression records. | planned |
| V2-100 | Public contracts, adapters, protocol objects, debug bundles, and extension points have executable examples and docs. | 09 | docs/tests/examples | Example tests, docs audit, contract construction/failure examples, final conformance report. | planned |

## Signoff Notes

Each phase must update this matrix in its exit-gate report. The matrix should remain conservative: a row is complete only when the implementation is wired into the intended runtime path and its old path is deleted or quarantined outside runtime.
