# Refactor Phase Checklist

Use this checklist during execution. A phase is complete only if every box is checked and the evidence is attached under `Docs/refactor/artifacts/phase_XX/`.

## Universal Checklist

- [ ] Phase scope is frozen and matches the V2 requirement matrix.
- [ ] Entry criteria from the previous phase are satisfied.
- [ ] Public contracts/interfaces for this phase are frozen before implementation work.
- [ ] Agent assignments use the Goal, Success Criteria, Constraints, Required Evidence, and Stop Rules format.
- [ ] Golden fixtures or baseline artifacts needed by this phase are identified.
- [ ] Golden fixtures include negative/corrupt cases, D6 variants, and expected semantic identities, not just shapes.
- [ ] New public contracts, adapters, protocol objects, and debug bundles have executable examples or tests showing correct construction and failure behavior.
- [ ] Centralized owners have extension-proof tests proving new capabilities can be registered without bypassing canonical validation or editing unrelated runtime internals.
- [ ] Unit tests for new behavior are added.
- [ ] Integration tests prove runtime consumers use the new path.
- [ ] Parity tests use frozen fixtures or migration tools only; old runtime code is not kept as an oracle.
- [ ] Cross-validation tests prove row identity, ordering, source, schema version, and hash stability across subsystem boundaries.
- [ ] Mutation-safety tests prove cached views, tensors, replay payloads, and target projections cannot silently change after validation.
- [ ] Corruption tests prove bad histories, illegal rows, stale hashes, bad masks, wrong D6 transforms, malformed targets, and protocol mismatches fail loudly.
- [ ] Rust-derived payloads are treated as canonical but not self-validating: legal rows, compact history, tactics, MCTS tokens, pair rows, and FFI bytes have semantic validation plus negative tests.
- [ ] Structured errors and logs identify whether a boundary failure came from Rust replay/legal/tactics/MCTS, PyO3 protocol decode, Python contract validation, inference decode, policy mapping, replay projection, or training target assembly.
- [ ] A single-position debug bundle or equivalent trace can localize failures to the owning subsystem.
- [ ] Performance smoke is compared to Phase 00 baseline where relevant, including host profile, throughput, latency, queue/backpressure behavior, and regression notes for hot-path changes.
- [ ] Structured telemetry/logging samples are attached where relevant.
- [ ] Import/code-search audits prove old phase-owned paths are removed or test-only.
- [ ] Deletion manifest lists removed files, quarantined migration tools, and banned imports.
- [ ] PR-required CI is green for phase-owned checks.
- [ ] Phase-owned architecture policy audits are active in CI or attached with a clear promotion rule.
- [ ] Deep/scheduled gate status is recorded; failures have owners and cannot be ignored for phase signoff.
- [ ] No flaky or quarantined test is the only proof for a phase-closing invariant.
- [ ] Rollback point is tagged and recovery smoke is documented.
- [ ] `agent_completion_packet.md` and `evidence_reconciliation.md` are attached and reconcile claimed work to matrix rows, tests, deletion proof, telemetry, docs, and performance evidence.
- [ ] Exit gate report names every V2 requirement row closed by this phase.

## Hard Failure Conditions

- [ ] No runtime compatibility shim remains in `Python/src/hexorl/` for a path replaced by this phase.
- [ ] No phase-owned implementation is deferred to a later phase.
- [ ] No phase-owned feature is implemented only in unit tests without runtime consumption.
- [ ] No old/new dual runtime path remains after the phase cutover.
- [ ] No dashboard/training/eval/self-play private reconstruction remains for data owned by contracts.
- [ ] No `skip`, `xfail`, TODO, placeholder, "manual-only", "later phase", or "not applicable" claim closes a required invariant without an explicit blocking gate record.
- [ ] No hot-path change removes batching, parallelism, backpressure, or cheap boundary validation without recorded benchmark evidence and orchestrator approval.

## Phase Quick Gates

### Phase 00
- [ ] Pre-refactor git tag and archive manifest recorded.
- [ ] Baseline command transcripts and config hashes saved.
- [ ] HostProfile, CPU/GPU utilization, inference throughput, self-play phase profile, replay projection throughput, and training step baseline are archived where available.
- [ ] `global_xattn` default pair strategy is `none`.
- [ ] Accidental pair scoring test exists and fails without explicit strategy opt-in.
- [ ] Structured self-play/autotune/no-progress trace samples archived.
- [ ] Legacy fallback and architecture-string inventories mapped to owner phases.

### Phase 01
- [ ] `contracts/` and `engine/` packages created.
- [ ] Rust is the production legal/history source only through validated `engine/` wrappers; direct `_engine` imports are removed from runtime.
- [ ] Python legal fallback is fixture-only.
- [ ] Rust/Python parity passes for legal rows, compact history, D6 coordinates/history/legal/dense tensors.
- [ ] Engine/legal verification checks semantic legality, row ordering, duplicate detection, terminal state, current player, and source/hash identity.
- [ ] Engine wrappers consume the existing Rust/PyO3 protocol helpers instead of duplicating legal/history/pair byte parsing in Python or Rust.
- [ ] Rust invariant hooks are exercised in engine parity/debug tests for representative legal, history, undo, tactical, and terminal states.
- [ ] Contract mutation tests prove zero-copy/cached views cannot invalidate validated hashes silently.
- [ ] Private production legal/history/D6 helpers removed or quarantined outside runtime.

### Phase 02
- [ ] Candidate, pair, and graph owners are the only semantic authorities, while approved extension points feed through their validation.
- [ ] Self-play, replay, training, eval, and dashboard fixtures consume the same builders.
- [ ] `PairCandidateBatch` is deleted or demoted to a semantics-free projection from `PairActionTable`.
- [ ] Graph semantic construction and tensorization are separate.
- [ ] Golden parity proves candidate/pair/graph equality across consumers.
- [ ] Candidate, pair, and graph tensor projections are proven to be pure, immutable projections from canonical contracts.
- [ ] D6 round-trip and inverse tests cover candidate rows, pair rows, graph tokens, graph relations, masks, and target mass.

### Phase 03
- [ ] `models/` registry/spec/capability system is authoritative and facet/descriptor based enough to add a fake family without editing trainer/inference/search/dashboard internals.
- [ ] No runtime imports from `hexorl/model`.
- [ ] Every family exposes model, train adapter, inference manifest/declaration, policy provider, loss plan, default recipe, tune space, and checkpoint manifest through registry facets/descriptors.
- [ ] Trainer runs one batch for every registered family through `TrainAdapter`.
- [ ] Checkpoint manifest strict round-trip and inspect-without-weights tests pass.
- [ ] Model target verification covers legal-row alignment, pair known-first semantics, opening no-pair targets, masks, finite losses, and D6 target mass.
- [ ] Training debug bundle proves replay record -> contracts -> tensors -> targets -> loss inputs without private reconstruction.
- [ ] TrainAdapter hot-path evidence covers vectorized projection, device transfer behavior, and one-batch throughput for each registered or approved representative family.

### Phase 04
- [ ] `InferenceProtocolManifest` is negotiated before request submission.
- [ ] Inference protocol uses a stable base envelope plus request-kind payload schemas so new request kinds do not require transport lifecycle rewrites.
- [ ] Batching/backpressure evidence proves bounded waits, queue telemetry, and GPU batching behavior under synthetic or self-play load.
- [ ] Protocol mismatch fails fast with structured error and no IPC hang.
- [ ] Transport owns pack/ready/wait/timeout/decode/reset lifecycle.
- [ ] Response telemetry includes protocol, contracts, model family, row/token counts, timings, and warnings.
- [ ] Old submit lifecycle/private tensor rebuild paths are removed.

### Phase 05
- [ ] All model families used by self-play expose priors through `PolicyProvider`.
- [ ] `EngineAdapter` is the only layer calling Rust MCTS APIs.
- [ ] `PairStrategySpec` validates root/leaf/full caps independently.
- [ ] Default pair strategy is `none`; `global_xattn` emits zero pair rows by default.
- [ ] Global graph pair heads have turn-aware, row-mapped, telemetry-visible contracts.
- [ ] Policy and MCTS verification proves raw model outputs map to exactly the intended legal rows before search consumes them.
- [ ] EngineAdapter rejects stale legal-row identity, stale pair-row identity, non-finite priors, and all-zero priors without explicit fallback reason.
- [ ] EngineAdapter preserves canonical MCTS root/batch token lifecycle and converts Rust `MCTSError` failures into structured Python errors without panic wrappers or stringly fallbacks.
- [ ] EngineAdapter/MCTS timing evidence covers root init, leaf selection, backprop, sampling, policy mapping, and token-failure paths without Python per-node hot loops.

### Phase 06
- [ ] `GameRunner` depends on providers/adapters/builders, not architecture/config strings.
- [ ] `GameRunner` composes service/pipeline outputs and does not call canonical builder internals directly.
- [ ] Self-play process/thread/resource ownership follows HostProfile budgets and propagates inference/replay backpressure.
- [ ] `SelfPlayWorker` is lifecycle/IPC only.
- [ ] Worker-owned game-loop, replay assembly, graph/candidate/pair chunking, and MCTS prior wiring are removed.
- [ ] Self-play heartbeat, no-progress, game summary, policy timing, pair summary, and `ContractTrace` events are asserted.
- [ ] Single-game debug trace can replay one game position-by-position and identify engine, contract, inference, MCTS, or replay-writer failures.
- [ ] Self-play verification catches legal-row disagreement, changed hashes, mutated model inputs, stale targets, and replay record mismatch.

### Phase 07
- [ ] New self-play writes only new replay records.
- [ ] Sampler reads only new replay records.
- [ ] Training batches come only through `replay/projector.py` from canonical contracts.
- [ ] Old replay/buffer decode code is absent from runtime imports.
- [ ] Round-trip, corruption, projection, and sample-to-loss tests pass.
- [ ] Replay write/read/project throughput, queue bounds, prefetch policy, memory profile, and vectorized projection evidence are attached.

### Phase 08
- [ ] Arena/eval uses `PolicyProvider` for every registered family.
- [ ] Dashboard routes use `ContractInspector` and read-only services only.
- [ ] Dashboard shows contract hash/source/version, trace IDs, and mismatch location.
- [ ] Dashboard can inspect a single-position debug bundle across engine, contracts, D6, targets, model outputs, policy priors, MCTS, and replay.
- [ ] Autotune uses typed `ModelRecipe` and family spaces, not raw config mutation.
- [ ] RuntimeSpec/autotune can search host-utilization knobs separately from model semantics and reports throughput, utilization, stability, and stall score components.
- [ ] Recipe dry-run, scheduler decision logging, and no-progress watchdog tests pass.

### Phase 09
- [ ] Final import graph has no banned runtime modules or compatibility facades.
- [ ] CI enforces all architecture invariants automatically.
- [ ] CI tiers, artifact retention, flaky/quarantine policy, performance budgets, and final closure gates are enforced and documented.
- [ ] Full self-play -> replay -> train -> eval -> dashboard smoke archived.
- [ ] V2 requirement matrix is fully closed.
