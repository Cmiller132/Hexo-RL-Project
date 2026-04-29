# Refactor Phase Checklist

Use this checklist during execution. A phase is complete only if every box is checked and the evidence is attached under `Docs/refactor/artifacts/phase_XX/`.

## Universal Checklist

- [ ] Phase scope is frozen and matches the V2 requirement matrix.
- [ ] Entry criteria from the previous phase are satisfied.
- [ ] Public contracts/interfaces for this phase are frozen before implementation work.
- [ ] Golden fixtures or baseline artifacts needed by this phase are identified.
- [ ] Golden fixtures include negative/corrupt cases, D6 variants, and expected semantic identities, not just shapes.
- [ ] Unit tests for new behavior are added.
- [ ] Integration tests prove runtime consumers use the new path.
- [ ] Parity tests use frozen fixtures or migration tools only; old runtime code is not kept as an oracle.
- [ ] Cross-validation tests prove row identity, ordering, source, schema version, and hash stability across subsystem boundaries.
- [ ] Mutation-safety tests prove cached views, tensors, replay payloads, and target projections cannot silently change after validation.
- [ ] Corruption tests prove bad histories, illegal rows, stale hashes, bad masks, wrong D6 transforms, malformed targets, and protocol mismatches fail loudly.
- [ ] A single-position debug bundle or equivalent trace can localize failures to the owning subsystem.
- [ ] Performance smoke is compared to Phase 00 baseline where relevant.
- [ ] Structured telemetry/logging samples are attached where relevant.
- [ ] Import/code-search audits prove old phase-owned paths are removed or test-only.
- [ ] Deletion manifest lists removed files, quarantined migration tools, and banned imports.
- [ ] CI is green on the branch.
- [ ] Rollback point is tagged and recovery smoke is documented.
- [ ] Exit gate report names every V2 requirement row closed by this phase.

## Hard Failure Conditions

- [ ] No runtime compatibility shim remains in `Python/src/hexorl/` for a path replaced by this phase.
- [ ] No phase-owned implementation is deferred to a later phase.
- [ ] No phase-owned feature is implemented only in unit tests without runtime consumption.
- [ ] No old/new dual runtime path remains after the phase cutover.
- [ ] No dashboard/training/eval/self-play private reconstruction remains for data owned by contracts.

## Phase Quick Gates

### Phase 00
- [ ] Pre-refactor git tag and archive manifest recorded.
- [ ] Baseline command transcripts and config hashes saved.
- [ ] `global_xattn` default pair strategy is `none`.
- [ ] Accidental pair scoring test exists and fails without explicit strategy opt-in.
- [ ] Structured self-play/autotune/no-progress trace samples archived.
- [ ] Legacy fallback and architecture-string inventories mapped to owner phases.

### Phase 01
- [ ] `contracts/` and `engine/` packages created.
- [ ] Rust is the production legal/history source.
- [ ] Python legal fallback is fixture-only.
- [ ] Rust/Python parity passes for legal rows, compact history, D6 coordinates/history/legal/dense tensors.
- [ ] Engine/legal verification checks semantic legality, row ordering, duplicate detection, terminal state, current player, and source/hash identity.
- [ ] Contract mutation tests prove zero-copy/cached views cannot invalidate validated hashes silently.
- [ ] Private production legal/history/D6 helpers removed or quarantined outside runtime.

### Phase 02
- [ ] `CandidateContractBuilder`, `PairActionTableBuilder`, `GraphSemanticBuilder`, `GraphTensorizer`, and collator are active.
- [ ] Self-play, replay, training, eval, and dashboard fixtures consume the same builders.
- [ ] `PairCandidateBatch` is deleted or demoted to a semantics-free projection from `PairActionTable`.
- [ ] Graph semantic construction and tensorization are separate.
- [ ] Golden parity proves candidate/pair/graph equality across consumers.
- [ ] Candidate, pair, and graph tensor projections are proven to be pure, immutable projections from canonical contracts.
- [ ] D6 round-trip and inverse tests cover candidate rows, pair rows, graph tokens, graph relations, masks, and target mass.

### Phase 03
- [ ] `models/` registry/spec/capability system is authoritative.
- [ ] No runtime imports from `hexorl/model`.
- [ ] Every family exposes model, train adapter, inference manifest/declaration, policy provider, loss plan, default recipe, and tune space.
- [ ] Trainer runs one batch for every registered family through `TrainAdapter`.
- [ ] Checkpoint manifest strict round-trip and inspect-without-weights tests pass.
- [ ] Model target verification covers legal-row alignment, pair known-first semantics, opening no-pair targets, masks, finite losses, and D6 target mass.
- [ ] Training debug bundle proves replay record -> contracts -> tensors -> targets -> loss inputs without private reconstruction.

### Phase 04
- [ ] `InferenceProtocolManifest` is negotiated before request submission.
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

### Phase 06
- [ ] `GameRunner` depends on providers/adapters/builders, not architecture/config strings.
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

### Phase 08
- [ ] Arena/eval uses `PolicyProvider` for every registered family.
- [ ] Dashboard routes use `ContractInspector` and read-only services only.
- [ ] Dashboard shows contract hash/source/version, trace IDs, and mismatch location.
- [ ] Dashboard can inspect a single-position debug bundle across engine, contracts, D6, targets, model outputs, policy priors, MCTS, and replay.
- [ ] Autotune uses typed `ModelRecipe` and family spaces, not raw config mutation.
- [ ] Recipe dry-run, scheduler decision logging, and no-progress watchdog tests pass.

### Phase 09
- [ ] Final import graph has no banned runtime modules or compatibility facades.
- [ ] CI enforces all architecture invariants automatically.
- [ ] Full self-play -> replay -> train -> eval -> dashboard smoke archived.
- [ ] V2 requirement matrix is fully closed.
