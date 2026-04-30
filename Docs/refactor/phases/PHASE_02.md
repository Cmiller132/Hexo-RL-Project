# Phase 02 - Candidate/Pair/Graph Builder Convergence

## Purpose
Converge candidate construction, pair-action construction, and global graph input construction onto one shared contract pipeline consumed by runtime, training, replay, evaluation, and dashboard tooling.

This phase removes private rebuild paths. A position should produce exactly one canonical `CandidateTable`, exactly one canonical phase-aware `PairActionTable` when pair rows are requested, and exactly one graph semantic contract that is tensorized by pure projection.

## Target Modules
Create or complete the shared builders:
- `Python/src/hexorl/contracts/candidates.py`
- `Python/src/hexorl/contracts/pairs.py`
- `Python/src/hexorl/graph/semantic_builder.py`
- `Python/src/hexorl/graph/tensorize.py`
- `Python/src/hexorl/graph/collate.py`
- shared fixture helpers under `Python/src/hexorl/replay/fixtures.py` or the equivalent test fixture owner

The canonical builder APIs are:
- `CandidateContractBuilder`
- `PairActionTableBuilder`
- `GraphSemanticBuilder`
- `GraphTensorizer`
- graph collator/batch projection utilities

## V2 Requirements to Implement
- `CandidateContractBuilder` is the only owner of `CandidateTable` rows, dense indices, candidate features, masks, targets, missing mass, recall, diagnostics, and contract hash.
- `PairActionTableBuilder` is the only owner of canonical pair-action rows, first/second row references, phase semantics, known-first handling, generation mode, possible-pair counts, selected-pair counts, and table hash.
- First-placement unordered pairs and second-placement known-first pairs are separate phases with explicit masks and validation.
- Crop pair candidates and global graph `PAIR_ACTION` rows must derive from `PairActionTable`.
- `PairCandidateBatch` must be deleted. If an intermediate tensor object is temporarily required, it must be a thin projection from `PairActionTable`, must not own semantics, and must be named/documented as a projection.
- Full `A * (A - 1) / 2` pair generation is forbidden unless an explicit `PairStrategy` requests it and supplies hard caps.
- Graph semantic construction and graph tensorization must be split. `GraphSemanticBuilder` owns tokens, semantic rows, relation identities, and debug metadata. `GraphTensorizer` and collator own only tensor layout, padding, batching, and device-facing projection.
- Graph tensorization must be a pure projection from graph semantic contract plus legal/candidate/pair contracts. It must not regenerate legal rows, candidates, pair rows, tactical facts, history, or D6 transforms.
- Self-play, replay projector/sampler, training adapters, evaluation debug paths, dashboard fixtures, dashboard inspectors, and model-input fixture generation must consume the same builders.
- Dashboard fixtures must prove the displayed candidate, pair, and graph views are sourced from the same contracts as training and self-play.

## Shared Consumption Cutover
Replace private construction paths in:
- self-play worker/game-runner preparation
- replay projection and sampler batch construction
- training adapter input assembly
- evaluation policy/debug payloads
- dashboard contract/model/graph inspectors and fixtures
- model cache or debug fixture generation

No runtime path may privately rebuild:
- candidate rows or candidate masks
- pair mini-contracts or pair-row references
- graph `LEGAL` or `PAIR_ACTION` rows
- graph semantic tokens or relations
- D6-transformed candidate/pair/graph inputs outside the contract APIs

## Detailed Verification Requirements
The builders introduced here must be verified as if the previous candidate, pair, and graph paths may contain subtle bugs. Do not use old runtime output as the only oracle.

Required verification:
- Golden positions must include opening, first-placement, second-placement known-first, terminal/near-terminal, pair-heavy, graph-token-heavy, and invalid-history cases.
- Candidate rows must be verified for legal-row identity, dense-index mapping, mask correctness, target mass, recall diagnostics, ordering, schema version, hash, and source.
- Pair rows must be verified for phase, unordered first-placement identity, ordered known-first second-placement semantics, selected-first validation, caps, row ordering, D6 behavior, schema version, hash, and source.
- Graph semantics must be verified for token identity, relation identity, legal-row links, candidate links, pair links, phase labels, relation schema version, graph schema version, and hash.
- Tensorized graph/candidate/pair batches must be pure projections from contracts. Mutating a tensor batch must not mutate the source contract unless the batch explicitly owns a copy and validation proves it.
- D6 transforms must be verified with inverse and composition tests across candidates, pairs, graph tokens, graph relations, masks, and targets.
- Negative tests must corrupt candidate rows, pair rows, graph row links, masks, counts, hashes, schema versions, known-first references, relation ids, and tensor shapes, and each failure must identify the owning builder/projection.
- A single-position debug bundle must extend the Phase 01 payload with candidates, pair rows, graph semantics, tensor shapes, masks, target mass, D6 variant identity, hashes, warnings, and builder timings.

## Delete or Demote
Delete:
- private candidate construction in sampler/dashboard/worker
- parallel crop/global pair mini-contracts
- graph batch code that owns semantic construction and tensorization together
- any production `PairCandidateBatch` semantic owner

Demote only as a short-lived projection:
- tensor-only pair batch views derived directly from `PairActionTable`
- tensor-only graph batches derived directly from `GraphSemanticBuilder` output

## Parallel Subagent Work
- S1: `CandidateContractBuilder` schema, validation, hashing, diagnostics, and golden fixture parity.
- S2: `PairActionTableBuilder` phases, known-first semantics, D6 behavior, caps, and `PairCandidateBatch` deletion/projection migration.
- S3: `GraphSemanticBuilder` extraction for tokens, relation identities, legal rows, candidate links, pair links, and debug payloads.
- S4: `GraphTensorizer`/collator rewrite as pure projection with batching, padding, masks, and shape/schema validation.
- S5: shared consumer cutover for self-play, replay sampler/projector, training, eval, dashboard fixtures, and import audits.

## Mandatory Tests
- Exact parity: self-play candidate table equals replay sampler candidate table equals dashboard candidate table for all golden positions.
- Exact parity: training adapter candidate tensors are projections of the same `CandidateTable` used by self-play and dashboard fixtures.
- Exact parity: crop pair rows and global graph `PAIR_ACTION` rows derive from the same `PairActionTable`.
- Exact parity: graph legal rows match the validated Rust-backed `LegalActionTable`; graph code does not build legal rows privately, and independent golden fixtures or hand-audited legal-row references catch cases where Rust itself would be wrong.
- Exact parity: graph tensor batches are pure projections from graph semantic contracts for golden positions.
- D6 parity: candidate target mass, pair target mass, unordered first-placement pair identity, and second-placement known-first semantics are preserved.
- Phase tests: opening, first-placement turn, second-placement known-first turn, pair-heavy state, and graph token-heavy state.
- Cap tests: full pair enumeration fails without an explicit capped strategy.
- Projection tests: any remaining pair tensor batch object has no semantic fields not derivable from `PairActionTable`.
- Mutation tests: graph/candidate/pair tensor projections cannot mutate canonical contracts or silently change validated identities.
- Corruption tests: stale hashes, wrong schema versions, wrong known-first rows, bad relation ids, bad masks, duplicate candidate rows, and illegal pair rows fail loudly.
- Debug bundle tests: the single-position debug payload can identify candidate-builder, pair-table-builder, graph-semantic-builder, or tensorizer ownership for failures.
- Import audits: no private candidate, pair, legal, D6, history, graph semantic, or graph tensor rebuild paths remain in runtime consumers.

## Required Artifacts
- Golden fixture bundle for candidate, pair, graph semantic, and graph tensor projection checks.
- Builder API documentation with field ownership and invariants.
- Migration notes listing deleted private builders and any temporary projection-only shims.
- Import audit output showing old private builder paths are absent from runtime imports.
- Trace examples showing candidate count, pair possible/selected counts, graph token/relation counts, and tensorization timing.
- Single-position debug bundle sample for candidate, pair, graph semantic, graph tensor, mask, target, D6, and mutation-safety verification.

## Hard Exit Gates
- `CandidateContractBuilder` is the only production candidate-table builder.
- `PairActionTableBuilder` is the only production pair-action-row builder.
- `PairCandidateBatch` is deleted or reduced to a tensor-only projection from `PairActionTable` with no independent semantics.
- `GraphSemanticBuilder` owns graph semantics; `GraphTensorizer`/collator only project, pad, batch, and validate tensors.
- Self-play, replay, sampler, training, evaluation, dashboard inspectors, and dashboard fixtures all consume shared builders.
- Golden-position parity tests pass exactly across self-play, replay sampler, training, evaluation debug, and dashboard fixtures.
- Pair tables are phase-aware, cap-aware, D6-equivariant, and telemetry-visible.
- Graph tensorization is proven to be a pure projection from graph semantic contract plus shared legal/candidate/pair contracts.
- Candidate, pair, and graph projections are mutation-safe and corruption-tested.
- The debug bundle localizes failures to the correct builder or projection owner.
- Import audits find no runtime-private candidate, pair, graph semantic, graph tensor, legal, D6, or compact-history reconstruction paths.
- No pair scoring or full pair enumeration is introduced by this phase without an explicit capped `PairStrategy`.
- CI fails hard on any parity mismatch, forbidden import, uncapped pair generation, or private builder reintroduction.
