# Phase 07 - Replay Cutover

## Purpose
Cut runtime replay over to the new canonical replay path only. After this phase, self-play writes only new replay records, sampling reads only new replay records, and training batches are produced only by `replay/projector.py` from canonical contracts.

Evaluation, dashboard, and autotune convergence are intentionally out of scope for this phase and remain Phase 08 work.

## Source Of Truth
This phase implements the Phase 7 replay cutover requirements from `Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_V2_20260429.md`.

Core invariant:

```text
Runtime code uses replay/codec.py, replay/storage.py, replay/sampler.py, and replay/projector.py.
Old replay and buffer decode paths may exist only in tools/migration or frozen fixtures.
```

The current runtime still uses `hexorl.buffer` paths. This phase must explicitly create `Python/src/hexorl/replay/` or move/rename the current buffer ownership into it, then delete runtime imports from the old buffer path for self-play, sampler, training, and epoch runtime. Dashboard/evaluation inspection removal is Phase 08 work unless a Phase 07 runtime import is actively used by sampling or training.

## Target Modules
- `Python/src/hexorl/replay/codec.py`
- `Python/src/hexorl/replay/storage.py`
- `Python/src/hexorl/replay/sampler.py`
- `Python/src/hexorl/replay/projector.py`
- `Python/src/hexorl/replay/fixtures.py`
- existing `Python/src/hexorl/buffer/` runtime paths as deletion or migration sources
- `Python/src/hexorl/selfplay/record_writer.py`
- `Python/src/hexorl/selfplay/records.py`
- `Python/src/hexorl/train/adapters.py`
- `Python/src/hexorl/train/trainer.py`
- `tools/migration/` for one-off old replay conversion only

## Required Cutover
- New self-play writes only new replay records.
- New replay records are versioned and validated at write time.
- Replay storage rejects records with unknown schema versions unless an explicit migration tool is being run.
- The sampler reads only new replay records.
- The sampler does not decode old buffer records, legacy replay rows, or compatibility payloads.
- Training batches are produced only through `replay/projector.py`.
- `replay/projector.py` projects from canonical contracts such as `MoveHistory`, `LegalActionTable`, `PositionContract`, `CandidateTable`, `PairActionTable`, and model-family train adapters.
- Training code may consume projected batches, but may not rebuild replay fields privately.
- Old replay/buffer decode code is removed from runtime imports.
- Any retained old decode logic lives only under `tools/migration/` or frozen test fixtures.
- There are no runtime imports from `buffer/` or old replay paths in self-play, sampler, training, or epoch runtime. Dashboard/eval import removal is enforced in Phase 08.

## Canonical Record Boundary
New replay records must preserve the data needed to reconstruct and validate canonical contracts without model-family-specific assumptions:

```text
record schema version
game identity and position identity
compact move history
Rust-derived legal action table identity/hash
Rust FFI protocol version or source marker for legal/history rows
compact history row count/hash
reconstructed Rust legal row hash
policy/value/search targets
policy target global row identity
candidate and pair target metadata where applicable, including pair known-first/completeness metadata
contract trace or validation metadata
writer version and config/checkpoint identity
```

Transient MCTS `root_generation` and `batch_generation` belong in debug traces only. They must not become replay semantics or training labels.

Rules:

- Replay records store facts and targets, not private training tensors.
- Dense, sparse, graph, and pair model batches are projections from the same record contracts.
- Projection failures are hard errors unless the record is explicitly marked as a frozen negative/corruption fixture.
- Corruption handling must identify the record, field, schema version, and failed invariant.
- Replay records must not store mutable views into self-play/game-runner payloads. Stored records must be immutable or copied at the boundary so later game-state mutation cannot change written data.

## Projection Requirements
- `replay/projector.py` is the single runtime path from replay sample to trainable batch payload.
- Projector output is typed and contract-versioned.
- Projector output is accepted by `train/adapters.py` without trainer architecture branches.
- Legal rows, candidates, pair rows, targets, masks, graph inputs, and loss inputs are derived from canonical contract builders and adapters.
- D6 replay augmentation, if active in this phase, must transform canonical contracts rather than mutate already-projected tensors privately.
- Sample-to-loss smoke coverage proves that a sampled record becomes a finite loss through the projector path.

## Detailed Replay Verification

Replay is a correctness boundary, not only a storage format. This phase must prove records preserve the semantics that training later consumes.

Required verification:

- self-play traced position -> replay record -> decoded record -> projected contracts must preserve history hash, legal row ids, source, schema version, target identity, D6 identity, candidate/pair/graph identities, and record hash
- record writing must reject stale legal hashes, stale schema versions, mismatched policy targets, mismatched value targets, bad pair known-first references, illegal masks, and non-finite target values
- record writing must reject records whose compact history or legal rows cannot be replayed through the current Rust engine and centralized FFI protocol
- storage round-trips must prove byte serialization does not reorder rows, drop source/version fields, alter target mass, or mutate nested payloads
- D6 replay augmentation must prove inverse/composition, target mass preservation, row identity preservation, and no in-place mutation of source records
- projector outputs must be reproducible from the record and canonical builders only
- corruption tests must identify whether the failure belongs to codec, storage, sampler, projector, contract builder, or train adapter
- the single-position behavior debug bundle must include replay record content, decoded content, projector output identities, and sample-to-loss identity checks

## Migration Boundary
Old replay data may be converted, but not consumed by runtime code.

Allowed locations:

```text
tools/migration/
Python/tests/**/fixtures/
```

Forbidden runtime behavior:

```text
import hexorl.buffer
import hexorl.replay_legacy
import old replay decode helpers from sampler/trainer/self-play
branch on legacy replay schema in runtime sampler
fallback to old buffer decode when new codec fails
GameRecord.from_compact_bytes accepting magic-less legacy records in runtime
```

Any migration script must be explicit about input schema, output schema, validation checks, and artifact location.

## Parallel Subagent Work
- S1: implement new replay record schema, codec roundtrip validation, and corruption diagnostics.
- S2: cut self-play record writing to new records only and remove runtime old-record write paths.
- S3: cut sampler reads to new records only and route all sample projection through `replay/projector.py`.
- S4: wire trainer/adapters to consume projector output and add sample-to-loss smoke coverage.
- S5: move old replay/buffer decode code to `tools/migration/` or frozen fixtures and add import audits.

## Mandatory Tests
- New replay codec roundtrip tests for golden records.
- Storage write/read roundtrip tests with schema-version validation.
- Corruption tests for malformed headers, truncated payloads, invalid schema versions, invalid legal hashes, invalid targets, and bad contract metadata.
- Corruption tests for stale row ids, stale hashes, bad masks, non-finite target values, bad known-first pair targets, reordered rows, and mutated nested payloads.
- Projector tests proving sampled records produce canonical train batches for supported model families.
- Sample-to-loss smoke tests proving projected batches produce finite losses through `train/adapters.py`.
- Tests proving self-play trace -> replay write -> replay read -> projector -> train adapter preserves semantic identities.
- Tests proving D6 replay augmentation does not mutate source records and preserves target mass and row identity.
- Import audit tests proving no runtime import of `buffer/` or old replay paths.
- Tests proving old replay/buffer decode exists only in `tools/migration/` or frozen fixtures.
- Tests proving sampler reads only new records and fails hard on old replay payloads.
- Tests proving records produced before the Rust hardening slice are either migrated offline with explicit validation or rejected by runtime readers.

## Required Artifacts
- New replay schema/version documentation.
- Golden new replay fixture set with generator command and fixed seed.
- Corruption fixture set or synthetic corruption test helper.
- Replay verification report proving trace-to-record-to-projector identity preservation.
- Rust replay/invariant verification report for golden replay records, including malformed-history and stale-legal negative cases.
- Single-position behavior debug bundle replay section sample.
- Migration tool notes for any retained old replay conversion path.
- Import audit output or CI check covering forbidden runtime imports.
- Sample-to-loss smoke output for each supported training adapter in scope.

## Hard Gates
- New self-play writes only new replay records.
- Sampler reads only new replay records.
- Training batches are produced only through `replay/projector.py`.
- Projector consumes canonical contracts rather than sampler-private reconstruction logic.
- Replay records, decoded records, and projector outputs preserve semantic identity across trace, storage, D6, and training boundaries.
- Replay mutation/corruption tests fail loudly with codec/storage/sampler/projector/train-adapter ownership.
- Old replay/buffer decode is absent from runtime imports.
- Old replay/buffer decode exists only in `tools/migration/` or frozen fixtures.
- Roundtrip, corruption, projection, and sample-to-loss tests pass.
- Import audits pass.
- No evaluation, dashboard, or autotune work is included in this phase.

## Exit Criteria
Runtime replay has one active path:

```text
self-play record writer
-> new replay codec/storage
-> new replay sampler
-> replay/projector.py
-> train adapters
-> finite loss
```

Legacy replay and buffer compatibility are no longer part of the runtime system. Any old-data support is isolated to migration tools or frozen fixtures, with hard import gates preventing it from re-entering self-play, sampling, or training.
