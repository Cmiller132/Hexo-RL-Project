# Phase 01 - Engine + Contracts Foundation

## Purpose
Establish the first hard boundary of the V2 architecture: Rust is the intended production game-state source, Python exposes that source only through validating `engine/` wrappers, and pure versioned contracts describe the data that every later subsystem consumes.

This phase is not contracts-only. It creates the shared foundation for history, legal rows, D6 symmetry, targets, candidates, pairs, telemetry, validation, and debug inspection while removing production paths that privately rebuild rules data in Python.

Source of truth: `Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_V2_20260429.md`.

## Scope

### 1. Contracts Package
Create `Python/src/hexorl/contracts/` as a pure data package.

Required modules:
- `__init__.py`
- `identity.py`
- `history.py`
- `coordinates.py`
- `symmetry.py`
- `legal.py`
- `actions.py`
- `targets.py`
- `tactical.py`
- `candidates.py`
- `pairs.py`
- `graph.py`
- `replay.py`
- `telemetry.py`
- `validation.py`
- `debug.py`

Required V2 objects and owners:
- `MoveHistory` in `contracts/history.py`
- `LegalActionTable` in `contracts/legal.py`
- D6 transform APIs in `contracts/symmetry.py`
- `PolicyTarget`, `PairPolicyTarget`, and related target payloads in `contracts/targets.py`
- `CandidateTable` and diagnostics types in `contracts/candidates.py`
- `PairActionTable` in `contracts/pairs.py`
- `ContractTrace` and contract telemetry payloads in `contracts/telemetry.py`
- validation failures and boundary assertions in `contracts/validation.py`
- debug/inspection payloads in `contracts/debug.py`

Contract rules:
- Contracts are plain typed data/value objects.
- Contracts are versioned.
- Contracts are comparable and hashable where useful.
- Contracts validate at construction or decode boundaries.
- Contracts do not import model, inference, search, train, dashboard, tuning, self-play orchestration, or process lifecycle code.
- Contracts may expose cached or zero-copy views when hot paths need them.
- Contracts may not hide subsystem behavior, perform inference, run search, build dashboards, or own process orchestration.

### 2. Engine Boundary
Create `Python/src/hexorl/engine/` as the only Python-facing Rust rules boundary. It wraps the compiled `_engine` PyO3 module; direct `_engine` imports are banned outside this package, tests, and explicit fixture tooling.

Required modules:
- `__init__.py`
- `rust.py`
- `legal.py`
- `history.py`
- `encoding.py`
- `parity.py`

Boundary rules:
- Rust is the production source of legal game state.
- `engine/legal.py` exposes the production `LegalTableProvider`.
- `engine/history.py` owns Rust replay/history parity helpers.
- `engine/encoding.py` owns Python-facing encode/decode calls over the Rust representation.
- `engine/parity.py` contains shared parity harness helpers for tests.
- Python wrappers must use the centralized Rust/PyO3 protocol owner in `crates/hexgame-py/src/protocol.rs` for legal rows, compact history rows, board-piece rows, and pair rows; they must not create a second byte layout or row-width interpretation.
- Production code may not silently fall back from Rust legal rows to Python legal rows.
- Any fixture-only fallback must be explicit, local to tests or fixture builders, and marked with `source="fixture"`.

Engine verification rule:
- Do not treat the current Rust/Python boundary as automatically correct.
- Treat the completed Rust hardening work as the canonical implementation baseline, but keep suspicion at the boundary: every Rust result must either validate into a Python contract or fail with a structured owner-specific error.
- Every Rust-exposed payload must be validated for semantic correctness, not only shape or row count.
- Legal rows must be checked for duplicate cells, occupied cells, current-player correctness, phase correctness, terminal-state consistency, row ordering, dense-index mapping, and stable source/hash identity.
- History replay must cross-check final board state, side to move, terminal result, placement counts, and rejected illegal transitions.
- Tactical responses must use `TacticalStatus`; `ThreatStatus` must not be recreated as a Python compatibility model.
- Debug and parity tests should exercise Rust invariant hooks, including candidate, hash, winner, move-history, eval/hot-window, and undo consistency where exposed.
- FFI boundary logs and debug bundles must include method name, trace id, history hash, legal hash, root/batch token when present, row counts, dtype/shape/contiguity, duration, and failure class.
- Engine output used by contracts must be copied, frozen, or guarded so later mutation cannot invalidate hashes, cached views, targets, or traces silently.

### 3. History Ownership
Centralize compact-history handling.

Required behavior:
- `MoveHistory` is the single Python contract for compact history.
- Invalid compact histories are rejected at decode time.
- Decode validation rejects invalid player order, duplicate cells, illegal placement counts, invalid radius, malformed bytes, and inconsistent current-player state.
- Bootstrap generation uses the same encoder as runtime paths.
- Graph, sampler, dashboard, tactical oracle, RGSC, epoch bootstrap, and self-play stop owning private compact-history parsers.

### 4. Legal Table Ownership
Centralize legal rows through Rust-backed engine APIs and a canonical contract.

`LegalActionTable` must include, at minimum:
- canonical rows
- dense indices
- source
- radius
- occupied count
- schema/version identifier
- table hash

Legal rules:
- Production legal rows come from Rust through `engine/legal.py`.
- Python legal generation is not allowed in production fallback paths.
- Fixture legal tables must explicitly use `source="fixture"`.
- Legal row ordering is part of the contract.
- Degraded or fallback sources must be telemetry-visible and test-visible.
- Dashboard, sampler, tactical oracle, graph, self-play, training, and evaluation debug may not generate private legal rows.

### 5. D6 Symmetry Ownership
Move all Python D6 code into `contracts/symmetry.py`.

Required APIs:
- `transform_qr`
- `transform_history`
- `transform_legal_table`
- `transform_policy_target`
- `transform_pair_policy_target`
- `transform_dense_policy`
- `transform_axis_label`
- `transform_axis_maps`
- `apply_tensor_symmetry`
- `compose_symmetries`
- `inverse_symmetry`

Required invariants:
- `transform(a then b) == transform(compose(a, b))`
- inverse transform restores original identity
- target mass is preserved
- dense policy mass is preserved
- legal-table hashes match transformed legal rows
- pair identity is preserved for unordered first-placement pairs
- ordered second-placement pairs preserve known-first semantics
- dashboard inputs match sampler inputs after the same symmetry

### 6. Schema, Version, Hash, And Source Rules
Every contract introduced in this phase must define:
- `schema_version` or equivalent version identity
- stable field semantics
- validation entry points
- stable equality behavior where applicable
- stable hash behavior for trace/debug comparison where applicable
- explicit source identity for externally produced data

Hash rules:
- Hashes are deterministic across processes for the same canonical payload.
- Hashes are based on canonical content, not object identity.
- Hash inputs include schema/version identity when a schema change could alter interpretation.
- Hashes for legal/history/pair/candidate/debug payloads are safe to log.

Source rules:
- `source="rust"` or a more specific Rust-provider source is required for production legal/history-derived data.
- `source="fixture"` is allowed only for tests and explicit fixture artifacts.
- `source="fallback"` is not allowed in production runtime.
- Any non-production source must fail hard unless the caller explicitly opted into fixture/test mode.

### 7. Hot-Path View Rules
Contracts must be cheap enough for self-play and search-adjacent paths.

Required behavior:
- Contract APIs can expose zero-copy views over engine-owned or NumPy-backed data where mutation safety is clear.
- Expensive derived views are cached when repeated access is expected.
- Debug dataclasses are not allocated on every leaf expansion unless explicitly requested.
- Validation can distinguish full debug validation from hot-path boundary validation.
- Cached views must not permit mutation that invalidates hashes or equality semantics.

### 8. Detailed Verification And Mutation Safety

This phase must establish the verification style used by all later phases. The existing runtime is not a trusted oracle.

Required verification patterns:
- golden histories with hand-audited board state, side to move, terminal status, legal rows, and D6 variants
- negative histories covering malformed bytes, duplicate placements, occupied-cell moves, illegal turn order, stale known-first placement, invalid radius, and impossible terminal states
- round-trip checks for Rust history replay -> Python `MoveHistory` -> Rust replay
- legal table checks that compare row ids, row order, dense indices, source, schema version, hash, and coordinate semantics
- D6 checks that validate inverse, composition, target mass, legal-row identity, and row ordering after canonicalization
- mutation tests for every ndarray/Torch/cached/zero-copy view exposed by contracts
- hash invalidation tests proving any payload mutation is either impossible or produces a different validated identity
- source enforcement tests proving fixture and fallback data cannot enter production paths accidentally

The phase artifact must include at least one single-position debug payload showing history, board state, legal rows, source/hash/version, D6 transforms, and validation outcome. Later phases extend this payload rather than inventing separate debug formats.

### 9. Production Fallback Removal
Remove or isolate production private implementations replaced by this phase.

Delete or make fixture-only:
- private compact-history parsers
- private D6 helpers
- production Python legal fallbacks
- legal row reconstruction in dashboard/sampler/graph/self-play paths
- history/legal/D6 helper clones used by tactical, replay, RGSC, or bootstrap code

Allowed temporary adapters:
- Thin call-boundary adapters that instantiate contracts.
- Fixture builders under tests or explicit fixture tooling.
- Migration-only helpers that are not imported by runtime code.

## Exact Tests

Add focused tests under `Python/tests/contracts/` and `Python/tests/engine/`.

Required contract tests:
- validation failure tests for every new contract type
- equality stability tests
- hash stability tests
- schema/version identity tests
- source enforcement tests
- fixture-source opt-in tests
- contract purity import test
- zero-copy/cached-view mutation safety tests

Required history tests:
- valid golden compact histories decode into `MoveHistory`
- malformed bytes are rejected
- invalid player order is rejected
- duplicate cells are rejected
- invalid radius is rejected
- invalid placement count/current-player state is rejected
- encode/decode round-trip passes for golden histories

Required legal tests:
- Rust legal table parity passes for golden positions
- legal row ordering is stable
- legal table hash is stable
- legal rows reject occupied cells, duplicate rows, stale side-to-move state, terminal-state inconsistencies, and phase mismatches
- legal row identity includes source, schema/version, dense index mapping, and canonical coordinate semantics
- production fallback source fails hard
- fixture source is accepted only in fixture/test mode
- dashboard/sampler/graph callers consume shared legal contracts

Required D6 tests:
- Python/Rust D6 parity for coordinates
- Python/Rust D6 parity for histories
- Python/Rust D6 parity for legal rows
- Python/Rust D6 parity for dense tensors where Rust support exists; otherwise use indirect engine-state parity and document the missing exposed Rust D6 surface
- composition invariant
- inverse invariant
- policy target mass preservation
- pair target mass preservation
- legal-table transformed hash consistency
- no D6 transform mutates the source history, legal table, target, tensor, or cached view in place unless explicitly documented and validated as an owned mutable buffer

Required mutation-safety tests:
- cached legal/history/D6 views cannot mutate canonical contract payloads
- zero-copy arrays are read-only or guarded by validation identity checks
- copying, tensor conversion, batching, and debug serialization preserve contract hashes
- corrupted hashes, stale schema versions, and changed source labels fail at decode/boundary validation

Required boundary/import tests:
- `contracts/` imports no forbidden runtime subsystem packages
- production code imports legal/history/D6 from `contracts/` or `engine/`, not private helpers
- no runtime import path uses fixture-only legal/history providers
- no runtime code imports `_engine` directly outside `Python/src/hexorl/engine/`
- no Python or Rust runtime path duplicates FFI byte decoding already owned by `crates/hexgame-py/src/protocol.rs`

## Import And `rg` Audits

Run and record audits before exit.

Required searches:
- private compact-history parser names and decode helpers
- private D6 helper names and transform helpers
- Python legal fallback names and legal row builders
- direct legal row reconstruction in dashboard, sampler, graph, self-play, tactical, replay, RGSC, and bootstrap code
- forbidden imports from `contracts/` into model/inference/search/train/dashboard/tuning/self-play orchestration
- runtime imports of fixture-only providers
- `source="fallback"` or equivalent degraded legal/history sources
- direct `_engine` imports outside `Python/src/hexorl/engine/`, tests, and explicit fixture tooling
- duplicate legal/history/pair byte parsers outside `crates/hexgame-py/src/protocol.rs`
- `ThreatStatus`, `threat_status`, or collapsed tactical compatibility names in new runtime code

Expected audit result:
- No production private legal/history/D6 owners remain.
- Any remaining old helper is either deleted, fixture-only, or tracked as a later-phase deletion with no runtime imports.

## Artifacts

Produce or update the following as part of this phase:
- `Python/src/hexorl/contracts/` package
- `Python/src/hexorl/engine/` package
- contract schema/version/hash/source definitions
- Rust/Python parity fixtures or golden fixture references
- single-position debug payload for engine/history/legal/D6 verification
- focused tests under `Python/tests/contracts/`
- focused tests under `Python/tests/engine/`
- import-boundary tests for contract purity
- audit notes showing private parser/helper/fallback removal
- telemetry-visible source fields for legal/history-derived data

## Parallel Subagent Work

Suggested work split:
- S1: contract dataclasses/types, schema/version policy, validation helpers, equality/hash behavior
- S2: `engine/` Rust boundary, legal/history providers, parity harness
- S3: D6 API consolidation and Rust/Python parity tests
- S4: call-boundary adapters for dashboard/sampler/graph/self-play/tactical/replay/bootstrap
- S5: production fallback deletion, import tests, `rg` audits, and artifact checklist

Coordination rule: adapters may be added before full call-site cutover, but no adapter may preserve a production fallback path that contradicts Rust-as-source-of-truth.

## Hard Exit Gates

All gates are required.

- `contracts/` package exists with required modules and pure import boundaries.
- `engine/` package exists and is the only Python-facing Rust rules boundary.
- `MoveHistory` is the single compact-history contract owner.
- `LegalActionTable` is the single legal table contract owner.
- `contracts/symmetry.py` is the single Python D6 owner.
- Production legal rows come from Rust-backed `LegalTableProvider`.
- Production Python legal fallback paths are removed or made fixture-only.
- Private compact-history parsers are removed from runtime imports.
- Private D6 helpers are removed from runtime imports.
- Dashboard, sampler, graph, self-play, tactical, replay, RGSC, and bootstrap paths use shared history/legal/D6 contracts or engine providers.
- Contract schema/version/hash/source rules are implemented and tested.
- Hot-path zero-copy/cached view rules are implemented or explicitly documented for each hot contract.
- Detailed verification fixtures cover positive, negative, D6, mutation, source, and semantic-identity cases.
- Single-position debug payload exists and can identify whether a failure belongs to engine replay, legal table construction, D6 transformation, or contract validation.
- Rust/Python D6 parity passes.
- Rust legal parity passes.
- History encode/decode parity passes for golden histories.
- Malformed FFI byte, invalid protocol row-width, direct `_engine` import, and Rust invariant-probe tests pass.
- Contract validation, equality, hash, source, and import-purity tests pass.
- `rg` audits find no production private legal/history/D6 parser or fallback owners.
- Fixture-only fallbacks cannot be imported by production runtime paths.

If any hard gate fails, Phase 01 is not complete.
