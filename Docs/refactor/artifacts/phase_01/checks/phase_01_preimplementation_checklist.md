# Phase 01 Pre-Implementation Checklist

- Created: `2026-04-30T03:35:54Z`
- Git SHA: `4055227a880a3f6995bc2d18e30f61a11b4a7ef4`
- Branch: `codex/phase-01-engine-contracts-foundation`
- Scope: artifact and audit support only. No runtime code or tests were edited.
- Source docs: `Docs/refactor/phases/PHASE_01.md`, `Docs/refactor/V2_REQUIREMENTS_MATRIX.md`

## Goal

Prepare Phase 01 artifact support for V2-010 through V2-016 without claiming implementation completion.

## Success Criteria

- [x] Create Phase 01 support artifacts under `Docs/refactor/artifacts/phase_01/`.
- [x] Build a pre-implementation checklist for V2-010 through V2-016.
- [x] Record interface-freeze notes for `contracts/` and `engine/`.
- [x] Record CI routing plan and required command list.
- [x] Prepare import-audit command plan for direct `_engine` imports, private legal/history/D6 helpers, forbidden contracts imports, fixture-only fallbacks, `source="fallback"`, and duplicate byte parsers.
- [x] Avoid edits outside `Docs/refactor/artifacts/phase_01/`.
- [x] Avoid claiming V2 row completion.

## Constraints

- Write scope is limited to `Docs/refactor/artifacts/phase_01/`.
- Runtime code, tests, and source docs are read-only for this setup pass.
- `rg` is preferred, but local execution failed with access denied; use `git grep` or PowerShell fallback commands.
- This artifact is a setup artifact only and cannot close any Phase 01 hard gate.

## Required Evidence

- Command index: `Docs/refactor/artifacts/phase_01/commands/COMMAND_INDEX.md`
- Artifact manifest: `Docs/refactor/artifacts/phase_01/MANIFEST.md`
- Interface freeze notes: `Docs/refactor/artifacts/phase_01/checks/interface_freeze_notes.md`
- CI routing plan: `Docs/refactor/artifacts/phase_01/commands/ci_routing_plan.md`
- Import audit plan: `Docs/refactor/artifacts/phase_01/import_audits/import_audit_command_plan.md`

## Stop Rules

- Stop before editing runtime code or tests.
- Stop if `Docs/refactor/artifacts/phase_01/` is missing.
- Stop if source docs conflict on V2-010 through V2-016 ownership or acceptance criteria.

No stop rule was triggered. The artifact directory exists, and `PHASE_01.md` and `V2_REQUIREMENTS_MATRIX.md` agree that V2-010 through V2-016 are Phase 01 rows.

## V2-010 Checklist: Pure Contracts Package

Requirement: `contracts/` is pure data/value objects and imports no model/inference/search/train/dashboard/tuning orchestration.

- [ ] Create `Python/src/hexorl/contracts/` with all Phase 01 required modules.
- [ ] Define contracts as plain typed data/value objects.
- [ ] Add schema/version identity for every public contract.
- [ ] Add construction or decode-boundary validation.
- [ ] Preserve comparability and hashability where useful.
- [ ] Prevent imports from model, inference, search, train, dashboard, tuning, self-play orchestration, and lifecycle code.
- [ ] Add executable examples or tests for construction from canonical source.
- [ ] Add malformed-input rejection tests for every public contract.
- [ ] Add stable schema/source/hash identity tests.
- [ ] Add mutation-safety tests for exposed views.
- [ ] Add projection example/test into at least one runtime consumer.
- [ ] Add debug payload example for failure explanation.
- [ ] Add import-purity test and audit proof.

## V2-011 Checklist: Engine Boundary

Requirement: `engine/` is the only Python-facing Rust rules boundary and runtime direct `_engine` imports are removed.

- [ ] Create `Python/src/hexorl/engine/` with `__init__.py`, `rust.py`, `legal.py`, `history.py`, `encoding.py`, and `parity.py`.
- [ ] Route production `_engine` imports through `engine/rust.py`.
- [ ] Remove or quarantine runtime direct `_engine` imports outside `Python/src/hexorl/engine/`.
- [ ] Allow direct `_engine` imports only in tests and explicit fixture tooling.
- [ ] Add direct `_engine` import audit.
- [ ] Add engine API tests covering import boundary behavior.
- [ ] Confirm no duplicate Python byte layout or row-width interpretation is created.
- [ ] Use `crates/hexgame-py/src/protocol.rs` as the centralized protocol owner.

## V2-012 Checklist: Rust Legal Rows And Fallback Removal

Requirement: Rust is production source for legal rows; Python legal fallback is fixture-only, and Rust legal rows are semantically validated before contract use.

- [ ] Implement Rust-backed `LegalTableProvider` in `engine/legal.py`.
- [ ] Implement `LegalActionTable` in `contracts/legal.py`.
- [ ] Include canonical rows, dense indices, source, radius, occupied count, schema/version identifier, and table hash.
- [ ] Require `source="rust"` or specific Rust-provider source for production legal rows.
- [ ] Reject `source="fallback"` in production paths.
- [ ] Allow fixture legal tables only through explicit fixture/test opt-in with `source="fixture"`.
- [ ] Validate duplicate cells, occupied cells, current-player correctness, phase correctness, terminal-state consistency, row ordering, dense-index mapping, and stable source/hash identity.
- [ ] Make degraded or non-production sources telemetry-visible and test-visible.
- [ ] Remove dashboard, sampler, tactical oracle, graph, self-play, training, and evaluation private legal generation from runtime paths.
- [ ] Add parity tests, fallback guard tests, source telemetry sample, and negative legal semantic tests.

## V2-013 Checklist: Compact History Contract

Requirement: Compact history has one Python contract and Rust parity, backed by the centralized PyO3 history byte protocol.

- [ ] Implement `MoveHistory` in `contracts/history.py`.
- [ ] Implement Rust replay/history parity helpers in `engine/history.py`.
- [ ] Decode compact histories through the centralized PyO3 protocol.
- [ ] Reject malformed bytes.
- [ ] Reject invalid player order.
- [ ] Reject duplicate cells.
- [ ] Reject illegal placement counts.
- [ ] Reject invalid radius.
- [ ] Reject inconsistent current-player state.
- [ ] Cross-check final board state, side to move, terminal result, placement counts, and rejected illegal transitions.
- [ ] Ensure bootstrap generation uses the same encoder as runtime paths.
- [ ] Remove runtime private compact-history parsers from graph, sampler, dashboard, tactical oracle, RGSC, epoch bootstrap, and self-play.
- [ ] Add encode/decode parity tests, malformed-byte tests, invalid history tests, and round-trip golden fixtures.

## V2-014 Checklist: D6 Symmetry Ownership

Requirement: D6 transforms live in one Python module and parity-test against Rust for coordinates/history/legal/dense tensors.

- [ ] Implement all required D6 APIs in `contracts/symmetry.py`.
- [ ] Remove private D6 helper imports from runtime paths.
- [ ] Validate composition invariant.
- [ ] Validate inverse invariant.
- [ ] Preserve policy target mass.
- [ ] Preserve pair target mass.
- [ ] Preserve dense policy mass.
- [ ] Match transformed legal-table hashes to transformed legal rows.
- [ ] Preserve unordered first-placement pair identity.
- [ ] Preserve ordered second-placement known-first semantics.
- [ ] Ensure dashboard inputs match sampler inputs after the same symmetry.
- [ ] Add Rust/Python parity tests for coordinates, histories, legal rows, and dense tensors where Rust support exists.
- [ ] Document missing Rust dense tensor D6 surface if not exposed.
- [ ] Add no-in-place-mutation tests for histories, legal tables, targets, tensors, and cached views.

## V2-015 Checklist: Schema, Version, Source, Hash

Requirement: Contract schema/version/source/hash policy is defined and tested for ndarray-backed contracts.

- [ ] Define stable field semantics for every Phase 01 contract.
- [ ] Define validation entry points.
- [ ] Define stable equality behavior where applicable.
- [ ] Define deterministic content-based hashes including schema/version identity where interpretation can change.
- [ ] Ensure legal/history/pair/candidate/debug hashes are safe to log.
- [ ] Require explicit source identity for externally produced data.
- [ ] Enforce production source rules: Rust source required, fixture source explicit, fallback source forbidden.
- [ ] Cache expensive derived views where expected on hot paths.
- [ ] Guard zero-copy or cached ndarray/Torch views against mutation.
- [ ] Add stable hash tests, schema version tests, source label tests, fixture opt-in tests, and mutation safety tests.

## V2-016 Checklist: Semantic Verification And Mutation Safety

Requirement: Engine/legal/history/D6 verification checks semantic identity, row ordering, terminal/current-player state, source/hash/version, Rust invariant hooks, and mutation safety.

- [ ] Add golden histories with hand-audited board state, side to move, terminal status, legal rows, and D6 variants.
- [ ] Add negative histories for malformed bytes, duplicate placements, occupied-cell moves, illegal turn order, stale known-first placement, invalid radius, and impossible terminal states.
- [ ] Add Rust history replay -> Python `MoveHistory` -> Rust replay round-trip checks.
- [ ] Add legal table checks for row ids, row order, dense indices, source, schema version, hash, and coordinate semantics.
- [ ] Add D6 checks for inverse, composition, target mass, legal-row identity, and row ordering after canonicalization.
- [ ] Add mutation tests for every ndarray/Torch/cached/zero-copy view exposed by contracts.
- [ ] Add hash invalidation tests proving mutation is impossible or produces a different validated identity.
- [ ] Add source enforcement tests proving fixture and fallback data cannot enter production paths accidentally.
- [ ] Add FFI boundary logs/debug bundles with method name, trace id, history hash, legal hash, root/batch token when present, row counts, dtype/shape/contiguity, duration, and failure class.
- [ ] Add Rust invariant hook tests for candidate, hash, winner, move-history, eval/hot-window, and undo consistency where exposed.
- [ ] Add single-position debug payload covering history, board state, legal rows, source/hash/version, D6 transforms, and validation outcome.

## Non-Claim Statement

No V2-010 through V2-016 requirement is claimed complete by this artifact. These checklists describe required implementation, verification, audit, telemetry, and documentation work still to be performed.
