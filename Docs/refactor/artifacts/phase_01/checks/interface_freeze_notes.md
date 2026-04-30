# Phase 01 Interface Freeze Notes

- Created: `2026-04-30T03:35:54Z`
- Git SHA: `4055227a880a3f6995bc2d18e30f61a11b4a7ef4`
- Scope: pre-implementation interface freeze notes only.

## Frozen Package Intent

`Python/src/hexorl/contracts/` is the Phase 01 pure contract package. It must expose versioned data/value objects, validation errors, debug payloads, and mutation-safe views without importing runtime subsystems.

`Python/src/hexorl/engine/` is the Phase 01 Python-facing Rust rules boundary. It must be the only runtime package that imports the compiled `_engine` module directly.

These notes freeze intended boundaries, not implementation completion.

## Contracts Package Freeze

Required modules from `PHASE_01.md`:

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

Required public objects and owners:

- `MoveHistory`: `contracts/history.py`
- `LegalActionTable`: `contracts/legal.py`
- D6 transform APIs: `contracts/symmetry.py`
- `PolicyTarget`, `PairPolicyTarget`, and related target payloads: `contracts/targets.py`
- `CandidateTable` and diagnostics types: `contracts/candidates.py`
- `PairActionTable`: `contracts/pairs.py`
- `ContractTrace` and contract telemetry payloads: `contracts/telemetry.py`
- Validation failures and boundary assertions: `contracts/validation.py`
- Debug/inspection payloads: `contracts/debug.py`

Import freeze:

- Allowed: Python standard library, typing/dataclasses/collections, NumPy/Torch only where required for typed payloads or views, and sibling `contracts` modules.
- Forbidden: `hexorl.model`, `hexorl.models` runtime builders, `hexorl.inference`, `hexorl.search`, `hexorl.train`, `hexorl.dashboard`, `hexorl.tuning`, `hexorl.selfplay` orchestration, process lifecycle, IPC, worker launch, checkpoint runtime, and direct `_engine`.
- Contract imports must be mechanically auditable by an import-purity test and code-search audit.

Behavior freeze:

- Contracts validate at construction or decode boundaries.
- Contracts carry schema/version identity where interpretation could change.
- Externally produced data carries explicit source identity.
- Hashes are deterministic and content based.
- Hot-path views are copied, read-only, or guarded so mutation cannot silently invalidate identity.
- Contracts must not own inference, search, dashboards, orchestration, or private rule reconstruction.

## Engine Package Freeze

Required modules from `PHASE_01.md`:

- `__init__.py`
- `rust.py`
- `legal.py`
- `history.py`
- `encoding.py`
- `parity.py`

Module responsibilities:

- `engine/rust.py`: owns direct `_engine` import, capability probing, structured import errors, and fixture/test opt-in behavior.
- `engine/legal.py`: exposes production `LegalTableProvider` and validates Rust legal payloads into `LegalActionTable`.
- `engine/history.py`: owns Rust replay/history parity helpers and validates compact history into `MoveHistory`.
- `engine/encoding.py`: owns Python-facing encode/decode calls over Rust representation.
- `engine/parity.py`: provides shared parity harness helpers for tests.
- `engine/__init__.py`: exports only stable boundary APIs, not raw `_engine` objects by default.

Protocol freeze:

- `crates/hexgame-py/src/protocol.rs` remains the canonical owner for legal rows, compact history rows, board-piece rows, and pair rows.
- Python engine wrappers must not create a second byte layout, duplicate row-width interpretation, or ad hoc parser for Rust FFI bytes.
- All Rust-exposed payloads used by contracts must validate semantically, not only by shape or row count.

Runtime source freeze:

- Production legal rows come from Rust through `engine/legal.py`.
- Direct runtime `_engine` imports outside `Python/src/hexorl/engine/` are banned after implementation.
- Production fallback from Rust legal rows to Python legal rows is banned.
- Fixture-only fallback must be local to tests or fixture builders and must use `source="fixture"`.
- `source="fallback"` is not a valid production source.

## Required Review Checks Before Closing Rows

- Import-purity test for `contracts/`.
- Direct `_engine` runtime import audit.
- Duplicate byte parser audit against `crates/hexgame-py/src/protocol.rs`.
- Source enforcement tests for Rust, fixture, and fallback labels.
- Mutation-safety tests for all exposed ndarray/Torch/cached views.
- Debug payload sample tying errors to engine replay, legal table construction, D6 transformation, or contract validation.

## Non-Claim Statement

This freeze note does not claim that the packages, modules, imports, tests, or runtime consumers already comply. It records the Phase 01 target boundary for implementation and review.
