# Phase 01 — Contracts Foundation

## Purpose
Create canonical, versioned Python contracts that replace implicit cross-subsystem assumptions.

## Required V2 Contract Set
Implement `Python/src/hexorl/contracts/` with owners and invariants:
- `identity.py`
- `history.py` (`MoveHistory` + compact-history decoding invariants)
- `symmetry.py` (D6 transform APIs + compose/inverse/mass preservation)
- `legal.py` (`LegalActionTable` and ordering/hash/source invariants)
- `candidates.py` (`CandidateTable` + recall/diagnostics/missing mass)
- `pairs.py` (`PairActionTable` for phase-specific pair semantics)
- `replay.py`, `targets.py`, `tactical.py`, `telemetry.py`, `validation.py`, `debug.py`

## Critical Invariants From V2
- Contracts are plain typed data, versioned, comparable/hashable.
- No imports of model/inference/search/train/dashboard orchestration into contracts.
- Hot-path rule: contract APIs can support zero-copy/cached views.
- `MoveHistory` is single compact-history owner; invalid histories rejected at decode.

## Parallel Subagent Work
- S1: dataclasses/types/version policy/validation and hash methods.
- S2: adapters at call boundaries that instantiate contracts (no cutover yet).
- S3: model/search requirements mapped to contract fields.
- S4: replay/training fixture payloads migrated to contracts.
- S5: contract-focused tests and docs.

## Mandatory Tests
- Validation failure tests for each contract.
- Equality/hash stability tests.
- D6 unit invariants: composition, inverse, mass preservation.
- Contract purity test: no forbidden dependency imports.

## Exit Criteria
- Contracts package complete and tested.
- No runtime subsystem-specific private contract clones introduced.
