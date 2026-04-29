# Phase 09 — Deletion Sweep and CI Hardening

## Purpose
Finalize the refactor by removing legacy paths and enforcing architecture invariants in CI.

## Required Deletions and Finalization
- Delete deprecated runtime modules/aliases (`action_contract/` path remnants, `buffer/` runtime path, legacy model/search shims once cutover complete).
- Remove old architecture aliases that preserve deprecated behavior.
- Ensure no production imports rely on compatibility facades.

## CI Policy Gates to Add
- Contract schema stability and validation suite.
- Rust/Python parity suite.
- Inference protocol compatibility suite.
- Registry/capability behavior suite.
- Import hygiene checks (no banned legacy module paths).

## Parallel Subagent Work
- S1: schema/alias removal and migration notes.
- S2: runtime import cleanup and dead path deletion.
- S3: model/search legacy utility removal.
- S4: replay/train/eval legacy path cleanup.
- S5: CI policy jobs + final conformance report.

## Mandatory Tests
- Full test matrix (Rust + Python) green.
- Import graph check for banned modules/paths.
- End-to-end smoke run covering self-play->replay->train->eval->dashboard.
- Rollback drill from final cut tag.

## Exit Criteria
- No compatibility shims in main runtime path.
- CI automatically enforces all architecture invariants.
- Final signoff report confirms complete, spec-compliant delivery.
