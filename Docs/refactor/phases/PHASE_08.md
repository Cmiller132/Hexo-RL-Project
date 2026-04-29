# Phase 08 — Dashboard and Debug Convergence

## Purpose
Ensure dashboard and debugging surfaces inspect the same contracts as runtime/training.

## Target Modules
- `dashboard/contract_inspector.py`, `model_inspector.py`, `graph_inspector.py`, `replay_views.py`
- existing dashboard services/routes updated to consume canonical contracts

## V2 Requirements
- Dashboard must not privately rebuild legal/candidate/pair/graph inputs.
- Replay inspection must display contract hashes/source/version for traceability.
- Debug UX should localize mismatches to one builder/adapter quickly.

## Parallel Subagent Work
- S1: inspector contract schemas and display metadata set.
- S2: backend route migration to canonical projection providers.
- S3: model/graph introspection alignment with new registry/adapters.
- S4: eval/debug tool alignment with replay contract shapes.
- S5: fixture-driven dashboard correctness tests.

## Mandatory Tests
- Dashboard fixture parity tests with sampler/trainer contract views.
- Route-level integration tests for replay and model inspectors.
- Contract hash/source/version display assertions.
- Smoke test for dashboard startup and inspection workflows.

## Exit Criteria
- Dashboard reconstructors removed from migrated scope.
- Debug outputs match runtime/training contract facts.
