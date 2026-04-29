# Phase 02 — Engine Boundary and Rust/Python Parity

## Purpose
Make Rust the authoritative source for legal state and replay legality through a strict Python boundary.

## Target Modules
Create `Python/src/hexorl/engine/`:
- `rust.py` (PyO3 bridge calls)
- `legal.py`
- `history.py`
- `encoding.py`
- `parity.py`

## V2 Requirements to Implement
- Production legal rows originate from Rust boundary only.
- Python legal fallback allowed only for explicit test fixtures (`source='fixture'`).
- Replay/history decode parity against Rust golden corpora.
- Telemetry must expose degraded/fallback source usage.

## Parallel Subagent Work
- S1: parity contract schema definitions and mismatch taxonomy.
- S2: engine API implementation and runtime integration.
- S3: update search/model prep paths to use engine legal tables.
- S4: replay ingestion parity and decode validation.
- S5: parity harness automation and artifact publishing.

## Mandatory Tests
- Golden corpus: legal row parity (ordering, radius, occupied count, hash).
- Golden corpus: compact-history decode parity.
- Negative tests for invalid history ordering/player/duplicate cells.
- Guardrail test that production mode cannot use Python fallback.

## Exit Criteria
- Zero critical parity mismatches.
- Runtime paths consume engine boundary for legal/history in production.
