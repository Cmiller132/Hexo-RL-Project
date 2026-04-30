# Phase 01 Exit Gate Report

## Decision

Phase 01 is closed for rows `V2-010` through `V2-016`.

## Gate Results

| Gate | Result | Evidence |
|---|---|---|
| Contracts package exists and is runtime-pure | GO | contract boundary tests |
| Engine package is only runtime Rust boundary | GO | direct `_engine` audit, engine boundary tests |
| `MoveHistory` is compact-history owner | GO | history tests, runtime cutover |
| `LegalActionTable` is legal table owner | GO | legal parity/mutation tests |
| D6 single Python owner | GO | symmetry tests and private helper audit |
| Rust legal rows validate before contract use | GO | `LegalTableProvider`, `decode_legal_bytes`, tests |
| Production Python legal/tensor fallbacks removed | GO | sampler cutover, private helper audit |
| Runtime legal-byte duplicate decoders removed | GO | protocol decode audit |
| Debug/telemetry sample exists | GO | `telemetry_samples/single_position_debug_payload.json` |
| Performance evidence exists for touched hot paths | GO | `performance/phase01_contract_engine_perf.json` |
| CI routing plan exists | GO | `commands/ci_routing_plan.md` |
| Adversarial review completed | GO | `checks/adversarial_review.md` |

## Closing Commands

- `python -m pytest ... -q`: `169 passed`, exit `0`.
- `python -m py_compile ...`: exit `0`.
- import/deletion audits: no production direct `_engine`, private D6/fallback helper, legal-byte decode, or source fallback hits.

## Non-Closing Deep Check Note

A full `python -m pytest Python/tests -q` run was attempted. It left inference-server shared-memory workers and exceeded local control reliability; after cleanup, the Phase 01 focused suite plus engine smoke is the deterministic closing gate. Phase 04/05 own inference/MCTS long-run server checks.
