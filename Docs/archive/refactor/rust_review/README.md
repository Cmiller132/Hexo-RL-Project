# Rust Review Repository

Date: 2026-04-29

Purpose: Rust review and hardening evidence for the Hexo game engine.

This directory began as a separate two-phase Rust review, but it is now a required evidence annex for the V2 modular refactor. Phase 1 formed hypotheses, Phase 2 verified and fixed the highest-confidence engine issues, and the remaining Rust/API/CI/performance items now feed the V2 gates for engine, search, replay, telemetry, and final CI closure.

Rust remains canonical but suspicious: improved Rust behavior is not a reason to skip Python contract validation, stale-token checks, malformed FFI tests, structured error handling, or debug-bundle evidence.

## Reading Order

1. `PHASE_1_HYPOTHESES.md` - consolidated hypotheses and direct issues.
2. `RISK_REGISTER.md` - severity/confidence triage.
3. `PHASE_2_VERIFICATION_PLAN.md` - concrete tests and probes to run next.
4. `STRUCTURE_REFACTOR_IDEAS.md` - maintainability and project-structure improvements.
5. `API_AND_FFI_PROTOCOL_PLAN.md` - stable facade API and Python FFI protocol ownership.
6. `INVARIANTS_AND_BOUNDS_PLAN.md` - rules, tactics, WindowKey, and evaluation-bound invariants.
7. `CI_AND_PERFORMANCE_BUDGET_PLAN.md` - fast CI, deep CI, and performance-budget gates.
8. `IMPLEMENTATION_SEQUENCE_AND_COMPLETENESS_CHECKLIST.md` - sequencing and acceptance checklist.
9. `subagents/` - detailed subagent reports by review area.
10. `evidence/` - command summaries, risky pattern inventory, and public API inventory.

## V2 Crosswalk

| Rust review area | V2 row / phase | Required closure evidence |
| --- | --- | --- |
| Stable facades and root export narrowing | `V2-095`, Phase 09 | public API drift check, no stale tactical names, no root implementation-detail re-exports except documented FFI exception. |
| PyO3 legal/history/pair protocol ownership | `V2-013`, `V2-095`, Phases 01/09 | malformed byte tests, duplicate parser audit, protocol source/version in debug bundles. |
| Tokenized fallible MCTS lifecycle | `V2-056`, Phase 05 | stale root token tests, stale batch token tests, structured `MCTSError` mapping, no panic/tokenless runtime calls. |
| Tactical source of truth | `V2-016`, Phases 01/09 | `TacticalStatus` consumed by contracts/search/debug; no recreated `ThreatStatus` compatibility model. |
| Invariants, WindowKey, eval bounds | `V2-016`, `V2-095`, Phases 01/09 | invariant-hook probes, far-coordinate fixtures, release-mode recompute/oracle tests. |
| CI and performance budgets | `V2-090`, `V2-094`, `V2-095`, Phase 09 | fast Rust/Python gates, deep oracle artifacts, checked-in perf metadata and scheduled comparison gates. |

## Review Semantics

Entries are classified as:

- `Hypothesis`: plausible issue needing targeted verification.
- `Direct issue`: obvious or unambiguous bug/error that still benefits from a regression test.
- `Structure recommendation`: maintainability/code-quality improvement.
- `Question`: behavior requiring an owner decision or clearer documentation.

Speculative risks are not presented as proven defects. Phase 2 should turn each hypothesis into one of: confirmed, disproven, accepted tradeoff, structure-only, or no-action.

## Phase 2 Oracle Assumption

For tactical verification, use a search radius of 3 around existing stones as the practical oracle bound. Players may legally place up to 8 hexes away from existing stones, but tactical wins and blocks should not require placements more than 2 hexes from an existing stone; radius 3 is the conservative bound until the radius-2 claim has a formal proof.

Winning/blocking oracle searches should only consider windows with 4 or more stones. With two placements, a 3-window cannot become a win in the current turn.

## Subagent Split

- `S1_core_rules_hypotheses.md`: board state, move legality, placement phases, undo, `set_position`, candidate sets, win detection, hashes.
- `S2_encoding_ffi_hypotheses.md`: tensor encoding, legal/history bytes, PyO3/numpy boundaries, stale data risks.
- `S3_mcts_search_hypotheses.md`: MCTS/search priors, value signs, virtual loss, rerooting, pair priors, extraction.
- `S4_eval_threats_hypotheses.md`: incremental eval, hot windows, threats, oracle coverage.
- `S5_structure_tests_perf_hypotheses.md`: crate boundaries, public APIs, CI, benches, panic policy, docs.
