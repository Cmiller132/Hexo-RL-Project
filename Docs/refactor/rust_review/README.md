# Rust Review Repository

Date: 2026-04-29

Purpose: Phase 1 of a two-phase Rust review for the Hexo game engine.

This repository is intentionally separate from the V2 refactor plan. Phase 1 does not attempt to prove every issue. It forms hypotheses, records direct issues where the evidence is unambiguous, and defines how Phase 2 should verify or falsify each risk.

## Reading Order

1. `PHASE_1_HYPOTHESES.md` - consolidated hypotheses and direct issues.
2. `RISK_REGISTER.md` - severity/confidence triage.
3. `PHASE_2_VERIFICATION_PLAN.md` - concrete tests and probes to run next.
4. `STRUCTURE_REFACTOR_IDEAS.md` - maintainability and project-structure improvements.
5. `subagents/` - detailed subagent reports by review area.
6. `evidence/` - command summaries, risky pattern inventory, and public API inventory.

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
