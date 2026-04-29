# Rust Review Phase 2 Verification Report

Date: 2026-04-29

## Summary

Phase 2 confirmed several Phase 1 defects and fixed the highest-confidence issues that could be safely completed in this pass.

Completed fixes:

- `set_position` is now transactional on every validation failure class.
- `set_position` semantics are documented as a synthetic fixture board loader, not chronological proof.
- `set_position` now rejects post-terminal extra stones.
- repeated `select_leaves` rolls back abandoned virtual loss before selecting a new batch.
- unordered root pair-prior rows now reject reversed duplicate pairs.
- Python MCTS root expansion now validates `legal_bytes` and tensor offsets against the latest `init_root`.
- Python dense root/leaf policy inputs now reject non-finite logits before entering Rust MCTS.
- CI Python integration now builds `hexgame-py` through the crate manifest and activates a venv before `maturin develop`.
- `hexgame-py` declares the `python` feature requested by CI.
- `PLACEMENT_RADIUS` rustdoc now matches the implemented rule.

## Verification Commands

Passed:

- `cargo test --workspace`
- `cargo test --workspace --release`
- `cargo clippy --workspace --release -- -D warnings`
- WSL Python integration:
  - `maturin develop --manifest-path crates/hexgame-py/Cargo.toml --features python`
  - `pytest Python/tests/test_engine_smoke.py -v`

Timed out / not suitable for PR CI:

- `cargo test --workspace --release -- --ignored`
  - exceeded 15 minutes locally and left long-running oracle test processes that had to be stopped.
  - keep this as a nightly/deep verification gate unless the ignored oracle tests are reduced or sharded.

## Phase 1 Item Status

| ID | Status | Result |
|---|---|---|
| D1 | Confirmed and fixed | `set_position` reset-before-validation was real. Added transactionality and regression tests. |
| D2 | Confirmed and fixed | repeated selection leaked virtual loss. Added rollback and regression test. |
| D3 | Confirmed and partially fixed | Python root expansion trusted stale legal rows/offsets. Added snapshot validation for dense/sparse/global root expansion. |
| D4 | Confirmed and fixed | CI requested an undeclared `python` feature and an ambiguous workspace-root maturin build. Added feature and manifest-path venv workflow. |
| D5 | Confirmed and fixed | reversed unordered pair rows were accepted. Added duplicate rejection and test. |
| D6 | Confirmed and fixed | placement radius docs contradicted behavior. |
| H1 | Partially resolved | `set_position` is explicitly a synthetic board loader; post-terminal extra stones are rejected. Chronological loading should become a separate API. |
| H2 | Partially verified | Added candidate-set property test against brute-force scans after place/unplace. |
| H3 | Open | Far-coordinate MCTS `i16` truncation still needs boundary tests and then either range enforcement or wider action storage. |
| H4 | Partially resolved | Python dense policy paths reject non-finite root/leaf logits. Core Rust direct APIs still use panic/assert-style contracts. |
| H5 | Open | Batch identity is still implicit. Shape checks exist, but stale batch generation IDs are not implemented. |
| H6 | Open | Finite eval-grid behavior is not fully classified. Needs far-grid tactical fixtures. |
| H7 | Partially satisfied | Existing oracle is strong, but full radius-3 independent scanner remains a planned refactor item. |
| H8 | Accepted current behavior | `ThreatStatus::WinningTurn` remains sufficient-move oriented, not a complete tactical mask. Document this more prominently before reuse as masks. |
| H9 | Open | Search candidate/pair caps still need radius-3 mandatory win/block coverage. |
| H10 | Partially verified | Release-mode fast tests pass; full ignored release oracle suite is too slow for local PR verification. |

## Complete Phase 2 Rust Refactor Plan

1. Board loading API split
   - Keep `set_position` as synthetic fixture loading.
   - Add `load_history` that replays chronological legal moves and rejects impossible turn order.
   - Move fixture constructors into a shared Rust test fixture module.

2. FFI protocol hardening
   - Move byte decoding/encoding into shared helpers for legal rows, pair rows, and histories.
   - Add Python-facing tests for malformed bytes, stale rows, shifted offsets, duplicate rows, and non-finite policies.
   - Add explicit root and batch generation tokens to prevent stale async submissions.

3. MCTS action-coordinate contract
   - Add far-coordinate tests around `i16::MIN`, `i16::MAX`, and `WindowKey` 15-bit bounds.
   - Decide between enforcing a public coordinate bound or widening node action storage to `i32`.
   - Make extraction/re-root/sample errors fallible wherever Python can trigger them.

4. Tactical oracle and search caps
   - Implement an independent radius-3 tactical scanner that does not depend on `CandidateSet`, `EvalState`, live cells, or hot windows.
   - Limit oracle tactical win/block checks to 4+ windows.
   - Add mandatory win/block cap tests for alpha-beta and MCTS root candidates.

5. Eval-grid boundary classification
   - Add far-grid 4/5/6-in-row fixtures.
   - Classify the finite eval grid as either a documented bounded approximation or a defect requiring full-board fallback for threats.
   - Keep release-mode differential tests for incremental eval/hot-window drift.

6. Public API narrowing
   - Inventory current public deep module paths.
   - Re-export stable rules/encoding/search/MCTS facades from the crate root.
   - Make implementation modules private or `pub(crate)` where downstream callers do not need internals.

7. CI split
   - Keep `cargo test --workspace`, `cargo test --workspace --release`, clippy, and Python smoke in PR CI.
   - Move ignored oracle tests to nightly or shard them by test name with longer timeouts.
   - Record timing budgets for candidate generation, encoding, threat status, MCTS selection/backprop, and tree extraction.
