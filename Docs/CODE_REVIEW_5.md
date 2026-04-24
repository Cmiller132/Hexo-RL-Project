# Code Review 5 — Hexgame Rust Foundation

**Scope:** Completeness check on [CODE_REVIEW_4.md](CODE_REVIEW_4.md). Verifies every CR4 finding against current source, corrects stale line numbers and inaccurate resolution calls, and flags new issues introduced since CR4 was written.

**Build status:**
- `cargo build --release` → clean, 0 warnings.
- `cargo test --release` → **121 passed, 0 failed, 6 ignored**.
- `cargo clippy --release` → **0 warnings**.

**Line counts (vs CR4 table — note increases since CR4 was written):**
| file | CR4 stated | actual now | plan target |
|------|-----------|-----------|-------------|
| core.rs | 305 | 317 | 250 |
| eval/state.rs | 424 | 459 | 400 |
| board.rs | 777 | 787 | 700 |
| threats.rs | 435 | 446 | 350 |
| encoder.rs | 478 | 498 | 280 |
| search.rs | 1201 | 1228 | 1000 |
| mcts.rs | 976 | 999 | 800 |
| pybridge/mod.rs | 726 | 759 | — |
| tests/threats.rs | 759 | 856 | — |
| tests/oracle.rs | 408 | 455 | — |

Most growth is additional tests and doc-comment expansion — not raw logic. No regressions in module structure.

---

## CR4 Resolution Audit

### Resolved since CR4

| CR4 ID | Finding | Verified at |
|--------|---------|-------------|
| H1 | `generate_threat_turns` is `pub` | ✅ RESOLVED — now `pub(crate)` at [src/threats.rs:354](src/threats.rs#L354) |
| H2 | `quiesce` allocates `Vec<Turn>` per node | ✅ RESOLVED — `ss.scratch_turns` on `SearchState` ([src/search.rs:172](src/search.rs#L172)); reused at [src/search.rs:726–731](src/search.rs#L726) |
| H4 | `generate_threat_turns` allocates `opp_buf`/`my_buf` internally | ✅ RESOLVED — signature now takes `opp_buf: &mut Vec<Hex>`, `my_buf: &mut Vec<Hex>` ([src/threats.rs:356–358](src/threats.rs#L356)); caller (`SearchState`) owns them |
| H5 | Dead conditional after `unmake_turn` | ✅ RESOLVED — `return Ok(score)` at [src/search.rs:717](src/search.rs#L717) with no post-unmake winner check |
| L4 | `src/pybridge.rs.bak` orphan file | ✅ RESOLVED — file deleted |

### Partially resolved since CR4

| CR4 ID | Finding | Status |
|--------|---------|--------|
| H3 | `encode_board_into` allocates `hot_buf` + `Vec<Hex>` return per MCTS leaf | ⚠️ PARTIAL — `hot_buf` is now a reusable field on `MCTSEngine` ([src/mcts.rs:267](src/mcts.rs#L267), passed at [src/mcts.rs:500](src/mcts.rs#L500)). The **legal-moves `Vec<Hex>` return** is still allocated per leaf: [src/encoder.rs:143](src/encoder.rs#L143) returns `Vec<Hex>`; stored as `legal_moves: Vec<Hex>` in `PendingLeaf` ([src/mcts.rs:169](src/mcts.rs#L169), set at [src/mcts.rs:511](src/mcts.rs#L511)). One heap allocation per MCTS leaf expansion remains. |
| H6 | All oracle proptests `#[ignore]`; zero CI coverage | ⚠️ PARTIAL — Medium tests (25 cases each, not `#[ignore]`) added at [src/tests/threats.rs:768–856](src/tests/threats.rs#L768); smoke tests (10 cases) still at line 595. Heavy 500-case tests remain `#[ignore]`. CI now runs 35+ oracle cases (better than CR4's 10), but the plan §11 requirement of 1000+ non-ignored cases is still not met. |

### Still open from CR4

All remaining CR4 items are reproduced verbatim in the consolidated list below.

---

## New Findings

### H-NEW1 — `ts_opt` is incorrectly threaded: recursive quiesce calls receive a stale parent `ThreatStatus`
**File:** [src/search.rs:748](src/search.rs#L748)

```rust
// After make_turn(game, t), game state has changed.
-quiesce(game, ss, -beta, -alpha, qdepth - 1, ply + 1, Some(&ts))?
//                                                      ^^^^^^^^^^
//                                              ts was computed for the PARENT position
```

`ts` at line 721 is `threat_status(game)` for the **current (parent)** position. After `make_turn(game, t)` at line 740, the game is at a **new (child)** position. The recursive `quiesce` at line 748 receives `Some(&ts)` — the parent's status — and uses it directly at line 721 (`ts_opt.cloned().unwrap_or_else(...)`) to skip recomputing.

Consequence: the child uses the parent's `ThreatStatus` for:
1. **Unblockable early-return** at line 722 — if the child is actually `Unblockable` (our blocking turn opened a new 2-window threat) but the parent was `MustBlock`, we skip the cut and waste tree depth.
2. Nothing else (move generation at lines 726–732 uses the live game state, not `ts`).

This is not a **correctness** bug (the score is eventually correct) but a **performance** regression that defeats the purpose of `ts_opt` and introduces a fragile assumption (parent ts ≈ child ts) that will misfire in tactical positions.

The winning-turn branch at line 712 correctly passes `None` (game state after placing the win is a different position). The threat-turns branch at 748 should do the same.

**Fix:** Change line 748 to `…quiesce(game, ss, -beta, -alpha, qdepth - 1, ply + 1, None)?`. The original intent of `ts_opt` (CR4 M3) was to pass `alphabeta`'s pre-computed ts to the **initial** quiesce call for the *same* position — that call at [src/search.rs:811](src/search.rs#L811) still passes `None`, so M3 remains unresolved regardless.

**Severity: HIGH** — introduced by the `ts_opt` optimization; silently degrades quiescence accuracy in multi-threat positions without any failing test.

---

### M-NEW1 — CR4 M3 (`alphabeta` → `quiesce` duplicate `threat_status`) remains unresolved
**File:** [src/search.rs:811](src/search.rs#L811)

CR4 marked M3 as unresolved; it's still open. `alphabeta` at line 830 computes `threat_status` for move ordering/filtering. When it calls `quiesce` at line 811, it passes `None`, so `quiesce` recomputes `threat_status` at line 721 for the same position. The `ts_opt` parameter exists on `quiesce` but the one call site that would benefit from it (line 811) doesn't use it.

**Fix:** Change line 811 to pass the already-computed `ts` from `alphabeta`.

---

### M-NEW2 — `generate_inner_turns` allocates `Vec<Turn>` on every alphabeta node
**File:** [src/search.rs:557–579](src/search.rs#L557)

`generate_inner_turns` returns `Vec<Turn>` (line 562). It is called on every alphabeta node. CR4's hot-path audit table listed this as "medium priority" but it was not given a finding number. For a search spending 80% of time in alphabeta, this is a constant allocation tax.

**Fix:** Add `scratch_inner: Vec<Turn>` to `SearchState`, change `generate_inner_turns` to take `out: &mut Vec<Turn>`, clear-and-fill on each call. Mirrors the pattern already applied to `quiesce`.

---

### L-NEW1 — `EncodedBoard.legal_moves` is `pub` — exposes heap-owned field to external mutation
**File:** [src/encoder.rs:51](src/encoder.rs#L51)

`pub legal_moves: Vec<Hex>` gives any caller the ability to call `.push()`, `.clear()`, or `.drain()` on the field. Plan §6.1 says "no more 12+ pub fields". This is the one remaining `pub` field on a public-facing struct. Since `encode_board` is only called from `pybridge/mod.rs` (Python interface), making `legal_moves` accessible via `fn legal_moves(&self) -> &[Hex]` would be sufficient.

**Fix:** `legal_moves: Vec<Hex>` → private field + `pub fn legal_moves(&self) -> &[Hex]`.

---

### L-NEW2 — `scratch_turns`, `scratch_opp`, `scratch_my` on `SearchState` are `pub(crate)` instead of private
**File:** [src/search.rs:172–174](src/search.rs#L172)

These scratch buffers should be implementation details of `SearchState`. `pub(crate)` allows test code in other `src/tests/` modules to poke at them. No test does so currently, but it widens the API surface unnecessarily.

**Fix:** Remove `pub(crate)` from all three fields.

---

### L-NEW3 — Medium oracle tests (25 cases) don't cover `turn_satisfies_status`
**File:** [src/tests/threats.rs:768–856](src/tests/threats.rs#L768)

Two medium tests exist: `threat_status_matches_oracle_medium` and `live_cells_matches_oracle_medium`. The third heavy test — `turn_satisfies_threats_matches_oracle` — has no medium equivalent. A regression in `turn_satisfies_status` would only be caught by the `#[ignore]`-gated heavy tests.

**Fix:** Add `turn_satisfies_status_matches_oracle_medium` (25 cases, not ignored) mirroring the heavy variant.

---

## Consolidated Open Findings (from CR4, still unresolved)

### HIGH

**H6 — Oracle heavy proptests all `#[ignore]`** *(CR4 H6, partially improved)*
[src/tests/threats.rs:223, 259, 355, 427, 454, 536](src/tests/threats.rs#L223) — 6 tests, 500 cases each, all ignored. CI now gets 35 oracle cases (medium + smoke). Plan §11 requires 1000+ non-ignored. Progress made; requirement not yet met. Add 500-case non-ignored test or set `cargo test -- --ignored` in CI.

---

### MEDIUM

**M1 — Duplicate window-scanning between `window_empties` and `live_cells`** *(CR4 M1)*
[src/threats.rs:138–147](src/threats.rs#L138) vs [src/threats.rs:440–452](src/threats.rs#L440). `live_cells` inlines the same 6-cell scan instead of calling `window_empties`.

**M2 — `generate_threat_turns` has no oracle-based correctness test** *(CR4 M2)*
[src/threats.rs:354](src/threats.rs#L354). Drives quiescence move generation; zero oracle coverage.

**M3 — `alphabeta` → `quiesce` duplicate `threat_status` computation** *(CR4 M3 + M-NEW1)*
[src/search.rs:811](src/search.rs#L811) passes `None`; `quiesce` recomputes at line 721 for the same position.

**M4 — `BlockConstraint` accessor inconsistency within `threats.rs`** *(CR4 M4)*
[src/threats.rs:270, 293, 313, 319, 324](src/threats.rs#L270) — internal code bypasses `cells()`/`pairs()` accessors.

**M5 — `win_grid_in_bounds` plan/spec divergence** *(CR4 M5)*
[src/eval/grid.rs:79–92](src/eval/grid.rs#L79) comment says must remain runtime; plan §3.2 says remove. One of them is wrong.

**M6 — Test code triplication in `tests/threats.rs`** *(CR4 M6)*
Three test groups (smoke / medium / heavy) share identical game-generation and oracle-analysis logic. Extract shared helper.

**M7 — `extract_tree_node_states` allocates per-candidate move-history clone** *(CR4 M7)*
[src/mcts.rs:811–817](src/mcts.rs#L811). 128 candidates × ~20-move histories = ~2560 tuple allocations per extraction.

**M8 — `visit_windows` callback passes flat `dir_idx`, caller re-derives `dir`/`off`** *(CR4 M8)*
[src/eval/state.rs:228–229](src/eval/state.rs#L228). 18 integer divides/mods per place/unplace.

---

### LOW

**L1 — `WindowKey::cell_at` has no production callers** *(CR4 L1)*
[src/core.rs:276](src/core.rs#L276). Only called from `tests/core.rs`. Gate with `#[cfg(test)]` or delete.

**L2 — `HotWindows::clear()` has no production callers** *(CR4 L2)*
[src/eval/hot.rs:113](src/eval/hot.rs#L113). Only called from `tests/hot.rs:41`. Gate or delete.

**L3 — `EvalState` derives `Clone`, copying ~22KB `Box<[u16]>`** *(CR4 L3)*
[src/eval/state.rs:113](src/eval/state.rs#L113). Acceptable now; dangerous if cloned in MCTS expansion. Add cost comment.

**L5 — Broken doc-comment in `pybridge/mod.rs`** *(CR4 L5 — not re-verified; check current state)*
[src/pybridge/mod.rs](src/pybridge/mod.rs) — `PyMCTSEngine` Python doc block may have unclosed ` ```python ` fence. Re-verify after pybridge growth from 726→759 lines.

**L6 — `encoder.rs` doc claims zero-allocation but `encode_board_into` returns `Vec<Hex>`** *(CR4 L6)*
[src/encoder.rs:133–136](src/encoder.rs#L133). Doc says "caller owns and reuses" but the Vec is created inside the function and returned. Docs and code disagree.

**L7 — `.unwrap()` in oracle test helpers** *(CR4 L7)*
[src/tests/oracle.rs:310, 326, 341, 357, 375, 392](src/tests/oracle.rs#L310). Use `.expect("oracle: set_position context")`.

**L8 — No Rust-level MCTS unit tests** *(CR4 L8)*
`select_child_puct`, `expand_and_backprop`, `re_root`, `gather_policy` tested only via Python integration.

**L9 — `lib.rs` 74 lines vs plan 40** *(CR4 L9)*
Excess is doc-comment prose. Low impact.

**L10 — `EvalState::place`/`unplace` lack `#[inline]`** *(CR4 L10)*
[src/eval/state.rs:211, 262](src/eval/state.rs#L211). All smaller helpers annotated; the two hottest functions are not.

---

### NIT

**N1 — Narrating 3-line comment in `eval/state.rs` hot loop** *(CR4 N1)*
[src/eval/state.rs:221–223](src/eval/state.rs#L221). Mirrors grid.rs docs; redundant.

**N2 — Missing `debug_assert` before subtraction in `unplace`** *(CR4 N2)*
[src/eval/state.rs:281](src/eval/state.rs#L281). `new_idx - cell_val * POW3[off]` could produce a wrap; add `debug_assert!(new_idx >= cell_val * POW3[off])`.

**N3 — Empty "Eval helpers" section header in `board.rs`** *(CR4 N3)*
[src/board.rs:755–757](src/board.rs#L755). No content under the header; remove.

**N4 — `threats.rs` module doc says "three public functions"** *(CR4 N4)*
[src/threats.rs:1–18](src/threats.rs#L1). Now has four (one `pub`, one `pub(crate)`, two `pub`). Update after H-NEW1 fix.

---

## Hot-Path Allocation Audit (updated)

| Location | Alloc | Hot? | Status |
|----------|-------|------|--------|
| `eval/state.rs` place/unplace | 0 | ✅ yes | ZERO-ALLOC ✓ |
| `threats.rs` threat_status | 0 (SmallVec) | ✅ yes | ZERO-ALLOC ✓ |
| `threats.rs` generate_threat_turns | 0 (caller scratch) | ✅ yes (quiesce) | ZERO-ALLOC ✓ |
| `search.rs` quiesce | 0 (scratch_turns/opp/my) | ✅ yes | ZERO-ALLOC ✓ |
| `encoder.rs` encode_board_into | `Vec<Hex>` legal return | ✅ yes (MCTS leaf) | **ALLOCATES** (H3 partial) |
| `mcts.rs` gather_policy | 0 (scratch_raw/priors) | ✅ yes | ZERO-ALLOC ✓ |
| `mcts.rs` select_leaves | 0 (batch_buf) | ✅ yes | ZERO-ALLOC ✓ |
| `search.rs` generate_inner_turns | `Vec<Turn>` per node | ✅ yes (alphabeta) | **ALLOCATES** (M-NEW2) |
| `mcts.rs` extract_tree_node_states | per-candidate clone | ❌ training only | ALLOCATES (M7, acceptable) |

---

## Severity Summary

- **CRITICAL:** 0.
- **HIGH (2 open + 1 new):** H6 (oracle CI coverage); H-NEW1 (`ts_opt` stale in recursive quiesce).
- **MEDIUM (8 open + 2 new):** M1–M8 from CR4; M-NEW1 (M3 still open); M-NEW2 (`generate_inner_turns` per-node alloc).
- **LOW (9 open + 3 new):** L1–L10 from CR4; L-NEW1 (`EncodedBoard.legal_moves` pub field); L-NEW2 (`scratch_*` fields pub(crate)); L-NEW3 (missing medium test for `turn_satisfies_status`).
- **NIT (4):** N1–N4 from CR4 — unchanged.

**Total open: 19 findings** (5 new since CR4, 14 carried over). H3 and H6 partially resolved; H1/H2/H4/H5/L4 fully resolved. The engine remains functionally correct and clippy-clean. The single new correctness-adjacent concern is H-NEW1 (`ts_opt` stale threading) which degrades quiescence precision without producing wrong scores.
