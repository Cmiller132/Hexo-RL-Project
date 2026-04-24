# Code Review 4 — Hexgame Rust Foundation

**Scope:** Independent re-review of the main worktree using 5 parallel subagents (core/eval, board/threats, search/mcts/encoder, pybridge/lib/tests). Every finding re-verified against source. Cross-referenced against `.claude/worktrees/lucid-diffie-634c8a/Docs/RUST_REWRITE_PLAN.md` (plan spec) and `Docs/CODE_REVIEW_3.md` (prior findings).

**Methodology:** Subagents reviewed file groups in parallel. All HIGH-severity findings re-read from source at cited lines. Claims that didn't reproduce were dropped.

**Build status:**
- `cargo build --release` → clean, 0 warnings.
- `cargo test --release` → **121 passed, 0 failed, 6 ignored**.
- `cargo clippy --release --all-targets` → **0 warnings** (was 39 in CR3).

**Line counts (actual vs plan target):**
| file | actual | plan target | delta |
|------|--------|-------------|-------|
| core.rs | 305 | 250 | +22% |
| eval/mod.rs | 23 | — | ✓ |
| eval/grid.rs | 94 | 80 | +18% |
| eval/hot.rs | 124 | 200 | ✓ under |
| eval/patterns.rs | 153 | 150 | ✓ |
| eval/state.rs | 424 | 400 | +6% |
| board.rs | 777 | 700 | +11% |
| threats.rs | 435 | 350 | +24% |
| encoder.rs | 478 | 280 | +71% |
| search.rs | 1201 | 1000 | +20% |
| mcts.rs | 976 | 800 | +22% |
| lib.rs | 74 | 40 | +85% |
| pybridge/mod.rs | 726 | — | — |
| pybridge/mcts.rs | 203 | — | — |
| pybridge total | 929 | 700 | +33% |

---

## Headline Assessment

Strong improvement since CR3. All 39 clippy warnings eliminated. `has_any_threats()` used consistently at all 5 call sites. `gather_policy` scratch buffers implemented correctly. Module boundaries are clean (no eval→board dependency violations). Fields are properly private throughout.

Remaining gaps cluster around:

1. **Allocation discipline on quiescence/encoding hot paths** — `quiesce`, `encode_board_into`, and `generate_threat_turns` still heap-allocate per call on the hottest paths.
2. **Plan spec has 1 unresolved violation** — `threats.rs` exports 4 public functions where the plan demands 3.
3. **Dead code and doc rot** — `WindowKey::cell_at`, `HotWindows::clear()`, `pybridge.rs.bak`, broken doc comment, encoder comment-code mismatch.
4. **Oracle proptests remain CI-invisible** — 6 of 6 heavy tests `#[ignore]`'d; only smoke tests (10 cases) run by default.
5. **Dead conditional branch** in `quiesce` — code checks `game.winner()` after `unmake_turn` where it's always `None`.

**Bottom line:** The engine is functionally correct, layered correctly, and clippy-clean. What remains is *discipline* work: zero-alloc follow-through on remaining hot paths, removing dead code, fixing doc mismatches, and making `generate_threat_turns` `pub(crate)`.

---

## CRITICAL

*(none)*

---

## HIGH

### H1 — Plan spec violation: `threats.rs` exports 4 public functions; plan §2.4 demands 3
**File:** [src/threats.rs:197, 301, 350, 415](src/threats.rs#L197)

Plan §2.4: *"Public API — the only three functions in threats.rs … That's it. Three functions."*

Four `pub fn` items exist — `threat_status`, `turn_satisfies_status`, `generate_threat_turns`, `live_cells`. `generate_threat_turns` is only called by `src/search.rs:710` (same crate). It should be `pub(crate)`.

**Fix:** `s/pub fn generate_threat_turns/pub(crate) fn generate_threat_turns/`

---

### H2 — `quiesce()` allocates `Vec<Turn>` per recursive node on the hottest search path
**File:** [src/search.rs:709](src/search.rs#L709)

```rust
let mut turns = Vec::new();
generate_threat_turns(game, &mut turns);
```

`quiesce` is called recursively from every alpha-beta leaf. Each invocation heap-allocates a fresh `Vec<Turn>`. Plan §7 mandates zero allocation on hot paths. `generate_threat_turns` already follows the caller-buffer pattern (`out: &mut Vec<Turn>`) — but the caller doesn't reuse the buffer.

**Fix:** Add `scratch_turns: Vec<Turn>` to `SearchState`, pass into `quiesce`, clear and reuse.

---

### H3 — `encode_board_into` allocates `Vec<Hex>` (return) + fresh `hot_buf` per call, contradicting its docs
**File:** [src/encoder.rs:127-132, 293-298](src/encoder.rs#L127)

```rust
pub fn encode_board_into(…) -> (i32, i32, Vec<Hex>) { … }  // allocates
let mut hot_buf = Vec::new();  // doc says "single reusable buffer" but created fresh
```

Doc at lines 293-297 claims *"single reusable buffer … avoids per-channel allocations"* — but `hot_buf` is created fresh on line 298. Called per leaf in MCTS `select_leaves` ([src/mcts.rs:487-488](src/mcts.rs#L487)).

**Fix:** Change signature to accept `&mut Vec<Hex>` for hot_buf and `&mut Vec<Hex>` for legal moves output, or update docs to reflect the allocation cost.

---

### H4 — `generate_threat_turns` heap-allocates two internal scratch `Vec`s per call
**File:** [src/threats.rs:355-360](src/threats.rs#L355)

```rust
// Comment: "Reusable scratch buffers — live_cells requires Vec"
let mut opp_buf = Vec::new();   // NOT reusable, created fresh each call
live_cells(game, opp, &mut opp_buf);
let mut my_buf = Vec::new();    // NOT reusable
live_cells(game, player, &mut my_buf);
```

The comment at line 355 says "Reusable scratch buffers" but they're instantiated as fresh `Vec::new()` every call. Two heap allocations per quiescence node on the hottest tactical path.

**Fix:** Add `scratch_opp: Vec<Hex>` and `scratch_my: Vec<Hex>` to `SearchState`, pass into `generate_threat_turns`, clear and reuse across calls.

---

### H5 — Dead conditional branch in `quiesce` after `unmake_turn`
**File:** [src/search.rs:694-700](src/search.rs#L694)

```rust
unmake_turn(game, placed);

return Ok(if over && game.winner() == Some(player) {
    WIN_SCORE - ply as i32    // ← DEAD: game.winner() is always None after unmake_turn
} else {
    score                     // ← always taken; `score` already has the correct value
});
```

After `unmake_turn`, `game.winner()` is `None` (the board returns to a non-terminal pre-turn state). The first branch is structurally unreachable. The `score` variable at line 685 already holds the correct value. The dead branch masks the simpler `return Ok(score);`.

**Fix:** Replace lines 696-700 with `return Ok(score);`.

---

### H6 — Oracle heavy proptests all `#[ignore]`; zero CI coverage of the oracle suite
**File:** [src/tests/threats.rs:217, 253, 349, 421, 448, 530](src/tests/threats.rs#L217)

All 6 heavy oracle proptests (500 cases each = 3000 total) are `#[ignore]`. Only smoke tests (10 cases × 3 groups) run by default. CR3 H1 flagged this — not resolved. Regressions in the threat fast path will not fail CI.

**Fix:** Un-ignore at least one 500-case test or add a `cases: 50` medium variant that runs by default. Add `cargo test -- --ignored` to CI.

---

## MEDIUM

### M1 — Duplicate window-scanning logic between `window_empties` and `live_cells`
**Files:** [src/threats.rs:136-144](src/threats.rs#L136) vs [src/threats.rs:423-432](src/threats.rs#L423)

`window_empties` iterates one `WindowKey`'s 6 cells → `SmallVec<[Hex; 2]>`. `live_cells` inlines an identical loop (iterate dir, loop `WIN_LENGTH`, check `stones().contains_key()`) but pushes into `Vec<Hex>` with dedup. `opponent_threat_windows` correctly calls `window_empties` at line 160; `live_cells` does not. Any change to window scanning must be made in two places.

**Fix:** Make `live_cells` call `window_empties` per hot window, then extend into its output Vec with dedup.

---

### M2 — `generate_threat_turns` has no oracle-based correctness test
**File:** [src/threats.rs:350](src/threats.rs#L350)

`threat_status`, `turn_satisfies_status`, and `live_cells` all have oracle proptest coverage (3000+ cases). `generate_threat_turns` — which drives quiescence move generation — has zero oracle comparisons. A bug in pair generation, cap logic, or missing block cells cannot be caught by the test suite.

**Fix:** Add a proptest comparing `generate_threat_turns` output against oracle enumeration of legal turns restricted to live cells.

---

### M3 — Duplicate `threat_status` computation in `quiesce` when called from `alphabeta`
**File:** [src/search.rs:704](src/search.rs#L704), [src/search.rs:803](src/search.rs#L803)

`alphabeta` computes `threat_status` at line 803. When it calls `-quiesce(game, …)`, the quiescence function recomputes `threat_status` at line 704 despite the caller already having it. Not a correctness issue, but wasted work on the hot path.

**Fix:** Thread `ts: Option<&ThreatStatus>` into `quiesce`. When called from `alphabeta`, pass `Some(&ts)`.

---

### M4 — Inconsistent `BlockConstraint` accessor usage within `threats.rs`
**File:** [src/threats.rs:106,111](src/threats.rs#L106) (definitions); [src/threats.rs:270,293,313,319,324](src/threats.rs#L270) (direct field access)

`cells()` and `pairs()` accessors exist but are never called within `threats.rs`. Internal code uses `bc.cells` / `bc.pairs` directly. External modules (encoder, pybridge, tests) use accessors. If an accessor ever adds logic, internal callers would bypass it.

**Fix:** Normalize to accessors throughout the module, or remove accessors and make fields `pub(crate)`.

---

### M5 — `win_grid_in_bounds` remains a runtime branch on the 18-iteration hot path
**File:** [src/eval/state.rs:224, 273, 406](src/eval/state.rs#L224); [src/eval/grid.rs:79-92](src/eval/grid.rs#L79)

Plan §3.2: *"bounds check becomes a debug_assert! … out-of-bounds is unreachable under game rules. Remove the runtime branch."*

Code at `grid.rs:79-86` explicitly documents this as deliberate: *"This check must remain a runtime guard (not a debug_assert!)"* — window origins can drift beyond radius 30 in long games. The plan and code directly contradict each other.

**Fix:** Update plan §3.2 to document this as a deliberate approximation, or expand the grid radius. Do not leave spec and code disagreeing.

---

### M6 — Test code duplication in `tests/threats.rs` — 3 near-identical proptest groups
**File:** [src/tests/threats.rs:192-758](src/tests/threats.rs#L192)

Three test groups (primary 500-case, offset 500-case, smoke 10-case) contain identical game-generation, oracle-analysis, and assertion logic. The `_b` variants differ only by `seed.wrapping_add(0xFEDC_BA98_7654_3210)`. Smoke tests differ only by `max_moves = 1 + rng.range(5)` vs `40`.

**Fix:** Extract `fn check_threat_props(seed_base: u64, max_moves: usize)` shared helper.

---

### M7 — `extract_tree_node_states` clones full move history per candidate node
**File:** [src/mcts.rs:812-818](src/mcts.rs#L812)

```rust
game.move_history().iter().map(…).collect()
```

For 128 candidates with ~20-move histories, ~2560 tuple allocations. Not on the critical search path (only training data export), so severity medium.

**Fix:** Reuse a scratch `Vec` across candidates, clearing between nodes.

---

### M8 — Redundant `dir`/`off` recomputation in `visit_windows` hot loop
**File:** [src/eval/state.rs:175-183](src/eval/state.rs#L175) → [src/eval/state.rs:228-229](src/eval/state.rs#L228)

`visit_windows` passes a flat `dir_idx = dir * 6 + off`, then the callback re-derives `dir = dir_idx / 6`, `off = dir_idx % 6`. This runs 18 times per `place`/`unplace`. The compiler likely optimizes it, but passing `(sq, sr, dir, off)` directly would be cleaner.

**Fix:** Change `visit_windows` callback signature to `fn(&mut self, i32, i32, u8, u8)`.

---

## LOW

### L1 — `WindowKey::cell_at` is dead code in production
**File:** [src/core.rs:262-265](src/core.rs#L262)

`#[inline(always)] pub fn cell_at(self, offset: i32) -> Hex` — only called from test code (`tests/core.rs:138-159`). Zero production callers.

**Fix:** Remove or gate with `#[cfg(test)]`.

---

### L2 — `HotWindows::clear()` is dead code in production
**File:** [src/eval/hot.rs:112-115](src/eval/hot.rs#L112)

Only called from `tests/hot.rs:41`. Board reset replaces the entire `EvalState` rather than calling `hot.clear()`. Dead code that could mislead maintainers.

**Fix:** Remove or `#[cfg(test)]`-gate.

---

### L3 — `EvalState` `Clone` cost: ~22KB deep copy
**File:** [src/eval/state.rs:113](src/eval/state.rs#L113)

`EvalState` derives `Clone`, copying the entire `Box<[u16; WIN_GRID_TOTAL]>` (~22KB). Currently cloned once per search invocation (acceptable). If ever cloned inside MCTS expansion, becomes a performance problem. No current misuse.

**Fix:** Document the cost with a comment on the `Clone` derive.

---

### L4 — Orphaned `src/pybridge.rs.bak` backup file (38KB)
**File:** `src/pybridge.rs.bak`

A 38,506-byte backup file remains after the `pybridge/` split. Not covered by `.gitignore`.

**Fix:** `rm src/pybridge.rs.bak`

---

### L5 — Broken doc-comment in `pybridge/mod.rs`: unclosed ```python block
**File:** [src/pybridge/mod.rs:615-636](src/pybridge/mod.rs#L615)

The `PyMCTSEngine` doc starts a ` ```python ` block at line 615 but never closes it. "Step 4" at line 631 is empty. Lines 633-636 use `//` instead of `///` syntax.

**Fix:** Close the code block after Step 3, remove or complete Step 4, convert section divider to `///` syntax.

---

### L6 — `encoder.rs` docs say "zero-allocation" but returns `Vec<Hex>`
**File:** [src/encoder.rs:124-132](src/encoder.rs#L124)

The function returns `Vec<Hex>` (heap-allocating), but docs claim zero-allocation semantics. Either the docs or the code is wrong.

**Fix:** Change to caller-owned buffer, or update documentation.

---

### L7 — `.unwrap()` instead of `.expect()` in oracle test helpers
**File:** [src/tests/oracle.rs:310,326,341,357,375,392](src/tests/oracle.rs#L310)

Six `.unwrap()` calls on `set_position()` in test code. Prefer `.expect("oracle: set_position")` for actionable panic messages.

---

### L8 — No Rust-level MCTS unit tests
MCTS correctness validated only through Python integration. No Rust unit tests for `select_child_puct`, `expand_and_backprop`, `re_root`, or `gather_policy`.

---

### L9 — `lib.rs` 74 lines vs plan target ~40
57 of 74 lines are doc comments. Code-only lines: ~17. Within the spirit but exceeds literal target.

---

### L10 — `EvalState::place`/`unplace` inconsistent `#[inline]` annotations
**File:** [src/eval/state.rs:211,262](src/eval/state.rs#L211)

`score()`, `counts()`, `has_threats()`, and all internal helpers carry `#[inline]`. `place` and `unplace` — the two hottest functions — are not annotated. May be intentional (larger functions) but inconsistent with surrounding style.

---

## NIT

### N1 — Narrating comment in hot loop at `eval/state.rs:221-223`
3-line comment inside the 18-iteration closure explaining bounds checks mirrors `grid.rs` module docs. Redundant on hot path.

### N2 — `unplace` lacks explicit `debug_assert` before subtraction
**File:** [src/eval/state.rs:281](src/eval/state.rs#L281)

`let old_idx = new_idx - cell_val * POW3[off]` — if a logic bug mismatches `cell_val`, wrapping could produce a spurious value. `debug_assert!(new_idx >= cell_val * POW3[off])` would fail-fast with a clearer message.

### N3 — Empty "Eval helpers" section header in `board.rs`
**File:** [src/board.rs:755-757](src/board.rs#L755)

```rust
// ── Eval helpers ────────────────────────────────────────────────────
// ── Candidate helpers ───────────────────────────────────────────────
```

No content under "Eval helpers". Remove or consolidate.

### N4 — `threats.rs` module doc still says "three public functions"
**File:** [src/threats.rs:1-18](src/threats.rs#L1)

Lists three functions but the module exports four. Needs updating after H1 is resolved.

---

## Hot-Path Allocation Audit

| Location | Allocation | Hot path? | Status |
|----------|-----------|-----------|--------|
| `eval/state.rs` place/unplace | 0 (SmallVec inline, const arrays) | ✅ yes | ZERO-ALLOC |
| `threats.rs` threat_status | 0 (SmallVec inline) | ✅ yes | ZERO-ALLOC |
| `threats.rs` generate_threat_turns | 2 × `Vec::new()` per call | ✅ yes (quiescence) | **ALLOCATES** (H4) |
| `search.rs` quiesce | `Vec::new()` per node | ✅ yes | **ALLOCATES** (H2) |
| `encoder.rs` encode_board_into | `Vec::new()` hot_buf + Vec<Hex> return | ✅ yes (MCTS leaf) | **ALLOCATES** (H3) |
| `mcts.rs` gather_policy | 0 (scratch buffers on MCTSEngine) | ✅ yes | ZERO-ALLOC ✓ |
| `mcts.rs` select_leaves | 0 (batch_buf reused) | ✅ yes | ZERO-ALLOC ✓ |
| `mcts.rs` extract_tree_node_states | per-candidate clone | ❌ no (training) | ALLOCATES (M7, acceptable) |
| `search.rs` generate_inner_turns | `Vec<Turn>` per node | ✅ yes | ALLOCATES (medium priority) |

---

## CR3 Resolution Verification Table

| CR3 ID | Description | Status | Evidence |
|--------|-------------|--------|----------|
| H1 | Oracle proptests `#[ignore]` | ❌ Still present | 6/6 ignored; smoke tests (10 cases) run in CI |
| H2 | `generate_threat_turns` 4th pub fn, allocates, untested | ❌ Still present | Still `pub`, still allocates (opp_buf/my_buf), still no oracle test |
| H3 | `gather_policy` allocates per expansion | ✅ RESOLVED | `scratch_raw`/`scratch_priors` on MCTSEngine |
| H4 | `win_grid_in_bounds` runtime branch | ❌ Deliberate deviation | Code comment at grid.rs:79-86 says must remain runtime guard |
| H5 | 39 clippy warnings | ✅ RESOLVED | `cargo clippy` → 0 warnings |
| H6 | `has_any_threats` used 1/5 sites | ✅ RESOLVED | Used at threats.rs:203, mcts.rs:640, search.rs:677 consistently |
| M1 | `opponent_threat_windows` re-checks guard | ✅ RESOLVED | Now `debug_assert!` only |
| M2 | `select_leaves` to_vec() buffer copy | ✅ RESOLVED | Returns slice reference, no to_vec() |
| M3 | `threat_constrained_moves` precomputes | ✅ PARTIAL | Only inside WinningTurn/MustBlock arms now |
| M4 | `extract_tree_node_states` DFS depth | ✅ RESOLVED | Stack-based DFS with Frame enum |
| M5 | `assert_invariants` double-gating | ✅ RESOLVED | `#[cfg(debug_assertions)]` on fn and call site |
| M6 | `generate_threat_turns` caller-owned buffer | ✅ PARTIAL | Takes `&mut Vec<Turn>` but internal scratch vecs still allocate |
| M7 | `pybridge.rs` vs plan `py.rs` | ✅ RESOLVED | Now `pybridge/` directory |
| M8 | File size budgets overrun | ❌ Still exceeds | encoder +71%, pybridge +33%, threats +24%, search +20% |
| M9 | Duplicate window-empties loops | ❌ Still present | `window_empties` and `live_cells` inline duplicated logic (M1) |
| M10 | `EvalDelta` stores `player` vs `cell_val` | ✅ RESOLVED | Now stores `cell_val: u8` |
| L1 | `tests/mod.rs` stale doc | ✅ RESOLVED | Updated to list all 10 modules |
| L2 | `any_winning_turn_for` dead code | ✅ RESOLVED | `all_winning_turns_for` is live |
| L3 | `pybridge` legal_moves_near unconditional | ✅ PARTIAL | Only in threat branches now |
| L4 | Clippy test warnings | ✅ RESOLVED | 0 warnings |
| L5 | `core.rs` over-commenting | 🟡 IMPROVED | Trimmed from 325 to 305 lines |
| L6 | Narrating comments | 🟡 PARTIAL | Some remain in hot loop (N1) |
| L7 | `Hex` pub fields | 🟡 DESIGN | Conventional for coordinate types |
| L8 | `Stones` type alias leak | 🟡 DESIGN | `pub(crate)` alias, deliberate transparency |
| L9 | `BlockConstraint` pub fields vs accessors | ❌ Still present | Accessors defined but unused within module (M4) |
| L10 | Encoder/pybridge over budget | ❌ Still present | See M8 |

**Score:** 14 resolved, 5 partial, 6 still present, 2 design choices.

---

## Appendix — Severity Summary

- **CRITICAL:** 0.
- **HIGH (6):** Plan spec violation — 4 pub fns in threats.rs (H1); quiesce() Vec<Turn> per node (H2); encode_board_into Vec<Hex> + hot_buf alloc (H3); generate_threat_turns internal Vec allocs (H4); dead conditional after unmake_turn (H5); oracle proptests all #[ignore] (H6).
- **MEDIUM (8):** Duplicate window-scanning (M1); generate_threat_turns no oracle test (M2); duplicate threat_status computation (M3); BlockConstraint accessor inconsistency (M4); win_grid_in_bounds plan/impl divergence (M5); test code duplication (M6); extract_tree_node_states per-candidate clone (M7); redundant dir/off recomputation (M8).
- **LOW (10):** WindowKey::cell_at dead code (L1); HotWindows::clear() dead code (L2); EvalState Clone cost (L3); pybridge.rs.bak orphan (L4); broken doc-comment (L5); encoder doc-code mismatch (L6); .unwrap()→.expect() (L7); no MCTS unit tests (L8); lib.rs line count (L9); #[inline] inconsistency (L10).
- **NIT (4):** Narration comments, debug_assert before subtraction, empty section header, stale module doc.

**Total: 28 findings.** No correctness bugs. Remaining work is allocation discipline on 3 hot paths, fixing the plan-spec violation (`pub`→`pub(crate)`), removing dead code, fixing the dead conditional branch, and doc reconciliation.
