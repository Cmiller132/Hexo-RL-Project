# Code Review 3 — Hexgame Rust Foundation

**Scope:** Follow-up review after the round-2 fixes landed. Evaluates the Rust engine against `.claude/worktrees/lucid-diffie-634c8a/Docs/RUST_REWRITE_PLAN.md` (plan spec) and `Docs/CODE_REVIEW_2.md` (prior findings).

**Methodology:** Three Sonnet subagents reviewed file groups in parallel (core/eval/board, threats/encoder/search, mcts/pybridge/tests). Every high-severity finding below was re-verified by reading the source at the cited line. Subagent claims that didn't reproduce were dropped.

**Build status:**
- `cargo build --release` → clean, 0 warnings.
- `cargo test --release` → **118 passed, 0 failed, 6 ignored**.
- `cargo clippy --release --all-targets` → **39 warnings** (see H5 / L4). Previously not surfaced.

**Line counts (actual vs plan target):**
| file | actual | target | delta |
|------|--------|--------|-------|
| core.rs | 325 | 250 | +30% |
| eval/mod.rs | 23 | — | ✓ clean |
| eval/grid.rs | 94 | 80 | +18% |
| eval/hot.rs | 124 | 200 | ✓ |
| eval/patterns.rs | 153 | 150 | ✓ |
| eval/state.rs | 470 | 400 | +18% |
| board.rs | 791 | 700 | +13% |
| threats.rs | 460 | 350 | +31% |
| encoder.rs | 482 | 280 | +72% |
| search.rs | 1217 | 1000 | +22% |
| mcts.rs | 947 | 800 | +18% |
| pybridge.rs | 990 | 700 | +41% |

---

## Headline assessment

Significant progress since CR2. Almost every verified CR2 finding is resolved (see table in appendix). The remaining gaps cluster around three themes:

1. **Quiescence move generation lives in `threats.rs` but is a 4th public function that the plan didn't sanction and the proptest oracle doesn't cover.** Its implementation is the only remaining hot-path heap allocator in the threat layer.
2. **CI actually runs zero oracle property tests** — all six are `#[ignore]`-gated. The plan's 1000-case proptest deliverable is met on paper but not exercised by default.
3. **Linter hygiene hasn't been a gate.** `cargo clippy` surfaces 39 warnings including `&mut Vec` misuse, redundant `Ok(?)` wrappers, and `is_multiple_of` / `is_none` / `is_err` lints. A "strong foundation" doesn't ship with that accumulation.

**Bottom line:** The correctness and layering problems from CR2 are largely resolved. What remains is *discipline* work — zero-alloc follow-through on the quiescence path, re-enabling the oracle tests for CI, a clippy pass, and finishing the plan/impl doc sync. No critical issues remain.

---

## CRITICAL

*(none)*

---

## HIGH

### H1 — Oracle property tests are `#[ignore]`-gated; CI runs zero oracle cases
**Files:** [src/tests/threats.rs:217, 253, 349, 421, 448, 530](src/tests/threats.rs#L217)

All six oracle proptests are marked `#[ignore = "slow oracle: run with cargo test --release -- --ignored"]`. `cases: 500` × 2 `proptest!` blocks = 1000 cases, which nominally satisfies plan §11 ("1000+ proptest cases"). But `cargo test --release` reports **6 ignored**; the plan deliverable only runs if someone remembers `--ignored`. Regressions in `threat_status` / `turn_satisfies_status` / `live_cells` will not fail CI.

**Fix:** Either (a) keep one shrunk-case fast variant un-ignored at `cases: 50` so CI gets signal, or (b) promote the oracle suite to the default run and budget for the wall time. Do not ship both in perpetual `#[ignore]`.

### H2 — `generate_threat_turns` is a 4th public function in `threats.rs`, not covered by the oracle suite, and allocates four `Vec`s per quiescence call
**File:** [src/threats.rs:359–419](src/threats.rs#L359)

Plan §2.4 names three public functions. `generate_threat_turns` is a fourth — fine in principle, but two concrete problems:

1. **Zero-alloc hot path violated.** `opp_threats: Vec::new()`, `my_threats: Vec::new()`, `all_threats: Vec<Hex>`, `turns: Vec::new()` — four heap allocations per call. The function runs on every quiescence node. Plan §7 performance budget mandates zero allocation on hot paths. `threat_status` itself was fixed to `SmallVec` in CR2; this function was moved into `threats.rs` without the same treatment.
2. **Untested against the oracle.** `src/tests/threats.rs` property-checks `threat_status`, `turn_satisfies_status`, `live_cells`. `generate_threat_turns` has no comparison path — a silent divergence (wrong pair cap, missed block cell, spurious move) cannot be caught by the suite. This function drives quiescence, so correctness here matters more than for `live_cells`.

**Fix:** Change signature to `fn generate_threat_turns(game: &HexGameState, out: &mut Vec<Turn>)` following the `live_cells` pattern; replace intermediates with `SmallVec<[Hex; 16]>` / `SmallVec<[Turn; 28]>`. Add a proptest that diffs its output against an oracle enumeration of legal turns restricted to live cells.

### H3 — `gather_policy` allocates `raw: Vec<f64>` and `priors: Vec<f32>` per expansion
**File:** [src/mcts.rs:190, 210](src/mcts.rs#L190)

`expand_node` → `gather_policy` is called once per batch leaf. At batch size 32 that is 64 heap allocations per MCTS step. The comment at [src/mcts.rs:201–204](src/mcts.rs#L201) justifies this as a borrow-checker limitation — "`expand_node` needs both the priors slice and `&mut self.arena` simultaneously." That argument doesn't hold: the function's signature is already `fn gather_policy(moves, policy_logits, offset_q, offset_r)` — it takes no `&mut self`. A scratch buffer stored on `MCTSEngine` can be passed in as `&mut Vec<f64>` / `&mut Vec<f32>` without aliasing the arena.

**Fix:** Add `scratch_raw: Vec<f64>` and `scratch_priors: Vec<f32>` on `MCTSEngine`; pass them into `gather_policy`, `clear()` then `extend` in place. Previous review H12 flagged this; the fix was not applied.

### H4 — `win_grid_in_bounds` is a runtime branch on the 18-iteration hot path, directly contradicting plan §3.2
**File:** [src/eval/state.rs:259, 318, 453](src/eval/state.rs#L259); check implemented at [src/eval/grid.rs:88](src/eval/grid.rs#L88)

Plan §3.2 is explicit: *"bounds check becomes a `debug_assert!` in `compute_delta` … out-of-bounds is unreachable under game rules. Remove the runtime branch in the 18-iteration hot loop."*

The implementation has `if !win_grid_in_bounds(sq, sr) { return; }` inside `visit_windows` at `state.rs:259` (place), `318` (unplace), and `453` (`hypothetical_score_delta`). Worse, the comment at `state.rs:255–258` reads: *"for stones near the grid boundary, some of the 18 windows extend outside the grid; those windows simply don't contribute to evaluation. This is a known approximation, not a bug."*

One of these must be wrong. If the plan is right, the branch is dead and should be a `debug_assert`. If the code comment is right (the grid silently drops boundary windows), the evaluation is a documented approximation and the plan's "unreachable under game rules" claim is false. Currently the user-facing doc and the production code disagree.

**Fix:** Either (a) enlarge the grid so bounds are unreachable and convert the check to `debug_assert!`, or (b) update plan §3.2 to describe this as a deliberate approximation with a bound on how many windows it drops. Don't ship both interpretations.

### H5 — 39 clippy warnings on `cargo clippy --release --all-targets`
**Command:** `cargo clippy --release --all-targets`

A "strong foundation" does not ship with clippy red. Representative lints:
- **`clippy::needless_range_loop`** — [src/tests/patterns.rs:82, 195](src/tests/patterns.rs#L82) (plus several in production files).
- **`clippy::manual_range_contains`** — 15+ occurrences; classic `if x >= 0 && x < N` should use `(0..N).contains(&x)`.
- **`clippy::needless_return` / `clippy::needless_ok` / `clippy::needless_question_mark`** — "enclosing `Ok` and `?` operator are unneeded."
- **`clippy::ptr_arg`** — "writing `&mut Vec` instead of `&mut [_]` involves a new object where a slice will do." Live-cells / generate-threat-turns signatures are the most visible offenders.
- **`clippy::large_enum_variant`** — `ThreatStatus` has a large `MustBlock(BlockConstraint)` variant vs unit variants; boxing may save stack.
- **`clippy::manual_is_multiple_of`**, **`clippy::redundant_pattern_matching`**, **`clippy::type_complexity`** — miscellaneous hygiene.

**Fix:** Run `cargo clippy --release --all-targets --fix` then review the non-autofixable items (type_complexity, large_enum_variant) manually. Add `cargo clippy -- -D warnings` to CI.

### H6 — `has_any_threats` helper exists but 3 of 4 sites still inline the 4-way check
**File:** [src/eval/state.rs:410–414](src/eval/state.rs#L410) defines it; [src/mcts.rs:631](src/mcts.rs#L631) uses it; [src/search.rs:407, 692–696](src/search.rs#L692), [src/threats.rs:136, 444](src/threats.rs#L136) still write `if c.fives() == 0 && c.fours() == 0 { … }` inline.

CR2's M7 was recognized — the helper was added — but the callers were never switched. Five remaining callers will drift independently on any future field addition.

**Fix:** Replace all four inline sites with `if !game.eval().has_any_threats() { … }` (or `has_threats(player)` where single-sided).

---

## MEDIUM

### M1 — `opponent_threat_windows` re-checks "has any threats" after its only caller already guarantees them
**File:** [src/threats.rs:131–138](src/threats.rs#L131)

CR2 M4. `threat_status` at [src/threats.rs:199–205](src/threats.rs#L199) returns `Quiet` when neither side has fours/fives, then calls `opponent_threat_windows` at [src/threats.rs:245](src/threats.rs#L245) — which re-checks the same condition at line 136. Dead branch on the only call path. Still present.

**Fix:** Inline `opponent_threat_windows` into `threat_status` (one caller, 30 lines), or drop the internal guard and document the precondition.

### M2 — `select_leaves` copies the entire batch buffer on every call
**File:** [src/pybridge.rs:773](src/pybridge.rs#L773)

```rust
let tensor_vec = tensors.to_vec();
let arr = ndarray::Array4::from_shape_vec((count, 13, 33, 33), tensor_vec)…
```

`MCTSEngine::batch_buf` is pre-allocated specifically to avoid per-call heap allocation ([src/mcts.rs:257–258](src/mcts.rs#L257)). `.to_vec()` defeats that — at batch size 32 it copies ~450 KB of f32 per step. This is on the MCTS critical path (every forward pass to the net).

**Fix:** Use `ndarray::ArrayView4::from_shape` + `PyArray4::from_array(py, &view)`, or pass the buffer by move and re-allocate `batch_buf` lazily. If numpy 0.24 genuinely requires owned data, document the copy as inherent and measure its cost.

### M3 — `threat_constrained_moves` calls `legal_moves_near` before `threat_status`, discarding it on the fast paths
**File:** [src/pybridge.rs:337–360](src/pybridge.rs#L337)

`legal = self.inner.legal_moves_near(radius)` runs unconditionally. When `threat_status(game)` returns `Quiet` or `Unblockable`, `legal` is thrown away. Under normal play the majority of positions are `Quiet`.

**Fix:** Compute `threat_status` first; only call `legal_moves_near` inside the `WinningTurn` / `MustBlock` arms.

### M4 — `extract_tree_node_states` linear-scans the arena, then uses a manual `cur_depth` tracker that overcounts for multi-child parents
**File:** [src/mcts.rs:733–832](src/mcts.rs#L733)

The arena scan at line 733 walks every slot (up to simulation count). The DFS at line 768 maintains `cur_depth` by incrementing per child push (line 818) and decrementing on `Frame::Unplace` (line 803). For a parent with N children, the counter increments N times but only one `Unplace` frame is pushed per parent — the arithmetic is salvaged by a `debug_assert_eq!(cur_depth, 0, …)` at line 832, which catches the off-by-one in debug builds only. Release builds would produce a wrong depth silently, although the rest of the function only uses `cur_depth` for placement symmetry so the user-visible effect may be zero.

**Fix:** Derive depth from a `parent` chain walk instead of tracking it manually. Simpler, correct by construction.

### M5 — `assert_invariants` is not `#[cfg(debug_assertions)]`-gated at the method level
**File:** [src/eval/state.rs:347–399](src/eval/state.rs#L347)

CR2 M13. The method body is conditionally compiled, but the method itself is a public-ish helper that unconditionally ships in release binaries as an empty function. Callers at [src/eval/state.rs:339](src/eval/state.rs#L339) make a call that becomes a no-op branch.

**Fix:** `#[cfg(debug_assertions)]` on the `fn` declaration; wrap the call site in `#[cfg(debug_assertions)]` and remove the inner gate.

### M6 — `generate_threat_turns` returns `Vec<Turn>` rather than writing into a caller-owned buffer
**File:** [src/threats.rs:359](src/threats.rs#L359)

`live_cells` takes `out: &mut Vec<Hex>` so quiescence can reuse a buffer across depths. `generate_threat_turns` forces a fresh allocation on every tactical node. Inconsistent with sibling API, and wastes the buffer-reuse discipline `live_cells` established.

**Fix:** Signature change to `fn generate_threat_turns(game: &HexGameState, out: &mut Vec<Turn>)`; `quiesce` owns the scratch buffer.

### M7 — `pybridge.rs` named differently from plan spec `py.rs`
**File:** [src/pybridge.rs](src/pybridge.rs); [plan §1](.claude/worktrees/lucid-diffie-634c8a/Docs/RUST_REWRITE_PLAN.md)

CR2 N11. Plan §1 and deliverables checklist list `py.rs`; actual file is `pybridge.rs`. Either rename the file or update the plan.

### M8 — `file_size` budget overruns remain for `pybridge.rs` (+41%), `encoder.rs` (+72%), and `threats.rs` (+31%)
**Files:** see table above

Excess in `encoder.rs` is mostly `extract_features` moved here per CR2 H1 fix (legitimate) plus dense doc comments. `pybridge.rs` excess is `classical_self_play` + `extract_tree_node_states`. `threats.rs` excess is `generate_threat_turns` + comments. None are critical individually, but the pattern says the foundation is 20–50% heavier than planned.

**Fix:** Treat the plan targets as budgets. Consider splitting `pybridge.rs` into `pybridge/classical.rs` + `pybridge/mcts.rs` + `pybridge/mod.rs`.

### M9 — Duplicate "threat cell empties" loop in `threat_status` and `opponent_threat_windows`
**Files:** [src/threats.rs:144–153](src/threats.rs#L144) vs [src/threats.rs:214–223](src/threats.rs#L214)

Both scan a hot window's length-6 line and push empties into a `SmallVec<[Hex; 2]>`. The two copies differ only in what they do with the result. Any change to "empty cell" semantics must be made twice.

**Fix:** Extract `fn window_empties(game: &HexGameState, key: WindowKey, out: &mut SmallVec<[Hex; 2]>)`.

### M10 — `EvalDelta` stores both `cell` and `player`; `player` is only used to derive `cell_val`
**File:** [src/eval/state.rs:83–89](src/eval/state.rs#L83)

`player: u8` is used exclusively to compute `cell_val = (player + 1) as usize` during `unplace`. Storing `cell_val: u8` directly (1 or 2) would remove the derivation and save a byte. Minor — structural tidiness, not performance.

---

## LOW

### L1 — `tests/mod.rs` doc-comment describes 3 modules; 10 are declared
**File:** [src/tests/mod.rs:1–16](src/tests/mod.rs#L1)

Doc header lists `patterns`, `threats`, `oracle`. File declares `board`, `core`, `encoder`, `eval_state`, `grid`, `hot`, `oracle`, `patterns`, `threats`, `threats_internal`. Stale since the test split landed.

### L2 — `any_winning_turn_for` in `oracle.rs` is `#[allow(dead_code)]` with no callers
**File:** [src/tests/oracle.rs:304](src/tests/oracle.rs#L304)

Remnant of the CR2 H14 fix (oracle circularity). Removed from the `analyse` hot path; function was not cleaned up. Dead code hiding behind `#[allow(dead_code)]` is an anti-pattern.

**Fix:** Delete the function, or convert to a `#[cfg(test)]` helper actually used by a regression test.

### L3 — `pybridge.rs:339` still allocates `legal` unconditionally; not a hot path but wasteful
See M3 — same file:line; flagging here if M3 is deferred.

### L4 — Clippy `needless_range_loop` / manual `is_none` in tests
**Files:** [src/tests/patterns.rs:82, 195](src/tests/patterns.rs#L82)

Test-only, low impact, but part of the 39-warning clippy pile surfaced in H5.

### L5 — `core.rs` 325 lines vs plan 250 target, primarily doc-comment narration
**File:** [src/core.rs](src/core.rs)

`hex_distance` doc at [src/core.rs:280–323](src/core.rs#L280) spends ~40 lines narrating cube-coordinate math that the 2-line formula already expresses. `HEX_DIRECTIONS` has a 7-line doc for a single constant. `Ord` impl doc spells out "this delegates to lexicographic ordering" across 8 lines.

**Fix:** Per repo CLAUDE.md (not read — but the guidance is universal): explain WHY, not WHAT. Trim step-by-step `// 1. Compute…` / `// 2. Recover…` commentary.

### L6 — Narrating comments elsewhere: `eval/state.rs::place`, `board.rs::commit_placement`, `search.rs::quiesce`
Same as L5 — `// 1. Game over?` / `// Commit the stone to the board.` and similar narration. CR2 L2 flagged this; partially addressed but not consistently.

### L7 — `Hex.q`, `Hex.r` are `pub` fields; plan §2.5 implies no public fields
**File:** [src/core.rs:47–48](src/core.rs#L47)

Plan spec §2.5 only describes `Turn`'s private-fields policy, not `Hex`. `pub` on coordinate types is conventional in Rust. Flagging for consistency only — low impact.

### L8 — `Stones` is a `pub(crate) type` alias over `FxHashMap<Hex, u8>` that leaks the concrete container type to every internal caller
**File:** [src/board.rs:215](src/board.rs#L215)

A type alias, not a newtype, so crate-internal code can call `.contains_key`, `.iter`, `.clone` directly. If `FxHashMap` is ever swapped for a different structure (e.g. a bitmap or `SmallBitSet`), every caller is affected.

**Fix:** Either (a) wrap as `pub(crate) struct Stones(FxHashMap<Hex, u8>)` with explicit delegator methods, or (b) accept the coupling as deliberate and document that `Stones` is intentionally transparent. Currently it is neither.

### L9 — Plan-prescribed public fields on `BlockConstraint`; code has private fields + accessors (stricter than plan)
**File:** [src/threats.rs:87–113](src/threats.rs#L87)

Plan §2.4 shows `pub cells: SmallVec<…>` / `pub pairs: SmallVec<…>`. Code has private fields + `fn cells()` / `fn pairs()` accessors. Code is *better* than plan — but the plan now lies about the API. Intra-module code at [src/threats.rs:319](src/threats.rs#L319), [src/threats.rs:325](src/threats.rs#L325) accesses the fields directly (same-module access), while cross-module code uses accessors. Inconsistent inside one file.

**Fix:** Update plan §2.4 to reflect accessor-based API, and normalize same-module code to use accessors as well.

### L10 — `N` encoder and pybridge files still above budget after CR2 fixes
See M8.

---

## NIT

### N1 — Stale inline comment at [src/mcts.rs:201–204](src/mcts.rs#L201) claims borrow-checker prevents scratch-buffer reuse
Justification is incorrect (see H3). Remove or rewrite.

### N2 — `oracle.rs` uses `.unwrap()` in test helpers at [src/tests/oracle.rs:258](src/tests/oracle.rs#L258), [src/tests/oracle.rs:297](src/tests/oracle.rs#L297); prefer `.expect("…oracle context…")` so panic messages are actionable

### N3 — `mcts.rs:631` uses `has_any_threats()`; `mcts.rs:644–647` inlines a similar check — pick one style

### N4 — `#[allow(dead_code)]` appears in production code at multiple places; audit whether each is still required after the refactor
```
src/tests/oracle.rs:304   any_winning_turn_for
```
(run `rg '#\[allow\(dead_code\)\]'` to enumerate.)

### N5 — `eval/state.rs:258` comment "known approximation, not a bug" contradicts plan §3.2; see H4
Resolve the doc/spec mismatch.

### N6 — Oracle doctest block in `src/board.rs - board::HexGameState (line 225)` runs but is an English description, not executable; consider `ignore` or removing

### N7 — Module-level doc in `eval/mod.rs` advertises "two complementary evaluation mechanisms" but only one (`EvalState`) is listed. Delete "two" or add the second

### N8 — `threats.rs:146` stack-allocated `SmallVec<[Hex; 2]>` name `empties` is re-declared at threats.rs:216 with an identical comment

---

## Verification table: which CR2 items are resolved

| CR2 ID | Status | Verified |
|---|---|---|
| C1 — `make_turn` unwrap_or silent error | ✅ fixed | [src/search.rs:246](src/search.rs#L246) returns `Result<(bool, u8), GameError>`; all call sites propagate with `?` |
| H1 — `eval/mod.rs` imports `board` | ✅ fixed | `eval/mod.rs` is 23 lines of re-exports only |
| H2 — `union_cells` permissive superset | ✅ fixed | `grep union_cells` → no matches |
| H3 — `_stones` dead parameter on `EvalState::place` | ✅ fixed | [src/eval/state.rs:245](src/eval/state.rs#L245) signature matches plan (modulo `board: &Stones`, see below) |
| H4 — `turn_satisfies_threats` wrapper recomputes `threat_status` | ✅ fixed | Only `turn_satisfies_status(&status, turn)` remains — no recomputation wrapper |
| H5 — `pub` fields on `ThreatCounts`/`EvalDelta`/etc | ✅ fixed | Fields are module-private; accessors expose read-only views |
| H6 — excess `pub` surface in `board.rs` | ✅ fixed | `CandidateSet`, `Stones`, `zobrist_piece`, `find_winning_line`, `validate_move` all `pub(crate)` |
| H7 — `place_unchecked` gating | ✅ fixed | [src/board.rs:496](src/board.rs#L496) `#[cfg(test)] pub(crate) fn` |
| H8 — `place`/`place_unchecked` body duplication | ✅ fixed | `commit_placement` at [src/board.rs:502](src/board.rs#L502) is called by both |
| H9 — `generate_threat_turns` lives in `search.rs` | ⚠️ partial | Moved to [src/threats.rs:359](src/threats.rs#L359). Still unplanned 4th public fn, still allocates, still untested (see new H2) |
| H10 — `re_root` BFS | ✅ fixed | Now scoped to root's children, not full arena |
| H11 — `extract_tree_node_states` replay-from-root | ✅ fixed | Now iterative DFS (see new M4 for residual concern) |
| H12 — hot-path allocations | ⚠️ partial | `threat_status` SmallVec-ified; `gather_policy`, `generate_threat_turns`, MCTS `search_path` still allocating (see new H2, H3) |
| H13 — proptest cases 200 vs plan 1000 | ⚠️ partial | `cases: 500` × 2 blocks = 1000 total, but all `#[ignore]` (see new H1) |
| H14 — oracle circularity on `live_cells` | ✅ fixed | Uses `player_candidates_near2` now |
| M1 — tests embedded in source | ✅ fixed | All `mod tests` blocks migrated to `src/tests/*.rs` |
| M2 — `unplace` reaches into `CandidateSet.rc` | — | not re-verified this round |
| M3 — `set_position` silent coercion | — | not re-verified this round |
| M4 — `classify_block` duplicate guard | ❌ still present | [src/threats.rs:136](src/threats.rs#L136) (see new M1) |
| M5 — GIL held across long calls | ✅ fixed | `classical_search` / `classical_self_play` wrapped in `py.allow_threads` |
| M6 — `unsafe` byte→i32 alignment UB | ✅ fixed | Now uses `chunks_exact(4).map(i32::from_le_bytes)` |
| M7 — duplicated "has any threats" idiom | ⚠️ partial | `has_any_threats()` added to `EvalState`, used in 1/5 call sites (see new H6) |
| M8 — `visit_windows` duplication | ✅ fixed | `visit_windows(cell, cb)` extracted |
| M9 — redundant Vec in `threat_constrained_moves` | ⚠️ partial | See new M3 — `legal_moves_near` still pre-computed |
| M10 — encoder `Vec::new()` in Quiet arm | ✅ fixed | Returns `None` now |
| M11 — `ThreatCountsDelta::Neg` missing | ✅ fixed | `impl Neg` at [src/eval/state.rs:68](src/eval/state.rs#L68) |
| M12 — `lib.rs` internal re-exports | ✅ fixed | Only `hex_distance`, `Hex`, `Turn`, `PLACEMENT_RADIUS`, `WIN_LENGTH`, `GameError`, `HexGameState` re-exported |
| M13 — `assert_invariants` double-gating | ❌ still present | See new M5 |
| M14 — `quiesce` double `threat_status` | ✅ fixed | Computed once at [src/search.rs:723](src/search.rs#L723) |
| L4 — `HotWindows::insert` redundant contains | ✅ fixed | Only `debug_assert!` remains |
| L5 — `collect_run` Vec → SmallVec | ✅ fixed | `SmallVec<[Hex; 6]>` |
| L6 — `PATTERN_COUNT` dead constant | ✅ removed | grep finds no declarations |
| L7 — `WindowKey::new` debug_assert bound `<4` | ✅ fixed | Now `dir < 3` |
| L8 — `Turn::pair(a,a)` accepts self-pair | ✅ fixed | `debug_assert_ne!` present + panic test in tests/core.rs |
| L9 — `PATTERN_VALUES` checksum test missing | ✅ fixed | Checksum test in tests/patterns.rs |
| L10 — `debug_failing_seed_*` stale artifacts | — | still present; cosmetic |
| L11 — `MCTSNode` Copy-eligible | ✅ fixed | `#[derive(Clone, Copy)]` |
| L12 — `i16` coord truncation | ✅ fixed | `i16::try_from(q).map_err(…)` |
| L13 — `placements_per_turn` instance fn | ✅ fixed | `#[staticmethod]` |
| L14 — `bankers_round` pub | ✅ fixed | `pub(crate)` |
| N5 — `// safety:` comment misuse | ✅ fixed | Removed |
| N9 — tests read private field `move_history` | — | not re-verified |
| N11 — `pybridge.rs` vs plan `py.rs` | ❌ still present | See new M7 |

**Score:** 32 resolved, 5 partial, 3 still present, 4 not re-verified.

---

## Appendix — Severity summary

- **CRITICAL:** 0.
- **HIGH (6):** ignored oracle proptests (H1); unplanned + allocating + untested `generate_threat_turns` (H2); `gather_policy` heap per expansion (H3); plan/impl divergence on `win_grid_in_bounds` (H4); clippy hygiene gap (H5); `has_any_threats` helper adopted at 1/5 sites (H6).
- **MEDIUM (10):** residual guard duplication, `select_leaves` buffer copy, `threat_constrained_moves` precompute, DFS depth tracker fragility, `assert_invariants` double-gating, caller-owned buffer discipline, file naming, size budgets, window-empties duplication, `EvalDelta.player` vs `cell_val` tidiness.
- **LOW (10):** stale test doc, dead oracle helper, clippy tail, core over-commenting, narration residue, `Hex` pub fields, `Stones` alias leak, plan-vs-code API discrepancy.
- **NIT (8):** comment rot, miscellaneous `.unwrap()` → `.expect()`, audit `#[allow(dead_code)]`, etc.

Total: **34 findings**. No functional correctness holes remain. Remaining work is hot-path allocation discipline, CI gating of the oracle suite, clippy cleanup, and plan/code doc reconciliation.

The engine has gone from "functional with silent-corruption holes" (CR2) to "functional and sound, short of final polish" (this round). One more focused pass — the six HIGH items above — and the foundation meets the bar.
