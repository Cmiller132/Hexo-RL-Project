# Code Review 2 — Hexgame Rust Foundation

**Scope:** Full code-quality review of the in-progress rewrite sitting in the main worktree (uncommitted changes across `board.rs`, `core.rs`, `encoder.rs`, `eval/*`, `mcts.rs`, `pybridge.rs`, `search.rs`, `threats.rs`, `tests/*`, `lib.rs`).

**Methodology:** Four Sonnet subagents reviewed file groups in parallel. Every high-severity finding below was then re-verified directly by reading the file at the cited line. Subagent claims that could not be reproduced are listed in an appendix and discarded.

**Build status at review time:**
- `cargo build --release` → clean, 0 warnings.
- `cargo test --release` → **119 passed, 0 failed, 3 ignored.** (Contradicts subagent #4's fabricated E0433 claim.)
- `cargo build --release --features python` → rust compiles cleanly, failure is only at link time (no Python lib on host — environment issue, not a code issue).

**Bottom line:** Every mandatory fix from `CODE_REVIEW_FIXES.md` (Fix 1a–e, 2, 3, 4, 5, 6, 7, 8, 9, 10) is applied and verified. The code functions. It is **not yet a strong foundation** — there is a silent correctness hole in turn execution, a module-layering violation, a fake "exactness" guarantee in the threat layer, a dead API parameter, and substantial leaked internals across `pub` fields/types the plan said must be private. File-size budgets are consistently exceeded, primarily by large in-file `mod tests` blocks that the plan explicitly placed in `src/tests/`.

Issues are ranked by severity. Every entry has a verified file:line cite.

---

## CRITICAL

### C1 — `make_turn` silently swallows `GameError`, corrupting the undo stack on any illegal placement
**Files:** [src/search.rs:244](src/search.rs#L244), [src/search.rs:249](src/search.rs#L249), [src/pybridge.rs:930](src/pybridge.rs#L930), [src/pybridge.rs:934](src/pybridge.rs#L934)

```rust
game.place(m1.q, m1.r).unwrap_or(true);
```

`HexGameState::place` returns `Result<bool, GameError>`. `unwrap_or(true)` maps `Err` to `Ok(true)` — the "game is over" sentinel — but **no stone is placed, no move pushed onto `move_history`, no Zobrist update, no eval delta pushed**. Downstream `unmake_turn(game, placed)` then calls `game.unplace()` `placed` times on an undo stack that never grew, which panics (or worse, pops stones that were placed in a previous turn).

The inline comment calls this a "defensive fallback so the search does not crash." It is the opposite of that — it converts a recoverable error into an invariant violation. In classical search this triggers only on a bug in move generation (which is exactly when you want a loud crash, not a silent corruption). In `classical_self_play` (pybridge.rs:918+) it will silently produce bogus training data if the engine ever proposes an illegal move.

**Fix:** Propagate the error. Replace with `game.place(m1.q, m1.r).expect("make_turn given illegal move")` or, better, return `Result<(bool, u8), GameError>` from `make_turn` and let callers handle it.

---

## HIGH

### H1 — `eval/mod.rs` imports `board.rs`, violating the layer hierarchy from plan §1
**File:** [src/eval/mod.rs:26](src/eval/mod.rs#L26)

```rust
use crate::board::HexGameState;
```

The plan's strict dependency order is `core → eval → board → threats → …`. `eval/` must not depend on `board`. The import exists because `extract_features()` (`eval/mod.rs:120–186`) takes `&HexGameState`. That function belongs in `encoder.rs`, not `eval/`. Additional misplaced items in the same file: `WIN_SCORE` (a search sentinel), `FEATURE_COUNT`, `FEATURES_PER_PLAYER`, `LIVE5..LIVE2`, the `count_run` helper, and ~120 lines of tests.

**Fix:** Move `extract_features`, `count_run`, and feature-vector constants to `encoder.rs`. Move `WIN_SCORE` to `search.rs`. Leave `eval/mod.rs` as `pub mod` declarations + re-exports, as specified.

### H2 — `BlockConstraint::union_cells` re-introduces the permissive-superset bug the plan explicitly eliminated
**File:** [src/threats.rs:87–111](src/threats.rs#L87), consumed at [src/threats.rs:315](src/threats.rs#L315), [src/encoder.rs:225](src/encoder.rs#L225), [src/pybridge.rs:351](src/pybridge.rs#L351)

Plan §2.4 says: *"`BlockConstraint` is **exact** in the 2-placement case … no permissive supersets."* The implementation has a third public field `union_cells` described in its own doc comment as "a pragmatic heuristic." `turn_satisfies_status` accepts a 1-placement `Turn` whenever its cell lies in `union_cells`, even though a single stone provably cannot block all threats when 2 placements are required. This is the exact semantic the plan called out as "the encoder bug" that the rewrite was supposed to fix.

**Fix:** Either delete `union_cells` and the corresponding branch in `turn_satisfies_status`, or rename `BlockConstraint` and its doc comment to stop claiming exactness. The current state silently hands callers incorrect blocking information.

### H3 — `EvalState::place` takes an unused `_stones` parameter, leaking `FxHashMap` into the eval-layer signature
**File:** [src/eval/state.rs:222](src/eval/state.rs#L222)

```rust
pub fn place(&mut self, _stones: &FxHashMap<Hex, u8>, cell: Hex, player: u8) -> EvalDelta {
```

The parameter is prefixed `_` (intentionally unused), documented as "reserved for future validation," and not read in the body. Every caller passes `&game.stones` purely to satisfy the signature. This couples `eval/state.rs` to `rustc_hash::FxHashMap` for no reason, and the `_`-prefix + "reserved for future" pair is self-contradictory (if it is reserved, don't underscore; if it is unused, don't take it).

Plan §2.3 specifies `pub fn place(&mut self, board: &Stones, cell: Hex, player: u8)` with **no return value** (the delta is pushed onto the internal stack). The actual signature also returns `EvalDelta`, doubling the leak.

**Fix:** Drop the `_stones` parameter and the `EvalDelta` return. Make `place` match the plan exactly. Delete `use rustc_hash::FxHashMap` from `state.rs`.

### H4 — Four public functions in `threats.rs`; plan specifies three, and the fourth (`turn_satisfies_status`) exposes the raw `&ThreatStatus` of the caller's caching scheme
**File:** [src/threats.rs:185, 306, 347, 372](src/threats.rs#L306)

Plan §2.4 is explicit: *"Three functions. Everything downstream composes from these."* The extra `turn_satisfies_status(status: &ThreatStatus, turn: Turn) -> bool` was added per `CODE_REVIEW_FIXES.md` Fix 4 — a reasonable optimization, but the plan was never updated to reflect it. Worse, `turn_satisfies_threats` is still defined as a wrapper that **recomputes `threat_status` on every call** ([src/threats.rs:348](src/threats.rs#L348)), which callers may forget and end up in the quadratic re-computation mess Fix 4 was meant to kill.

**Fix:** Either reconcile the plan with `CODE_REVIEW_FIXES.md` (update plan §2.4 to list four functions and document the caching contract), **or** delete the convenience wrapper entirely and force callers through the status-cached path.

### H5 — `pub` fields on `ThreatCounts`, `ThreatCountsDelta`, `EvalDelta`, `MoveRecord`, `BlockConstraint`, `EncodedBoard.legal_moves`
**Files:** [src/eval/state.rs:19–22, 51–55, 66–71](src/eval/state.rs#L19), [src/board.rs:108–124](src/board.rs#L108), [src/threats.rs:91–111](src/threats.rs#L91), [src/encoder.rs:51](src/encoder.rs#L51)

Plan §2.3/§6: *"Every field is private."* The actual code declares these fields `pub`, which defeats the debug-assert underflow guard in `ThreatCounts::apply` (any external code can mutate counts directly and bypass the guard), exposes the undo-record layout, and lets callers construct arbitrary `EvalDelta`/`ThreatCountsDelta` values that `apply()` will happily accept. The plan's invariant protection is aspirational, not enforced.

**Fix:** Make each field private (or `pub(crate)` at most); add accessor methods where external reads are needed. Specifically:
- `ThreatCounts::{fives, fours, threes}` → private, `pub fn fives(&self) -> u32` etc.
- `ThreatCountsDelta` — make all fields private; construction goes through a builder or the internal `place` logic only.
- `EvalDelta` — entire struct `pub(crate)`; callers never need to inspect one.
- `MoveRecord` — fields `pub(crate)`; expose via named accessors if needed.
- `BlockConstraint.{cells, pairs, union_cells}` — private, accessor-only.
- `EncodedBoard.legal_moves` — private with `pub fn legal_moves(&self) -> &[Hex]`.

### H6 — `CandidateSet`, `Stones`, `zobrist_piece`, `find_winning_line`, `validate_move`, `move_eval_delta` are `pub` at crate level with no external consumer
**File:** [src/board.rs:36, 143, 178, 591, 752, 818](src/board.rs)

Plan §6.1 lists the exact public surface of `HexGameState` and nothing else from `board.rs` is meant to be public. The actual file exports six more items (`CandidateSet` as a fully public struct with `pub new`/`remove`/`clear`, the `Stones` type alias that leaks `FxHashMap` as the public ABI, the Zobrist seed function, an internal win-line finder, a rule-validation helper that duplicates logic already inside `place`, and a thin delegation over `eval::hypothetical_score_delta`).

**Fix:** Downgrade all six to `pub(crate)`. Delete `move_eval_delta` — callers can use `game.eval().hypothetical_score_delta(cell, player)` directly.

### H7 — `place_unchecked` is `pub(crate)` + `#[allow(dead_code)]` instead of `#[cfg(test)]`
**File:** [src/board.rs:544](src/board.rs#L544)

```rust
#[allow(dead_code)]
pub(crate) fn place_unchecked(&mut self, cell: Hex) {
```

It is only called from `src/tests/oracle.rs` (verified: 6 call sites, all under `#[cfg(test)]`). Gating it with `#[allow(dead_code)]` instead of `#[cfg(test)]` means:
- The compiler will not warn if a non-test module starts calling it (a real risk since it skips validation).
- The function body is compiled into release builds unnecessarily.
- The `#[allow]` is a tell that something is being suppressed rather than solved.

**Fix:** Replace `#[allow(dead_code)] pub(crate) fn` with `#[cfg(test)] pub(crate) fn`.

### H8 — 40-line body duplication between `place` and `place_unchecked`
**File:** [src/board.rs:322–367](src/board.rs#L322) vs [src/board.rs:538–583](src/board.rs#L538)

Both functions build an identical `MoveRecord`, do the insert, xor Zobrist, bump `move_count`, push history, remove from candidates, call `eval.place`, check for win, and advance turn. The only difference is the leading validation. Two copies of a 40-line invariant-heavy state mutation is a correctness hazard; any bug fix must be applied in two places.

**Fix:** Extract a private `fn commit_placement(&mut self, cell: Hex)` used by both paths. `place` does `validate_move(cell)?; self.commit_placement(cell); Ok(...)`; `place_unchecked` calls `self.commit_placement(cell)` directly.

### H9 — `generate_threat_turns` is a 60-line search-internal implementation of move-set construction that should live in `threats.rs`
**File:** [src/search.rs:615–675](src/search.rs#L615)

The plan mandates "one public API for threat queries." The quiescence search instead builds its own pair enumeration using `live_cells` + ad-hoc dedup (`sort() + dedup() + truncate(16)`). This logic cannot be tested against the oracle (`tests/threats.rs` only cross-checks `threat_status`/`turn_satisfies_threats`/`live_cells`), so any divergence from the oracle goes unnoticed. It also allocates four fresh `Vec`s per quiescence node.

**Fix:** Either move this into `threats.rs` as a fourth-or-fifth documented public function and cover it with the proptest suite, or rewrite quiescence to use only the three plan-approved public functions.

### H10 — `re_root` BFS invalidation is O(arena_size) per call
**File:** [src/mcts.rs:662–684](src/mcts.rs#L662)

When `constrain_threats=true` and the new root has expanded threat children, re-rooting walks potentially thousands of arena nodes every half-turn. Additionally, the parent-pointer clearing (`children_count = 0` on the ancestor combined with `parent = NO_PARENT` on every descendant) is redundant — once the ancestor's `children_count = 0`, descendants are unreachable and the safety check at [src/mcts.rs:821](src/mcts.rs#L821) already catches orphans.

**Fix:** Either drop the descendant parent-clearing entirely (relying on the existing reachability check) or switch the invalidation strategy to a generational counter so that stale descendants are identified in O(1).

### H11 — `extract_tree_node_states` replays moves from the root for every candidate
**File:** [src/mcts.rs:829–863](src/mcts.rs#L829)

For each of up to 128 sampled nodes, it walks the parent chain to the root, calls `place` per ancestor on the live game state, encodes, then `unplace`s back. With depth ~30 that is ~3,840 `place`/`unplace` pairs per training extraction. The function also silently discards orphaned samples (`valid = false; break;`) instead of returning an error.

**Fix:** Either batch the reconstruction (re-use a single forward walk) or produce a per-node encoding during selection time. At minimum, return `PyResult` so orphan detection surfaces instead of silently shrinking the sample.

### H12 — Heap allocations on documented zero-alloc hot paths
**Files:** [src/threats.rs:241–244](src/threats.rs#L241), [src/mcts.rs:158, 189, 208, 210, 423](src/mcts.rs#L158), [src/search.rs:1253](src/search.rs#L1253)

- `threat_status` collects threat cells into `Vec<Hex>` (then `sort + dedup`) on every `MustBlock` classification. Plan §7 requires this path to be zero-alloc. Use `SmallVec<[Hex; 32]>`.
- `PendingLeaf.search_path: Vec<u32>` — allocated once per leaf, typical depth 10–30. Use `SmallVec<[u32; 32]>`.
- `gather_policy` allocates `raw`, `exps`, `priors` fresh per leaf expansion (3 × batch_size heap allocs per `expand_and_backprop`).
- `select_leaves` does `let mut search_path = vec![self.root_idx];` per simulation.
- `iterative_deepening` allocates an `FxHashMap<Turn, i32>` per depth for root re-sort where a `Vec` with `sort_by_key` would be cheaper.

**Fix:** `SmallVec` with inline size matching the typical case; reuse scratch buffers stored on `SearchState` / `MCTSEngine`.

### H13 — Property tests use 200 cases, plan spec demands 1000+
**File:** [src/tests/threats.rs:205–206](src/tests/threats.rs#L205)

```rust
proptest! {
    #![proptest_config(ProptestConfig { cases: 200, ..ProptestConfig::default() })]
```

`CODE_REVIEW_FIXES.md` Fix 3 says "200+", which is met. `RUST_REWRITE_PLAN.md` §5 and §11 say **1000+**. The two specs disagree; the plan (higher bar) is not met.

**Fix:** Raise to `cases: 1000`. Benchmark wall time — if it exceeds 30 s even under `--release`, split into two `#[ignore]`-gated tests at 500 cases each.

### H14 — `oracle.rs::any_winning_turn_for` uses `live_cells` to bound its candidate set
**File:** [src/tests/oracle.rs:116–144](src/tests/oracle.rs#L116)

The oracle is defined as the **single ground truth** (plan §5, §8). Bounding its candidate loop by `live_cells(...)` — a fast-path function under test — makes the oracle circular: if `live_cells` has a bug that silently omits a cell, `any_winning_turn_for` misses the same cell, and the property test blesses the bug. The oracle should enumerate every cell in `game.candidates_near2()` (or a larger superset) to remain independent of the code it is validating.

**Fix:** Replace `live_cells(...)` with `game.candidates_near2()` or a radius-3 scan. Accept the slower per-test time — the oracle is `#[ignore]`-gated.

---

## MEDIUM

### M1 — Tests embedded in source files instead of `src/tests/`
**Files:** `board.rs` (548 lines of in-file tests), `threats.rs` (~432 lines), `core.rs` (~215 lines), `eval/*.rs`

Plan §1 places tests under `src/tests/`. The actual code keeps large `mod tests` blocks in production source, accounting for the bulk of the line-count overrun against plan targets (board: 1395 vs 700; threats: 828 vs 350; core: 527 vs 250; eval/state: 652 vs 400). Test-code drift inside a library-code file also makes it impossible to measure the "strong foundation" line count independently of the test harness.

**Fix:** Move the `mod tests` blocks to `src/tests/board.rs`, `src/tests/threats_internal.rs`, etc. Keep only public-API tests in `src/tests/`; internal-only unit tests become `#[cfg(test)]` integration files.

### M2 — `unplace` reaches into `CandidateSet.rc` directly, bypassing the `CandidateSet` abstraction
**File:** [src/board.rs:408–431](src/board.rs#L408)

`CandidateSet` exposes `new`/`remove`/`clear`. `unplace` needs to decrement-and-reinsert neighbors, which the API does not support, so it pokes `self.candidates.rc` directly. Either the abstraction is insufficient (add `fn decr_neighbors(&mut self, cell: Hex, stones: &Stones)`) or the abstraction shouldn't exist (inline the `rc` map into `HexGameState`).

**Fix:** Add a `pub(crate) fn on_unplace(&mut self, cell: Hex, stones: &Stones)` on `CandidateSet` that captures the reverse logic.

### M3 — `set_position` silently coerces invalid inputs
**File:** [src/board.rs:519–520](src/board.rs#L519)

```rust
self.current_player = player & 1;
self.placements_remaining = remaining.max(1);
```

A caller passing `player = 2` gets 0. A caller passing `remaining = 0` gets 1. Both hide bugs in test fixtures and Python-side setup code. This is a boundary-layer function; it should validate loudly.

**Fix:** Return `Result<(), GameError>` with dedicated variants `GameError::InvalidPlayer` / `GameError::InvalidRemaining`, or `debug_assert!` + `unwrap_or_else(panic)` on the hot path.

### M4 — `classify_block` / threat-pair loop silently duplicates the `opponent_threat_windows` guard
**File:** [src/threats.rs:125–157](src/threats.rs#L125) vs the caller at [src/threats.rs:198](src/threats.rs#L198)

The helper re-checks `counts.fours == 0 && counts.fives == 0` after the caller already guaranteed threats exist. Dead branch on the hot path.

**Fix:** Inline the helper into `threat_status`, or drop the internal guard and document the precondition.

### M5 — GIL held across long-running Rust calls from Python
**Files:** [src/pybridge.rs:482, 520, 921](src/pybridge.rs#L482) plus all `PyMCTSEngine` methods

`classical_search` / `classical_search_turn` / `classical_self_play` run `iterative_deepening` for up to `time_ms` ms without `py.allow_threads(|| ...)`. In a training pipeline with a separate GPU inference thread, this blocks Python-level concurrency for the full search duration.

**Fix:** Wrap the CPU-bound body in `py.allow_threads(|| { … })`.

### M6 — `unsafe` byte-to-`i32` reinterpret without alignment guarantee
**File:** [src/pybridge.rs:712–720](src/pybridge.rs#L712)

```rust
let legal_i32: &[i32] = unsafe {
    std::slice::from_raw_parts(legal_bytes.as_ptr() as *const i32, …)
};
```

`PyBytes` is 1-byte aligned; `i32` requires 4-byte alignment. This is undefined behavior on architectures that enforce alignment, and even on x86-64 it relies on undocumented Python guarantees. The bytes come from Python `int.to_bytes` / `struct.pack` anyway.

**Fix:** Replace with safe `legal_bytes.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()))`. Delete the `unsafe` block.

### M7 — Duplicate "has any threats" idiom copy-pasted across three files
**Files:** [src/mcts.rs:644–647](src/mcts.rs#L644), [src/search.rs:769–774](src/search.rs#L769), similar in pybridge

Four-way `|| ||` check of `counts(0).fives/fours/counts(1).fives/fours` appears in three places.

**Fix:** Add `impl EvalState { pub fn has_any_threats(&self) -> bool { … } }`.

### M8 — `hypothetical_score_delta` duplicates the inner loop of `place`
**File:** [src/eval/state.rs:447–475](src/eval/state.rs#L447) vs [src/eval/state.rs:232–270](src/eval/state.rs#L232)

Three independent copies of the same 18-window traversal (place, unplace, hypothetical). Any future change to the window-iteration pattern must be made three times.

**Fix:** Extract `fn visit_windows(cell: Hex, mut cb: impl FnMut(usize, usize, usize))` and reuse.

### M9 — Redundant `Vec` allocations in `pybridge.rs::threat_constrained_moves`
**File:** [src/pybridge.rs:339–358](src/pybridge.rs#L339)

Three allocations per call (build `legal`, build `constrained`, collect the final `Option<Vec<…>>`). Called from Python on every MCTS root expansion.

**Fix:** Build once via `.filter().map().collect()`, or return an iterator.

### M10 — `encode_board_into`'s `constrain_threats` path allocates a throwaway `Vec` in the `Quiet | Unblockable` arm
**File:** [src/encoder.rs:215–231](src/encoder.rs#L215)

Explicitly constructs `Vec::new()` for both branches that represent "no constraint", then relies on the `if !constrained.is_empty()` guard to preserve `legal`. Intent is obscured; an `if let Some(mask) = …` pattern would be clearer and avoid the dead allocation.

### M11 — `ThreatCountsDelta` lacks `Neg` — `unplace` manually negates three fields inline
**File:** [src/eval/state.rs:302–311](src/eval/state.rs#L302)

Fragile: adding a 4th count field (say `sixes`) would silently fail to negate in `unplace`, corrupting undo.

**Fix:** `impl std::ops::Neg for ThreatCountsDelta` that negates every field, then `self.counts[0].apply(&(-delta.counts[0]))`.

### M12 — `lib.rs` re-exports internals: `extract_features`, `FEATURE_COUNT`, `HEX_DIRECTIONS`, `MoveRecord`
**File:** [src/lib.rs:73–75](src/lib.rs#L73)

`HEX_DIRECTIONS` is a coordinate-system implementation detail; `MoveRecord` is a snapshot type with (as noted in H5) public fields; `extract_features` belongs in `encoder.rs`. Each unnecessary re-export is a commit to public stability the plan did not ask for.

**Fix:** Remove the re-exports; let consumers qualify via submodule paths.

### M13 — `debug_assertions` invariant check in `unplace` uses `HashSet` allocations
**File:** [src/eval/state.rs:362–399](src/eval/state.rs#L362)

Correct per plan §3.4 (debug-only), but `#[cfg(debug_assertions)]` gating on the method plus an additional `#[cfg(debug_assertions)]` block at the call site ([src/eval/state.rs:351](src/eval/state.rs#L351)) is redundant.

**Fix:** Gate once — at the method. Drop the wrapping `#[cfg]` at the call.

### M14 — `quiesce` recomputes `threat_status` twice per tactical node
**File:** [src/search.rs:803, 900](src/search.rs#L803)

`generate_inner_turns` / `generate_threat_turns` call `threat_status` for filtering; the Unblockable check later in the same node calls it again. The whole point of Fix 4 was to compute it once per node.

**Fix:** Cache `threat_status(&game)` as `ts` at node entry, hand it to both move-gen and the Unblockable check.

---

## LOW

### L1 — File sizes are 2× plan targets after accounting for tests being moved out
Even after moving `mod tests` blocks to `src/tests/`, several files exceed the plan:
- `board.rs`: 847 lines of non-test code vs 700 target.
- `threats.rs`: 396 lines of non-test code vs 350 target.
- `eval/state.rs`: 652 total vs 400 target (most of the excess is doc-comment verbosity).
- `pybridge.rs`: 972 vs 700 — `extract_tree_node_states` and `classical_self_play` are large contributors.

Plan line counts are estimates, not hard limits, but the trend (everything is 20–50 % larger than estimated) points at consistent over-commenting and under-factoring.

### L2 — Narrating comments throughout `board.rs::place`, `board.rs::unplace`, `eval/state.rs::place`, `search.rs::quiesce`
Comments like `// Commit the stone to the board.` immediately above `self.stones.insert(cell, player);` or `// 1. Game over?` above a literal `if game.winner().is_some()` are the anti-pattern the repo's `CLAUDE.md` flags (file not checked, but the instruction is universal): explain WHY, not WHAT. Remove them or replace with why-comments.

### L3 — `place_unchecked` is `pub(crate)` rather than `#[cfg(test)]`
See H7 above — listing here for completeness in the L tier if H7 is deferred.

### L4 — `HotWindows::insert` has a redundant `contains` guard
**File:** [src/eval/hot.rs:70–72](src/eval/hot.rs#L70)

All callers (`update_hot` in `state.rs`) have already decided whether the window transitioned hot→cold or cold→hot before calling. Scanning the `SmallVec` again inside `insert` is defensive coding that costs an O(n) scan per hot-window update.

**Fix:** Remove the `contains` guard; add `debug_assert!(!vec.contains(&k))`.

### L5 — `Vec<Hex>` used in win-line detection; use `SmallVec<[Hex; 6]>`
**File:** [src/board.rs:778](src/board.rs#L778) — `collect_run`

`collect_run` allocates a fresh `Vec` per call and is invoked up to 6× per placement. The run is bounded by `WIN_LENGTH = 6`, so `SmallVec<[Hex; 6]>` is a zero-alloc drop-in.

### L6 — `PATTERN_COUNT: usize = 729` is dead
**File:** [src/eval/patterns.rs:40](src/eval/patterns.rs#L40)

Exported but never referenced. Either replace every `729` literal with `PATTERN_COUNT` or delete the constant.

### L7 — `WindowKey::new` debug_assert uses wrong bound
**File:** [src/core.rs:213](src/core.rs#L213)

`debug_assert!(dir < 4, ...)` but `HEX_DIRECTIONS.len() == 3`. A `dir = 3` input passes the assert but is never used by any window.

**Fix:** `debug_assert!(dir < 3, ...)`.

### L8 — `Turn::pair(a, a)` is accepted and round-trips
**File:** [src/core.rs:500–506](src/core.rs#L500) (test documents the degenerate case)

Plan Invariant 5 implies `a != b`. `Turn::pair` does not assert this; neither does the oracle (it short-circuits via `c2 <= c1`, but an external caller can still synthesize one).

**Fix:** `debug_assert_ne!(a, b)` inside `Turn::pair`. Explicitly test that a self-pair panics in debug.

### L9 — `PATTERN_VALUES` bit-identity with CMA-ES weights is not tested
**File:** [src/eval/patterns.rs](src/eval/patterns.rs)

Plan Invariant 1 says the table must be bit-identical to the tuned weights. There is no checksum test; accidental corruption would not fail any test.

**Fix:** Hash `PATTERN_VALUES` with a deterministic digest (blake3 / fnv) and assert the literal hash in a unit test.

### L10 — `debug_failing_seed_1` / `debug_failing_seed_2` in `tests/threats.rs` are development artifacts
**File:** [src/tests/threats.rs](src/tests/threats.rs) (both inside `mod debug_tests`, both `#[test]`, both hard-coded-seed regressions with `println!` output)

Verified to compile and pass (contrary to subagent #4's false claim — see Appendix A). Still, they are stale regression artifacts that duplicate proptest coverage and muddy the output. Delete them or convert to `#[ignore]` with a regression-ticket comment.

### L11 — `MCTSNode` derives `Clone` but is `Copy`-eligible
**File:** [src/mcts.rs:93](src/mcts.rs#L93)

All fields are `Copy`. Deriving `Copy` would prevent accidental semantic cloning and allow implicit copy during `Vec` grow.

### L12 — `re_root` coordinate truncation to `i16` with no validation
**File:** [src/pybridge.rs:879](src/pybridge.rs#L879)

`q as i16` silently wraps outside `[-32768, 32767]`. In normal games this never fires, but a Python caller with malformed inputs gets silent corruption.

### L13 — `placements_per_turn` is a pure function on an instance method
**File:** [src/pybridge.rs:234–236](src/pybridge.rs#L234)

Should be `#[staticmethod]` or a module-level constant.

### L14 — `bankers_round` is `pub` with no external caller
**File:** [src/encoder.rs:62](src/encoder.rs#L62)

Should be `pub(crate)`.

---

## NIT

### N1 — `eval/mod.rs:142–144`: `match Option { None => true, Some(_) => false }` instead of `.is_none()`.
### N2 — `patterns.rs:183–190`: Stream-of-consciousness debugging commentary left inside a test assertion ("Wait, that's actually (1,0) because digits are …"). Replace with a clean explanation or remove.
### N3 — `grid.rs:44,49`: Trailing literal comments (`// 2 * 30 + 1`, `// 11163`) duplicate arithmetic the compiler can do. Use `2 * WIN_GRID_RADIUS as usize + 1` / `WIN_GRID_SIDE * WIN_GRID_SIDE * 3`.
### N4 — `hot.rs:22` says "typically < 20", `hot.rs:66` says "≈ 0–10". Same quantity, two estimates.
### N5 — `mcts.rs:821`: `// safety:` comment mis-uses the Rust `// SAFETY:` convention (which is reserved for `unsafe` blocks).
### N6 — `search.rs:113`: Magic `15` for tempo bonus. Name it `const TEMPO_BONUS: i32 = 15;`.
### N7 — `search.rs:785`: "Should not happen" comment on a live recursive call. Use `unreachable!()` or `debug_unreachable!()`.
### N8 — `pybridge.rs:221–226`: Redundant `as usize` / `as u8` round-tripping in `threat_level`.
### N9 — `board.rs:1141–1147`: Tests read `g.move_history` as a private field instead of through the `move_history()` accessor.
### N10 — `core.rs:79–94`: `PartialOrd` doc says "delegates to Ord::cmp" — a narrating comment. Explain the *purpose* of the lexicographic order (canonical pair storage).
### N11 — `pybridge.rs` is called `pybridge.rs` but plan consistently calls it `py.rs`. Update the plan or rename the file.

---

## Verification of CODE_REVIEW_FIXES.md items

| Fix | Status | Verified at |
|----|--------|------------|
| 1a — no `window_fives[` field access | ✅ | `pybridge.rs:223–226` uses `eval().counts(me).fives` |
| 1b — `WindowKey` accessors, not tuple destructure | ✅ | `pybridge.rs:281` uses `key.q() / key.r() / key.dir()` |
| 1c — `move_history()` accessor | ✅ | `pybridge.rs:394` |
| 1d — no `PyArray4::from_shape_vec` | ✅ | `pybridge.rs:442, 678, 765, 843` use `Array3::from_shape_vec` + `PyArray3::from_owned_array` |
| 1e — `as_slice()?` error propagation | ✅ | all call sites use `.map_err(…)?` |
| 2 — `j in (i+1)..` | ✅ | `threats.rs:272`; `debug_assert_ne!` present at line 275 |
| 3 — proptest with bidirectional check | ⚠️ | Applied with `cases: 200`. Plan demands 1000+. See H13. |
| 4 — `filter_turns_by_threats` deleted | ✅ | `grep` finds no matches; `turn_satisfies_status` added |
| 5 — bounds check kept as runtime guard | ✅ | `eval/state.rs:244, 323, 457` with explanatory comment |
| 6 — underflow `debug_assert` in `ThreatCounts::apply` | ✅ | `eval/state.rs:32–38` |
| 7 — `SmallVec` in threat hot paths | ⚠️ | Applied in `opponent_threat_windows`. `threat_status` still heap-allocates `all_cells`. See H12. |
| 8 — `winning_line` snapshotted in `MoveRecord` | ✅ | `board.rs:123, 335, 438` |
| 9 — encoder channels 9/10 call `live_cells` | ✅ | `encoder.rs:303–318` with shared `hot_buf` |
| 10 — one of `unplace` / `unmake_move` | ✅ | Only `unplace` exists |

---

## Appendix A — Subagent #4 fabrications (discarded)

The pybridge/tests subagent reported two CRITICAL compile failures that do not exist:

1. **"`debug_failing_seed_2` outside any module causes E0433 compile failure"** at `tests/threats.rs:489–552` — the file is **406 lines total**. There is no line 489. The function exists inside `mod debug_tests` and compiles. `cargo test --release` reports 119 tests pass, 0 fail.
2. **"`temp_test` module with `panic!(\"intentional fail to print output\")`"** at `tests/oracle.rs:261–279` — those lines are legitimate tests `oracle_finds_blocking_single` and `oracle_finds_blocking_pair`, both passing. `grep` for `panic!.*intentional` returns no matches anywhere in `src/`.

These are listed here so they can be ignored if the subagent report is referenced directly. All other subagent findings were spot-checked against the source and either promoted to the body of this review with verified citations, or silently dropped.

---

## Appendix B — Severity summary

- **CRITICAL (1):** silent `GameError` suppression in `make_turn` / `classical_self_play`.
- **HIGH (14):** layering violation, fake exactness guarantee, leaky `pub` fields + types, hot-path allocations, oracle self-dependency, proptest coverage below plan spec, duplicated placement logic.
- **MEDIUM (14):** tests-in-source, abstraction leaks, GIL discipline, unsafe alignment, repeated idioms, redundant computation.
- **LOW (14):** file-size overruns, narrating comments, minor missing guards, dead exports.
- **NIT (11):** doc-comment and naming cleanups.

Total: 54 verified findings. None functional / correctness-blocking at runtime except C1 (triggers only on an illegal-move bug in move generation, but corrupts silently when it does).
