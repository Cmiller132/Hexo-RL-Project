# Post-Review Fix Plan — Hexgame Rust Rewrite

**Source:** Orchestrated 4-agent code review (Claude Sonnet 4.6 subagents).  
**Build baseline:** Default build passes (119 tests). `--features python` build: **13 compile errors**.  
**All file paths are absolute** from the repo root `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project`.

Issues are ordered by execution priority: fix blockers first, then correctness, then performance.

---

## Fix 1 — `pybridge.rs` does not compile with `--features python`

**Severity: CRITICAL — blocks all Python training**

The Python bridge was never updated to use the new board/eval/threats API. 13 compile errors when building with `--features python`. Fix all of them in one pass.

### Sub-issue 1a — Direct field access to `window_fives` / `window_fours` (non-existent fields)

**File:** `src/pybridge.rs:165–168`

```rust
// BROKEN — these fields no longer exist on HexGameState
let own = (self.inner.window_fives[me] > 0 || self.inner.window_fours[me] > 0) as u8;
let opp_threat = (self.inner.window_fives[opp] > 0 || self.inner.window_fours[opp] > 0) as u8;
```

**Fix:** Use the `eval()` accessor + `ThreatCounts`:

```rust
let own = (self.inner.eval().counts(me as u8).fives > 0
        || self.inner.eval().counts(me as u8).fours > 0) as u8;
let opp_threat = (self.inner.eval().counts(opp as u8).fives > 0
               || self.inner.eval().counts(opp as u8).fours > 0) as u8;
```

Also verify `window_fours` and `window_fives` *method* delegates (around line 187–194) compile — those already use the accessor correctly and should be fine.

### Sub-issue 1b — `hot_windows()` iterator yields `WindowKey`, not a `(i32, i32, u8)` tuple

**File:** `src/pybridge.rs:212`

```rust
// BROKEN — WindowKey is not destructurable as a reference to a tuple
for &(wq, wr, dir) in &game.eval().hot_windows(player) {
```

**Fix:** Use the `WindowKey` accessor methods:

```rust
for k in game.eval().hot_windows(player) {
    let (wq, wr, dir) = (k.q(), k.r(), k.dir());
    // rest of body unchanged
}
```

Also remove the now-unused variable `pi` at line 210 (`let pi = player as usize;`).

### Sub-issue 1c — `move_history` is a private field

**File:** `src/pybridge.rs:310`

```rust
// BROKEN — move_history is private
let hist = &self.inner.move_history;
```

**Fix:** Use the public accessor:

```rust
let hist = self.inner.move_history();
```

### Sub-issue 1d — `PyArray4::from_shape_vec` no longer exists in numpy 0.24

**File:** `src/pybridge.rs:625, 687` (and any other `PyArray4::from_shape_vec` calls)

The numpy 0.24 API removed `from_shape_vec`. Use `ndarray::Array4::from_shape_vec` then convert:

```rust
// Example pattern (adapt shape dims to match existing code):
let arr = ndarray::Array4::from_shape_vec((d0, d1, d2, d3), data)
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
let arr = PyArray4::from_owned_array_bound(py, arr);
```

Check every `PyArray3::from_shape_vec` call too — apply the same pattern there.

### Sub-issue 1e — `as_slice()` now returns `Result`, not `&[T]`

**File:** `src/pybridge.rs:606, 649` (and any `expand_and_backprop` / `add_dirichlet_noise` call sites)

```rust
// BROKEN — as_slice() returns Result<&[f32], NotContiguousError>
self.inner.add_dirichlet_noise(noise_slice, noise_fraction);
self.inner.expand_and_backprop(policies_slice, values_slice);
```

**Fix:** Propagate the error properly. Since PyResult already flows, use `?` with a map:

```rust
let noise_slice = noise.as_slice()
    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("array must be contiguous"))?;
self.inner.add_dirichlet_noise(noise_slice, noise_fraction);
```

Apply the same pattern to every `as_slice()` call that isn't already handling the Result.

### Verification steps for Fix 1

```bash
# Must produce zero errors, zero pybridge-related warnings:
cargo build --release --features python 2>&1 | grep -E "error|warning.*pybridge"

# Must print the module name (proves .so linked):
python3 -c "import hexgame; print(hexgame.__name__)"

# Smoke test the repaired methods:
python3 -c "
import hexgame
g = hexgame.HexGame()
g.place(0, 0)
print('threat_level:', g.threat_level)
print('window_fours(0):', g.window_fours(0))
print('threat_windows:', g.get_threat_windows(0))
"
```

---

## Fix 2 — `BlockConstraint::pairs` contains degenerate self-pairs `(c, c)`

**Severity: HIGH — semantically incorrect output**

**File:** `src/threats.rs:149`

The pair-enumeration loop includes the diagonal `j == i`, generating `(c, c)` fake pairs:

```rust
// BROKEN — j starts at i, producing (c, c) self-pairs
for j in i..all_cells.len() {
```

**Fix:** Start inner loop at `i + 1`:

```rust
for j in (i + 1)..all_cells.len() {
```

This makes `BlockConstraint::pairs` exactly what the plan specifies: pairs of **distinct** cells that together cover all threat windows.

### Verification steps for Fix 2

```bash
# Unit test that catches self-pairs — run the existing exact-pair test:
cargo test --release threats::tests::block_constraint_two_placements_exact_pairs -- --nocapture

# Manual check: add this temporary assertion to the test or REPL:
# For each pair (a, b) in block_constraint.pairs, assert a != b
# Run the full threat suite:
cargo test --release threats:: -- --nocapture
```

**Code-review step:** Read `src/threats.rs` around the pairs loop. Confirm `for j in (i + 1)..` and that no `j == i` diagonal remains. Add a `debug_assert_ne!(all_cells[i], all_cells[j])` inside the loop body as a guard.

---

## Fix 3 — Expand property tests from 1 seed / 6 moves to 1000+ positions

**Severity: HIGH — insufficient correctness validation**

**File:** `src/tests/threats.rs`

Current property tests use a single seed and 6-move trajectories. The oracle is correct and fast enough to run 1000+ positions. Add `proptest` and rewrite the three slow tests.

### Step 1: Add `proptest` to `Cargo.toml`

```toml
[dev-dependencies]
proptest = "1"
```

### Step 2: Replace the three `#[ignore]` tests

Replace each existing single-seed test with a proptest that:
- Takes a `u64` seed via `prop_assume!` / strategy
- Plays between 1 and 40 random moves using `game.candidates_near2()`
- After each completed turn, calls both the fast path and oracle and diffs

**Key correctness property to check for each position:**

```rust
// Fast path finds no winning turn that oracle misses:
assert!(fast_winning.iter().all(|t| oracle.winning.contains(t)));
// Oracle finds no winning turn that fast path misses:
assert!(oracle.winning.iter().all(|t| fast_winning_set.contains(t)));
// Same bidirectional check for blocking cells
```

Both directions must be checked — current code only checks one.

### Step 3: Rename `#[ignore]` appropriately

Keep the `#[ignore]` attribute (tests are slow) but add a comment: `// Run with: cargo test --release -- --ignored`.

### Verification steps for Fix 3

```bash
# Confirm proptest is available:
cargo test --release -p hexgame -- --ignored 2>&1 | head -5

# Run the full oracle suite (slow):
cargo test --release -- --ignored --nocapture 2>&1 | grep -E "FAILED|ok|ignored"

# Confirm 0 failures across the 3 property tests:
cargo test --release -- --ignored 2>&1 | tail -5
```

**Code-review step:** Read `src/tests/threats.rs`. Confirm:
- At least 200 distinct board positions covered per test (check iteration count or proptest config `cases = 200`)
- Bidirectional assertion (oracle ⊆ fast AND fast ⊆ oracle)
- All three tests: `threat_status`, `turn_satisfies_threats`, `live_cells`

---

## Fix 4 — Delete `filter_turns_by_threats`; compute `threat_status` once per node

**Severity: HIGH — plan violation, redundant computation**

**File:** `src/search.rs`

`filter_turns_by_threats` (around line 370–384) wraps `turn_satisfies_threats` but is called per-turn in the generation loop, re-computing `threat_status` once per turn instead of once per node. Delete the wrapper; apply `threat_status` once and pass the result to generation.

### What to do

1. **Delete** `fn filter_turns_by_threats(...)` entirely.
2. In root-turn generation: call `threat_status(&game)` once, store in `let ts = threat_status(&game);`.
3. Replace every call to `filter_turns_by_threats(...)` with a `retain` / filter using `turn_satisfies_threats_with_status(&ts, turn)`.
4. If `turn_satisfies_threats` in `threats.rs` always recomputes `threat_status` internally, add a variant `fn turn_satisfies_status(status: &ThreatStatus, turn: Turn) -> bool` that takes the pre-computed status. Move the match body there. Keep `turn_satisfies_threats` as a convenience wrapper for one-off call sites.

### Verification steps for Fix 4

```bash
# Confirm the function is gone:
grep -n "filter_turns_by_threats" src/search.rs && echo "STILL PRESENT - FAIL" || echo "DELETED - PASS"

# Confirm threat_status called once per node (spot-check with grep):
grep -n "threat_status" src/search.rs

# Run full test suite to confirm search behavior unchanged:
cargo test --release board:: search:: -- --nocapture
```

---

## Fix 5 — `win_grid_in_bounds` branch → `debug_assert!` in `EvalState::place`

**Severity: MEDIUM — missed performance optimization, plan deviation**

**File:** `src/eval/state.rs:103–105, 159–161, 255–257`

**Before making this change**, verify the invariant: with `PLACEMENT_RADIUS = 8` and `WIN_GRID_RADIUS = 30`, can a stone ever legally reach a window origin outside radius 30?

- A stone placed at `(q, r)` generates windows with origins as far as `(q + 5, r)` in one direction. If `q = 22` (reachable in ~3 moves along one axis), origin `q + 5 = 27` — within 30. Worst case: `q = 8k` for `k` moves, generating origin `8k + 5`. For `k = 3`, that's 29. For `k = 4`, that's 37 > 30. So after ~4 moves along one axis, windows are clipped.
- **Conclusion:** the bounds check IS exercised in long games. Converting to `debug_assert!` would cause a panic in release builds on long games.

**Correct fix:** Do NOT convert to `debug_assert!`. Update the plan doc (`RUST_REWRITE_PLAN.md §3.2`) to acknowledge that `win_grid_in_bounds` must stay as a runtime guard, and add a comment in `state.rs` explaining why:

```rust
// WIN_GRID_RADIUS (30) caps evaluation at ~3–4 moves from origin per axis.
// For stones near the grid boundary, some of the 18 windows extend outside
// the grid; those windows simply don't contribute to evaluation. This is a
// known approximation, not a bug.
if !win_grid_in_bounds(sq, sr) {
    continue;
}
```

### Verification steps for Fix 5

```bash
# Confirm comment is present and branch is still a runtime guard (not debug_assert):
grep -A2 "win_grid_in_bounds" src/eval/state.rs | head -20

# Confirm long-game positions don't panic:
cargo test --release board::tests::random_game_terminates -- --nocapture
```

---

## Fix 6 — Add `debug_assert` in `ThreatCounts::apply` to catch underflow

**Severity: MEDIUM — silent bug masking**

**File:** `src/eval/state.rs:15–19`

`ThreatCounts` uses `u32` but `apply` does the arithmetic through `i32` and casts back. A negative intermediate silently wraps to a huge `u32`.

**Fix:** Add debug assertions before the cast:

```rust
impl ThreatCounts {
    pub fn apply(&mut self, delta: &ThreatCountsDelta) {
        debug_assert!((self.fives as i32 + delta.fives) >= 0,
            "fives underflow: {} + {}", self.fives, delta.fives);
        debug_assert!((self.fours as i32 + delta.fours) >= 0,
            "fours underflow: {} + {}", self.fours, delta.fours);
        debug_assert!((self.threes as i32 + delta.threes) >= 0,
            "threes underflow: {} + {}", self.threes, delta.threes);
        self.fives  = (self.fives  as i32 + delta.fives)  as u32;
        self.fours  = (self.fours  as i32 + delta.fours)  as u32;
        self.threes = (self.threes as i32 + delta.threes) as u32;
    }
}
```

### Verification steps for Fix 6

```bash
# Run all eval tests in debug mode (debug_assert fires in debug builds):
cargo test eval:: -- --nocapture

# Run a game and confirm no panics:
cargo test --release board::tests::random_game_terminates -- --nocapture

# Confirm the assert is present:
grep -n "debug_assert.*fives\|debug_assert.*fours\|debug_assert.*threes" src/eval/state.rs
```

---

## Fix 7 — Zero-alloc pass: `threat_status` and `find_instant_win`

**Severity: MEDIUM — plan §7 performance target not met**

### 7a: `threat_status` allocates `Vec<SmallVec<…>>` at `src/threats.rs:42–49`

The `opponent_threat_windows` binding heap-allocates a Vec on every call where threats exist. Replace with a fixed-size stack buffer:

```rust
// Replace Vec<SmallVec<[Hex; 2]>> with ArrayVec or a SmallVec of SmallVecs:
use smallvec::SmallVec;
// Typical game has <20 hot windows, each with ≤2 empties:
let mut threat_windows: SmallVec<[SmallVec<[Hex; 2]>; 16]> = SmallVec::new();
```

Ensure `smallvec` is already in `Cargo.toml` (it should be, since `eval/hot.rs` uses it). If not, add it.

### 7b: `find_instant_win` in `src/search.rs:339` allocates `Vec::new()` per window

Replace the per-iteration `Vec::new()` with a `SmallVec<[Hex; 2]>`:

```rust
// Replace:
let mut empties = Vec::new();
// With:
let mut empties: SmallVec<[Hex; 2]> = SmallVec::new();
```

### Verification steps for Fix 7

```bash
# Confirm no Vec::new in the hot paths:
grep -n "Vec::new\|Vec::with_capacity" src/threats.rs src/search.rs | grep -v "//\|result\|buf\|legal\|turns"

# Confirm smallvec is in Cargo.toml:
grep "smallvec" Cargo.toml

# Confirm builds and tests still pass:
cargo test --release 2>&1 | tail -5
```

---

## Fix 8 — Snapshot `winning_line` in `MoveRecord`

**Severity: LOW — technical invariant violation, harmless in practice**

**File:** `src/board.rs` — `MoveRecord` struct and `unplace` logic

`winning_line` is never snapshotted, so `unplace` unconditionally sets it to `None`. In practice this is safe (a won game has no further moves to undo), but the "every field bit-identical after place/unplace" invariant is technically broken.

**Fix:** Add `winning_line_before: Option<Vec<Hex>>` to `MoveRecord`, clone on place, restore on unplace:

```rust
// In MoveRecord:
pub winning_line_before: Option<Vec<Hex>>,

// In place():
let record = MoveRecord {
    // ... existing fields ...
    winning_line_before: self.winning_line.clone(),
};

// In unplace():
self.winning_line = rec.winning_line_before;
// Remove the old: self.winning_line = None;
```

Alternatively, add a `debug_assert!(self.winner.is_none() || self.move_count == ... )` at the top of `unplace` that asserts the game isn't over before any undo, making the invariant explicit rather than fixing the snapshot.

### Verification steps for Fix 8

```bash
# Confirm winning_line_before field exists in MoveRecord:
grep -n "winning_line" src/board.rs | head -20

# Confirm unplace restores rather than clears:
grep -n "winning_line = " src/board.rs

# Run place/unplace round-trip tests:
cargo test --release board::tests -- --nocapture
```

---

## Fix 9 — Encoder channels 9/10 should call `live_cells`

**Severity: LOW — maintainability, abstraction contract**

**File:** `src/encoder.rs:285–300`

Channels 9 and 10 inline the hot-window iteration instead of calling `threats::live_cells`. This means two implementations of the same logic that can drift.

**Fix:** Replace inline iteration with:

```rust
// Shared buffer — reuse across channels to avoid allocation:
let mut hot_buf: Vec<Hex> = Vec::new();

// Channel 10 (own live cells):
threats::live_cells(game, current, &mut hot_buf);
for h in &hot_buf {
    // write into tensor as before
}

// Channel 9 (opponent live cells):
threats::live_cells(game, opp, &mut hot_buf);
for h in &hot_buf {
    // write into tensor as before
}
```

`hot_buf` is allocated once per encode call (not per cell), keeping the per-channel cost zero-alloc.

### Verification steps for Fix 9

```bash
# Confirm live_cells import and usage in encoder:
grep -n "live_cells" src/encoder.rs

# Confirm no inline hot_windows loop in encoder (other than via live_cells):
grep -n "hot_windows" src/encoder.rs

# Regression-test encoding is unchanged:
cargo test --release encoder:: -- --nocapture
```

---

## Fix 10 — Remove duplicate `unmake_move` alias; standardize to `unplace`

**Severity: LOW — surface area clutter**

**File:** `src/board.rs`

`unmake_move` is a `pub` alias for `unplace`. Pick one name; remove the other.

- `unplace` is shorter and matches the plan's naming.
- Search every consumer (`search.rs`, `mcts.rs`, `pybridge.rs`, `tests/`) for the deprecated name and update.

### Verification steps for Fix 10

```bash
# Confirm only one of the two names exists:
grep -rn "unmake_move\|\.unplace(" src/ | grep -v "//\|test"

# Rebuild:
cargo build --release 2>&1 | grep error
```

---

## Master verification checklist

After all fixes are applied, run these in order:

```bash
# 1. Default build (no python):
cargo build --release 2>&1 | grep -E "^error" && echo "FAIL" || echo "PASS"

# 2. Python feature build (was broken — must now pass):
cargo build --release --features python 2>&1 | grep -E "^error" && echo "FAIL" || echo "PASS"

# 3. Full unit test suite:
cargo test --release 2>&1 | tail -6

# 4. Slow oracle property tests (was 3 ignored — must still pass):
cargo test --release -- --ignored 2>&1 | tail -6

# 5. Grep for known bad patterns:
echo "=== self-pairs check ===" && grep -n "for j in i\.\." src/threats.rs && echo "FAIL: self-pair loop" || echo "PASS"
echo "=== filter_turns_by_threats deleted ===" && grep -n "fn filter_turns_by_threats" src/search.rs && echo "FAIL: still present" || echo "PASS"
echo "=== no window_fives field access ===" && grep -n "\.window_fives\[" src/pybridge.rs && echo "FAIL: direct field" || echo "PASS"
echo "=== no move_history field access ===" && grep -n "\.move_history;" src/pybridge.rs && echo "FAIL: direct field" || echo "PASS"

# 6. Python smoke test (requires python build + maturin):
python3 -c "
import hexgame
g = hexgame.HexGame()
g.place(0, 0)
g.place(1, 0)
g.place(-1, 0)
print('threat_level:', g.threat_level)
print('window_fours(0):', g.window_fours(0))
print('threat_windows:', g.get_threat_windows(0))
print('encode shape:', g.encode_board_and_legal(8)[0].shape)
print('ALL PYTHON SMOKE TESTS PASSED')
"
```

---

## Summary table

| # | Severity | File(s) | Issue | One-line fix |
|---|----------|---------|-------|-------------|
| 1 | 🔴 CRITICAL | pybridge.rs:165,212,310,625,649 | 13 compile errors, python feature broken | Update all broken call sites to new API |
| 2 | 🟠 HIGH | threats.rs:149 | `pairs` includes `(c,c)` self-pairs | Change `j in i..` to `j in (i+1)..` |
| 3 | 🟠 HIGH | tests/threats.rs | 1 seed / 6 moves per test, one-directional | Add proptest, 200+ positions, bidirectional |
| 4 | 🟠 HIGH | search.rs:370-384 | `filter_turns_by_threats` not deleted | Delete fn; cache `threat_status` per node |
| 5 | 🟡 MEDIUM | eval/state.rs:103,159,255 | Plan said debug_assert, but check IS needed | Keep runtime guard; update plan doc comment |
| 6 | 🟡 MEDIUM | eval/state.rs:15-19 | `ThreatCounts::apply` wraps silently on underflow | Add `debug_assert!(count + delta >= 0)` |
| 7 | 🟡 MEDIUM | threats.rs:42, search.rs:339 | Heap alloc in hot paths | Swap `Vec::new()` → `SmallVec<[…; N]>` |
| 8 | 🟢 LOW | board.rs | `winning_line` not snapshotted in `MoveRecord` | Clone on place, restore on unplace |
| 9 | 🟢 LOW | encoder.rs:285-300 | Channels 9/10 inline instead of `live_cells` | Call `threats::live_cells` with shared buf |
| 10 | 🟢 LOW | board.rs | Duplicate `unmake_move` / `unplace` | Pick one, delete other, update callers |
