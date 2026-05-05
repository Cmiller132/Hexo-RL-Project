# Rust Code Review & Implementation Plan — Hexo-RL-Project

**Scope:** Rust crate under `src/` (9,528 LoC), Cargo configuration, test organization.
**Goal:** Performant, multithread-capable core game engine producing correct features for RL training.

---

## Contents

1. [Project Structure](#1-project-structure)
2. [Findings Summary](#2-findings-summary)
3. [Implementation Plan — Tier 1: Correctness Blockers](#3-implementation-plan--tier-1-correctness-blockers)
4. [Implementation Plan — Tier 2: Performance](#4-implementation-plan--tier-2-performance)
5. [Implementation Plan — Tier 3: Structure & Hygiene](#5-implementation-plan--tier-3-structure--hygiene)
6. [Implementation Plan — Tier 4: Tests](#6-implementation-plan--tier-4-tests)

---

## 1. Project Structure

```
Cargo.toml                 single package, cdylib + rlib, optional `python` feature
src/
├── lib.rs                 (74)   crate root — all sub-modules re-exported as pub
├── core.rs               (317)   hex coords, turns, window keys
├── board.rs              (787)   HexGameState, rules, legal-move generation
├── threats.rs            (446)   threat detection, forced-move logic
├── search.rs            (1239)   alpha-beta search
├── mcts.rs               (999)   MCTS + PUCT + virtual-loss batching
├── encoder.rs            (508)   13-channel NN feature encoder
├── eval/
│   ├── state.rs          (459)   incremental EvalState
│   ├── patterns.rs       (153)   ternary pattern tables
│   ├── grid.rs            (94)   spatial indexing
│   └── hot.rs            (123)   zero-alloc threat-window cache
├── pybridge/
│   ├── mod.rs            (759)   PyO3 bindings
│   └── mcts.rs           (205)   MCTS Python wrapper
└── tests/                        10 files, 3,798 LoC total
```

**Source: 6,530 LoC · Tests: 3,798 LoC.**
Test coverage is radically uneven: `search.rs` (1,239), `mcts.rs` (999), and all of `pybridge/` (964) have **zero tests**.

---

## 2. Findings Summary

| # | Severity | Location | Issue |
|---|----------|----------|-------|
| T1-1 | CRITICAL | `mcts.rs:578-584` | Backprop sign flip uses `node.player` which is `255` on fresh nodes |
| T1-2 | CRITICAL | `mcts.rs:462-465` | Virtual-loss numerator not adjusted; Q distorted during batch selection |
| T1-3 | CRITICAL | `mcts.rs:519` | `sims_done` incremented before backprop; `done()` fires prematurely |
| T1-4 | CRITICAL | `mcts.rs:561-562` | No bounds check on `policies`/`values` slices; wrong-length Python batch panics or silently corrupts |
| T1-5 | CRITICAL | `pybridge/mcts.rs:117,143` | GIL held throughout `select_leaves` and `expand_and_backprop`; blocks GPU inference |
| T1-6 | CRITICAL | `board.rs:429-488` | `set_position` does not validate per-stone radius or origin-first rule |
| T1-7 | CRITICAL | `eval/state.rs:265,316,453` | Pattern index bounds guards are `debug_assert!` only; UB in release on bad input |
| T2-1 | MAJOR | `board.rs:459` | `winning_line_before: Option<Vec<Hex>>` — heap alloc on every placement |
| T2-2 | MAJOR | `board.rs:500` | `EvalState::reset` drops+reallocates 22 KB box; should zero in place |
| T2-3 | MAJOR | `board.rs:610-670` | `legal_moves*` unconditionally sort; hot-path waste |
| T2-4 | MAJOR | `threats.rs:275,298` | `ThreatStatus::MustBlock(Box<BlockConstraint>)` — unnecessary heap alloc |
| T2-5 | MAJOR | `threats.rs:171` | `opponent_threat_windows` clones `SmallVec` per window |
| T2-6 | MAJOR | `threats.rs:440` | `live_cells` uses `Vec::contains` for dedup — O(n²) |
| T2-7 | MAJOR | `encoder.rs:154` | `encode_board_into` returns freshly allocated `Vec<Hex>` per call |
| T2-8 | MAJOR | `encoder.rs:298-306` | Centroid-distance channel recomputed per encode; is grid-positional constant |
| T2-9 | MAJOR | `encoder.rs:312` | `opponent_last_turn_cells()` allocates `Vec` for ≤2 cells |
| T2-10 | MAJOR | `mcts.rs:986` | `children_count: u16` silently truncates on large move sets |
| T2-11 | MAJOR | `mcts.rs:392-401` | `add_dirichlet_noise` silently accepts short slice |
| T2-12 | MAJOR | `board.rs:570-577` | `validate_move` radius check is O(n) hash-map scan |
| T3-1 | MEDIUM | `Cargo.toml` | Missing `panic = "abort"` and `[profile.bench]` |
| T3-2 | MEDIUM | `mcts.rs:252` | `c_puct_init` is `pub` field; should be wired through constructor |
| T3-3 | MEDIUM | `pybridge/mcts.rs:87` | Legal-bytes parsing panics instead of returning `PyErr` |
| T3-4 | MEDIUM | `pybridge/mod.rs:659` | Docstring references `engine.root_child_stats()` which does not exist |
| T3-5 | MEDIUM | `pybridge/mod.rs:755-758` | Missing `BOARD_SIZE`, `NUM_CHANNELS` exports to Python |
| T3-6 | MEDIUM | `lib.rs:58-64` | All modules `pub`; internal types leak into crate API |
| T3-7 | MEDIUM | `board.rs:757` | `select_segment` returns `Vec<Hex>` — should be `[Hex; WIN_LENGTH]` |
| T3-8 | MEDIUM | Project | No `rust-toolchain.toml`, no `rustfmt.toml`, no `clippy.toml` |
| T4-1 | CRITICAL | `mcts.rs`, `search.rs` | 2,238 LoC with zero tests |
| T4-2 | CRITICAL | `pybridge/` | 964 LoC, zero tests, not compiled in CI |
| T4-3 | HIGH | `src/tests/` | Test placement hurts compile time; integration tests in wrong location |
| T4-4 | HIGH | Project | No benchmarks (`benches/`) |

---

## 3. Implementation Plan — Tier 1: Correctness Blockers

Each fix is self-contained. Do them in the order listed — T1-1 through T1-5 are all in `mcts.rs`/`pybridge/mcts.rs` and can be done in one pass.

---

### T1-1 — Fix MCTS Backprop Sign Flip

**File:** `src/mcts.rs` ~line 575  
**Problem:** The backup loop does `if n.player == leaf_player { value } else { -value }`. Nodes have `player: 255` until first visited as a leaf. On a fresh branch every node has `player == 255`, so the condition `255 == 0` is false and every node receives the same-sign value — no alternation.  
**Fix:** Replace player-identity comparison with depth-parity. The search path is `[root, n1, n2, ..., leaf]`. The leaf is at position `search_path.len() - 1`. Each step back from the leaf flips the sign.

**Before (`mcts.rs` ~line 575):**
```rust
let leaf_player = self.arena[leaf.node_idx as usize].player;
for &ni in leaf.search_path.iter().rev() {
    let n = &mut self.arena[ni as usize];
    let pv_value = if n.player == leaf_player {
        value
    } else {
        -value
    };
    n.visit_count += 1;
    n.total_value += pv_value;
}
```

**After:**
```rust
// Flip sign by depth-parity from the leaf.
// search_path is [root, ..., leaf]; iterating rev() the first element
// is the leaf itself (distance 0, same sign), then each step back flips.
let mut parity_value = value;
for &ni in leaf.search_path.iter().rev() {
    let n = &mut self.arena[ni as usize];
    n.visit_count += 1;
    n.total_value += parity_value;
    parity_value = -parity_value;
}
```

> **Note:** This also makes `node.player` irrelevant to backprop. The `player` field on `MCTSNode` is still useful for PUCT (`select_child_puct` uses it to flip Q when looking at children from the parent's perspective). Keep the field; just stop using it in the backup loop.

---

### T1-2 — Fix Virtual-Loss Q Arithmetic

**File:** `src/mcts.rs`  
**Problem:** Virtual loss adds `VIRTUAL_LOSS_VISITS` to `visit_count` but leaves `total_value` unchanged. During the next batch slot's PUCT selection, `Q = total_value / (real_n + VL)` which is less pessimistic than intended. Standard VL is `Q = (total_value - VL) / (real_n + VL)`.

**Step 1 — When applying virtual loss** (~line 462), also subtract from `total_value`:
```rust
// Before:
for &ni in &search_path {
    let n = &mut self.arena[ni as usize];
    n.visit_count += VIRTUAL_LOSS_VISITS;
}

// After:
for &ni in &search_path {
    let n = &mut self.arena[ni as usize];
    n.visit_count += VIRTUAL_LOSS_VISITS;
    n.total_value -= VIRTUAL_LOSS_VISITS as f32;  // pessimize Q
}
```

**Step 2 — When removing virtual loss** (~line 552, inside `expand_and_backprop`), restore `total_value`:
```rust
// Before:
for &ni in &leaf.search_path {
    let n = &mut self.arena[ni as usize];
    n.visit_count -= VIRTUAL_LOSS_VISITS;
}

// After:
for &ni in &leaf.search_path {
    let n = &mut self.arena[ni as usize];
    n.visit_count -= VIRTUAL_LOSS_VISITS;
    n.total_value += VIRTUAL_LOSS_VISITS as f32;  // restore pessimization
}
```

---

### T1-3 — Move `sims_done` Increment into `expand_and_backprop`

**File:** `src/mcts.rs`  
**Problem:** `sims_done += actual_batch` is at the end of `select_leaves` (line 519). `done()` checks `sims_done >= num_simulations`. If Python calls `done()` between `select_leaves` and `expand_and_backprop`, the engine falsely reports completion with VL leaves still outstanding that will never be reverted.

**Step 1 — Remove the increment from `select_leaves`:**
```rust
// Delete this line at the end of select_leaves (~line 519):
self.sims_done += actual_batch;
```

**Step 2 — Add the increment at the start of `expand_and_backprop`, counting only leaves that were actually processed.** Add this line immediately after `let leaves = std::mem::take(&mut self.pending);` (~line 548):
```rust
let leaves = std::mem::take(&mut self.pending);
self.sims_done += leaves.len() as u32;  // count completions, not selections
```

---

### T1-4 — Validate `policies`/`values` Slice Lengths in `expand_and_backprop`

**File:** `src/mcts.rs`  
**Problem:** `expand_and_backprop` indexes `policies[eval_idx * BOARD_AREA .. +BOARD_AREA]` without checking that the slice is long enough. A miscounted Python batch panics inside the Rust hot loop with an index-out-of-bounds that is hard to diagnose.

Add validation at the entry of `expand_and_backprop` (~line 540), after the function signature and before `let mut eval_idx`:
```rust
pub fn expand_and_backprop(&mut self, policies: &[f32], values: &[f32]) {
    let non_terminal_count = self.pending.iter().filter(|l| !l.is_terminal).count();
    assert!(
        policies.len() == non_terminal_count * BOARD_AREA,
        "expand_and_backprop: policies length {} != {} (non_terminal={} * BOARD_AREA={})",
        policies.len(), non_terminal_count * BOARD_AREA,
        non_terminal_count, BOARD_AREA,
    );
    assert!(
        values.len() == non_terminal_count,
        "expand_and_backprop: values length {} != non_terminal_count {}",
        values.len(), non_terminal_count,
    );
    // ... rest of function unchanged
```

---

### T1-5 — Release GIL in `select_leaves` and `expand_and_backprop`

**File:** `src/pybridge/mcts.rs`  
**Problem:** Both hot-path methods hold the GIL while doing heavy Rust work (board traversal, tensor encoding, backprop). Classical search already uses `py.allow_threads()` correctly at `pybridge/mod.rs:504-505`. MCTS must do the same or Python's GPU inference thread blocks for the duration of every Rust step.

**`select_leaves` — before:**
```rust
fn select_leaves<'py>(
    &mut self,
    py: Python<'py>,
    batch_size: u32,
) -> PyResult<(Bound<'py, PyArray4<f32>>, u32)> {
    let (tensors, count) = self.inner.select_leaves(batch_size);
    let view = ndarray::ArrayView4::from_shape(...)?;
    let arr = PyArray4::from_array(py, &view);
    Ok((arr, count))
}
```

**After:** Run the pure-Rust work with GIL released; re-acquire only for numpy construction.
```rust
fn select_leaves<'py>(
    &mut self,
    py: Python<'py>,
    batch_size: u32,
) -> PyResult<(Bound<'py, PyArray4<f32>>, u32)> {
    // Release GIL for the entire Rust traversal + encoding.
    let (count, tensor_vec) = py.allow_threads(|| {
        let (tensors, count) = self.inner.select_leaves(batch_size);
        (count, tensors.to_vec())  // copy into owned Vec before re-acquiring GIL
    });
    let view = ndarray::ArrayView4::from_shape(
        (count as usize, NUM_CHANNELS, BOARD_SIZE as usize, BOARD_SIZE as usize),
        &tensor_vec,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let arr = PyArray4::from_array(py, &view);
    Ok((arr, count))
}
```

> The `to_vec()` inside `allow_threads` copies the slice into an owned `Vec<f32>` so the borrow of `self.inner.batch_buf` ends before the GIL is re-acquired. This is one extra copy; the zero-copy improvement (T2 below) will eliminate it later.

**`expand_and_backprop` — before:**
```rust
fn expand_and_backprop<'py>(
    &mut self,
    policies: PyReadonlyArray1<'py, f32>,
    values: PyReadonlyArray1<'py, f32>,
) -> PyResult<()> {
    let policies_slice = policies.as_slice()...?;
    let values_slice = values.as_slice()...?;
    self.inner.expand_and_backprop(policies_slice, values_slice);
    Ok(())
}
```

**After:**
```rust
fn expand_and_backprop<'py>(
    &mut self,
    policies: PyReadonlyArray1<'py, f32>,
    values: PyReadonlyArray1<'py, f32>,
) -> PyResult<()> {
    let policies_slice = policies
        .as_slice()
        .map_err(|_| PyErr::new::<PyValueError, _>("policies array must be contiguous"))?;
    let values_slice = values
        .as_slice()
        .map_err(|_| PyErr::new::<PyValueError, _>("values array must be contiguous"))?;
    // Copy to owned Vecs so we can drop the GIL before calling Rust.
    let p = policies_slice.to_vec();
    let v = values_slice.to_vec();
    py.allow_threads(|| {
        self.inner.expand_and_backprop(&p, &v);
    });
    Ok(())
}
```

> `policies` and `values` arrays can be large (batch × 1089 floats). The `.to_vec()` copies are necessary because `PyReadonlyArray` borrows the GIL. The zero-copy path requires `unsafe` lifetime extension — defer to T2.

---

### T1-6 — Validate Per-Stone Inputs in `set_position`

**File:** `src/board.rs` ~line 447  
**Problem:** `set_position` validates `player` and `remaining` at the top but does not validate each stone in the `stones` slice. Specifically: (1) it doesn't check that each stone's `player` field is 0 or 1; (2) it doesn't enforce the radius constraint (all stones must be within `PLACEMENT_RADIUS` of an existing stone or be the origin); (3) the first stone must be at `Hex::ORIGIN`.

Add per-stone validation inside the loop, before insertion. Replace:
```rust
for &(q, r, player) in stones {
    let cell = Hex::new(q, r);
    if self.stones.contains_key(&cell) {
        return Err(GameError::CellOccupied(cell));
    }
    // ... rest of loop
```

With:
```rust
for &(q, r, stone_player) in stones {
    let cell = Hex::new(q, r);
    if stone_player > 1 {
        return Err(GameError::InvalidPlayer(stone_player));
    }
    if self.stones.contains_key(&cell) {
        return Err(GameError::CellOccupied(cell));
    }
    if self.stones.is_empty() && cell != Hex::ORIGIN {
        return Err(GameError::MustPlaceAtOrigin);
    }
    if !self.stones.is_empty()
        && !self.stones.keys().any(|&e| hex_distance(e, cell) <= PLACEMENT_RADIUS)
    {
        return Err(GameError::OutOfRadius(cell));
    }
    // ... rest of loop
```

> This is O(n²) over the stones slice — acceptable because `set_position` is called only at position setup, not during search. The hot-path `validate_move` has a separate O(n) fix in T2-12.

---

### T1-7 — Promote Pattern-Index `debug_assert!` to `assert!`

**File:** `src/eval/state.rs` lines 265, 316, 453  
**Problem:** The only guard against indexing `PATTERN_VALUES` and `PATTERN_COUNTS` out-of-bounds is `debug_assert!(new_idx < 729)`, which compiles away in release builds. Malformed input (from T1-6 being unfixed, or future bugs) silently reads a wrong value or indexes out-of-bounds.

Three sites to change. The fix is identical at each:

**Line 265:**
```rust
// Before:
debug_assert!(new_idx < 729);
// After:
assert!(new_idx < 729, "pattern index out of range: {} (cell_val={}, off={})", new_idx, cell_val, off);
```

**Line 316:**
```rust
// Before:
debug_assert!(old_idx < 729);
// After:
assert!(old_idx < 729, "pattern index out of range on unplace: {}", old_idx);
```

**Line 453:**
```rust
// Before:
debug_assert!(new_idx < 729);
// After:
assert!(new_idx < 729, "pattern index out of range in score_delta: {}", new_idx);
```

> These are on the hot path. Once T1-6 is enforced, `new_idx < 729` should be structurally guaranteed, and these assertions will add near-zero overhead (branch-predictor-friendly always-true path). They can be demoted back to `debug_assert!` later once you are confident the invariant holds.

---

## 4. Implementation Plan — Tier 2: Performance

### T2-1 — Replace `Option<Vec<Hex>>` with `Option<[Hex; WIN_LENGTH]>` in `MoveRecord`

**File:** `src/board.rs`  
**Problem:** `winning_line_before: Option<Vec<Hex>>` in `MoveRecord` triggers a heap allocation on every `commit_placement`, even when the game is not won (which is almost always during search). `WIN_LENGTH = 6` is a const, so the array fits on the stack.

**Step 1 — Change the field type in `MoveRecord`** (~line 123):
```rust
// Before:
pub(crate) winning_line_before: Option<Vec<Hex>>,
// After:
pub(crate) winning_line_before: Option<[Hex; WIN_LENGTH as usize]>,
```

**Step 2 — Change the accessor** (~line 148):
```rust
// Before:
pub fn winning_line_before(&self) -> Option<&[Hex]> {
    self.winning_line_before.as_deref()
}
// After:
pub fn winning_line_before(&self) -> Option<&[Hex]> {
    self.winning_line_before.as_ref().map(|a| a.as_slice())
}
```

**Step 3 — Change `select_segment` return type** (~line 757) and update `winning_line` field on `HexGameState` from `Option<Vec<Hex>>` to `Option<[Hex; WIN_LENGTH as usize]>`:
```rust
// Before:
fn select_segment(line: &[Hex], pivot: usize) -> Vec<Hex> {
    // ...
    line[start..start + wl].to_vec()
}
// After:
fn select_segment(line: &[Hex], pivot: usize) -> [Hex; WIN_LENGTH as usize] {
    let wl = WIN_LENGTH as usize;
    let lo = pivot.saturating_sub(wl - 1);
    let hi = pivot.min(line.len() - wl);
    let preferred = pivot.saturating_sub((wl - 1) / 2);
    let start = hi.min(lo.max(preferred));
    line[start..start + wl].try_into().expect("select_segment: slice length != WIN_LENGTH")
}
```

**Step 4 — Update all sites that clone `self.winning_line`:**
In `set_position` (~line 459), change:
```rust
winning_line_before: self.winning_line.clone(),
```
This is already correct — `Option<[Hex; 6]>` is `Copy`, so `.clone()` compiles to a copy. No other change needed.

**Step 5 — Update `unplace`** wherever `winning_line` is restored from `MoveRecord`. Search for `winning_line_before` in `board.rs` and ensure the assignment is `self.winning_line = record.winning_line_before;` (unchanged — value semantics).

---

### T2-2 — Zero EvalState In Place Instead of Reallocating

**File:** `src/board.rs` ~line 500 (`reset` method)  
**Problem:** `self.eval = EvalState::new()` drops the existing `Box<[u16; WIN_GRID_TOTAL]>` and allocates a fresh one. `reset` is called from `set_position` before every training game.

**Step 1 — Add a `clear` method to `EvalState`** in `src/eval/state.rs`:
```rust
/// Zero all state in place, avoiding reallocation.
pub fn clear(&mut self) {
    self.indices.fill(0);
    self.score = 0.0;
    self.counts = ThreatCounts::default();
    self.hot = HotWindows::new();
}
```

**Step 2 — In `board.rs` `reset`**, replace:
```rust
self.eval = EvalState::new();
```
With:
```rust
self.eval.clear();
```

---

### T2-3 — Make Sorting in `legal_moves*` Opt-In

**File:** `src/board.rs`  
**Problem:** `legal_moves`, `legal_moves_near`, and `candidates_near2` all call `result.sort()` unconditionally. MCTS immediately reorders candidates by PUCT score; the sort is wasted work. Deterministic output (needed by tests and Python bindings) should be opt-in.

**Step 1 — Remove `result.sort()` from the hot-path methods.** The three call sites to change:

In `legal_moves` (~line 609): remove `result.sort();`  
In `legal_moves_near` (~line 633 — the fast-path branch): remove `result.sort();`  
In `legal_moves_near` (~line 648 — the fallback branch): remove `result.sort();`  
In `candidates_near2` (~line 665): remove `result.sort();`

**Step 2 — Add sorted variants for callers that need determinism:**
```rust
/// Sorted version of legal_moves_near — for tests and Python export only.
pub fn legal_moves_near_sorted(&self, radius: i32) -> Vec<Hex> {
    let mut v = self.legal_moves_near(radius);
    v.sort();
    v
}

pub fn candidates_near2_sorted(&self) -> Vec<Hex> {
    let mut v = self.candidates_near2();
    v.sort();
    v
}
```

**Step 3 — Update callers.** In `pybridge/mod.rs` wherever `legal_moves_near` or `candidates_near2` results are sent to Python, switch to the `_sorted` variants. In `encoder.rs`, use the unsorted path since order does not matter for tensor encoding. In test code, switch to `_sorted` variants.

---

### T2-4 — Remove `Box` from `ThreatStatus::MustBlock`

**File:** `src/threats.rs` ~lines 270-298  
**Problem:** `ThreatStatus::MustBlock(Box<BlockConstraint>)` boxes a struct that fits on the stack (two `SmallVec`s). `threat_status` is on the search hot path.

**Step 1 — Remove the `Box`** from the enum variant. Find the definition of `ThreatStatus` (search for `enum ThreatStatus`) and change:
```rust
// Before:
MustBlock(Box<BlockConstraint>),
// After:
MustBlock(BlockConstraint),
```

**Step 2 — Update all construction sites.** Both return sites in the `compute_block_constraint` function (~lines 275, 298):
```rust
// Before:
return ThreatStatus::MustBlock(Box::new(BlockConstraint { cells, pairs }));
// After:
return ThreatStatus::MustBlock(BlockConstraint { cells, pairs });
```

**Step 3 — Update all match arms** that destructure `MustBlock`. Search for `MustBlock` throughout `threats.rs` and `search.rs`. The pattern `ThreatStatus::MustBlock(ref bc)` or `ThreatStatus::MustBlock(bc)` works the same way whether or not the inner type is boxed — most match arms need no change. Verify by compiling.

---

### T2-5 — Flatten `opponent_threat_windows` to Avoid Per-Window Clone

**File:** `src/threats.rs` ~line 165  
**Problem:** `result.push(empties.clone())` clones a `SmallVec<[Hex; 2]>` per hot window. The function is called from `threat_status` on every search node.

Replace the function's return type and body:

**Before:**
```rust
fn opponent_threat_windows(game: &HexGameState, player: u8)
    -> SmallVec<[SmallVec<[Hex; 2]>; 16]>
{
    let mut result = SmallVec::<[SmallVec<[Hex; 2]>; 16]>::new();
    let mut empties = SmallVec::<[Hex; 2]>::new();
    for key in game.eval().hot_windows(1 - player) {
        empties.clear();
        window_empties(game, key, &mut empties);
        if !empties.is_empty() {
            result.push(empties.clone());
        }
    }
    result
}
```

**After:** Use a flat representation — one contiguous `SmallVec` of empties and a parallel `SmallVec` of per-window lengths (always 1 or 2):
```rust
/// Returns (flat_empties, window_lengths) where flat_empties is all empty cells
/// across all opponent hot windows concatenated, and window_lengths[i] is how
/// many cells belong to window i.
fn opponent_threat_windows(
    game: &HexGameState,
    player: u8,
) -> (SmallVec<[Hex; 32]>, SmallVec<[u8; 16]>) {
    let mut flat = SmallVec::<[Hex; 32]>::new();
    let mut lengths = SmallVec::<[u8; 16]>::new();
    let mut empties = SmallVec::<[Hex; 2]>::new();
    for key in game.eval().hot_windows(1 - player) {
        empties.clear();
        window_empties(game, key, &mut empties);
        if !empties.is_empty() {
            lengths.push(empties.len() as u8);
            flat.extend_from_slice(&empties);
        }
    }
    (flat, lengths)
}
```

Update all callers of `opponent_threat_windows` to use the new flat API. In `compute_block_constraint` (the main caller), iterate windows by reconstructing slices:
```rust
let (flat_empties, window_lengths) = opponent_threat_windows(game, player);
let mut offset = 0usize;
let must_hit: SmallVec<[&[Hex]; 16]> = window_lengths.iter().map(|&len| {
    let s = &flat_empties[offset..offset + len as usize];
    offset += len as usize;
    s
}).collect();
```

---

### T2-6 — Fix O(n²) Dedup in `live_cells`

**File:** `src/threats.rs` ~line 439  
**Problem:** `if !out.contains(&h) { out.push(h); }` is O(n) per insertion. `live_cells` is called from the encoder on the MCTS hot path.

**Before:**
```rust
for &h in &empties {
    if !out.contains(&h) {
        out.push(h);
    }
}
```

**After:** Use a local `FxHashSet` for O(1) membership test, then drain into `out`:
```rust
// At the top of live_cells, add a local seen-set:
let mut seen = rustc_hash::FxHashSet::default();
// Pre-populate seen with anything already in out (for callers that
// pass a partially-filled buffer):
seen.extend(out.iter().copied());

// Then in the loop:
for &h in &empties {
    if seen.insert(h) {   // insert returns true if the value was new
        out.push(h);
    }
}
```

> `rustc_hash` is already in `Cargo.toml`. `FxHashSet` is `rustc_hash::FxHashSet`.

---

### T2-7 — Eliminate Per-Encode `Vec<Hex>` Allocation in `encode_board_into`

**File:** `src/encoder.rs`  
**Problem:** `encode_board_into` returns `(i32, i32, Vec<Hex>)` where the `Vec<Hex>` is freshly allocated every call. The MCTS engine stores this in `PendingLeaf::legal_moves` — one allocation per simulated leaf.

**Step 1 — Change the signature** to accept an output buffer:
```rust
// Before:
pub fn encode_board_into(
    game: &HexGameState,
    near_radius: i32,
    constrain_threats: bool,
    out: &mut [f32],
    hot_buf: &mut Vec<Hex>,
) -> (i32, i32, Vec<Hex>)

// After:
pub fn encode_board_into(
    game: &HexGameState,
    near_radius: i32,
    constrain_threats: bool,
    out: &mut [f32],
    hot_buf: &mut Vec<Hex>,
    legal_out: &mut Vec<Hex>,   // caller-provided; cleared on entry
) -> (i32, i32)
```

**Step 2 — Inside the function body**, replace:
```rust
let legal = game.legal_moves_near(near_radius);
// ... use legal ...
(offset_q, offset_r, legal)
```
With:
```rust
legal_out.clear();
legal_out.extend(game.legal_moves_near(near_radius));
// ... use legal_out wherever legal was used ...
(offset_q, offset_r)
```

**Step 3 — Add a `legal_moves_buf` field to `MCTSEngine`** (`src/mcts.rs`):
```rust
// In MCTSEngine struct (~line 268), alongside hot_buf:
legal_buf: Vec<Hex>,
```
Initialize to `Vec::new()` in `with_arena_sim_hint`.

**Step 4 — Update the call site in `select_leaves`** (~line 494):
```rust
// Before:
let (oq, or_, legal) = encoder::encode_board_into(
    &self.game, self.near_radius, false, tensor_slice, &mut self.hot_buf,
);
// After:
self.legal_buf.clear();
let (oq, or_) = encoder::encode_board_into(
    &self.game, self.near_radius, false, tensor_slice,
    &mut self.hot_buf, &mut self.legal_buf,
);
```

**Step 5 — Update `PendingLeaf`** to store a `Vec<Hex>` that is moved out of `self.legal_buf`. After calling `encode_board_into`, push a clone of `legal_buf` into the pending leaf — OR restructure `expand_node` to accept the legal moves directly from `legal_buf` without storing them in `PendingLeaf` at all (preferred, since `legal_buf` is not needed after `expand_and_backprop`). The simplest version: `legal_moves: self.legal_buf.clone()` in `PendingLeaf` construction until the deeper refactor.

**Step 6 — Update `encode_board` (the non-`_into` public variant)** to pass a temporary:
```rust
pub fn encode_board(...) -> EncodedBoard {
    let mut tensor = vec![0.0f32; TENSOR_SIZE];
    let mut hot_buf = Vec::new();
    let mut legal_moves = Vec::new();
    let (offset_q, offset_r) = encode_board_into(
        game, near_radius, constrain_threats, &mut tensor, &mut hot_buf, &mut legal_moves,
    );
    EncodedBoard { tensor, offset_q, offset_r, legal_moves }
}
```

---

### T2-8 — Pre-Compute Centroid-Distance Channel

**File:** `src/encoder.rs`  
**Problem:** Channel 11 (distance from board centroid) recomputes `hex_distance(h, center)` for all 33×33 = 1,089 cells on every encode call. But `center` always maps to `(HALF_BOARD, HALF_BOARD)` in tensor space, so the distances are grid-positional constants independent of any game state. The only thing that varies is `offset_q`/`offset_r`, which shift which board cell maps to which tensor cell — but the *distance to tensor centre* for cell `(gi, gj)` is purely a function of `gi - HALF_BOARD` and `gj - HALF_BOARD`.

**Step 1 — Add a module-level static lookup table** at the top of `encoder.rs`:
```rust
use std::sync::OnceLock;

static CENTROID_DIST_CHANNEL: OnceLock<[f32; BOARD_AREA]> = OnceLock::new();

fn centroid_dist_channel() -> &'static [f32; BOARD_AREA] {
    CENTROID_DIST_CHANNEL.get_or_init(|| {
        let center = Hex::new(HALF_BOARD, HALF_BOARD);
        let mut buf = [0.0f32; BOARD_AREA];
        for gi in 0..BOARD_SIZE {
            for gj in 0..BOARD_SIZE {
                let h = Hex::new(gi, gj);
                buf[(gi * BOARD_SIZE + gj) as usize] =
                    hex_distance(h, center) as f32 / HALF_BOARD as f32;
            }
        }
        buf
    })
}
```

**Step 2 — Replace the per-encode channel 11 loop** (~lines 298-306):
```rust
// Before:
{
    let center = Hex::new(offset_q + HALF_BOARD, offset_r + HALF_BOARD);
    for gi in 0..BOARD_SIZE {
        for gj in 0..BOARD_SIZE {
            let h = Hex::new(gi + offset_q, gj + offset_r);
            let dist = hex_distance(h, center) as f32 / HALF_BOARD as f32;
            out[idx(11, gi, gj)] = dist;
        }
    }
}

// After:
{
    let ch11_start = 11 * BOARD_AREA as usize;
    out[ch11_start..ch11_start + BOARD_AREA as usize]
        .copy_from_slice(centroid_dist_channel());
}
```

> `BOARD_AREA = BOARD_SIZE * BOARD_SIZE`. Verify that `idx(11, gi, gj)` is row-major and matches `11 * BOARD_AREA + gi * BOARD_SIZE + gj` before landing this — if the layout differs, adjust the copy accordingly.

---

### T2-9 — Eliminate Vec Allocation in `opponent_last_turn_cells`

**File:** `src/board.rs` ~line 677  
**Problem:** `opponent_last_turn_cells` returns a `Vec<Hex>` of at most 2 cells, allocated on every encode call.

**Change the return type** to `SmallVec<[Hex; 2]>`:
```rust
// Before:
pub fn opponent_last_turn_cells(&self) -> Vec<Hex> {
    let mut result = Vec::new();
    // ...
    result
}

// After:
pub fn opponent_last_turn_cells(&self) -> smallvec::SmallVec<[Hex; 2]> {
    let mut result = smallvec::SmallVec::new();
    // ...
    result
}
```

`SmallVec` is already in `Cargo.toml`. The call site in `encoder.rs` (~line 312) iterates with `for h in game.opponent_last_turn_cells()` — no change needed there.

---

### T2-10 — Add Overflow Guard on `children_count`

**File:** `src/mcts.rs` ~line 986  
**Problem:** `children_count: u16`; `legal_moves.len() as u16` silently wraps past 65,535.

In `expand_node`, immediately before assigning `children_count`, add:
```rust
let child_count = legal_moves.len();
assert!(
    child_count <= u16::MAX as usize,
    "expand_node: legal_moves count {} exceeds u16::MAX",
    child_count
);
node.children_count = child_count as u16;
```

---

### T2-11 — Assert Noise Length in `add_dirichlet_noise`

**File:** `src/mcts.rs` ~line 392  
**Problem:** Short noise slice silently produces partially-noised priors with no warning.

Add at the start of `add_dirichlet_noise`:
```rust
let count = self.arena[self.root_idx as usize].children_count as usize;
assert!(
    noise.len() >= count,
    "add_dirichlet_noise: noise length {} < root children count {}",
    noise.len(), count
);
```

---

### T2-12 — Fix O(n) Radius Check in `validate_move`

**File:** `src/board.rs` ~line 570  
**Problem:** The radius guard iterates `self.stones.keys()` on every call — O(n) in stone count.

The existing `CandidateSet` tracks radius-2 neighbours. Add a second radius-8 candidate set to `HexGameState` for placement validation. This is a more involved refactor; here is the exact approach:

**Step 1 — Add a `placement_candidates: CandidateSet` field** to `HexGameState` initialized with `CandidateSet::new(PLACEMENT_RADIUS)`.

**Step 2 — Update `commit_placement`** to call `self.placement_candidates.remove(cell)` and the increment helper for `placement_candidates` in parallel with the existing `candidates` updates.

**Step 3 — Replace the O(n) check in `validate_move`:**
```rust
// Before:
if !self.stones.is_empty()
    && !self.stones.keys().any(|&e| hex_distance(e, cell) <= PLACEMENT_RADIUS)
{
    return Err(GameError::OutOfRadius(cell));
}

// After:
if !self.stones.is_empty() && !self.placement_candidates.rc.contains_key(&cell) {
    return Err(GameError::OutOfRadius(cell));
}
```

> `CandidateSet::rc` is a `FxHashMap<Hex, u32>` — lookup is O(1). The `CandidateSet` already handles radius arithmetic correctly; using radius 8 simply makes its incremental neighbour ring larger (~169 cells per stone instead of ~18).

---

## 5. Implementation Plan — Tier 3: Structure & Hygiene

### T3-1 — Add `panic = "abort"` and `[profile.bench]` to `Cargo.toml`

**File:** `Cargo.toml`

The current release profile:
```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
```

**Change to:**
```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
panic = "abort"      # removes unwinding machinery from hot loops

[profile.bench]
opt-level = 3
lto = "fat"
codegen-units = 1
# no strip — keep symbols for flamegraph/perf
inherits = "release"
```

---

### T3-2 — Wire `c_puct_init` Through the Constructor

**File:** `src/mcts.rs`, `src/pybridge/mcts.rs`  
**Problem:** `c_puct_init` is `pub` on `MCTSEngine`; set via post-construction mutation `engine.c_puct_init = c_puct_init` in the pybridge.

**Step 1 — Make `c_puct_init` private** in `MCTSEngine`:
```rust
// Before:
pub c_puct_init: f32,
// After:
c_puct_init: f32,
```

**Step 2 — Add `c_puct_init` as a parameter to `with_arena_sim_hint`:**
```rust
pub fn with_arena_sim_hint(
    game: HexGameState,
    num_simulations: u32,
    arena_hint: usize,
    c_puct: f32,
    near_radius: i32,
    constrain_threats: bool,
    c_puct_init: f32,     // add this
) -> Self {
    Self {
        // ...
        c_puct_init,
        // ...
    }
}
```

**Step 3 — Update `pybridge/mcts.rs`** constructor:
```rust
// Before:
let mut engine = MCTSEngine::with_arena_sim_hint(
    game.inner.clone(), num_simulations, hint, c_puct, near_radius, constrain_threats,
);
engine.c_puct_init = c_puct_init;

// After:
let engine = MCTSEngine::with_arena_sim_hint(
    game.inner.clone(), num_simulations, hint, c_puct, near_radius, constrain_threats,
    c_puct_init,
);
```

---

### T3-3 — Return `PyErr` Instead of Panicking in Legal-Bytes Parser

**File:** `src/pybridge/mcts.rs` ~line 85  
**Problem:** `c.try_into().unwrap()` panics on malformed input.

**Before:**
```rust
let mut ints = legal_bytes
    .chunks_exact(4)
    .map(|c| i32::from_le_bytes(c.try_into().unwrap()));
```

**After:**
```rust
if legal_bytes.len() % 8 != 0 {
    return Err(PyErr::new::<PyValueError, _>(
        format!("legal_bytes length {} is not a multiple of 8", legal_bytes.len())
    ));
}
let mut legal = Vec::with_capacity(legal_bytes.len() / 8);
for chunk in legal_bytes.chunks_exact(8) {
    let q = i32::from_le_bytes(chunk[0..4].try_into().unwrap());
    let r = i32::from_le_bytes(chunk[4..8].try_into().unwrap());
    legal.push(Hex::new(q, r));
}
```

> The `chunks_exact(4)` original parsed coordinates as sequential 4-byte ints (two per hex). This version parses 8-byte pairs directly and is clearer. Verify this matches `init_root`'s serialization format (`h.q.to_le_bytes()` then `h.r.to_le_bytes()`, 4 bytes each, 8 per hex — it does).

---

### T3-4 — Fix Docstring: `root_child_stats()` Does Not Exist

**File:** `src/pybridge/mod.rs` ~line 659  
**Change:**
```python
# Before (in the docstring):
visits, q_values, priors = engine.root_child_stats()

# After:
moves_q, moves_r, visit_counts, root_value = engine.get_results()
# For priors and Q-values per child:
priors = engine.root_child_priors()
q_values = engine.root_child_q_values()
```

Replace the entire example block to match the actual API.

---

### T3-5 — Export `BOARD_SIZE` and `NUM_CHANNELS` to Python

**File:** `src/pybridge/mod.rs` ~line 755  
**Before:**
```rust
m.add("FEATURE_COUNT", encoder::FEATURE_COUNT)?;
m.add("WIN_LENGTH", crate::core::WIN_LENGTH)?;
m.add("PLACEMENT_RADIUS", crate::core::PLACEMENT_RADIUS)?;
```

**After:**
```rust
m.add("FEATURE_COUNT", encoder::FEATURE_COUNT)?;
m.add("WIN_LENGTH", crate::core::WIN_LENGTH)?;
m.add("PLACEMENT_RADIUS", crate::core::PLACEMENT_RADIUS)?;
m.add("BOARD_SIZE", encoder::BOARD_SIZE)?;
m.add("NUM_CHANNELS", encoder::NUM_CHANNELS)?;
m.add("TENSOR_SIZE", encoder::TENSOR_SIZE)?;
```

Ensure `BOARD_SIZE`, `NUM_CHANNELS`, and `TENSOR_SIZE` are `pub` in `encoder.rs`. Update any Python scripts that hardcode `(13, 33, 33)` to use `_engine.NUM_CHANNELS` and `_engine.BOARD_SIZE`.

---

### T3-6 — Restrict Visibility of Internal Modules

**File:** `src/lib.rs`  
**Problem:** All modules are `pub mod`, exposing every type as part of the crate's public API.

**Before:**
```rust
pub mod board;
pub mod core;
pub mod encoder;
pub mod eval;
pub mod mcts;
pub mod search;
pub mod threats;
```

**After:**
```rust
pub mod board;       // HexGameState, GameError — intentionally public
pub mod core;        // Hex, Turn, constants — intentionally public
pub mod encoder;     // encode_board — intentionally public
pub mod eval;        // EvalState — public for downstream Rust users
pub(crate) mod mcts;     // internal search engine; not a stable API
pub(crate) mod search;   // internal search engine
pub(crate) mod threats;  // internal threat types; ThreatStatus leaked via board API
```

> `threats` may need to remain `pub` if `ThreatStatus` is part of the Python API surface. Audit `pybridge/mod.rs` for references to `threats::ThreatStatus` — if Python never sees it directly, make it `pub(crate)`. If it does, add a re-export via `pub use threats::ThreatStatus;` at the crate root instead of exposing the whole module.

---

### T3-7 — Add `rust-toolchain.toml`

**Create** `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/rust-toolchain.toml`:
```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
```

Pin to a specific version (e.g. `"1.87"`) once you confirm the toolchain works for your CI runner. Using `"stable"` without a version is better than nothing — it at least declares intent and allows `rustup` to manage the install.

---

### T3-8 — Add `rustfmt.toml` and `clippy.toml`

**Create** `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/rustfmt.toml`:
```toml
edition = "2021"
max_width = 100
use_small_heuristics = "Default"
```

**Create** `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/.clippy.toml`:
```toml
msrv = "1.80"
```

And add a `clippy.toml` (or `.cargo/config.toml`) to pin the deny list:
```toml
# .cargo/config.toml
[build]
rustflags = ["-W", "clippy::all", "-W", "clippy::pedantic", "-A", "clippy::module_name_repetitions"]
```

---

## 6. Implementation Plan — Tier 4: Tests

### T4-1 — Add MCTS Tests

Create `src/tests/mcts.rs` and register it in `src/tests/mod.rs`.

**Required test cases:**

#### Deterministic replay
```rust
#[test]
fn mcts_deterministic_replay() {
    // Run MCTS twice with the same seed path and verify visit distributions match.
    // Since MCTS is deterministic for a given policy sequence, build a mock
    // policy that always returns uniform distribution and value=0.0.
    let game = HexGameState::new();
    let mut engine = MCTSEngine::with_arena_sim_hint(
        game, 50, 200, 1.5, 2, false, 19652.0
    );
    // init_root with uniform policy
    let uniform = vec![1.0 / BOARD_AREA as f32; BOARD_AREA];
    engine.init_root();  // calls encode; we need to call expand_root manually
    engine.expand_root(&uniform, 0.0, 0, 0, &[Hex::ORIGIN]);

    // Run to completion with uniform inference
    while !engine.done() {
        let (_, count) = engine.select_leaves(8);
        let policies = vec![1.0 / BOARD_AREA as f32; count as usize * BOARD_AREA];
        let values = vec![0.0f32; count as usize];
        engine.expand_and_backprop(&policies, &values);
    }
    let (_, _, visits1, _) = engine.get_results();

    // Repeat identically — same game position, same policy → same visits.
    // (MCTS is deterministic with no Dirichlet noise and uniform priors)
    // ... second engine run ...
    assert_eq!(visits1, visits2);
}
```

#### `re_root` consistency
```rust
#[test]
fn mcts_reroot_visit_counts_preserved() {
    // After re-rooting to child c, the visit count of the new root must equal
    // the visit count that c had as a child before re-rooting.
    let game = HexGameState::new();
    // ... run MCTS, get results, pick best child, re_root, verify ...
    let visits_before = visits_of_best_child; // from get_results()
    engine.re_root(best_q, best_r, 50);
    // The new root's visit count should equal the child's visit count before re-root.
    let root_visits = engine.arena[engine.root_idx as usize].visit_count;
    assert_eq!(root_visits, visits_before);
}
```

#### Backprop monotonicity
```rust
#[test]
fn mcts_root_value_bounded() {
    // After any number of simulations with values in [-1, 1], root Q must stay in [-1, 1].
    // This catches sign-flip bugs.
    // ... run MCTS with random values in [-1,1] ...
    let (_, _, _, root_q) = engine.get_results();
    assert!(root_q >= -1.0 && root_q <= 1.0,
        "root Q {} out of range [-1, 1]", root_q);
}
```

#### Policy/value roundtrip
```rust
#[test]
fn mcts_expand_and_backprop_wrong_length_panics() {
    // Verify that T1-4's assertion fires on wrong-length batch.
    // ... set up engine with 2 non-terminal leaves pending ...
    let wrong_policies = vec![0.0f32; 1 * BOARD_AREA]; // should be 2 * BOARD_AREA
    let values = vec![0.0f32; 2];
    let result = std::panic::catch_unwind(|| {
        engine.expand_and_backprop(&wrong_policies, &values);
    });
    assert!(result.is_err());
}
```

#### `sims_done` only counts completed simulations
```rust
#[test]
fn mcts_done_not_true_before_backprop() {
    // After select_leaves but before expand_and_backprop, done() must be false
    // (assuming num_simulations > batch_size).
    let mut engine = MCTSEngine::with_arena_sim_hint(game, 100, 200, 1.5, 2, false, 0.0);
    // ... init and expand root ...
    let (_, count) = engine.select_leaves(8);
    assert!(!engine.done(), "done() must be false after select_leaves but before backprop");
    let policies = vec![...];
    let values = vec![...];
    engine.expand_and_backprop(&policies, &values);
    // Now 8 sims are done, 100 total → still not done
    assert!(!engine.done());
}
```

---

### T4-2 — Add Python Integration CI Job

**Step 1 — Create `Python/tests/test_engine_smoke.py`:**
```python
"""Smoke tests for the Rust-compiled Python extension."""
import numpy as np
import pytest
import _engine  # maturin-built extension

def test_constants_exported():
    assert hasattr(_engine, "BOARD_SIZE")
    assert hasattr(_engine, "NUM_CHANNELS")
    assert _engine.NUM_CHANNELS == 13
    assert _engine.BOARD_SIZE == 33

def test_game_basic():
    g = _engine.PyHexGame()
    g.place(0, 0)
    assert not g.is_over()
    g.unplace()
    assert not g.is_over()

def test_encode_shape():
    import _engine
    g = _engine.PyHexGame()
    g.place(0, 0)
    tensor, oq, or_, legal_bytes = g.encode_board(near_radius=2, constrain_threats=False)
    assert tensor.shape == (_engine.NUM_CHANNELS, _engine.BOARD_SIZE, _engine.BOARD_SIZE)
    assert tensor.dtype == np.float32

def test_mcts_runs_to_completion():
    g = _engine.PyHexGame()
    g.place(0, 0)
    engine = _engine.PyMCTSEngine(g, num_simulations=20, c_puct=1.5,
                                   near_radius=2, constrain_threats=False,
                                   c_puct_init=19652.0)
    result = engine.init_root()
    assert result is not None
    tensor, oq, or_, legal_bytes = result
    policy = np.ones(_engine.BOARD_SIZE ** 2, dtype=np.float32)
    policy /= policy.sum()
    engine.expand_root(policy, 0.0, oq, or_, legal_bytes)

    while not engine.done():
        tensor_batch, count = engine.select_leaves(4)
        policies = np.ones((count, _engine.BOARD_SIZE ** 2), dtype=np.float32)
        policies /= policies.sum(axis=1, keepdims=True)
        values = np.zeros(count, dtype=np.float32)
        engine.expand_and_backprop(policies.flatten(), values)

    moves_q, moves_r, visits, root_q = engine.get_results()
    assert len(visits) > 0
    assert sum(visits) == 20
    assert -1.0 <= root_q <= 1.0
```

**Step 2 — Add CI job.** If using GitHub Actions, add to `.github/workflows/ci.yml`:
```yaml
  python-integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install maturin pytest numpy
      - run: maturin develop --features python
        working-directory: .
      - run: pytest Python/tests/test_engine_smoke.py -v
```

---

### T4-3 — Move Integration Tests to Top-Level `tests/`

Tests that exercise the full game pipeline across multiple modules belong in `tests/` as integration tests. Keep only private-API unit tests in `src/tests/`.

**Move these files** from `src/tests/` to `tests/`:
- `threats.rs` (full-game property tests; uses only public API)
- `board.rs` (full-game scenario tests)
- `encoder.rs` (exercises public `encode_board`)

**Keep in `src/tests/`:**
- `oracle.rs` (needs `pub(crate)` access)
- `threats_internal.rs` (tests internal threat helpers)
- `eval_state.rs`, `hot.rs`, `grid.rs`, `patterns.rs` (unit-test internal state)

**In `tests/threats.rs`**, change the import from:
```rust
use hexgame::...; // already correct for integration tests
```
No other change needed — the tests already use the public API.

Register integration tests: Rust auto-discovers `tests/*.rs` files; no `mod.rs` registration needed.

---

### T4-4 — Add Benchmarks

**Create** `benches/engine.rs`:
```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use hexgame::{HexGameState, encoder};
use hexgame::mcts::MCTSEngine;

fn bench_single_mcts_sim(c: &mut Criterion) {
    let game = HexGameState::new();
    // Set up a mid-game position: place ~20 stones
    // ...

    c.bench_function("mcts_select_leaves_batch8", |b| {
        b.iter(|| {
            let mut engine = MCTSEngine::with_arena_sim_hint(
                game.clone(), 8, 100, 1.5, 2, false, 0.0,
            );
            // init_root + expand_root with uniform policy
            // select_leaves(8) + expand_and_backprop with uniform inference
        });
    });
}

fn bench_threat_status(c: &mut Criterion) {
    use hexgame::threats::threat_status;
    // Build a position with active threats
    let game = /* mid-game position */;
    c.bench_function("threat_status_mid_game", |b| {
        b.iter(|| threat_status(&game));
    });
}

fn bench_encode_board(c: &mut Criterion) {
    let game = /* mid-game position */;
    let mut out = vec![0.0f32; encoder::TENSOR_SIZE];
    let mut hot_buf = Vec::new();
    let mut legal_out = Vec::new();
    c.bench_function("encode_board_into", |b| {
        b.iter(|| {
            encoder::encode_board_into(&game, 2, false, &mut out, &mut hot_buf, &mut legal_out);
        });
    });
}

criterion_group!(benches, bench_single_mcts_sim, bench_threat_status, bench_encode_board);
criterion_main!(benches);
```

**Add to `Cargo.toml`:**
```toml
[dev-dependencies]
proptest = "1"
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "engine"
harness = false
```

---

### T4-5 — Broaden Proptest Coverage

Add property tests to `src/tests/board.rs`:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn place_unplace_is_identity(
        moves in prop::collection::vec((-8i32..=8, -8i32..=8), 1..15)
    ) {
        let mut game = HexGameState::new();
        // Place origin first
        let _ = game.place(0, 0);
        let mut placed = 0;
        for (q, r) in moves {
            if game.place(q, r).is_ok() {
                placed += 1;
            }
        }
        let hash_before = game.zobrist();
        // Undo all placed moves
        for _ in 0..placed + 1 {
            game.unplace();
        }
        assert_eq!(game.zobrist(), 0, "zobrist after full unplace must be zero");
        assert_eq!(game.move_count(), 0);
    }

    #[test]
    fn zobrist_changes_on_every_valid_placement(
        q in -8i32..=8,
        r in -8i32..=8,
    ) {
        let mut game = HexGameState::new();
        let _ = game.place(0, 0);
        let before = game.zobrist();
        if game.place(q, r).is_ok() {
            assert_ne!(game.zobrist(), before, "zobrist must change on placement");
        }
    }
}
```

Add property tests to `src/tests/encoder.rs`:

```rust
proptest! {
    #[test]
    fn encode_output_range(moves in prop::collection::vec((-4i32..=4, -4i32..=4), 0..10)) {
        let mut game = HexGameState::new();
        let _ = game.place(0, 0);
        for (q, r) in moves {
            let _ = game.place(q, r);
        }
        let encoded = encoder::encode_board(&game, 2, false);
        for &v in &encoded.tensor {
            assert!(v >= 0.0 && v <= 1.0,
                "tensor value {} out of [0, 1]", v);
        }
        assert_eq!(encoded.tensor.len(), encoder::TENSOR_SIZE);
    }
}
```

---

## Change Order Summary

Execute in this order to minimise broken intermediate states:

```
T1-7  eval/state.rs   promote debug_assert → assert (safe, no API change)
T1-6  board.rs        set_position validation (safe, additive)
T1-3  mcts.rs         sims_done timing fix
T1-1  mcts.rs         backprop parity fix
T1-2  mcts.rs         virtual loss numerator fix
T1-4  mcts.rs         slice bounds assertion
T1-5  pybridge/mcts.rs GIL release (run Python smoke tests after)

T2-4  threats.rs      unbox MustBlock (mechanical, check all match arms compile)
T2-1  board.rs        Option<[Hex;6]> winning_line (cascading type change, do in one commit)
T2-2  board.rs        EvalState::clear() + reset()
T2-10 mcts.rs         children_count assert
T2-11 mcts.rs         dirichlet noise assert
T2-3  board.rs        remove unconditional sort (add _sorted variants)
T2-5  threats.rs      flatten opponent_threat_windows
T2-6  threats.rs      FxHashSet dedup in live_cells
T2-7  encoder.rs      legal_out parameter (signature change; update all callers)
T2-8  encoder.rs      static centroid channel
T2-9  board.rs        SmallVec return from opponent_last_turn_cells
T2-12 board.rs        radius-8 CandidateSet (larger refactor; do last in T2)

T3-1  Cargo.toml      panic=abort, profile.bench
T3-7  rust-toolchain  create file
T3-2  mcts.rs         c_puct_init private + constructor param
T3-3  pybridge/mcts.rs legal-bytes PyErr
T3-4  pybridge/mod.rs fix docstring
T3-5  pybridge/mod.rs export BOARD_SIZE/NUM_CHANNELS
T3-6  lib.rs          pub(crate) for mcts/search

T4-4  benches/        add criterion benchmarks (run after T2 to measure gains)
T4-1  src/tests/      add mcts tests
T4-2  Python/tests/   add smoke test + CI job
T4-3  tests/          move integration tests
T4-5  src/tests/      broaden proptest
```
