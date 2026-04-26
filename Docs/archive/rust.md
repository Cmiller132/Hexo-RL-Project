# Hexgame Rust Engine — Complete Architecture Reference

**Project:** `hexgame` — Hexo engine with PyO3 bindings  
**Location:** `src/` (11 Rust modules, ~6,500 lines)  
**Edition:** Rust 2021  
**Key deps:** `rustc-hash = "2"`, `pyo3 = "0.24"`, `numpy = "0.24"`

---

## Table of Contents

1. [Game Rules](#game-rules)
2. [Module Overview](#module-overview)
3. [Core Primitives (`core.rs`)](#core-primitives-corers)
4. [Game State (`board.rs`)](#game-state-boardrs)
5. [Pattern Evaluation (`patterns.rs`)](#pattern-evaluation-patternsrs) — *deep dive*
6. [Threat Analysis (`threats.rs`)](#threat-analysis-threatsrs) — *deep dive*
7. [Neural Encoder (`encoder.rs`)](#neural-encoder-encoderrs)
8. [Classical Evaluation (`eval.rs`)](#classical-evaluation-evalrs)
9. [Alpha-Beta Search (`search.rs`)](#alpha-beta-search-searchrs)
10. [MCTS Engine (`mcts.rs`)](#mcts-engine-mctsrs)
11. [Python Bridge (`pybridge.rs`)](#python-bridge-pybridgers)
12. [Module Re-exports (`game.rs`)](#module-re-exports-gamers)
13. [How Threats and Patterns Interact](#how-threats-and-patterns-interact)

---

## Game Rules

- Two players (0 and 1) on an **infinite hexagonal grid** using axial coordinates `(q, r)`. Hexo is a new game most similar to Connect 6; it shares the hexagonal board geometry with Hex but has no relation to Hex as a game.
- **Player 0** opens with **one** tile at the origin `(0, 0)`.
- Every subsequent turn consists of **two** placements.
- Each placement must land on an empty hex within `PLACEMENT_RADIUS = 8` of any existing tile.
- First player to form `WIN_LENGTH = 6` tiles in a contiguous straight line along any of the three hex axes wins.
- The board is infinite — **no draws**.

---

## Module Overview

| Module | File | Lines | Responsibility |
|--------|------|-------|----------------|
| `core` | `core.rs` | 215 | Axial hex coordinates, distance, three principal directions |
| `board` | `board.rs` | 1,194 | Game state, rules, placement/undo, win detection, Zobrist hash |
| `patterns` | `patterns.rs` | 520 | Ternary 6-cell window encoding, incremental evaluation, hot-window tracking |
| `threats` | `threats.rs` | 870 | Hot-window enumeration, threat detection, solver-based verification |
| `encoder` | `encoder.rs` | 289 | 13-channel 33×33 NN tensor encoder (canonical implementation) |
| `eval` | `eval.rs` | 318 | Classical feature extraction (13-element vector for bootstrap training) |
| `search` | `search.rs` | 1,156 | Turn-based alpha-beta with iterative deepening, TT, PVS, LMR, quiescence |
| `mcts` | `mcts.rs` | 891 | Arena-allocated MCTS with PUCT, virtual loss, batch leaf selection |
| `pybridge` | `pybridge.rs` | 788 | PyO3 bindings exposing engine to Python |
| `game` | `game.rs` | 8 | Thin re-export module for backward compatibility |
| `lib` | `lib.rs` | 52 | Module declarations, public re-exports, crate docs |

**Total:** ~5,500 lines of library code + ~1,000 lines of tests = **~6,500 lines**.

**Test count:** 73 unit tests + 2 doc tests, all passing.

---

## Core Primitives (`core.rs`)

The spatial foundation of the entire engine.

### `Hex` struct

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hex { pub q: i32, pub r: i32 }
```

- `Copy` — passed by value thousands of times per search node
- `Eq + Hash` — used as keys in `FxHashMap`/`FxHashSet` for the board
- `Ord` — lexicographic `(q, r)` ordering for deterministic collections
- `Hex::ORIGIN = (0, 0)` — the mandatory opening move

### Three Principal Directions

```rust
pub const HEX_DIRECTIONS: [(i32, i32); 3] = [(1, 0), (0, 1), (1, -1)];
```

A regular hexagon has three pairs of parallel sides. Any straight line on the grid must be parallel to one of these orientations. Win detection scans forward and backward along these three axes (and their negatives) and is guaranteed to find every possible winning line.

| Axis | Direction | Geometric meaning |
|------|-----------|-------------------|
| A | `(1, 0)` | East (column / "q" axis) |
| B | `(0, 1)` | South-east (row / "r" axis) |
| C | `(1, -1)` | North-east (diagonal axis) |

### Hex Distance

```rust
pub fn hex_distance(a: Hex, b: Hex) -> i32 {
    let dq = a.q - b.q;
    let dr = a.r - b.r;
    let ds = dq + dr;  // implicit third cube component
    (dq.abs() + dr.abs() + ds.abs()) / 2
}
```

The minimum number of steps between two hexes. Derived from cube coordinates `(x, y, z)` with constraint `x + y + z = 0`, where `x = q`, `z = r`, `y = -q - r`.

---

## Game State (`board.rs`)

### `HexGameState`

The central mutable game state. Contains everything needed to play, evaluate, search, and undo moves.

```rust
pub struct HexGameState {
    // ── Rule state ──
    pub board: FxHashMap<Hex, u8>,                    // stone lookup
    pub current_player: u8,                            // 0 or 1
    pub placements_remaining: u8,                      // 1 or 2
    pub winner: Option<u8>,
    pub winning_line: Option<Vec<Hex>>,
    pub move_count: u32,
    pub move_history: Vec<MoveRecord>,
    pub zobrist_hash: u64,

    // ── Incremental evaluation ──
    pub window_eval: i32,
    pub window_fives: [i32; 2],   // near-win window counts per player
    pub window_fours: [i32; 2],   // hot window counts per player
    pub window_threes: [i32; 2],  // developing window counts per player
    pub hot_windows: [FxHashSet<(i32, i32, u8)>; 2],  // active threat windows
    pub window_indices: Vec<u16>,                     // ternary pattern indices

    // ── Internal ──
    pub(crate) eval_stack: Vec<EvalDelta>,            // undo stack for eval
    pub(crate) candidate_rc: FxHashMap<Hex, u32>,     // ref-counted radius-2 candidates
    pub(crate) candidate_radius: i32,                 // always 2 for search
}
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `place(q, r)` | Validate, place stone, update eval, check win, advance turn |
| `unmake_move()` | Pop eval delta, reverse window indices, restore candidate set, revert turn state |
| `set_position(pieces, player, pr)` | Bulk-load a position for testing/analysis |
| `legal_moves()` | Exhaustive radius-8 scan (expensive, used sparingly) |
| `legal_moves_near(radius)` | Fast candidate generation near existing stones |
| `candidates_near2()` | Radius-2 incremental candidate set (primary search generator) |
| `find_winning_line(last, player)` | Scan 3 axes from last placement for 6-in-a-row |

### Zobrist Hashing (Infinite Board)

Since the board is infinite, a precomputed Zobrist table is impossible. Instead, a **deterministic mixing function** based on FNV-1a with final avalanche produces a pseudo-random 64-bit value for any `(player, q, r)` triple in O(1) time with no memory overhead:

```rust
pub fn zobrist_piece(player: u8, cell: Hex) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
    h ^= player as u64;
    h = h.wrapping_mul(0x100_0000_01b3);    // FNV prime
    h ^= cell.q as u64;
    h = h.wrapping_mul(0x100_0000_01b3);
    h ^= cell.r as u64;
    h = h.wrapping_mul(0x100_0000_01b3);
    // Final avalanche (3 rounds)
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    h
}
```

### Win Detection

After every placement, `find_winning_line` scans the three principal axes from the newly-placed stone, collecting contiguous same-player cells forward and backward. If the total run length ≥ 6, it extracts a preferred 6-cell segment centered around the pivot and declares a win.

### Candidate Generation

The incremental `candidate_rc` map uses **reference counting**: every time a stone is placed, all empty cells within radius 2 get their count incremented. When a stone is removed (unmake), counts are decremented. Cells with count > 0 are legal radius-2 candidates. This makes `candidates_near2()` O(number of candidates) instead of O(board_size × radius²).

---

## Pattern Evaluation (`patterns.rs`)

### Overview

The heart of the positional evaluator. Every 6-cell sliding window along the three hex axes is encoded as a **base-3 (ternary) number**:

| Digit | Meaning |
|-------|---------|
| 0 | Empty cell |
| 1 | Player 0 stone |
| 2 | Player 1 stone |

With 6 cells, there are `3⁶ = 729` possible patterns. Each pattern has a **precomputed static evaluation weight** (CMA-ES optimized, stored in `PATTERN_VALUES: [i32; 729]`).

When a stone is placed, only the windows containing that cell need updating — at most **18 windows** (3 directions × 6 possible origin offsets). This makes evaluation **O(18)** per placement instead of O(board_size × windows).

### The Ternary Index System

#### Flat Grid Storage

Window origins are stored in a flat array indexed by `(q, r, dir)`:

```rust
pub const WIN_GRID_RADIUS: i32 = 30;
const WIN_GRID_SIDE: usize = 61;           // 2*30 + 1
pub const WIN_GRID_TOTAL: usize = 61*61*3; // 11,163 entries

pub fn win_grid_idx(q: i32, r: i32, dir: u8) -> usize {
    ((q + 30) as usize) * 61 * 3
        + ((r + 30) as usize) * 3
        + dir as usize
}
```

A radius of 30 comfortably covers all reachable window origins since pieces are constrained to be within radius 8 of existing pieces.

#### Pattern Count Table (Compile-Time)

```rust
const fn build_pattern_counts() -> [(u8, u8); 729] {
    // For each ternary index 0..728:
    //   Count digits == 1 → p0_count
    //   Count digits == 2 → p1_count
}
pub static PATTERN_COUNTS: [(u8, u8); 729] = build_pattern_counts();
```

This table is generated at compile time and stored in `.rodata`. Every pattern lookup is a single array access.

#### Pattern Values (CMA-ES Optimized)

```rust
const PATTERN_VALUES: [i32; 729] = [
    0, -26, 26, -349, -323, -37, 349, 37, 323, 
    // ... 729 values
];
```

These values represent the static evaluation contribution of each window from **player 0's perspective** (positive = good for P0). They were optimized using CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to maximize win rate in self-play.

### `EvalDelta` — Per-Move Delta for Undo

```rust
pub struct EvalDelta {
    pub score: i32,           // change in window_eval
    pub five_delta: [i32; 2], // near-win count changes
    pub four_delta: [i32; 2], // hot window count changes
    pub three_delta: [i32; 2],// developing window count changes
}
```

When a stone is placed, `compute_eval_delta` returns this struct. The caller applies it to running totals and pushes it onto `eval_stack`. `unmake_move` pops the delta and subtracts it, restoring exact eval state.

### Incremental Eval Algorithm (`compute_eval_delta`)

For each of the 18 windows containing the newly-placed cell:

1. **Read old ternary index** from `window_indices[gi]`
2. **Compute new index**: `old_idx + cell_val * POW3[off]`
   - `cell_val = 1` for P0, `2` for P1
   - `POW3 = [1, 3, 9, 27, 81, 243]`
   - `off` = offset of the cell within the 6-cell window (0..5)
3. **Score delta**: `PATTERN_VALUES[new_idx] - PATTERN_VALUES[old_idx]`
4. **Threat count deltas**: Use `PATTERN_COUNTS` to get old/new `(p0_count, p1_count)`, then call `classify()` with `sign = -1` (remove old contribution) and `sign = +1` (add new contribution)
5. **Update `hot_windows`**: A window is "hot" when it has ≥4 stones of one player and 0 opponent stones. If the window transitions into or out of this state, insert/remove from the appropriate `hot_windows` set.
6. **Write new index** back to `window_indices`

### Hot Windows

```rust
pub hot_windows: [FxHashSet<(i32, i32, u8)>; 2]
```

A window key is `(origin_q, origin_r, direction_index)`. A window is hot for player P when:

- `p0_count >= 4 && p1_count == 0` (hot for P0)
- `p1_count >= 4 && p0_count == 0` (hot for P1)

Hot windows are the **foundation of threat detection**. They represent windows where a player has a serious threat (4+ stones, unblocked). The `threats` module queries these sets to find instant wins, forced blocks, and unblockable threats.

### Threat Classification (`classify`)

```rust
pub fn classify(own: u8, other: u8, fives: &mut i32, fours: &mut i32, threes: &mut i32, sign: i32) {
    if other == 0 {  // uncontested window
        match own {
            5..=6 => *fives += sign,
            4     => *fours += sign,
            3     => *threes += sign,
            _     => {}
        }
    }
}
```

Only **uncontested** windows (0 opponent stones) are classified into threat tiers. A 3-window with an opponent stone inside is irrelevant for threat counting.

### Read-Only Hypothetical Eval (`move_eval_delta`)

```rust
pub fn move_eval_delta(&self, cell: Hex, player: u8) -> i32
```

A read-only version of `compute_eval_delta` used for **move ordering**. It computes what the score delta would be without modifying any state. Used by the search engine to sort candidates before turn generation.

---

## Threat Analysis (`threats.rs`)

### Philosophy: Fast Path + Solver Ground Truth

The module provides **two tiers** of threat analysis:

1. **Fast path (production)** — Uses precomputed `hot_windows` and hitting-set logic. O(n) to O(n²) depending on placements remaining. Used in search hot loops and MCTS leaf encoding.
2. **Solver (verification)** — Brute-force simulates placements and checks the resulting board state. Used in tests to validate that the fast path is correct.

> **Key invariant:** For this game (win = 6 contiguous, max 2 placements/turn), `collect_winning_threat_cells` is **exact**. Every cell it returns wins either alone (5-window) or with its partner empty (4-window), and no winning cell exists outside a hot window.

### Fast-Path Methods

#### `collect_threat_window_empties(player)`

```rust
pub fn collect_threat_window_empties(&self, player: u8) -> Vec<FxHashSet<Hex>>
```

Iterates over `hot_windows[player]`, reconstructs each 6-cell window, and collects the empty cells. Only returns windows with **1 or 2 empties** (3+ empties cannot be completed in a single turn with at most 2 placements).

**Returns:** A vector where each element is a set of empty cells for one hot window.

**Example:** A bare 5-stone run `(0,0)..(4,0)` produces 4 overlapping hot windows:
- `(-2,0)..(3,0)` → empties `{(-2,0), (-1,0)}`
- `(-1,0)..(4,0)` → empties `{(-1,0)}`
- `(0,0)..(5,0)` → empties `{(5,0)}`
- `(1,0)..(6,0)` → empties `{(5,0), (6,0)}`

Only the windows with ≤2 empties are returned. For a bare 5-run, this filters to the 5-windows (1 empty each).

#### `collect_threat_cells(player)`

Union of all empty cells from `collect_threat_window_empties`. Simple flatten + collect.

#### `collect_winning_threat_cells(player)`

```rust
pub fn collect_winning_threat_cells(&self, player: u8) -> FxHashSet<Hex>
```

**The most important fast path.** Returns cells that would **complete a win** for `player` on this turn.

Logic per window:
- **1 empty + ≥1 placement remaining** → that empty cell wins immediately (5-window)
- **2 empties + ≥2 placements remaining** → both empty cells win together (4-window)
- Otherwise → skip

**Why this is exact for this game:**
- A 5-window (1 empty) requires 1 placement to become 6. Since the game ends immediately on win, even with 2 placements remaining, `Turn::one(empty)` is sufficient.
- A 4-window (2 empties) requires 2 placements to become 6. Only wins when `placements_remaining >= 2`.
- No other window configuration can produce a win in one turn.

#### `collect_blocking_threat_cells(player)`

```rust
pub fn collect_blocking_threat_cells(&self, player: u8) -> FxHashSet<Hex>
```

Returns cells that can **block** `player`'s threats.

- **1 placement remaining:** Returns the **intersection** of all threat-window empties. A single cell must hit every threat window to block all threats.
- **2 placements remaining:** Returns the **union** of all threat-window empties. This is a safe superset — the search engine does precise turn-level filtering via `filter_turns_by_threats`.

> **Design decision:** With 2 placements, returning the union instead of trying to compute exact covering pairs is intentional. Computing exact pairs is O(n²) and can miss edge cases. The union is safe because any cell outside it cannot possibly block. The search engine's `filter_turns_by_threats` checks each generated turn against all threat windows.

#### `is_player_win_unblockable(player, available_placements)`

A **hitting-set check**: can all of `player`'s hot windows be blocked with `available_placements` cells?

- **0 placements** → always true (cannot block anything)
- **1 placement** → Is there a single cell present in every threat window? If no such cell exists, the threats are unblockable.
- **2 placements** → Is there a pair of cells `(c1, c2)` such that every threat window contains at least one of them? Brute-force O(n²) over deduplicated candidate cells.

**Used by:** The search engine to detect forced losses and prune branches.

#### `compute_threat_constrained_moves(legal, constrain)`

The top-level API for move filtering:

1. If current player has winning cells → return only legal moves in those cells
2. If opponent has unblockable threats → return `None` (all moves are losing)
3. If opponent has threats → return only legal moves in blocking cells
4. Otherwise → return `None` (no constraint)

**Used by:**
- The search engine's root turn generation (when threats exist)
- The encoder's channel 3 (legal moves mask, when `constrain_threats=true`)
- MCTS root initialization (when `constrain_threats=true`)

### Solver Methods (Ground Truth)

#### `solve_winning_cells(player)`

```rust
pub fn solve_winning_cells(&mut self, player: u8) -> FxHashSet<Hex>
```

**Brute-force solver:** Which cells from `collect_winning_threat_cells` actually participate in a winning turn?

Algorithm:
1. Get candidate cells from the fast path
2. **Single placements:** For each candidate, call `place()`, check if `winner == Some(player)`, then `unmake_move()`. Record winning cells.
3. **Pairs:** For each pair of distinct candidates (when `placements_remaining >= 2`), place both, check for win, unmake both. Record both cells if the pair wins.

**Why this validates the fast path:** The solver starts from the fast path's candidates and confirms each one actually wins. If the fast path ever returns a false positive, this test catches it. If the fast path ever misses a winning cell, the solver would find a winning cell not in the fast path set.

> For this game, the solver and fast path are **provably equivalent** (tested on thousands of random positions), but the solver remains as a safety net.

#### `solve_blocking_cells(player)`

```rust
pub fn solve_blocking_cells(&mut self, player: u8) -> FxHashSet<Hex>
```

**Brute-force solver:** Which single cells actually block the opponent?

Algorithm:
1. Iterate over all `candidates_near2()` cells
2. For each cell, call `place()`, then check `collect_winning_threat_cells(opp)`
3. If opponent has no winning threats after the placement, this cell blocks → record it
4. `unmake_move()` and continue

**Validates:** `collect_blocking_threat_cells` with 1 placement. The solver checks every candidate cell, not just cells in hot windows.

#### `solve_blocking_pairs(player)`

```rust
pub fn solve_blocking_pairs(&mut self, player: u8) -> Vec<(Hex, Hex)>
```

**Brute-force solver:** Which pairs of cells block the opponent?

Algorithm:
1. Iterate over all `candidates_near2()` cells for the first placement
2. If first placement wins the game → record `(c1, c1)` as sentinel
3. Otherwise, if second placement is available, iterate all candidates for second placement
4. After both placements, check if opponent has no winning threats
5. Record blocking pairs

**Validates:** Turn-level blocking with 2 placements. Used by tests to verify that the search engine's `filter_turns_by_threats` does not allow illegal blocking turns through.

### Edge Cases Tested

#### 1. Five-Window (Single Empty)

```rust
// P0 has 5-stone run (0,0)..(4,0)
game.set_position(&[(0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0)], 0, 2);
let cells = game.collect_winning_threat_cells(0);
// Returns: {(-2,0), (-1,0), (5,0), (6,0)} — empties from 4 overlapping 5-windows
```

**Tested:** `collect_winning_threat_cells_five_window`

#### 2. Four-Window (Two Empties)

```rust
// P0 has 4-stone run (0,0)..(3,0)
game.set_position(&[(0,0,0),(1,0,0),(2,0,0),(3,0,0)], 0, 2);
let cells = game.collect_winning_threat_cells(0);
// Returns: {(-2,0), (-1,0), (4,0), (5,0)} — all empties from 3 hot 4-windows
```

**Tested:** `collect_winning_threat_cells_four_window` — also verifies that with only 1 placement, no 4-window can be completed (returns empty set).

#### 3. Single-Cell Block (Intersection)

```rust
// P1 has 4-run (0,0)..(3,0), P0 already blocked at (-2,0)
// Remaining hot windows: (-1,0)..(4,0) → {(-1,0),(4,0)}
//                       (0,0)..(5,0)  → {(4,0),(5,0)}
// Both contain (4,0) → single placement there blocks all
game.set_position(&[(-2,0,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1)], 0, 1);
let cells = game.collect_blocking_threat_cells(1);
// Returns: {(4,0)}
```

**Tested:** `collect_blocking_threat_cells_single_placement` — solver agrees exactly.

#### 4. Two-Placement Block (Union)

```rust
// P1 has bare 4-run (0,0)..(3,0)
// Hot windows: (-2,0)..(3,0) → {(-2,0),(-1,0)}
//              (-1,0)..(4,0) → {(-1,0),(4,0)}
//              (0,0)..(5,0)  → {(4,0),(5,0)}
// With 2 placements: return union of all empties
game.set_position(&[(0,0,1),(1,0,1),(2,0,1),(3,0,1)], 0, 2);
let cells = game.collect_blocking_threat_cells(1);
// Returns: {(-2,0), (-1,0), (4,0), (5,0)}
```

**Tested:** `collect_blocking_threat_cells_two_placements` — solver finds valid pairs from this set.

#### 5. Disjoint Threats (Unblockable)

```rust
// P0 has two disjoint 5-runs: (0,0)..(4,0) and (10,0)..(14,0)
// Each 5-run has multiple disjoint empties
// With 2 placements: cannot block both → unblockable
game.set_position(&[(0,0,0),...,(4,0,0),(10,0,0),...,(14,0,0)], 0, 2);
assert!(game.is_player_win_unblockable(0, 2));
```

**Tested:** `is_player_win_unblockable_disjoint_threats`

#### 6. Three-Axis Star

```rust
// P0 has 4 stones on each axis through (0,0):
// q-axis: (-1,0),(0,0),(1,0),(2,0)
// r-axis: (0,-1),(0,0),(0,1),(0,2)
// diag:   (1,-1),(0,0),(-1,1),(-2,2)
```

**Tested:** `hot_window_on_all_three_axes` — verifies hot windows exist on all 3 axes, solver finds pair-winning cells, and fast path contains all solver cells.

#### 7. Six-Stone Window (Game Over)

```rust
// P0 already has 6 in a row — game is over
game.set_position(&[(0,0,0),...,(5,0,0)], 0, 2);
// The 6-window has 0 empties, but overlapping 5-windows still exist structurally
// collect_winning_threat_cells reports those empties (correct for the position)
// solve_winning_cells returns empty (place() fails on won game)
```

**Tested:** `six_stone_window_is_not_a_threat` — demonstrates that the fast path and solver can diverge on already-won positions (by design; the game is over).

#### 8. Blocked Window

```rust
// P0 has 4 stones but P1 is inside the window at (2,0)
// Window (-1,0)..(4,0) has P1 at offset 3 → NOT hot
game.set_position(&[(-1,0,0),(0,0,0),(2,0,1),(3,0,0),(4,0,0)], 0, 2);
```

**Tested:** `blocked_window_is_not_hot` — verifies no hot window contains an opponent stone.

#### 9. Three-Window Is Not Hot

```rust
// P0 has only 3 stones — no hot windows
game.set_position(&[(0,0,0),(1,0,0),(2,0,0)], 0, 2);
assert!(game.hot_windows[0].is_empty());
```

**Tested:** `three_window_is_not_hot` — confirms the ≥4 threshold.

#### 10. Overlapping Hot Windows

```rust
// P0 5-run: (0,0)..(4,0)
// Two overlapping 5-windows with different empties:
// (-1,0)..(4,0) → empty at (-1,0)
// (0,0)..(5,0)  → empty at (5,0)
```

**Tested:** `overlapping_hot_windows_share_empties` — verifies both empties are winning cells and solver confirms both.

#### 11. Partial Block

```rust
// P1 has two 5-windows that don't share an empty cell
// q-axis: (0,0)..(4,0), empty at (5,0)
// r-axis: (2,0)..(2,4), empty at (2,5)
// Block (5,0) — P1 still threatens (2,5)
```

**Tested:** `solver_blocks_after_partial_block` — confirms that blocking one of two disjoint threats leaves the other active.

#### 12. Random Game Validation (The Big One)

```rust
// Deterministic "random" playout: hash-based move selection
// After EVERY completed turn, cross-check:
//   - fast_win == solver_win (exact equality)
//   - fast_block == solver_block (with 1 placement, exact equality)
```

**Tested:** `fast_blocking_matches_solver_random_positions` — plays out ~80 moves of deterministic pseudo-random Hex, validating the fast path against solvers at every turn. This is the **primary safety mechanism** that catches any divergence between fast path and ground truth.

---

## Neural Encoder (`encoder.rs`)

The single canonical implementation for encoding a `HexGameState` into a 13-channel 33×33 float32 tensor. Both MCTS and the Python training pipeline call into here, eliminating previous duplication.

### Constants

```rust
pub const BOARD_SIZE: i32 = 33;       // tensor width/height
pub const HALF_BOARD: i32 = 16;       // center coordinate
pub const NUM_CHANNELS: usize = 13;
pub const BOARD_AREA: usize = 1089;   // 33×33
pub const TENSOR_SIZE: usize = 14157; // 13×1089
```

### Channel Layout

| Ch | Content |
|----|---------|
| 0 | Current player's stones (1.0 where they have a stone) |
| 1 | Opponent's stones |
| 2 | Empty cell mask (`1.0 - ch0 - ch1`) |
| 3 | Legal moves mask (constrained if `constrain_threats=true`) |
| 4 | Turn phase: all 1.0 if on second placement |
| 5 | First stone of current turn (phase 2 only) |
| 6 | Current player color: all 1.0 if player 0 |
| 7 | Own stone recency: `1/(1 + plies_ago)` |
| 8 | Opponent stone recency |
| 9 | Opponent "hot cells" (empties in opponent's hot windows) |
| 10 | Own "hot cells" |
| 11 | Distance from centroid: `hex_dist(cell, center) / HALF_BOARD` |
| 12 | Opponent's most recent completed turn cells |

### Centroid Computation

The board centroid (mean of all occupied cells) is computed and **banker's-rounded** (round half to even) to match Python's `round()` behavior exactly. The offset is chosen so the centroid maps to tensor center `(16, 16)`.

### Threat-Constrained Encoding

When `constrain_threats=true`, channel 3 only includes moves from `compute_threat_constrained_moves`. This means:
- If the current player has an instant win → only winning cells are legal
- If the opponent has threats → only blocking cells are legal
- If the opponent has unblockable threats → no constraint (all moves are losing)

MCTS internal nodes always use `constrain_threats=false` to avoid the O(n²) unblockable check on every leaf expansion.

---

## Classical Evaluation (`eval.rs`)

### `extract_features` — 13-Element Feature Vector

Used by the classical self-play pipeline to generate training data. Scans the entire board to count live and dead runs of various lengths.

**Features per player (indices 0–5 for P0, 6–11 for P1):**

| Index | Feature | Definition |
|-------|---------|------------|
| 0,6 | live-5 | 5+ in a row with ≥1 open end |
| 1,7 | dead-5 | 5 in a row, blocked on one end |
| 2,8 | live-4 | 4 in a row, both ends open |
| 3,9 | dead-4 | 4 in a row, one end open |
| 4,10 | live-3 | 3 in a row, both ends open |
| 5,11 | live-2 | 2 in a row, both ends open |
| 12 | tempo | 1.0 if P0 to move, -1.0 otherwise |

### Run Detection Algorithm

For each occupied cell and each of the 3 principal directions:

1. **Skip if not a run start** — if the predecessor cell in the negative direction is the same player, this cell is in the middle of a longer run
2. **Count forward** — contiguous same-player cells + whether the end is open (empty) or blocked (opponent)
3. **Determine backward open end** — the predecessor is known to not be the same player; if it's empty, the end is open
4. **Classify** by run length and total open ends (0, 1, or 2)

This guarantees each run is counted **exactly once**.

---

## Alpha-Beta Search (`search.rs`)

Turn-based alpha-beta with a full suite of modern chess-engine techniques adapted for a 2-placement-per-turn hex game.

### Key Design Decisions

| Feature | Rationale |
|---------|-----------|
| **Turn-based search** | A "turn" (1–2 placements) is the atomic search unit. Doubles effective depth vs placement-based search. |
| **Instant-win detection** | Checks `hot_windows` at every node. A 5-window (1 empty) is an instant win even with 2 placements remaining — the game ends after the 6th stone. |
| **Threat-filtered move generation** | `filter_turns_by_threats` prunes turns that don't block opponent threats. Combined with `is_opponent_win_unblockable` for early forced-loss detection. |
| **Quiescence search** | Extends search on threat moves only (depth 6). Only explores cells in hot windows. |
| **PVS + LMR** | Principal Variation Search with Late Move Reduction. Aggressive: reduction starts at move 2. |
| **Killer / history heuristics** | Killer moves per ply + history scores (depth² on cutoff). |
| **Aspiration windows** | Narrow window around previous best score, re-search with full window on failure. |
| **Mate-distance TT scoring** | Stores `WIN_SCORE ± ply` in transposition table, adjusts on load/store. |
| **Noise injection** | Deterministic pseudo-random noise in root candidate ordering for training data variety. Threat handling remains deterministic. |

### `Turn` Type

```rust
pub struct Turn {
    pub m1: Hex,
    pub m2: Option<Hex>,
}
```

- `Turn::one(m)` — single placement (opening move or when 1 placement remains)
- `Turn::two(a, b)` — canonical ordering (`a <= b`) for TT consistency

### Instant Win Detection (`find_instant_win`)

```rust
fn find_instant_win(game: &HexGameState, player: u8) -> Option<Turn>
```

Iterates over `hot_windows[player]`, counts empties in each window:
- **1 empty** → `Turn::one(empty)` wins immediately (even with 2 placements, game ends on 6th stone)
- **2 empties + remaining ≥ 2** → `Turn::two(empty1, empty2)`
- Otherwise → continue searching

**Bug fix history:** An earlier version searched `candidates_near2()` for a distinct second cell when a 5-window existed with 2 placements remaining. This could fail if no other candidate existed, causing the search to miss a guaranteed win. The fix: always return `Turn::one` for any 5-window — the game engine ends the turn immediately on win.

### Turn Generation

**Root turns:**
1. Check opening move → `Turn::one(ORIGIN)`
2. Check instant win → return winning turn
3. Generate sorted candidates (eval delta + history + tactical bonuses + optional noise)
4. Add "colony candidate" — a cell far from the centroid to encourage expansion
5. If single-placement turn → map candidates to `Turn::one`
6. Generate pairs with pair-sum constraint (`i + j <= 12`)
7. Filter by threats

**Inner turns:** Same but without opening/colony logic, and always deterministic (`noise_level = 0`).

### Candidate Scoring

```rust
score = delta * sign * 15 + history + tactical + noise
```

- `delta` = `move_eval_delta` for this cell
- `sign` = +1 for P0, -1 for P1 (eval is from P0's perspective)
- `history` = cumulative history heuristic score (capped at 500,000)
- `tactical` = +50,000 for cells in opponent's hot windows (blocking), +40,000 for cells in own hot windows (completing)
- `noise` = pseudo-random perturbation proportional to base score magnitude (only at root, only when `noise_level > 0`)

Tactical bonuses are **never affected by noise** — blocking must remain deterministic.

### Transposition Table

```rust
pub struct TTEntry {
    pub depth: i32,
    pub score: i32,
    pub flag: TTFlag,      // Exact, LowerBound, UpperBound
    pub best_turn: Option<Turn>,
}
```

- Hash = `zobrist_hash ^ side_code ^ phase_code` (encodes player-to-move and placements-remaining)
- Mate-distance adjustment: store `score + ply` for wins, `score - ply` for losses
- TT cleared when size exceeds 2,000,000 entries

---

## MCTS Engine (`mcts.rs`)

Arena-allocated MCTS with PUCT, virtual loss, and batch leaf selection. The tree lives entirely in Rust; Python drives the loop.

### Python Usage Pattern

```python
engine = MCTSEngine(game, num_sims, c_puct=1.4, near_radius=8)
tensor, oq, or_, legal = engine.init_root()
# ... GPU inference → policy, value ...
engine.expand_root(policy, value, oq, or_, legal)
engine.add_dirichlet_noise(noise, 0.25)

while not engine.done():
    tensors, count = engine.select_leaves(batch_size)
    # ... GPU inference → policies, values ...
    engine.expand_and_backprop(policies, values)

moves_q, moves_r, visits, root_value = engine.get_results()
```

### `MCTSNode`

Stored contiguously in a `Vec<MCTSNode>` (the arena). Each node tracks:
- Parent index, action `(q, r)`, policy prior
- Visit count, total accumulated value
- Child range (`children_start`, `children_count`)
- Player whose turn it is at this node

### PUCT Selection

```text
score = Q + c_puct * prior * sqrt(parent_visits) / (1 + child_visits)
```

- Dynamic `c_puct`: grows logarithmically with parent visits (`c_puct + ln((N + init) / init)`)
- FPU (First-Play Urgency): unvisited children get `Q = parent_Q - 0.2`
- Virtual loss: +1 visit added during selection, removed during backprop (prevents parallel batch slots from converging on the same branch)

### Re-Rooting and Subtree Reuse

After placement 1 is selected, `re_root` advances the tree so placement 2's MCTS starts from the surviving subtree. The arena is **not compacted** — dead siblings remain in memory but are never referenced.

**Threat constraint safety:** When `constrain_threats=true` and the new root was expanded with unconstrained children (internal nodes always use `constrain_threats=false`), the old children are **BFS-invalidated** so they don't leak into training data extraction as spurious off-policy positions.

### Training Data Extraction

`extract_tree_node_states(min_visits)` replays move sequences from high-visit expanded nodes, encodes their board states, and returns packed tensors + move histories for the neural network training pipeline.

---

## Python Bridge (`pybridge.rs`)

PyO3 bindings exposing the engine to Python. Two main Python-facing types:

### `PyHexGame`

Wrapper around `HexGameState` with methods for:
- Placement, undo, query state
- Legal move generation (various radii, byte-packed formats)
- Threat window inspection
- Board encoding (`encode_board_and_legal`)
- Classical search (`classical_search`, `classical_search_turn`)
- Position setup (`set_position`)

### `PyMCTSEngine`

Wrapper around `MCTSEngine` with methods for:
- Root initialization and expansion
- Dirichlet noise injection
- Batch leaf selection and backpropagation
- Results extraction
- Subtree re-rooting
- Training data extraction (`extract_tree_node_states`)

### Bulk Self-Play

```rust
#[pyfunction]
fn classical_self_play(num_games, time_ms, max_depth, near_radius, max_moves)
    -> Vec<(Vec<f32>, f32, Vec<(i32, i32, u8)>)>
```

Generates self-play data using the classical alpha-beta engine. Fast path for bootstrap training data before the neural network is strong enough for MCTS self-play.

---

## Module Re-Exports (`game.rs`)

A thin 8-line module for backward compatibility:

```rust
pub use crate::board::{GameError, HexGameState, MoveRecord};
pub use crate::patterns::{PLACEMENT_RADIUS, WIN_LENGTH};
```

Existing code using `crate::game::HexGameState` continues to work.

---

## How Threats and Patterns Interact

The relationship between `patterns.rs` and `threats.rs` is the core tactical engine:

```
Placement ──► compute_eval_delta ──► updates window_indices
                   │                     │
                   ▼                     ▼
            PATTERN_VALUES          PATTERN_COUNTS
            (score delta)           (p0_count, p1_count)
                   │                     │
                   ▼                     ▼
            window_eval += delta    classify() ──► window_fives/fours/threes
                   │                     │
                   ▼                     ▼
                                    hot_windows transition
                                           │
                                           ▼
                              threats::collect_threat_window_empties
                                           │
                                           ▼
                              ┌─────────────────────────────┐
                              │  Fast path (production)     │
                              │  - collect_winning_cells    │
                              │  - collect_blocking_cells   │
                              │  - is_win_unblockable       │
                              └─────────────────────────────┘
                                           │
                              ┌─────────────────────────────┐
                              │  Solver (verification)      │
                              │  - solve_winning_cells      │
                              │  - solve_blocking_cells     │
                              │  - solve_blocking_pairs     │
                              └─────────────────────────────┘
                                           │
                                           ▼
                              search::filter_turns_by_threats
                              encoder::compute_threat_constrained_moves
```

**Critical invariant:** `patterns.rs` maintains `hot_windows` incrementally on every place/unmake. `threats.rs` reads these sets (O(1) access) and derives tactical information. The solvers validate that this tactical information is correct by brute-force simulation.

---

## Build Configuration

```toml
[package]
name = "hexgame"
version = "0.2.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
rustc-hash = "2"
pyo3 = { version = "0.24", features = ["extension-module", "abi3-py310"], optional = true }
numpy = { version = "0.24", optional = true }

[features]
default = []
python = ["pyo3", "numpy"]

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
```

- `cdylib` — Python extension module
- `rlib` — Rust library for internal use
- `python` feature — enables PyO3 + numpy bindings
- Release profile: full LTO, single codegen unit, stripped — optimized for inference speed
