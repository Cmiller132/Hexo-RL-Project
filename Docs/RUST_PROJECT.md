# Rust Project Overview — Hexgame Engine

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Dependency Graph](#architecture--dependency-graph)
3. [Module Layout](#module-layout)
4. [Detailed File Reference](#detailed-file-reference)
   - [Core Layer](#core-layer)
   - [Evaluation Layer](#evaluation-layer)
   - [Board Layer](#board-layer)
   - [Threat Layer](#threat-layer)
   - [Search & Learning Layer](#search--learning-layer)
   - [Python Bridge Layer](#python-bridge-layer)
   - [Test Suite](#test-suite)
5. [Key Design Decisions](#key-design-decisions)
6. [Performance Budget](#performance-budget)
7. [Configuration & Tooling](#configuration--tooling)
8. [Testing Strategy](#testing-strategy)
9. [Python Integration](#python-integration)

---

## Project Overview

The `hexgame` crate is a high-performance game engine for **Hexo**, a new game most similar to Connect 6 played on an infinite hexagonal grid where two players alternate placing stones with the goal of forming a contiguous 6-in-a-row line. The engine supports both classical alpha-beta search and neural-network-guided MCTS, with PyO3 bindings exposing the engine to Python for reinforcement learning training.

Hexo has no relation to Hex other than sharing hexagonal board geometry.

### Game Rules

- Two players (0 and 1) alternate turns on an infinite board using axial coordinates `(q, r)`.
- Player 0 opens with **one** stone at the origin `(0, 0)`.
- Every subsequent turn consists of **two** stone placements (except the opening).
- Each placement must land on an empty hex within radius 8 of any existing stone.
- The first player to form 6 contiguous stones along any of the three hex axes wins.
- The board is infinite — there is no draw condition.

### Project Goals

- **Correctness:** All game rules and threat analysis validated against a brute-force oracle via property-based testing.
- **Performance:** Zero heap allocation on all hot paths; incremental evaluation updates only 18 windows per move; `O(1)` move generation and radius validation.
- **Neuro-Symbolic Architecture:** Shares a single encoder between classical bootstrap training and neural MCTS; the same `EvalState` drives both the classical evaluator and the live-cell channels used by the neural network.
- **Python-RL Ready:** MCTS tree lives entirely in Rust; Python calls `select_leaves` → GPU inference → `expand_and_backprop` in a tight training loop with GIL released during all Rust work.

---

## Architecture & Dependency Graph

The codebase follows a strict layered architecture with zero circular dependencies. Each layer depends only on the layers below it.

```
┌─────────────────────────────────────────────────────────────────┐
│                        pybridge (PyO3)                          │
│  src/pybridge/mod.rs  —  PyHexGame, classical_self_play         │
│  src/pybridge/mcts.rs —  PyMCTSEngine                           │
├─────────────────────────────────────────────────────────────────┤
│              encoder            search            mcts          │
│         src/encoder.rs    src/search.rs     src/mcts.rs         │
│         (13-ch tensor)    (alpha-beta)     (neural MCTS)        │
├─────────────────────────────────────────────────────────────────┤
│                         threats                                 │
│                     src/threats.rs                               │
│            ThreatStatus, blocking, live cells                   │
├─────────────────────────────────────────────────────────────────┤
│                          board                                  │
│                      src/board.rs                               │
│          HexGameState, rules, win detection, Zobrist            │
├─────────────────────────────────────────────────────────────────┤
│                     eval (evaluation)                           │
│         src/eval/state.rs   — EvalState (incremental)           │
│         src/eval/hot.rs     — HotWindows cache                  │
│         src/eval/patterns.rs — PATTERN_VALUES[729]              │
│         src/eval/grid.rs    — WIN_GRID spatial indexing         │
├─────────────────────────────────────────────────────────────────┤
│                         core                                    │
│                      src/core.rs                                │
│            Hex, Turn, WindowKey, hex_distance                   │
└─────────────────────────────────────────────────────────────────┘
                              ↑
                         tests/oracle  (test-only brute-force)
```

**Dependency rule:** `core → eval → board → threats → {search, mcts, encoder} → pybridge`

**Visibility:** `board`, `core`, `encoder`, `eval` are `pub` — these form the stable public API. `mcts`, `search`, and `threats` are `pub(crate)` — internal search engines are not yet stable for downstream Rust consumers, but are accessible to the Python bridge and tests within the crate.

---

## Module Layout

```
src/
├── lib.rs                  # Crate root: module declarations + re-exports (67 lines)
├── core.rs                 # Hex, Turn, WindowKey, hex_distance (~305 lines)
├── eval/
│   ├── mod.rs              # Re-exports EvalState, ThreatCounts
│   ├── patterns.rs         # PATTERN_VALUES[729], PATTERN_COUNTS[729], POW3[6]
│   ├── grid.rs             # WIN_GRID constants, win_grid_idx
│   ├── hot.rs              # HotWindows (SmallVec-backed threat cache)
│   └── state.rs            # EvalState: incremental score, counts, delta stack
├── board.rs                # HexGameState: rules, placement, undo, Zobrist, win detection
├── threats.rs              # ThreatStatus, threat_status(), live_cells()
├── encoder.rs              # 13-channel NN tensor encoder + classical feature extraction
├── search.rs               # Alpha-beta with iterative deepening, TT, quiescence
├── mcts.rs                 # Arena-allocated MCTS with PUCT, virtual loss, batch leaves
├── pybridge/
│   ├── mod.rs              # PyHexGame + classical_self_play
│   └── mcts.rs             # PyMCTSEngine (GIL-free hot path)
└── tests/                  # Crate-internal unit tests (pub(crate) access)
    ├── mod.rs              # Test module declarations
    ├── oracle.rs           # Brute-force solver (test-only ground truth)
    ├── threats.rs          # Proptest: fast path vs oracle (1000+ cases)
    ├── threats_internal.rs # Hand-crafted threat unit tests
    ├── mcts.rs             # MCTS engine correctness tests (5 tests)
    ├── core.rs             # Hex, Turn, WindowKey unit tests
    ├── patterns.rs         # Pattern table integrity, checksum tests
    ├── eval_state.rs       # EvalState place/unplace round-trip tests
    ├── grid.rs             # Win grid indexing bijection tests
    └── hot.rs              # HotWindows insert/remove/clear tests

tests/                      # Integration tests (public API only)
├── board.rs                # Full-game scenario tests + proptests
└── encoder.rs              # Encode/decode tests + proptests

benches/
└── engine.rs               # Criterion benchmarks (encode, legal_moves, candidates)

Python/tests/
└── test_engine_smoke.py    # PyO3 extension smoke tests (4 tests)
```

---

## Detailed File Reference

### Core Layer

#### `src/core.rs` — Fundamental Primitives

Defines the three foundational types used throughout the engine.

**Key types:**

- **`Hex`** — Axial coordinate `(q: i32, r: i32)`. Supports arithmetic, `Ord` (lexicographic), `Hash`, `Display`. Used as the universal board coordinate.
- **`Turn`** — A single placement (`Turn::single(h)`) or a pair (`Turn::pair(a, b)`). Enforces canonical ordering: `first <= second` so `Turn::pair(a,b) == Turn::pair(b,a)`. This invariant is essential for transposition-table consistency. Fields are private with accessors.
- **`WindowKey`** — A compact 32-bit key packing `(dir: 2 bits, r: 15 bits signed, q: 15 bits signed)`. Enables `O(1)` hashing and comparison of win-threat window origins. Fits in one word; no heap allocation.

**Key functions:**

- `hex_distance(a, b) -> i32` — Manhattan distance on the axial hex grid.
- `HEX_DIRECTIONS: [(i32, i32); 3]` — The three principal hex axes `(1,0)`, `(0,1)`, `(1,-1)`.

**Constants:** `WIN_LENGTH = 6`, `PLACEMENT_RADIUS = 8`.

**Depends on:** Nothing (leaf module — only `std::fmt`).

---

### Evaluation Layer

#### `src/eval/patterns.rs` — Static Pattern Tables

Contains compile-time-generated ternary (base-3) pattern lookup tables for 6-cell hex windows. Every contiguous 6-cell window is encoded as an integer `0..728` using base-3 digits (0=empty, 1=P0, 2=P1; LSB = window offset 0).

**Key data:**

- **`PATTERN_VALUES: [i32; 729]`** — CMA-ES optimized static evaluation scores for each pattern. Bit-identical to the original tuned weights.
- **`PATTERN_COUNTS: [(u8, u8); 729]`** — Per-player stone counts for each pattern `(P0_count, P1_count)`.
- **`POW3: [usize; 6]`** — Powers of 3: `[1, 3, 9, 27, 81, 243]`.

Everything is `const` — zero runtime initialization cost. The `build_pattern_counts()` function runs at compile time. A FNV-1a checksum test (`0x9f5d14a209044de4`) guards against accidental table corruption.

**Depends on:** Nothing (leaf).

---

#### `src/eval/grid.rs` — Spatial Win Grid Indexing

Defines the dense spatial grid used by `EvalState` for incremental pattern tracking.

**Key constants:**

- `WIN_GRID_RADIUS = 30` — Covers `[-30, 30]` on both axes.
- `WIN_GRID_SIDE = 61` — 61 cells per axis.
- `WIN_GRID_TOTAL = 11,163` — `61 × 61 × 3 directions`.

**Key functions:**

- `win_grid_idx(q, r, dir) -> usize` — Flatten 3D coordinate `(q, r, dir)` into a 1D array index.
- `win_grid_in_bounds(q, r) -> bool` — Runtime bounds guard. Window origins can drift beyond radius 30 in long games; out-of-bounds windows are silently skipped.

**Depends on:** Nothing (leaf).

---

#### `src/eval/hot.rs` — Hot Windows Cache

Zero-allocation cache tracking which windows currently represent win threats (4+ stones of one player, 0 opponent stones). Uses `SmallVec<[WindowKey; 32]>` per player — inline storage for 32 entries, no heap allocation in practice.

**Key type:** `HotWindows` with methods:

- `new()` — Empty cache.
- `insert(player, key)` — Add a hot window.
- `remove(player, key)` — Remove via `swap_remove` (`O(n)` find, `O(1)` removal).
- `iter(player) -> impl Iterator<Item = WindowKey>` — Iterate hot windows.
- `len(player) -> usize`, `is_empty(player) -> bool` — Query methods.
- `clear()` — Zero all state in place, preserving allocations.

Linear scan beats hashing for the typical small-N use case (≤20 hot windows per player). `WindowKey` compares in one instruction.

**Depends on:** `core::WindowKey`, `smallvec::SmallVec`.

---

#### `src/eval/state.rs` — Incremental Evaluation State

The central evaluation subsystem. Maintains a dense win grid (`Box<[u16; 11163]>`), accumulated score, per-player `ThreatCounts`, hot windows, and an undo delta stack. Every field is private; mutation only through `place()`/`unplace()`.

**Key types:**

- **`EvalState`** — Core struct with `score`, `counts`, `hot`, `indices`, `delta_stack`.
- **`ThreatCounts`** — Per-player `fives`, `fours`, `threes` counts.
- **`ThreatCountsDelta`** — Signed deltas (`+1`/`-1`) per threat tier.
- **`EvalDelta`** — Snapshot of `(cell, cell_val, score, counts_delta)` for undo.

**Key public API:**
| Method | Purpose |
|---|---|
| `new()` | Allocate a fresh `Box<[u16; 11163]>` (22 KB heap). |
| `clear()` | Zero all state in place, preserving the existing heap allocation. Used in `HexGameState::reset()` to avoid reallocation. |
| `place(cell, player)` | Incremental placement. Touches 18 windows (3 directions × 6 offsets). Returns delta pushed onto stack. |
| `unplace()` | Pops delta stack, reverses the placement. Exactly restores pre-place state. |
| `score() -> i32` | Current static evaluation (from P0's perspective). |
| `counts(player) -> ThreatCounts` | Read threat counts for one player. |
| `has_threats(player) -> bool` | Fast check for any fours or fives. |
| `has_any_threats() -> bool` | Fast check for any threats on either side. |
| `hot_windows(player) -> impl Iterator<Item = WindowKey>` | Iterate hot windows for one player. |
| `hot_len(player) -> usize` | Number of hot windows. |
| `hypothetical_score_delta(cell, player) -> i32` | Read-only score change estimate for move ordering. |

**Correctness guards:** Pattern index bounds checks (`new_idx < 729`) use runtime `assert!` (not `debug_assert!`) in `place()`, `unplace()`, and `hypothetical_score_delta()`. These remain in release builds so malformed input cannot silently corrupt the pattern tables. An underflow guard in `unplace()` uses `debug_assert!` since the invariant is structurally guaranteed.

**Key implementation details:**

- `place()` uses the `visit_windows` iterator to enumerate 18 window origins touching the placed stone. For each, it looks up the old pattern index, computes the new index by adding `cell_val * POW3[offset]`, updates score/counts/hot, and writes the new index back.
- `unplace()` has an `#[cfg(debug_assertions)]` invariant check that recomputes hot windows from scratch and asserts equality with the incremental cache.
- `Clone` copies ~22 KB (the boxed indices array). Acceptable for search-invocation clones but would be costly inside MCTS node expansion.

**Depends on:** `core` (Hex, WindowKey, directions), `eval/grid`, `eval/hot`, `eval/patterns`.

---

#### `src/eval/mod.rs` — Evaluation Module Hub

Re-exports `EvalState` and `ThreatCounts` from `state`. Declares the four submodules (`grid`, `hot`, `patterns`, `state`). No logic.

---

### Board Layer

#### `src/board.rs` — Game State & Rules Engine

The authoritative game state. All fields are private; access is through read-only accessors and mutation methods.

**Key type:** `HexGameState`

| Field                  | Type                      | Purpose                                                                              |
| ---------------------- | ------------------------- | ------------------------------------------------------------------------------------ |
| `stones`               | `FxHashMap<Hex, u8>`      | All occupied cells (sparse infinite-board representation)                            |
| `eval`                 | `EvalState`               | Composed evaluation state (board never touches eval internals)                       |
| `candidates`           | `CandidateSet` (radius 2) | Reference-counted empty cells near stones for fast `O(1)` move generation            |
| `placement_candidates` | `CandidateSet` (radius 8) | Reference-counted empty cells within `PLACEMENT_RADIUS` for `O(1)` radius validation |
| `zobrist`              | `u64`                     | Incremental Zobrist hash (deterministic FNV-1a, no precomputed table)                |
| `winner`               | `Option<u8>`              | Winner when game over, `None` otherwise                                              |
| `winning_line`         | `Option<[Hex; 6]>`        | The exact 6-in-a-row that produced the win (stack-allocated, no heap)                |
| `move_history`         | `Vec<MoveRecord>`         | Undo stack recording pre-move state                                                  |

**`MoveRecord`** captures the full pre-move state for exact restoration: `cell`, `player`, `current_player_before`, `placements_remaining_before`, `winner_before`, and `winning_line_before: Option<[Hex; 6]>` (stack-allocated array, no per-move heap allocation).

**Key public API:**
| Method | Purpose |
|---|---|
| `new()` | Empty board, P0 to move, 1 placement remaining. |
| `place(q, r) -> Result<bool, GameError>` | Validate + place stone. Returns whether the turn ended. |
| `unplace()` | Pop move history, restore previous state. |
| `set_position(stones, player, remaining) -> Result<(), GameError>` | Bulk-load a position (test/oracle use). **Validates:** first stone must be at `Hex::ORIGIN`, player ∈ {0,1}, every stone within radius of predecessor, no duplicates. |
| `reset()` | Clear all state to initial. Uses `EvalState::clear()` (zero-in-place) instead of reallocating. |
| Accessors | `stones()`, `eval()`, `current_player()`, `placements_remaining()`, `winner()`, `winning_line()`, `move_count()`, `move_history()`, `zobrist()`, `is_over()` |
| `legal_moves() -> Vec<Hex>` | All legal cells (unsorted — fast path). |
| `legal_moves_near(radius) -> Vec<Hex>` | Legal cells within `radius` of any stone (**unsorted** — hot-path optimization). |
| `legal_moves_near_sorted(radius) -> Vec<Hex>` | Sorted variant (for tests and Python export only). |
| `candidates_near2() -> Vec<Hex>` | Fast radius-2 candidate cells (**unsorted**). |
| `candidates_near2_sorted() -> Vec<Hex>` | Sorted variant (for tests and Python export). |
| `opponent_last_turn_cells() -> SmallVec<[Hex; 2]>` | Cells placed in opponent's last completed turn (encoder channel 12). Stack-allocated. |

**Key implementation details:**

- **Radius validation is `O(1)`**: `validate_move` queries `placement_candidates.contains(cell)` (an `FxHashMap` lookup) instead of scanning all stones. The `placement_candidates` set is maintained incrementally by `commit_placement`/`unplace` alongside the existing `candidates` set.
- **`CandidateSet`** implements reference-counted incremental updates: `on_place` increments neighbor refs, `on_unplace` decrements, `contains()` is `O(1)`. Two separate sets with different radii (2 for move gen, 8 for validation) are maintained in parallel.
- **Win detection** scans 3 hex axes from the newly placed stone, extracting a 6-stone segment centered on the pivot. `select_segment` returns a stack-allocated `[Hex; 6]` (no allocation).
- **Opening move enforcement**: first move in `set_position` must be `Hex::ORIGIN`. `place()` enforces this independently for the normal game flow.
- Sorting in hot-path move generation methods has been removed (MCTS re-sorts by PUCT score; deterministic output is opt-in via `_sorted` variants).

**Depends on:** `core`, `eval/state`.

---

### Threat Layer

#### `src/threats.rs` — Tactical Threat Analysis

Classifies the tactical situation, checks turn legality under threat constraints, and enumerates live cells.

**Key types:**

- **`ThreatStatus`** — Enum: `Quiet` (no threats), `WinningTurn(Turn)` (forced win exists), `MustBlock(BlockConstraint)` (must play specific cells/pairs), `Unblockable` (cannot stop opponent). **`MustBlock` stores `BlockConstraint` directly** (no heap `Box` — the struct fits on the stack).
- **`BlockConstraint`** — Exact blocking sets: `cells: SmallVec<[Hex; 16]>` (single-placement intersection) and `pairs: SmallVec<[(Hex, Hex); 32]>` (exact 2-placement covering pairs). Accessors: `cells() -> &[Hex]`, `pairs() -> &[(Hex, Hex)]`.

**Key public functions:**

1. **`threat_status(game) -> ThreatStatus`** — Full tactical classification. Short-circuits: checks instant wins first, then builds `BlockConstraint`.
2. **`turn_satisfies_status(status, turn) -> bool`** — Membership test against pre-computed constraint. `O(1)` for single-placement, `O(pairs)` for 2-placement.
3. **`live_cells(game, player, out: &mut Vec<Hex>)`** — Enumerates empty cells in hot windows (caller-owned buffer, reusable). Uses a local `FxHashSet` for `O(1)` deduplication instead of `O(n²)` linear scan.

**Crate-private:** `generate_threat_turns(game, out, opp_buf, my_buf)` — Produces candidate turns for quiescence search from live cells. Takes caller-owned scratch buffers.

**Key implementation details:**

- **`BlockConstraint` is exact** — in the 2-placement case it enumerates covering pairs up front, rather than returning a permissive superset. This eliminates the need for callers to re-validate.
- **`opponent_threat_windows`** returns a **flat representation**: `(SmallVec<[Hex; 32]>, SmallVec<[u8; 16]>)` — all empty cells concatenated plus per-window length markers. No per-window `SmallVec` clones.
- `threat_status` returns `Quiet` immediately when `game.winner().is_some()` (no threats matter after game ends).
- All hot-path data structures use `SmallVec` with inline capacity; no heap allocation in `threat_status`.

**Depends on:** `board`, `core`.

### Search & Learning Layer

#### `src/search.rs` — Classical Alpha-Beta Search

Turn-based alpha-beta with iterative deepening, transposition table (TT), quiescence search, and multiple pruning strategies. This module is `pub(crate)` — internal API, not exposed to downstream Rust consumers.

**Key types:**

- **`SearchState`** — Holds TT (`FxHashMap<u64, TTEntry>`), killer moves per ply, history heuristic, scratch buffers for quiescence, noise seed/level. Public fields: `tt`, `nodes`, `deadline`, `aborted`.
- **`SearchResult`** — `best_turn`, `best_move`, `score`, `depth_reached`, `nodes`, `root_candidates`.
- **`TTEntry`** — `(score, depth, flag: TTFlag, best_turn)` with `TTFlag` enum (`Exact`, `LowerBound`, `UpperBound`).

**Key public function:**

- `iterative_deepening(game, time_limit, max_depth, near_radius, collect_candidates, noise_level) -> Result<SearchResult, GameError>`

**Pruning stack (applied in order):**

1. Instant-win detection (check before move generation)
2. Unblockable-loss pruning (return large negative score)
3. Threat-filtered move generation (only blocking turns kept)
4. Reverse futility pruning (depth ≤ 2, static eval margin)
5. Late-move pruning (skip moves late in ordering at low depth)
6. PVS + LMR (Principal Variation Search with Late Move Reduction)

**Key implementation details:**

- Move ordering: eval delta → history heuristic (depth²) → tactical bonuses (+50k blocking, +40k completing own threats) → noise injection.
- TT uses Zobrist hash XOR side-to-move XOR phase. Mate scores adjusted by ply distance. Auto-clears at 2M entries.
- Quiescence search extends up to 6 turns deep using only threat-generated turns.
- Scratch buffers (`scratch_turns`, `scratch_opp`, `scratch_my`) on `SearchState` eliminate per-node allocations in quiescence.

**Depends on:** `board`, `core`, `encoder::WIN_SCORE`, `threats`.

---

#### `src/mcts.rs` — Neural MCTS Engine

Arena-allocated MCTS with PUCT, virtual loss, batch leaf selection, and subtree reuse. Tree lives entirely in Rust; Python calls `select_leaves(batch_size)` → GPU inference → `expand_and_backprop(policies, values)`. This module is `pub(crate)` — internal search engine, not yet stable for external Rust consumers.

**Key type:** `MCTSEngine`

| Field               | Type               | Visibility   | Purpose                                                                           |
| ------------------- | ------------------ | ------------ | --------------------------------------------------------------------------------- |
| `arena`             | `Vec<MCTSNode>`    | `pub(crate)` | All tree nodes in a single contiguous allocation                                  |
| `root_idx`          | `u32`              | `pub(crate)` | Index of current root node in arena                                               |
| `game`              | `HexGameState`     | private      | Mutable board state used during tree traversal                                    |
| `batch_buf`         | `Vec<f32>`         | private      | Pre-allocated tensor buffer for batch leaf encoding                               |
| `pending`           | `Vec<PendingLeaf>` | private      | Leaves waiting for GPU evaluation                                                 |
| `scratch_raw`       | `Vec<f64>`         | private      | Scratch buffer for softmax in f64                                                 |
| `scratch_priors`    | `Vec<f32>`         | private      | Scratch buffer for gathered priors                                                |
| `hot_buf`           | `Vec<Hex>`         | private      | Reusable buffer for `encode_board_into` live-cells channels                       |
| `legal_buf`         | `Vec<Hex>`         | private      | Reusable buffer for `encode_board_into` legal-moves output                        |
| `c_puct`            | `f32`              | private      | PUCT exploration constant                                                         |
| `c_puct_init`       | `f32`              | private      | KataGo-style dynamic c_puct initialization value (wired through constructor)      |
| `sims_done`         | `u32`              | private      | Completed simulations (incremented in `expand_and_backprop`, not `select_leaves`) |
| `num_simulations`   | `u32`              | private      | Total simulations to perform                                                      |
| `near_radius`       | `i32`              | private      | Radius for legal-move generation during encoding                                  |
| `constrain_threats` | `bool`             | private      | Whether to threat-constrain moves at root                                         |

**Key public API:**
| Method | Purpose |
|---|---|
| `new(game, num_sims, c_puct, near_radius, constrain_threats)` | Convenience constructor (c_puct_init defaults to 19652.0). |
| `with_arena_sim_hint(game, num_sims, arena_hint, c_puct, near_radius, constrain_threats, c_puct_init)` | Full constructor with arena size hint and explicit c_puct_init. |
| `init_root() -> Option<(tensor, oq, or, legal)>` | Encode root position for GPU inference. Returns `None` if game over. |
| `expand_root(policy_logits, value, oq, or, legal)` | Expand root node from GPU policy output. |
| `add_dirichlet_noise(noise, fraction)` | Inject exploration noise at root. **Validates** noise length ≥ children count. |
| `select_leaves(batch_size) -> (&[f32], u32)` | Select batch leaves, return flat tensor slice + non-terminal count. |
| `expand_and_backprop(policies, values)` | GPU output → node expansion → backprop. **Validates** policy/value slice lengths against non-terminal leaf count. Increments `sims_done`. |
| `done() -> bool` | Whether `sims_done >= num_simulations`. |
| `re_root(q, r, new_num_sims)` | Subtree reuse after opponent's move. |
| `get_results() -> (Vec<i32>, Vec<i32>, Vec<u32>, f32)` | Final statistics: moves_q, moves_r, visit_counts, root_value. |
| `root_child_count() -> u16`, `root_child_priors() -> Vec<f32>`, `root_child_q_values() -> Vec<f32>` | Child statistics accessors. |
| `extract_tree_node_states(min_visits) -> Result<TreeNodeStates, &str>` | Training data export. |

**Key implementation details:**

- **Backprop sign flip uses depth parity**: iterates the search path from leaf → root, flipping `-value` at each step. This eliminates a correctness bug where fresh nodes (player=255) caused incorrect sign using the old player-identity comparison.
- **Virtual loss correctly pessimizes Q**: when applying VL, `total_value -= VIRTUAL_LOSS_VISITS as f32`; when removing VL, `total_value += VIRTUAL_LOSS_VISITS as f32`. This produces the correct pessimistic `Q = (total_value - VL) / (visit_count + VL)` during batch selection.
- **`sims_done` is incremented in `expand_and_backprop`** (after actual backprop), not in `select_leaves`. This prevents `done()` from firing prematurely with VL leaves still outstanding.
- **`expand_and_backprop` validates** both `policies.len()` and `values.len()` against the non-terminal leaf count. Mismatched Python batches panic with a clear assertion message rather than silently corrupting the tree or indexing out of bounds.
- **Children count overflow guard**: `children_count: u16` assignment checks `legal_moves.len() <= u16::MAX` with an `assert!`.
- **`c_puct_init` is private**, passed through the constructor. No post-construction mutation — avoids accidentally running the first simulation with the wrong value.
- PUCT formula: `Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`.
- Dynamic c_puct (KataGo-style): `c_puct_eff = c_puct + ln((N + c_puct_init) / c_puct_init)`.
- FPU reduction of 0.2 for unvisited children.
- Policy gathering uses softmax in `f64` for numerical stability.
- Subtree reuse: `re_root` clears children when threats exist (internal nodes skip threat constraints).
- `legal_buf` is cleared before each `encode_board_into` call, eliminating per-leaf `Vec` allocations.

**Depends on:** `board`, `core`, `encoder`.

---

#### `src/encoder.rs` — Neural Network Board Encoder

Produces a 13-channel `33×33` float32 tensor for neural network input. Computes board centroid (banker's rounding), maps the infinite board into a fixed window, and populates 13 feature planes.

**Key types:**

- **`EncodedBoard`** — `tensor: Vec<f32>`, `offset_q: i32`, `offset_r: i32`, `legal_moves: Vec<Hex>`.

**Key constants:**

- `BOARD_SIZE = 33`, `HALF_BOARD = 16`, `NUM_CHANNELS = 13`, `TENSOR_SIZE = 14,157` (`33 × 33 × 13`).
- `BOARD_AREA = 1,089` (`33 × 33`).
- `WIN_SCORE = 1,000,000` (used across search and MCTS).

**13-channel layout:**

| Ch  | Name               | Description                                                                                   |
| --- | ------------------ | --------------------------------------------------------------------------------------------- |
| 0   | Own stones         | 1.0 for current player's stones                                                               |
| 1   | Opponent stones    | 1.0 for opponent's stones                                                                     |
| 2   | Empty mask         | 1.0 - (ch0 + ch1)                                                                             |
| 3   | Legal moves        | 1.0 for legal move cells (threat-constrained when enabled)                                    |
| 4   | Turn phase         | All 1.0 during second placement                                                               |
| 5   | First stone        | 1.0 for the first stone of the current turn                                                   |
| 6   | Player color       | All 1.0 (P0) or all 0.0 (P1)                                                                  |
| 7   | Own recency        | `1.0 / (1.0 + plies_ago)` for own stones                                                      |
| 8   | Opp recency        | `1.0 / (1.0 + plies_ago)` for opponent stones                                                 |
| 9   | Opponent hot cells | 1.0 for cells in opponent's hot windows                                                       |
| 10  | Own hot cells      | 1.0 for cells in own hot windows                                                              |
| 11  | Distance           | Normalized hex distance from board centroid ∈ `[0, 1]` (**pre-computed once** via `OnceLock`) |
| 12  | Opp last turn      | 1.0 for opponent's most recent placement(s)                                                   |

**Channel 11** is a static pre-computation stored in `CENTROID_DIST_CHANNEL: OnceLock<[f32; BOARD_AREA]>`. Initialized lazily on first encode call and simply `copy_from_slice`-d into each output tensor — zero per-call computation. The distances are grid-positional constants independent of game state; only `offset_q`/`offset_r` shift which board cells map to which tensor cells.

**Key public functions:**

- `encode_board(game, near_radius, constrain_threats) -> EncodedBoard` — Convenience wrapper with fresh allocation.
- `encode_board_into(game, near_radius, constrain_threats, out, hot_buf, legal_out) -> (i32, i32)` — Zero-alloc variant. Returns `(offset_q, offset_r)`. The caller provides `out: &mut [f32]` (tensor buffer), `hot_buf: &mut Vec<Hex>` (live-cells scratch), and `legal_out: &mut Vec<Hex>` (legal moves output — cleared and repopulated).
- `extract_features(game) -> [f32; FEATURE_COUNT]` — Classical 13-element feature vector for bootstrap training.
- `bankers_round(v: f64) -> i32` — Matches Python's `round()` exactly.

**Depends on:** `board`, `core`, `threats`.

---

### Python Bridge Layer

#### `src/pybridge/mod.rs` — PyO3 Bindings (Main)

Exposes the Rust engine to Python via two Python classes and a self-play function. All methods release the GIL during long computations via `py.allow_threads()`.

**Key types:**

- **`PyHexGame`** (Python class `HexGame`) — Wraps `HexGameState`. Methods: `place()`, `unplace()`, `legal_moves()`, `is_over()`, `winner()`, `current_player()`, `threat_constrained_moves()`, `classical_search()`, `classical_search_turn()`, `candidates()`, `zobrist()`, `encode_board()`, `opponent_last_turn()`, `clone()`.

**Key function:** `classical_self_play(num_games, time_ms, ...) -> Vec<(features, outcome, board_snap)>` — Generates bootstrap training data using the alpha-beta engine.

**Python-exported constants:** `FEATURE_COUNT`, `WIN_LENGTH`, `PLACEMENT_RADIUS`, `BOARD_SIZE`, `NUM_CHANNELS`, `TENSOR_SIZE`.

**Module function:** `hexgame(m)` — Registers all Python classes, constants, and functions with the PyO3 module.

**Depends on:** `pyo3`, `numpy`, `board`, `core`, `encoder`, `search`, `threats`, `pybridge/mcts`.

---

#### `src/pybridge/mcts.rs` — PyO3 Bindings (MCTS)

Python-facing wrapper for the neural MCTS engine. **All hot-path methods release the GIL** during Rust computation to avoid blocking Python's GPU inference thread.

**Key type:** `PyMCTSEngine` (Python class `MCTSEngine`)

| Method                                                                     | GIL behavior                                                                                                                           |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `new(game, num_sims, c_puct, near_radius, constrain_threats, c_puct_init)` | Constructor — Rust allocation only (no Python interaction needed)                                                                      |
| `init_root() -> PyResult<Option<(PyArray3, i32, i32, PyBytes)>>`           | Returns `None` if game over. Tensor returned as owned numpy array.                                                                     |
| `expand_root(policy, value, oq, or, legal)`                                | Legal bytes validated (must be multiple of 8); returns `PyErr` on malformed input.                                                     |
| `add_dirichlet_noise(noise, fraction)`                                     | Simple delegation — no GIL considerations needed.                                                                                      |
| `select_leaves(batch) -> PyResult<PyArray4<f32>>`                          | **GIL released** during `MCTSEngine::select_leaves`. Copies tensor to owned `Vec<f32>` before re-acquiring GIL for numpy construction. |
| `expand_and_backprop(policies, values)`                                    | **GIL released** during `MCTSEngine::expand_and_backprop`. Policies and values arrays copied to owned `Vec`s before GIL drop.          |
| `get_results() -> (Vec<i16>, Vec<i16>, Vec<f32>, f32)`                     | Pure Rust — no Python interaction.                                                                                                     |
| `root_child_count()`, `root_child_priors()`, `root_child_q_values()`       | Statistics accessors.                                                                                                                  |
| `extract_tree_node_states(min_visits) -> PyResult<...>`                    | Training data export.                                                                                                                  |
| `re_root(q, r, num_sims)`                                                  | Subtree reuse.                                                                                                                         |

**Key implementation details:**

- All methods that receive NumPy arrays validate contiguity and return `PyValueError` on failure.
- Legal moves are decoded from packed byte buffers (pairs of little-endian `i32`). The parser validates that `legal_bytes.len() % 8 == 0` and returns `PyErr` instead of panicking on malformed input.
- `c_puct_init` is passed through the constructor directly to `MCTSEngine::with_arena_sim_hint` — no post-construction field mutation.

**Depends on:** `pyo3`, `numpy`, `core`, `encoder`, `mcts::MCTSEngine`, `pybridge::PyHexGame`.

---

### Test Suite

The test suite is split into two locations:

- **`src/tests/`** — crate-internal unit tests with access to `pub(crate)` modules (`mcts`, `search`, `threats`).
- **`tests/`** — integration tests that exercise only the public API (`board`, `core`, `encoder`, `eval`).

#### `src/tests/oracle.rs` — Brute-Force Solver (Test-Only Ground Truth)

Exhaustive brute-force solver used as ground truth for property-based tests. Enumerates every legal turn, simulates to completion, and classifies results.

**Key type:** `TurnAnalysis { legal, winning, blocking_single, blocking_pairs }`.

**Key function:** `analyse(game: &mut HexGameState) -> TurnAnalysis` — Leaves the game state unchanged.

**Key detail:** Uses `player_candidates_near2()` (radius-2 superset) for candidate generation — intentionally independent of `live_cells` so it serves as true ground truth.

---

#### `src/tests/threats.rs` — Proptest: Fast Path vs Oracle

Property-based tests comparing `threat_status`, `turn_satisfies_status`, and `live_cells` against the oracle on randomized game positions. Three test tiers:

- **Heavy:** 500 cases, `#[ignore]` — run manually with `--ignored`.
- **Medium:** 25 cases, NOT ignored — runs in CI.
- **Smoke:** 10 cases, NOT ignored — fast CI feedback.

The `assert_matches` helper performs bidirectional verification (fast→oracle and oracle→fast).

---

#### `src/tests/threats_internal.rs` — Hand-Crafted Threat Unit Tests

Specific test scenarios: winning turns (5-window, 4-window), blocking (single/2 placements), unblockable detection, live cell enumeration, edge cases (blocked windows, 3-window not hot, game over).

---

#### `src/tests/mcts.rs` — MCTS Engine Correctness Tests (5 tests)

| Test                                           | What it verifies                                                                    |
| ---------------------------------------------- | ----------------------------------------------------------------------------------- |
| `mcts_deterministic_replay`                    | Two identical MCTS runs produce identical visit distributions                       |
| `mcts_reroot_visit_counts_preserved`           | After `re_root`, the new root's visit count equals the child's count before re-root |
| `mcts_root_value_bounded`                      | Root Q always stays in `[-1, 1]` with values in that range (catches sign-flip bugs) |
| `mcts_expand_and_backprop_wrong_length_panics` | Wrong-length policy batch triggers the T1-4 assertion                               |
| `mcts_done_not_true_before_backprop`           | `done()` is false after `select_leaves` but before `expand_and_backprop`            |

---

#### `src/tests/core.rs` — Core Primitives Tests

Validates `hex_distance` symmetry, `Hex` display/ordering, `WindowKey` round-trip and size validation, `Turn` canonical ordering and rejection of self-pairs.

---

#### `src/tests/patterns.rs` — Pattern Table Integrity Tests

Validates base-3 encoding, `POW3` correctness, `PATTERN_VALUES` checksum, incremental `place` matches brute-force recompute, `unplace` restores default, and hot windows match brute-force recomputation.

---

#### `src/tests/eval_state.rs`, `src/tests/grid.rs`, `src/tests/hot.rs`

Small, focused unit tests for their respective modules. Grid tests verify the 11,163-element bijection. Hot tests verify insert/remove/clear. EvalState tests verify place/unplace consistency and threat count updates.

---

#### `tests/board.rs` — Full-Game Scenario Tests + Proptests

Covers opening rules, placement validation, win detection on all 3 axes, move tracking, legal moves, clone independence, Zobrist round-trip, `set_position` correctness, and candidate set behavior.

**Proptests** (100 cases each):

- `place_unplace_is_identity` — Random placements followed by full undo restores Zobrist to zero and move count to zero.
- `zobrist_changes_on_every_valid_placement` — Every valid placement changes the Zobrist hash.

---

#### `tests/encoder.rs` — Tensor Encoding Tests + Proptests

Classical feature extraction tests: empty board features, tempo flip, live/dead five counting, six-in-a-row, live fours/threes/twos, opponent feature separation.

**Proptest** (100 cases):

- `encode_output_range` — All tensor values stay in `[0.0, ∞)`. Tensor always has exactly `TENSOR_SIZE` elements.

---

## Key Design Decisions

### 1. Incremental Evaluation

`EvalState` updates only 18 windows per stone placement (3 directions × 6 offsets) rather than re-scanning the board. Score, threat counts, and hot windows are updated incrementally. Undo is supported via a delta stack. `EvalState::clear()` zeros state in place rather than deallocating and reallocating the 22 KB box.

### 2. Zero Heap Allocation on Hot Paths

All hot-path data structures use stack-allocated or inline storage:

- `WindowKey` — packed `u32` (no heap).
- `HotWindows` — `SmallVec<[WindowKey; 32]>` (inline, 512 bytes total).
- `BlockConstraint` — stored directly in `ThreatStatus::MustBlock` (no `Box`). Inner vectors use `SmallVec<[Hex; 16]>` and `SmallVec<[(Hex, Hex); 32]>`.
- `winning_line` — `Option<[Hex; 6]>` (stack array, no per-placement allocation).
- `opponent_last_turn_cells` — `SmallVec<[Hex; 2]>` (inline for typical 1–2 cells).
- `opponent_threat_windows` — flat `(SmallVec<[Hex; 32]>, SmallVec<[u8; 16]>)` tuple; no per-window clones.
- Quiescence and MCTS scratch buffers are pre-allocated and reused across iterations.
- `legal_buf` and `hot_buf` on `MCTSEngine` are cleared and reused instead of freshly allocated per leaf.
- Channel 11 (centroid distance) is pre-computed once via `OnceLock` and copied; no per-encode recomputation.

### 3. Exact BlockConstraint

In the 2-placement case, `BlockConstraint` enumerates exact covering pairs upfront rather than returning a permissive superset. This eliminates re-validation at call sites and fixes an encoder bug (channel 3 was previously a superset).

### 4. Strict Module Boundaries

Each file owns one concept. `EvalState` fields are private; mutation only through `place()`/`unplace()`. `HexGameState` fields are private; access through accessors. `MCTSEngine` fields are private (except `arena` and `root_idx` which are `pub(crate)` for test access and re-rooting). `c_puct_init` is set at construction time only — no post-construction mutation.

### 5. Single Oracle

One `analyse()` function serves as the single ground truth for all threat-related property tests — replacing three separate solver functions that could drift independently.

### 6. Visibility Discipline

- `pub` — `board`, `core`, `encoder`, `eval`. These form the stable public API for downstream Rust consumers and Python bindings.
- `pub(crate)` — `mcts`, `search`, `threats`. Internal search engines and threat logic accessible to tests and the Python bridge but not yet committed to external stability.
- Private (`mod pybridge`) — Entirely gated behind `#[cfg(feature = "python")]`. Not compiled in standard builds.

### 7. Correctness-First Guard Rails

- Pattern index bounds checks use runtime `assert!` (not `debug_assert!`) — UB prevention in release builds.
- `set_position` validates per-stone inputs: player bounds, origin-first rule, radius proximity — preventing silent corruption from test and oracle callers.
- `expand_and_backprop` validates policy/value slice lengths against actual leaf count — catches Python-side batch mismatches with a clear assertion.
- MCTS backprop uses depth parity (not player field comparison) — eliminates the fresh-node player=255 bug.
- Virtual loss correctly adjusts both `visit_count` (up) and `total_value` (down) — producing correct pessimistic Q values during batch selection.
- `sims_done` increments after actual backprop, not after selection — prevents `done()` from firing with VL leaves still pending.
- Legal-bytes parser returns `PyErr` instead of panicking on malformed Python input.

### 8. Cargo Feature Gating

- `#[cfg(feature = "python")]` gates the entire `pybridge` module (requires PyO3 + numpy).
- `#[cfg(test)]` gates the `tests` module and test-only production code (`place_unchecked`, `HotWindows::clear()`).
- `#[cfg(debug_assertions)]` gates the `unplace` invariant recomputation check in `EvalState`.

---

## Performance Budget

| Path                         | Allocation                                                        | Target   |
| ---------------------------- | ----------------------------------------------------------------- | -------- |
| `EvalState::place`           | 1 push to `delta_stack` (amortized no-alloc)                      | < 90 ns  |
| `EvalState::unplace`         | Pop from stack                                                    | < 50 ns  |
| `threat_status` (Quiet)      | 0                                                                 | < 20 ns  |
| `threat_status` (full)       | 0 (SmallVec inline)                                               | < 1 µs   |
| `turn_satisfies_status`      | 0                                                                 | < 50 ns  |
| `live_cells`                 | 0 (caller-owned Vec + local FxHashSet)                            | < 500 ns |
| `hypothetical_score_delta`   | 0                                                                 | < 50 ns  |
| `validate_move` radius check | 0 (O(1) FxHashMap lookup via `placement_candidates.contains()`)   | < 30 ns  |
| `encode_board_into`          | 0 (all buffers caller-provided; centroid channel copy_from_slice) | < 15 µs  |

No function in the hot path allocates. SmallVec inline sizes are chosen so typical games never spill to heap.

---

## Configuration & Tooling

### Cargo Configuration (`Cargo.toml`)

**Release profile:**

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
panic = "abort"
```

- `panic = "abort"` removes unwinding machinery from hot loops, reducing binary size and improving performance.
- `lto = "fat"` enables link-time optimization across the entire dependency graph.
- `codegen-units = 1` maximizes inlining opportunities.

**Bench profile:**

```toml
[profile.bench]
opt-level = 3
lto = "fat"
codegen-units = 1
inherits = "release"
```

Inherits from release but without `strip` (preserves symbols for profilers).

**Dev dependencies:**

```toml
[dev-dependencies]
proptest = "1"
criterion = { version = "0.5", features = ["html_reports"] }
```

**Benchmarks:**

```toml
[[bench]]
name = "engine"
harness = false
```

### Toolchain & Linting

| File                  | Purpose                                                                                         |
| --------------------- | ----------------------------------------------------------------------------------------------- |
| `rust-toolchain.toml` | Pins toolchain to `stable` with `rustfmt` and `clippy` components. Ensures reproducible builds. |
| `rustfmt.toml`        | Edition 2021, max width 100, default small heuristics. Prevents format churn.                   |
| `.clippy.toml`        | Sets MSRV to 1.80 for compatibility-aware linting.                                              |
| `.cargo/config.toml`  | Clippy deny list: `clippy::all`, `clippy::pedantic`, with `module_name_repetitions` allowed.    |

---

## Testing Strategy

### Unit Tests (crate-internal — `src/tests/`)

- **Module-specific tests**: `core.rs`, `patterns.rs`, `grid.rs`, `hot.rs`, `eval_state.rs`
- **Threat logic**: `threats_internal.rs` (hand-crafted scenarios), `threats.rs` (proptest: fast path vs oracle)
- **MCTS engine**: `mcts.rs` (5 correctness tests: determinism, re-root, value bounds, panic guards, done timing)
- **Oracle**: `oracle.rs` (brute-force solver for ground-truth comparison)

### Integration Tests (public API — `tests/`)

- **`board.rs`**: Full-game scenarios (36 tests) covering all game rules, win detection on all axes, placement validation, undo/Zobrist, move tracking, candidate sets, error messages. Includes proptests for `place_unplace_is_identity` and `zobrist_changes`.
- **`encoder.rs`**: Tensor encoding and classical feature extraction (9 tests). Includes a proptest for `encode_output_range`.

### Benchmarks (`benches/`)

- **`engine.rs`**: Criterion benchmarks for `encode_board_into`, `legal_moves_near(2)`, and `candidates_near2()` on mid-game positions. Run with `cargo bench`.

### CI (`.github/workflows/ci.yml`)

- **`test` job**: `cargo build --release && cargo test --release && cargo test --release -- --ignored && cargo clippy --release -- -D warnings`
- **`python-integration` job**: `maturin develop --features python && pytest Python/tests/test_engine_smoke.py -v`

---

## Python Integration

### Building the Extension

```bash
maturin develop --features python
```

### Python Module Structure

```python
import _engine

# Classes
game = _engine.PyHexGame()
mcts = _engine.PyMCTSEngine(game, num_simulations=800, c_puct=1.5, ...)

# Constants
_engine.BOARD_SIZE      # 33
_engine.NUM_CHANNELS    # 13
_engine.TENSOR_SIZE     # 14157
_engine.WIN_LENGTH      # 6
_engine.PLACEMENT_RADIUS  # 8
_engine.FEATURE_COUNT   # 13
```

### Typical RL Training Loop

```python
engine = _engine.PyMCTSEngine(game, num_simulations=800, c_puct=1.5,
                               near_radius=2, constrain_threats=False,
                               c_puct_init=19652.0)

# 1. Initialize
result = engine.init_root()
tensor, oq, or_, legal_bytes = result
policy = model.predict(tensor[np.newaxis, ...])[0]
engine.expand_root(policy, 0.0, oq, or_, legal_bytes)

# 2. Run MCTS
while not engine.done():
    tensor_batch, count = engine.select_leaves(64)
    policies, values = model.predict_batch(tensor_batch, count)
    engine.expand_and_backprop(policies.flatten(), values)

# 3. Extract results
moves_q, moves_r, visits, root_q = engine.get_results()
```

### Smoke Tests

`Python/tests/test_engine_smoke.py` contains 4 pytest tests verifying: constant exports, basic game play/unplay, encode tensor shape, and full MCTS run-to-completion. Run with:

```bash
pytest Python/tests/test_engine_smoke.py -v
```
