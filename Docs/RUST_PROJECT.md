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

---

## Project Overview

The `hexgame` crate is a high-performance game engine for **Infinity Hexagonal Tic-Tac-Toe**, a variant of Hex played on an infinite hexagonal grid where two players alternate placing stones with the goal of forming a contiguous 6-in-a-row line. The engine supports both classical alpha-beta search and neural-network-guided MCTS, with PyO3 bindings exposing the engine to Python for RL training.

### Game Rules
- Two players (0 and 1) alternate turns on an infinite board using axial coordinates `(q, r)`.
- Player 0 opens with **one** stone at the origin `(0, 0)`.
- Every subsequent turn consists of **two** stone placements (except the opening).
- Each placement must land on an empty hex within radius 8 of any existing stone.
- The first player to form 6 contiguous stones along any of the three hex axes wins.
- The board is infinite — there is no draw condition.

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

---

## Module Layout

```
src/
├── lib.rs                  # Crate root: module declarations + re-exports (74 lines)
├── core.rs                 # Hex, Turn, WindowKey, hex_distance (~305 lines)
├── eval/
│   ├── mod.rs              # Re-exports EvalState, ThreatCounts
│   ├── patterns.rs         # PATTERN_VALUES[729], PATTERN_COUNTS[729], POW3[6]
│   ├── grid.rs             # WIN_GRID constants, win_grid_idx
│   ├── hot.rs              # HotWindows (SmallVec-backed threat cache)
│   └── state.rs            # EvalState: incremental score, counts, delta stack
├── board.rs                # HexGameState: rules, placement, undo, Zobrist, win detect
├── threats.rs              # ThreatStatus, threat_status(), live_cells()
├── encoder.rs              # 13-channel NN tensor encoder + classical feature extraction
├── search.rs               # Alpha-beta with iterative deepening, TT, quiescence
├── mcts.rs                 # Arena-allocated MCTS with PUCT, virtual loss, batch leaves
├── pybridge/
│   ├── mod.rs              # PyHexGame + classical_self_play
│   └── mcts.rs             # PyMCTSEngine
└── tests/
    ├── mod.rs              # Test module declarations
    ├── oracle.rs           # Brute-force solver (test-only ground truth)
    ├── threats.rs          # Proptest: fast path vs oracle (1000+ cases)
    ├── threats_internal.rs # Hand-crafted threat unit tests
    ├── board.rs            # Game rules, win detection, placement tests
    ├── core.rs             # Hex, Turn, WindowKey unit tests
    ├── patterns.rs         # Pattern table consistency, checksum tests
    ├── eval_state.rs       # EvalState place/unplace round-trip tests
    ├── encoder.rs          # Classical feature extraction tests
    ├── grid.rs             # Win grid indexing bijection tests
    └── hot.rs              # HotWindows insert/remove/clear tests
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
- `win_grid_in_bounds(q, r) -> bool` — Runtime bounds guard. This must remain a runtime check (not `debug_assert!`) because window origins can drift beyond radius 30 in long games. Out-of-bounds windows are silently skipped — a deliberate approximation.

**Depends on:** Nothing (leaf).

---

#### `src/eval/hot.rs` — Hot Windows Cache
Zero-allocation cache tracking which windows currently represent win threats (4+ stones of one player, 0 opponent stones). Uses `SmallVec<[WindowKey; 32]>` per player — inline storage for 32 entries, no heap allocation in practice.

**Key type:** `HotWindows` with methods:
- `new()` — Empty cache.
- `insert(player, key)` — Add a hot window (with `debug_assert!` against duplicates).
- `remove(player, key)` — Remove via `swap_remove` (`O(n)` find, `O(1)` removal).
- `iter(player) -> impl Iterator<Item = WindowKey>` — Iterate hot windows.
- `len(player) -> usize`, `is_empty(player) -> bool` — Query methods.

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
- `place(cell, player)` — Incremental placement. Touches 18 windows (3 directions × 6 offsets). Returns delta pushed onto stack.
- `unplace()` — Pops delta stack, reverses the placement. Exactly restores pre-place state.
- `score() -> i32` — Current static evaluation (P0's perspective).
- `counts(player) -> ThreatCounts` — Read threat counts.
- `has_threats(player) -> bool` — Fast check for any fours or fives.
- `has_any_threats() -> bool` — Fast check for any threats on either side.
- `hot_windows(player) -> impl Iterator<Item = WindowKey>` — Iterate hot windows.
- `hypothetical_score_delta(cell, player) -> i32` — Score change if stone were placed (read-only, used for move ordering).

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
- **`stones: FxHashMap<Hex, u8>`** — All occupied cells (sparse infinite-board representation).
- **`eval: EvalState`** — Composed evaluation state (board never touches eval internals).
- **`candidates: CandidateSet`** — Reference-counted empty cells within radius 2 of any stone (enables fast `O(1)` move generation with incremental updates).
- **`zobrist: u64`** — Incremental Zobrist hash (deterministic FNV-1a, no precomputed table).
- **`move_history: Vec<MoveRecord>`** — Undo stack recording pre-move state.

**Key public API:**
- `new()` — Empty board, P0 to move.
- `place(q, r) -> Result<bool, GameError>` — Validate + place stone. Returns whether the turn ended.
- `unplace()` — Pop move history, restore previous state.
- `set_position(stones, player, remaining)` — Bypass turn rules for test/oracle setup.
- Accessors: `stones()`, `eval()`, `current_player()`, `placements_remaining()`, `winner()`, `winning_line()`, `move_count()`, `move_history()`, `zobrist()`, `is_over()`.
- `legal_moves_near(radius) -> Vec<Hex>` — Legal move cells within `radius` of any stone.
- `candidates_near2() -> Vec<Hex>` — Fast radius-2 candidate cells (for search).
- `opponent_last_turn_cells() -> Vec<Hex>` — Cells placed in opponent's last turn (encoder channel 12).

**Key implementation details:**
- Win detection scans 3 hex axes from the newly placed stone, extracting a 6-stone segment centered on the pivot.
- `CandidateSet` implements reference-counted incremental updates: `on_place` increments neighbor refs, `on_unplace` decrements. Enables `O(1)` candidate query.
- Opening move enforcement: first move must be `(0, 0)` and only 1 placement.
- `place_unchecked` is `#[cfg(test)] pub(crate)` for oracle use.

**Depends on:** `core`, `eval/state`.

---

### Threat Layer

#### `src/threats.rs` — Tactical Threat Analysis
Classifies the tactical situation, checks turn legality under threat constraints, and enumerates live cells.

**Key types:**
- **`ThreatStatus`** — Enum: `Quiet` (no threats), `WinningTurn(Turn)` (forced win exists), `MustBlock(Box<BlockConstraint>)` (must play specific cells/pairs), `Unblockable` (cannot stop opponent).
- **`BlockConstraint`** — Exact blocking sets: `cells: SmallVec<[Hex; 16]>` (single-placement intersection) and `pairs: SmallVec<[(Hex, Hex); 32]>` (exact 2-placement covering pairs). Accessors: `cells() -> &[Hex]`, `pairs() -> &[(Hex, Hex)]`.

**Key public functions:**
1. **`threat_status(game) -> ThreatStatus`** — Full tactical classification. Short-circuits: checks instant wins first, then builds `BlockConstraint`.
2. **`turn_satisfies_status(status, turn) -> bool`** — Membership test against pre-computed constraint. `O(1)` for single-placement, `O(pairs)` for 2-placement.
3. **`live_cells(game, player, out: &mut Vec<Hex>)`** — Enumerates empty cells in hot windows (caller-owned buffer, reusable).

**Crate-private:** `generate_threat_turns(game, out, opp_buf, my_buf)` — Produces candidate turns for quiescence search from live cells. Takes caller-owned scratch buffers.

**Key implementation details:**
- `BlockConstraint` is **exact** — in the 2-placement case it enumerates covering pairs up front, rather than returning a permissive superset. This eliminates the need for callers to re-validate.
- `threat_status` returns `Quiet` immediately when `game.winner().is_some()` (no threats matter after game ends).
- `live_cells` reuses `window_empties()` internally to avoid code duplication.
- All hot-path data structures use `SmallVec` with inline capacity; no heap allocation in `threat_status`.

**Depends on:** `board`, `core`.

---

### Search & Learning Layer

#### `src/search.rs` — Classical Alpha-Beta Search
Turn-based alpha-beta with iterative deepening, transposition table (TT), quiescence search, and multiple pruning strategies.

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
Arena-allocated MCTS with PUCT, virtual loss, batch leaf selection, and subtree reuse. Tree lives entirely in Rust; Python calls `select_leaves(batch_size)` → GPU inference → `expand_and_backprop(policies, values)`.

**Key type:** `MCTSEngine`
- **`arena: Vec<MCTSNode>`** — All tree nodes in a single contiguous allocation.
- **`game: HexGameState`** — Mutable board state used during tree traversal.
- **`batch_buf: Vec<f32>`** — Pre-allocated tensor buffer for batch leaf encoding.
- **`scratch_raw: Vec<f64>`**, **`scratch_priors: Vec<f32>`**, **`hot_buf: Vec<Hex>`** — Reusable scratch buffers.

**Key public API:**
- `new(game, num_sims, c_puct, near_radius, constrain_threats)`
- `init_root() -> Option<(tensor, oq, or, legal)>` — Encode root for GPU inference.
- `expand_root(policy_logits, value, oq, or, legal)` — Expand root from GPU output.
- `add_dirichlet_noise(noise, fraction)` — Inject exploration noise at root.
- `select_leaves(batch_size) -> (&[f32], u32)` — Select batch leaves, return encoded tensors.
- `expand_and_backprop(policies, values)` — GPU output → node expansion → backprop.
- `re_root(q, r, new_num_sims)` — Subtree reuse after opponent's move.
- `get_results() -> (moves_q, moves_r, visits, root_value)` — Final statistics.
- `extract_tree_node_states(min_visits) -> Result<TreeNodeStates, &str>` — Training data export.

**Key implementation details:**
- PUCT formula: `Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`.
- Dynamic c_puct (KataGo-style): `c_puct_eff = c_puct + ln((N + c_puct_init) / c_puct_init)`.
- FPU reduction of 0.2 for unvisited children.
- Virtual loss (1 visit per path node) during batch selection discourages duplicate paths.
- Values stored per-node from own player perspective (sign flips at each ply).
- Policy gathering uses softmax in `f64` for numerical stability.
- Subtree reuse: `re_root` clears children when threats exist (internal nodes skip threat constraints).

**Depends on:** `board`, `core`, `encoder`.

---

#### `src/encoder.rs` — Neural Network Board Encoder
Produces a 13-channel `33×33` float32 tensor for neural network input. Computes board centroid (banker's rounding), maps the infinite board into a fixed window, and populates 13 feature planes.

**Key types:**
- **`EncodedBoard`** — `tensor: Vec<f32>`, `offset_q: i32`, `offset_r: i32`, `legal_moves: Vec<Hex>`.

**Key constants:**
- `BOARD_SIZE = 33`, `HALF_BOARD = 16`, `NUM_CHANNELS = 13`, `TENSOR_SIZE = 14157`.
- `WIN_SCORE = 1_000_000` (used across search and MCTS).

**13-channel layout:**
| Ch | Name | Description |
|----|------|-------------|
| 0 | Own stones | 1.0 for current player's stones |
| 1 | Opponent stones | 1.0 for opponent's stones |
| 2 | Empty mask | 1.0 - (ch0 + ch1) |
| 3 | Legal moves | 1.0 for legal move cells (threat-constrained when enabled) |
| 4 | Turn phase | All 1.0 during second placement |
| 5 | First stone | 1.0 for the first stone of the current turn |
| 6 | Player color | All 1.0 (P0) or all 0.0 (P1) |
| 7 | Own recency | `1.0 / (1.0 + plies_ago)` for own stones |
| 8 | Opp recency | `1.0 / (1.0 + plies_ago)` for opponent stones |
| 9 | Opponent hot cells | 1.0 for cells in opponent's hot windows |
| 10 | Own hot cells | 1.0 for cells in own hot windows |
| 11 | Distance | Normalized hex distance from centroid ∈ `[0, 1]` |
| 12 | Opp last turn | 1.0 for opponent's most recent placement(s) |

**Key public functions:**
- `encode_board(game, near_radius, constrain_threats) -> EncodedBoard` — Convenience wrapper.
- `encode_board_into(game, near_radius, constrain_threats, out, hot_buf) -> (i32, i32, Vec<Hex>)` — Zero-alloc variant with reusable buffers.
- `extract_features(game) -> [f32; 13]` — Classical 13-element feature vector for bootstrap training.
- `bankers_round(v: f64) -> i32` — Matches Python's `round()` exactly.

**Depends on:** `board`, `core`, `threats`.

---

### Python Bridge Layer

#### `src/pybridge/mod.rs` — PyO3 Bindings (Main)
Exposes the Rust engine to Python via two Python classes and a self-play function. All methods release the GIL during long computations via `py.allow_threads()`.

**Key types:**
- **`PyHexGame`** (Python class `HexGame`) — Wraps `HexGameState`. Methods: `place()`, `unplace()`, `legal_moves()`, `is_over()`, `winner()`, `current_player()`, `threat_constrained_moves()`, `classical_search()`, `classical_search_turn()`, `candidates()`, `zobrist()`, `encode_board()`, `opponent_last_turn()`, `clone()`.
- **`PyMCTSEngine`** (re-exported from `mcts.rs`) — Python class `MCTSEngine`.

**Key function:** `classical_self_play(num_games, time_ms, ...) -> Vec<(features, outcome, board_snap)>` — Generates bootstrap training data using the alpha-beta engine.

**Depends on:** `pyo3`, `numpy`, `board`, `core`, `encoder`, `search`, `threats`, `pybridge/mcts`.

---

#### `src/pybridge/mcts.rs` — PyO3 Bindings (MCTS)
Python-facing wrapper for the neural MCTS engine.

**Key type:** `PyMCTSEngine` (Python class `MCTSEngine`)
- `new(game, num_sims, c_puct, near_radius, constrain_threats)` — Construction.
- `init_root() -> PyResult<Option<(PyArray3, i32, i32, PyBytes)>>` — Encode root for GPU (returns `None` if game over).
- `expand_root(policy, value, oq, or, legal)` — Expand root from GPU output.
- `add_dirichlet_noise(noise, fraction)` — Inject exploration noise.
- `select_leaves(batch) -> PyResult<PyArray4<f32>>` — Batch leaf selection, returns `(batch, 13, 33, 33)` tensor.
- `expand_and_backprop(policies, values)` — GPU output → expansion → backprop.
- `get_results() -> (Vec<i16>, Vec<i16>, Vec<f32>, f32)` — Final statistics.
- `root_child_count() -> usize`, `root_child_priors()`, `root_child_q_values()` — Statistics accessors.
- `extract_tree_node_states(min_visits) -> PyResult<(Vec<f32>, Vec<PyObject>, usize)>` — Training data export.
- `re_root(q, r, num_sims)` — Subtree reuse.

All methods that receive NumPy arrays validate contiguity. Legal moves are decoded from packed byte buffers (pairs of little-endian `i32`).

**Depends on:** `pyo3`, `numpy`, `core`, `encoder`, `mcts::MCTSEngine`, `pybridge::PyHexGame`.

---

### Test Suite

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

#### `src/tests/board.rs` — Game Rules & Win Detection Tests
Covers opening rules, placement validation, win detection on all 3 axes, move tracking, legal moves, clone independence, Zobrist round-trip, `set_position` correctness, and candidate set behavior.

---

#### `src/tests/core.rs` — Core Primitives Tests
Validates `hex_distance` symmetry, `Hex` display/ordering, `WindowKey` round-trip and size validation, `Turn` canonical ordering and rejection of self-pairs.

---

#### `src/tests/patterns.rs` — Pattern Table Integrity Tests
Validates base-3 encoding, `POW3` correctness, `PATTERN_VALUES` checksum, incremental `place` matches brute-force recompute, `unplace` restores default, and hot windows match brute-force recomputation.

---

#### `src/tests/eval_state.rs` — EvalState Round-Trip Tests
Place/unplace consistency, threat count updates, hot window tracking, hypothetical score delta accuracy, debug invariant checks.

---

#### `src/tests/grid.rs`, `src/tests/hot.rs`, `src/tests/encoder.rs`
Small, focused unit tests for their respective modules. Grid tests verify the 11,163-element bijection. Hot tests verify insert/remove/clear. Encoder tests verify classical feature extraction indexing.

---

## Key Design Decisions

### 1. Incremental Evaluation
`EvalState` updates only 18 windows per stone placement (3 directions × 6 offsets) rather than re-scanning the board. Score, threat counts, and hot windows are updated incrementally. Undo is supported via a delta stack.

### 2. Zero Heap Allocation on Hot Paths
All hot-path data structures use stack-allocated or inline storage:
- `WindowKey` — packed `u32` (no heap).
- `HotWindows` — `SmallVec<[WindowKey; 32]>` (inline, 512 bytes total).
- `BlockConstraint` — `SmallVec<[Hex; 16]>` + `SmallVec<[(Hex, Hex); 32]>`.
- Quiescence and MCTS scratch buffers are pre-allocated and reused.

### 3. Exact BlockConstraint
In the 2-placement case, `BlockConstraint` enumerates exact covering pairs upfront rather than returning a permissive superset. This eliminates re-validation at call sites and fixes an encoder bug (channel 3 was previously a superset).

### 4. Strict module boundaries
Each file owns one concept. `EvalState` fields are private; mutation only through `place()`/`unplace()`. `HexGameState` fields are private; access through accessors. No back-door `pub(crate)` field access across module boundaries.

### 5. Single Oracle
One `analyse()` function serves as the single ground truth for all threat-related property tests — replacing three separate solver functions that could drift independently.

### 6. Cargo Feature Gating
- `#[cfg(feature = "python")]` gates the entire `pybridge` module (requires PyO3 + numpy).
- `#[cfg(test)]` gates dead production code used only in tests (`WindowKey::cell_at`, `HotWindows::clear()`, `place_unchecked`).
- `#[cfg(debug_assertions)]` gates the `unplace` invariant recomputation check.

---

## Performance Budget

| Path | Allocation | Target |
|------|-----------|--------|
| `EvalState::place` | 1 push to `delta_stack` (amortized no-alloc) | < 90 ns |
| `EvalState::unplace` | Pop from stack | < 50 ns |
| `threat_status` (Quiet) | 0 | < 20 ns |
| `threat_status` (full) | 0 (SmallVec inline) | < 1 µs |
| `turn_satisfies_status` | 0 | < 50 ns |
| `live_cells` | 0 (caller-owned Vec) | < 500 ns |
| `hypothetical_score_delta` | 0 | < 50 ns |

No function in the hot path allocates. SmallVec inline sizes are chosen so typical games never spill to heap.
