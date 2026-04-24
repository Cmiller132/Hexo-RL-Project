//! PyO3 Python bindings — exposes the Rust Hex game engine to Python.
//!
//! This module defines two Python-facing classes:
//!
//! * `HexGame` — a thin wrapper around [`HexGameState`] that exposes board
//!   manipulation, query methods, and neural-network encoding.
//! * `MCTSEngine` — a wrapper around the Rust MCTS tree that handles root
//!   expansion, leaf selection, Dirichlet noise, and back-propagation.
//!
//! # PyO3 / numpy patterns used throughout
//!
//! * **`Python<'py>` lifetime** — every method that allocates Python objects
//!   (e.g. `numpy.ndarray`, `PyBytes`) takes a `Python<'py>` token so that
//!   the returned `Bound<'py, T>` values are known to be valid for the
//!   duration of the call.
//! * **`Bound<'py, PyArray>` instead of `&PyArray`** — PyO3 0.22+ moved to the
//!   `Bound` API for all object references. This is why init-methods return
//!   `Bound<'py, PyArray3<f32>>` rather than `&PyArray3<f32>`.
//! * **`ndarray::ArrayX::from_shape_vec` + `PyArrayX::from_owned_array`** —
//!   `numpy` 0.24 removed the old `PyArrayX::from_shape_vec` constructor.
//!   The new two-step pattern (build a Rust `ndarray::Array`, then transfer
//!   ownership into Python) is used by `encode_board_and_legal`, `init_root`,
//!   `select_leaves`, and `extract_tree_node_states`.
//! * **Contiguous-slice validation** — `PyReadonlyArray1::as_slice()` returns
//!   `Result<&[T], NotContiguousError>` because numpy arrays may be strided.
//!   Fallible methods propagate this as a Python `ValueError` instead of
//!   panicking so that the caller can catch it.
//! * **Packed byte buffers** — methods ending in `_bytes` return `PyBytes`
//!   containing little-endian `i32` values. This avoids the overhead of
//!   constructing millions of tiny Python tuples when shipping move lists or
//!   board states across the FFI boundary.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use numpy::{ndarray, PyArray3};

use crate::core::Hex;
use crate::threats::{threat_status, ThreatStatus};
use crate::encoder;
use crate::board::{GameError, HexGameState};
use crate::search;

pub mod mcts;
use mcts::PyMCTSEngine;

use std::cell::Cell;
use std::time::{Duration, SystemTime};

// Re-export encoder constants so Python can query shapes that match the
// canonical implementation in `encoder.rs`.
use encoder::{BOARD_SIZE, NUM_CHANNELS};

// -------------------------------------------------------------------------
// XOR-shift RNG (thread-local, seeded from system time)
// -------------------------------------------------------------------------

thread_local! {
    static RNG_STATE: Cell<u64> = Cell::new({
        // Seed from system time mixed with stack address for uniqueness per thread.
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x4da7_b0e3_9a1f_2c85);
        // Mix so that identical-nanosecond seeds (e.g. same start time) differ.
        let ptr = &nanos as *const u64 as u64;
        let v = nanos ^ ptr.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        if v == 0 { 1 } else { v }
    });
}

/// Next 64-bit XOR-shift value from the thread-local generator.
fn rng_next() -> u64 {
    RNG_STATE.with(|s| {
        let mut v = s.get();
        v ^= v << 13;
        v ^= v >> 7;
        v ^= v << 17;
        s.set(v);
        v
    })
}

/// Epsilon top-k sampling around the deterministic best move.
///
/// With probability `noise_level` a candidate from the top `k` root moves is
/// sampled uniformly, otherwise the best move is returned. Used by
/// `classical_search` to inject variability into self-play games.
fn epsilon_topk_sample(best: Hex, candidates: &[(Hex, i32)], noise_level: f32) -> Hex {
    if candidates.is_empty() {
        return best;
    }

    let eps = (noise_level as f64).clamp(0.0, 1.0);
    let r = rng_next() as f64 / u64::MAX as f64;
    if r > eps {
        return best;
    }

    let k = if noise_level < 0.05 {
        2usize
    } else if noise_level < 0.15 {
        3usize
    } else {
        5usize
    }
    .min(candidates.len());

    let idx = (rng_next() as usize) % k;
    let sampled = candidates[idx].0;
    if sampled == best && k > 1 {
        candidates[(idx + 1) % k].0
    } else {
        sampled
    }
}

// -------------------------------------------------------------------------
// Python-facing wrapper for HexGameState
// -------------------------------------------------------------------------

/// Python-facing wrapper around [`HexGameState`].
///
/// Provides board manipulation, threat queries, legal-move generation, and
/// neural-network encoding for a 6-in-a-row Hex variant on an infinite board.
#[pyclass(name = "HexGame")]
pub struct PyHexGame {
    inner: HexGameState,
}

#[pymethods]
impl PyHexGame {
    /// Create a new game in the initial empty state.
    ///
    /// Player 0 moves first with a single opening stone; all subsequent turns
    /// consist of two stone placements.
    #[new]
    fn new() -> Self {
        Self {
            inner: HexGameState::new(),
        }
    }

    /// Place the current player's tile at `(q, r)`.
    ///
    /// Returns `True` when the turn ends (i.e. the second stone of a turn was
    /// just placed, or the single opening stone), `False` when the player has
    /// another placement remaining this turn.
    ///
    /// Raises `ValueError` if the placement is illegal.
    fn place(&mut self, q: i32, r: i32) -> PyResult<bool> {
        self.inner
            .place(q, r)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Undo the last placement.
    ///
    /// If the game is over this resets the winner as well. Safe to call even
    /// when the history is empty (no-op).
    fn unplace(&mut self) {
        self.inner.unplace();
    }

    /// Whether the game has ended.
    #[getter]
    fn is_over(&self) -> bool {
        self.inner.is_over()
    }

    /// The winning player (`0` or `1`), or `None` if the game is still ongoing.
    #[getter]
    fn winner(&self) -> Option<u8> {
        self.inner.winner()
    }

    /// Current player (`0` or `1`).
    #[getter]
    fn current_player(&self) -> u8 {
        self.inner.current_player()
    }

    /// Placements remaining in the current turn (`1` or `2`).
    #[getter]
    fn placements_remaining(&self) -> u8 {
        self.inner.placements_remaining()
    }

    /// Total number of individual tile placements so far.
    #[getter]
    fn move_count(&self) -> u32 {
        self.inner.move_count()
    }

    /// Incremental Zobrist hash of the board position.
    ///
    /// This hash is updated in O(1) on every placement and is suitable for
    /// transposition tables.
    #[getter]
    fn zobrist_hash(&self) -> u64 {
        self.inner.zobrist()
    }

    /// Tactical threat summary for the current position.
    ///
    /// With two placements per turn, both 4-windows (2 empty cells) and
    /// 5-windows (1 empty cell) are single-turn wins. This property returns a
    /// 2-bit value:
    ///
    /// | Value | Meaning |
    /// |---|---|
    /// | `0` | No immediate threats for either side. |
    /// | `1` | Current player has at least one 4-window or 5-window. |
    /// | `2` | Opponent has at least one 4-window or 5-window. |
    /// | `3` | Both sides have threats. |
    ///
    /// The opponent bit is used by noise suppression (must block). Zero extra
    /// cost — reads the incremental threat counters maintained by the evaluator.
    #[getter]
    fn threat_level(&self) -> u8 {
        let me = self.inner.current_player();
        let opp = 1 - me;
        let own = self.inner.eval().has_threats(me) as u8;
        let opp_threat = self.inner.eval().has_threats(opp) as u8;
        own | (opp_threat << 1)
    }

    /// How many placements a turn allows given a move count.
    ///
    /// The very first turn (`move_count == 0`, player 0 to move) allows only
    /// one placement (the opening stone). All subsequent turns allow two.
    #[staticmethod]
    fn placements_per_turn(move_count: u32) -> u8 {
        if move_count == 0 { 1 } else { 2 }
    }

    /// Raw window-based positional evaluation from player 0's perspective.
    ///
    /// This is turn-independent: it does **not** include threat bonuses or
    /// tempo adjustments. Positive values favour player 0; negative values
    /// favour player 1.
    #[getter]
    fn window_eval(&self) -> i32 {
        self.inner.eval().score()
    }

    /// Number of hot windows containing exactly 4 stones for `player`.
    ///
    /// A "hot window" is a 6-cell line with 4+ stones of the given player and
    /// zero opponent stones. These are the tactical lines that matter for
    /// threat detection.
    fn window_fours(&self, player: u8) -> i32 {
        self.inner.eval().counts(player).fours() as i32
    }

    /// Number of hot windows containing 5+ stones for `player`.
    ///
    /// A 5-window is one empty cell away from an instant win.
    fn window_fives(&self, player: u8) -> i32 {
        self.inner.eval().counts(player).fives() as i32
    }

    /// Number of hot windows containing 3+ stones for `player`.
    fn window_threes(&self, player: u8) -> i32 {
        self.inner.eval().counts(player).threes() as i32
    }

    /// Return active threat windows for the given player.
    ///
    /// Each window is a list of 6 `(q, r, occupied)` tuples where `occupied`
    /// is `True` if the player already has a stone on that cell and `False`
    /// if the cell is empty (i.e. must be filled to complete the line).
    ///
    /// Only "hot" windows are returned — lines with 4+ player stones and no
    /// blocking opponent stones.
    fn get_threat_windows(&self, player: u8) -> Vec<Vec<(i32, i32, bool)>> {
        use crate::core::{HEX_DIRECTIONS, WIN_LENGTH};
        let game = &self.inner;
        let mut result = Vec::new();
        for key in game.eval().hot_windows(player) {
            let (wq, wr, dir) = (key.q(), key.r(), key.dir());
            let (dq, dr) = HEX_DIRECTIONS[dir as usize];
            let mut cells = Vec::with_capacity(WIN_LENGTH as usize);
            for k in 0..WIN_LENGTH {
                let cq = wq + dq * k;
                let cr = wr + dr * k;
                let h = crate::core::Hex::new(cq, cr);
                let occupied = game.stones().get(&h) == Some(&player);
                cells.push((cq, cr, occupied));
            }
            result.push(cells);
        }
        result
    }

    /// Dict-like list of all occupied cells: `[(q, r, player), ...]`.
    fn board_pieces(&self) -> Vec<(i32, i32, u8)> {
        self.inner
            .stones()
            .iter()
            .map(|(&h, &p)| (h.q, h.r, p))
            .collect()
    }

    /// All legal placements (exhaustive radius-8 scan).
    ///
    /// This is slower but guarantees completeness. For most use-cases prefer
    /// `legal_moves_near` with a small radius.
    fn legal_moves(&self) -> Vec<(i32, i32)> {
        self.inner
            .legal_moves()
            .into_iter()
            .map(|h| (h.q, h.r))
            .collect()
    }

    /// Legal placements within `radius` of any occupied cell (fast heuristic).
    ///
    /// The NN encoder uses `radius = 8` during self-play. For classical search
    /// `radius = 2` is usually sufficient.
    fn legal_moves_near(&self, radius: i32) -> Vec<(i32, i32)> {
        self.inner
            .legal_moves_near(radius)
            .into_iter()
            .map(|h| (h.q, h.r))
            .collect()
    }

    /// Threat-constrained legal placements when a forced tactical state exists.
    ///
    /// Returns `None` when no threat-based hard constraint applies (the caller
    /// should fall back to the full legal move set).
    ///
    /// When a constraint *does* apply the returned list is a subset of
    /// `legal_moves_near(radius)`:
    /// * **Winning turn** — only the 1 or 2 cells that complete a 6-line.
    /// * **Must-block** — only the cells that block the opponent's immediate win.
    fn threat_constrained_moves(&self, radius: i32) -> Option<Vec<(i32, i32)>> {
        match threat_status(&self.inner) {
            ThreatStatus::Quiet | ThreatStatus::Unblockable => None,
            ThreatStatus::WinningTurn(t) => {
                let first = t.first();
                let second = t.second();
                let legal = self.inner.legal_moves_near(radius);
                let result: Vec<(i32, i32)> = legal
                    .into_iter()
                    .filter(|h| *h == first || second == Some(*h))
                    .map(|h| (h.q, h.r))
                    .collect();
                if result.is_empty() { None } else { Some(result) }
            }
            ThreatStatus::MustBlock(b) => {
                let legal = self.inner.legal_moves_near(radius);
                let result: Vec<(i32, i32)> = legal
                    .into_iter()
                    .filter(|h| b.cells().contains(h))
                    .map(|h| (h.q, h.r))
                    .collect();
                if result.is_empty() { None } else { Some(result) }
            }
        }
    }

    /// Legal moves as packed bytes: pairs of little-endian `i32` `(q, r)`.
    ///
    /// Use `np.frombuffer(data, dtype=np.int32).reshape(-1, 2)` in Python to
    /// decode. This is much faster than returning a list of tuples when
    /// shipping large move lists across the FFI boundary.
    fn legal_moves_near_bytes<'py>(&self, py: Python<'py>, radius: i32) -> Bound<'py, PyBytes> {
        let moves = self.inner.legal_moves_near(radius);
        let mut buf: Vec<u8> = Vec::with_capacity(moves.len() * 8);
        for h in &moves {
            buf.extend_from_slice(&h.q.to_le_bytes());
            buf.extend_from_slice(&h.r.to_le_bytes());
        }
        PyBytes::new(py, &buf)
    }

    /// Board pieces as packed bytes: triples of little-endian `i32` `(q, r, player)`.
    ///
    /// Use `np.frombuffer(data, dtype=np.int32).reshape(-1, 3)` in Python.
    fn board_pieces_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let board = self.inner.stones();
        let mut buf: Vec<u8> = Vec::with_capacity(board.len() * 12);
        for (&h, &p) in board.iter() {
            buf.extend_from_slice(&h.q.to_le_bytes());
            buf.extend_from_slice(&h.r.to_le_bytes());
            buf.extend_from_slice(&(p as i32).to_le_bytes());
        }
        PyBytes::new(py, &buf)
    }

    /// Move history as packed bytes: triples of little-endian `i32` `(player, q, r)`.
    ///
    /// Use `np.frombuffer(data, dtype=np.int32).reshape(-1, 3)` in Python.
    fn move_history_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let hist = self.inner.move_history();
        let mut buf: Vec<u8> = Vec::with_capacity(hist.len() * 12);
        for r in hist {
            buf.extend_from_slice(&(r.player as i32).to_le_bytes());
            buf.extend_from_slice(&r.cell.q.to_le_bytes());
            buf.extend_from_slice(&r.cell.r.to_le_bytes());
        }
        PyBytes::new(py, &buf)
    }

    /// Encode the board as a `(13, 33, 33)` float32 tensor and return legal moves.
    ///
    /// Delegates to the canonical encoder in `encoder.rs`.
    ///
    /// Returns `(tensor, offset_q, offset_r, legal_moves_bytes)` where:
    ///
    /// * `tensor` — `numpy.ndarray` of shape `(13, 33, 33)` and dtype `float32`.
    /// * `offset_q`, `offset_r` — board coordinates that map to tensor index `(0, 0)`.
    ///   To recover board coords: `q = gi + offset_q`, `r = gj + offset_r`.
    /// * `legal_moves_bytes` — packed `i32` pairs `(q, r)` for the legal moves
    ///   inside `near_radius` (the same set used for channel 3).
    ///   Decode with `np.frombuffer(data, dtype=np.int32).reshape(-1, 2)`.
    ///
    /// # Channel reference
    ///
    /// | Ch | Name | Description |
    /// |---|---|---|
    /// | 0 | Own stones | `1.0` on each cell occupied by the current player. |
    /// | 1 | Opponent stones | `1.0` on each cell occupied by the opponent. |
    /// | 2 | Empty mask | `1.0 - ch0 - ch1`. |
    /// | 3 | Legal moves | `1.0` on each legal move (threat-constrained when `constrain_threats=True`). |
    /// | 4 | Turn phase | All `1.0` during the second placement of a turn. |
    /// | 5 | First stone | `1.0` on the first placement of the current turn (phase 2 only). |
    /// | 6 | Player colour | All `1.0` if current player is 0, else all `0.0`. |
    /// | 7 | Own recency | `1/(1+plies_ago)` for each own stone. |
    /// | 8 | Opponent recency | `1/(1+plies_ago)` for each opponent stone. |
    /// | 9 | Opponent hot cells | Empty cells in opponent's 4+ windows (must-block candidates). |
    /// | 10 | Own hot cells | Empty cells in own 4+ windows (attacking candidates). |
    /// | 11 | Distance from centre | Normalised hex distance to board centroid, in `[0, 1]`. |
    /// | 12 | Opponent's last turn | Cells placed by opponent in their most recent completed turn. |
    #[pyo3(signature = (near_radius, constrain_threats=true))]
    fn encode_board_and_legal<'py>(
        &self,
        py: Python<'py>,
        near_radius: i32,
        constrain_threats: bool,
    ) -> PyResult<(Bound<'py, PyArray3<f32>>, i32, i32, Bound<'py, PyBytes>)> {
        let encoded = encoder::encode_board(&self.inner, near_radius, constrain_threats);
        let arr = ndarray::Array3::from_shape_vec(
            (NUM_CHANNELS, BOARD_SIZE as usize, BOARD_SIZE as usize),
            encoded.tensor,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let arr = PyArray3::from_owned_array(py, arr);
        let mut legal_buf: Vec<u8> = Vec::with_capacity(encoded.legal_moves.len() * 8);
        for h in &encoded.legal_moves {
            legal_buf.extend_from_slice(&h.q.to_le_bytes());
            legal_buf.extend_from_slice(&h.r.to_le_bytes());
        }
        Ok((arr, encoded.offset_q, encoded.offset_r, PyBytes::new(py, &legal_buf)))
    }

    /// Extract the 13-element classical feature vector.
    ///
    /// This is a compact hand-crafted feature set used by the shallow
    /// evaluation function in classical alpha-beta search.
    fn extract_features(&self) -> Vec<f32> {
        encoder::extract_features(&self.inner).to_vec()
    }

    /// Run iterative-deepening alpha-beta search and return the best single placement.
    ///
    /// Parameters:
    /// * `time_ms` — search time budget in milliseconds.
    /// * `max_depth` — hard cap on search depth (measured in **turns**, not individual placements).
    /// * `near_radius` — candidate-generation radius (currently ignored; always uses 2).
    /// * `noise_level` — `0.0` = deterministic (default); `>0` = sample from the
    ///   top candidates with epsilon-greedy noise for variety.
    ///
    /// Returns `(best_q, best_r, score, depth_reached, nodes)`.
    #[pyo3(signature = (time_ms, max_depth, near_radius, noise_level=0.0))]
    fn classical_search(
        &self,
        py: Python<'_>,
        time_ms: u64,
        max_depth: i32,
        near_radius: i32,
        noise_level: f32,
    ) -> PyResult<(i32, i32, i32, i32, u64)> {
        let result = py
            .allow_threads(|| {
                search::iterative_deepening(
                    &self.inner,
                    Duration::from_millis(time_ms),
                    max_depth,
                    near_radius,
                    noise_level > 0.0,
                    noise_level,
                )
            })
            .map_err(|e: GameError| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let chosen = if noise_level > 0.0 {
            epsilon_topk_sample(result.best_move, &result.root_candidates, noise_level)
        } else {
            result.best_move
        };

        Ok((
            chosen.q,
            chosen.r,
            result.score,
            result.depth_reached,
            result.nodes,
        ))
    }

    /// Run turn-based search and return the full turn (1 or 2 moves).
    ///
    /// Returns `(moves, score, depth_reached, nodes)` where `moves` is a list
    /// of `(q, r)` tuples:
    /// * Opening move — `[(0, 0)]`
    /// * Normal turn — `[(q1, r1), (q2, r2)]`
    #[pyo3(signature = (time_ms, max_depth, near_radius=2, noise_level=0.0))]
    fn classical_search_turn(
        &self,
        py: Python<'_>,
        time_ms: u64,
        max_depth: i32,
        near_radius: i32,
        noise_level: f32,
    ) -> PyResult<(Vec<(i32, i32)>, i32, i32, u64)> {
        let result = py
            .allow_threads(|| {
                search::iterative_deepening(
                    &self.inner,
                    Duration::from_millis(time_ms),
                    max_depth,
                    near_radius,
                    false,
                    noise_level,
                )
            })
            .map_err(|e: GameError| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let turn = result.best_turn;
        let mut moves = vec![(turn.first().q, turn.first().r)];
        if let Some(m2) = turn.second() {
            moves.push((m2.q, m2.r));
        }

        Ok((moves, result.score, result.depth_reached, result.nodes))
    }

    /// Deep copy of this game state.
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    /// Reset to the initial empty state.
    ///
    /// All stones, history, and derived evaluator state are cleared.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Set a custom board position, bypassing normal turn rules.
    ///
    /// * `pieces` — list of `(q, r, player)` tuples.
    /// * `current_player` — whose turn it will be after setup (`0` or `1`).
    /// * `placements_remaining` — defaults to `2` (or `1` for a fully empty
    ///   board with player 0 to move).
    ///
    /// Raises `ValueError` if the position is inconsistent (e.g. overlapping stones).
    #[pyo3(signature = (pieces, current_player, placements_remaining=None))]
    fn set_position(
        &mut self,
        pieces: Vec<(i32, i32, u8)>,
        current_player: u8,
        placements_remaining: Option<u8>,
    ) -> PyResult<()> {
        let pr = placements_remaining.unwrap_or_else(|| if pieces.is_empty() && current_player == 0 { 1 } else { 2 });
        self.inner
            .set_position(&pieces, current_player, pr)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))
    }

    /// Move history as a list of `(player, q, r)` tuples.
    fn move_history(&self) -> Vec<(u8, i32, i32)> {
        self.inner
            .move_history()
            .iter()
            .map(|r| (r.player, r.cell.q, r.cell.r))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "HexGame(player={}, placements={}, moves={}, over={})",
            self.inner.current_player(),
            self.inner.placements_remaining(),
            self.inner.move_count(),
            self.inner.is_over()
        )
    }
}

// -------------------------------------------------------------------------
// Python-facing MCTS engine wrapper
// -------------------------------------------------------------------------

/// Python-facing MCTS engine that keeps the entire tree in Rust memory.
///
/// Typical usage from Python:
///
/// ```python
/// engine = MCTSEngine(game, num_sims, c_puct=1.4, near_radius=8)
///
/// # 1. Encode root board for GPU inference
/// tensor, oq, or_, legal_bytes = engine.init_root()
///
/// # 2. Run GPU inference → root_policy, root_value
/// engine.expand_root(root_policy, root_value, oq, or_, legal_bytes)
/// engine.add_dirichlet_noise(noise_array, noise_fraction)
///
/// # 3. Search loop
/// while not engine.done():
///     tensor, count = engine.select_leaves(batch_size)
///     # ... GPU inference → policies, values ...
///     engine.expand_and_backprop(policies, values)
///
/// # 4. Extract results


// -------------------------------------------------------------------------
// Bulk classical self-play (for bootstrap training data)
// -------------------------------------------------------------------------

/// Generate self-play data using classical alpha-beta search.
///
/// Returns a list of `(features, outcome, board_snap)` tuples from completed
/// games. This is fast because it uses the Rust alpha-beta engine rather than
/// neural-net MCTS.
///
/// * `features` — 13-element classical feature vector at the position.
/// * `outcome` — `1.0` if the player to move eventually won, `-1.0` if they
///   lost, `0.0` for a draw.
/// * `board_snap` — list of `(q, r, player)` tuples representing the board
///   state at that point.
#[pyfunction]
fn classical_self_play(
    py: Python<'_>,
    num_games: u32,
    time_ms: u64,
    max_depth: i32,
    near_radius: i32,
    max_moves: u32,
) -> PyResult<Vec<(Vec<f32>, f32, Vec<(i32, i32, u8)>)>> {
    py.allow_threads(|| -> Result<Vec<(Vec<f32>, f32, Vec<(i32, i32, u8)>)>, GameError> {
        let mut results = Vec::new();

        for _ in 0..num_games {
            let mut game = HexGameState::new();
            let mut positions: Vec<(Vec<f32>, u8, Vec<(i32, i32, u8)>)> = Vec::new();
            let mut move_num = 0u32;

            while !game.is_over() && move_num < max_moves {
                let feats = encoder::extract_features(&game).to_vec();
                let player = game.current_player();
                let board_snap: Vec<(i32, i32, u8)> =
                    game.stones().iter().map(|(&h, &p)| (h.q, h.r, p)).collect();
                positions.push((feats, player, board_snap));

                let result = search::iterative_deepening(
                    &game,
                    Duration::from_millis(time_ms),
                    max_depth,
                    near_radius,
                    false,
                    0.0,
                )?;
                let turn = result.best_turn;
                game.place(turn.first().q, turn.first().r)?;
                move_num += 1;
                if !game.is_over() {
                    if let Some(m2) = turn.second() {
                        game.place(m2.q, m2.r)?;
                        move_num += 1;
                    }
                }
            }

            let winner = game.winner();
            for (feats, player, board_snap) in positions {
                let outcome = match winner {
                    Some(w) if w == player => 1.0f32,
                    Some(_) => -1.0f32,
                    None => 0.0f32,
                };
                results.push((feats, outcome, board_snap));
            }
        }

        Ok(results)
    })
    .map_err(|e: GameError| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

// -------------------------------------------------------------------------
// Module definition
// -------------------------------------------------------------------------

/// The Python module definition.
///
/// When built with `maturin` the compiled shared object is named `_engine`
/// and re-exported by the Python wrapper as `hexgame`.
#[pymodule]
#[pyo3(name = "_engine")]
fn hexgame(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHexGame>()?;
    m.add_class::<PyMCTSEngine>()?;
    m.add_function(wrap_pyfunction!(classical_self_play, m)?)?;
    m.add("FEATURE_COUNT", encoder::FEATURE_COUNT)?;
    m.add("WIN_LENGTH", crate::core::WIN_LENGTH)?;
    m.add("PLACEMENT_RADIUS", crate::core::PLACEMENT_RADIUS)?;
    Ok(())
}
