//! PyO3 bindings — exposes the Rust game engine to Python.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::core::{hex_distance, Hex};
use crate::eval;
use crate::game::HexGameState;
use crate::mcts::MCTSEngine;
use crate::search;

use std::cell::Cell;
use std::time::{Duration, SystemTime};

// ── Board encoding constants (must match Python features.py) ─────────
const BOARD_SIZE: i32 = 33;
const HALF_BOARD: i32 = BOARD_SIZE / 2; // 16
const NUM_CHANNELS: usize = 13;
const BOARD_AREA: usize = (BOARD_SIZE * BOARD_SIZE) as usize; // 1089
const TENSOR_SIZE: usize = NUM_CHANNELS * BOARD_AREA; // 14157

/// Python-compatible "banker's rounding" (round half to even).
fn bankers_round(v: f64) -> i32 {
    let frac = v - v.floor();
    if (frac - 0.5).abs() < 1e-9 {
        // Exactly half: round to even
        let lo = v.floor() as i32;
        let hi = v.ceil() as i32;
        if lo % 2 == 0 {
            lo
        } else {
            hi
        }
    } else {
        v.round() as i32
    }
}

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

/// Next 64-bit XOR-shift value.
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
// Python-facing wrapper
// -------------------------------------------------------------------------

/// Python-facing wrapper around `HexGameState`.
#[pyclass(name = "HexGame")]
pub struct PyHexGame {
    inner: HexGameState,
}

#[pymethods]
impl PyHexGame {
    #[new]
    fn new() -> Self {
        Self {
            inner: HexGameState::new(),
        }
    }

    /// Place the current player's tile at (q, r).
    /// Returns True when the turn ends, False when the player has another placement.
    fn place(&mut self, q: i32, r: i32) -> PyResult<bool> {
        self.inner
            .place(q, r)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Undo the last placement.
    fn unmake_move(&mut self) {
        self.inner.unmake_move();
    }

    /// Whether the game has ended.
    #[getter]
    fn is_over(&self) -> bool {
        self.inner.is_over()
    }

    /// The winning player (0 or 1), or None.
    #[getter]
    fn winner(&self) -> Option<u8> {
        self.inner.winner
    }

    /// Current player (0 or 1).
    #[getter]
    fn current_player(&self) -> u8 {
        self.inner.current_player
    }

    /// Placements remaining in the current turn (1 or 2).
    #[getter]
    fn placements_remaining(&self) -> u8 {
        self.inner.placements_remaining
    }

    /// Total number of individual tile placements so far.
    #[getter]
    fn move_count(&self) -> u32 {
        self.inner.move_count
    }

    /// Incremental Zobrist hash of the board.
    #[getter]
    fn zobrist_hash(&self) -> u64 {
        self.inner.zobrist_hash
    }

    /// Whether a forced position exists (hot windows with 4+ own, 0 opponent).
    ///
    /// With 2 placements per turn, both 4-windows (2 empty) and 5-windows
    /// (1 empty) are single-turn wins. Returns a 2-bit value:
    ///   0 = no threats, 1 = own threat only, 2 = opponent threat only, 3 = both.
    /// Noise suppression only uses the opponent bit (must block).
    /// Zero extra cost (reads incremental counters).
    #[getter]
    fn threat_level(&self) -> u8 {
        let me = self.inner.current_player as usize;
        let opp = 1 - me;
        let own = (self.inner.window_fives[me] > 0 || self.inner.window_fours[me] > 0) as u8;
        let opp_threat =
            (self.inner.window_fives[opp] > 0 || self.inner.window_fours[opp] > 0) as u8;
        own | (opp_threat << 1)
    }

    /// How many placements the current turn allows.
    ///
    /// The first turn (move_count == 0, i.e. the very first placement) allows
    /// only 1 placement (P0's opening stone). All subsequent turns allow 2.
    fn placements_per_turn(&self, move_count: u32) -> u8 {
        if move_count == 0 { 1 } else { 2 }
    }

    /// Raw window-based positional eval, from player-0's perspective.
    /// Turn-independent: does NOT include threat bonuses or tempo.
    #[getter]
    fn window_eval(&self) -> i32 {
        self.inner.window_eval
    }

    /// Number of windows with 4+ pieces for the given player (0 or 1).
    fn window_fours(&self, player: u8) -> i32 {
        self.inner.window_fours[player as usize]
    }

    /// Number of windows with 5+ pieces (one-move wins) for the given player.
    fn window_fives(&self, player: u8) -> i32 {
        self.inner.window_fives[player as usize]
    }

    /// Number of windows with 3+ pieces for the given player.
    fn window_threes(&self, player: u8) -> i32 {
        self.inner.window_threes[player as usize]
    }

    /// Return active threat windows (4+ stones, unblocked) for the given player.
    /// Each window is a list of 6 `(q, r, occupied)` tuples where `occupied`
    /// is `true` if the player has a stone there, `false` if the cell is empty
    /// (i.e. needs to be filled to complete the line).
    fn get_threat_windows(&self, player: u8) -> Vec<Vec<(i32, i32, bool)>> {
        use crate::core::HEX_DIRECTIONS;
        use crate::game::WIN_LENGTH;
        let game = &self.inner;
        let pi = player as usize;
        let mut result = Vec::new();
        for &(wq, wr, dir) in &game.hot_windows[pi] {
            let (dq, dr) = HEX_DIRECTIONS[dir as usize];
            let mut cells = Vec::with_capacity(WIN_LENGTH as usize);
            for k in 0..WIN_LENGTH {
                let cq = wq + dq * k;
                let cr = wr + dr * k;
                let h = crate::core::Hex::new(cq, cr);
                let occupied = game.board.get(&h) == Some(&player);
                cells.push((cq, cr, occupied));
            }
            result.push(cells);
        }
        result
    }

    /// Dict of {(q, r): player} for all occupied cells.
    fn board_pieces(&self) -> Vec<(i32, i32, u8)> {
        self.inner
            .board
            .iter()
            .map(|(&h, &p)| (h.q, h.r, p))
            .collect()
    }

    /// All legal placements (exhaustive radius-8 scan).
    fn legal_moves(&self) -> Vec<(i32, i32)> {
        self.inner
            .legal_moves()
            .into_iter()
            .map(|h| (h.q, h.r))
            .collect()
    }

    /// Legal placements within `radius` of any occupied cell (fast).
    fn legal_moves_near(&self, radius: i32) -> Vec<(i32, i32)> {
        self.inner
            .legal_moves_near(radius)
            .into_iter()
            .map(|h| (h.q, h.r))
            .collect()
    }

    /// Threat-constrained legal placements when a forced tactical state exists.
    /// Returns None when no threat-based hard constraint applies.
    fn threat_constrained_moves(&self, radius: i32) -> Option<Vec<(i32, i32)>> {
        let legal = self.inner.legal_moves_near(radius);
        self.inner
            .compute_threat_constrained_moves(&legal, true)
            .map(|moves| moves.into_iter().map(|h| (h.q, h.r)).collect())
    }

    /// Legal moves as packed bytes: pairs of little-endian i32 (q, r).
    /// Use ``np.frombuffer(data, dtype=np.int32).reshape(-1, 2)`` in Python.
    fn legal_moves_near_bytes<'py>(&self, py: Python<'py>, radius: i32) -> Bound<'py, PyBytes> {
        let moves = self.inner.legal_moves_near(radius);
        let mut buf: Vec<u8> = Vec::with_capacity(moves.len() * 8);
        for h in &moves {
            buf.extend_from_slice(&h.q.to_le_bytes());
            buf.extend_from_slice(&h.r.to_le_bytes());
        }
        PyBytes::new(py, &buf)
    }

    /// Board pieces as packed bytes: triples of little-endian i32 (q, r, player).
    /// Use ``np.frombuffer(data, dtype=np.int32).reshape(-1, 3)`` in Python.
    fn board_pieces_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let board = &self.inner.board;
        let mut buf: Vec<u8> = Vec::with_capacity(board.len() * 12);
        for (&h, &p) in board.iter() {
            buf.extend_from_slice(&h.q.to_le_bytes());
            buf.extend_from_slice(&h.r.to_le_bytes());
            buf.extend_from_slice(&(p as i32).to_le_bytes());
        }
        PyBytes::new(py, &buf)
    }

    /// Move history as packed bytes: triples of little-endian i32 (player, q, r).
    /// Use ``np.frombuffer(data, dtype=np.int32).reshape(-1, 3)`` in Python.
    fn move_history_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let hist = &self.inner.move_history;
        let mut buf: Vec<u8> = Vec::with_capacity(hist.len() * 12);
        for r in hist {
            buf.extend_from_slice(&(r.player as i32).to_le_bytes());
            buf.extend_from_slice(&r.cell.q.to_le_bytes());
            buf.extend_from_slice(&r.cell.r.to_le_bytes());
        }
        PyBytes::new(py, &buf)
    }

    /// Encode the board as a (13, 33, 33) float32 tensor in Rust.
    ///
    /// Returns ``(board_bytes, offset_q, offset_r, legal_moves_bytes)`` where:
    /// - ``board_bytes``: packed little-endian f32, shape (13, 33, 33).
    ///   Use ``np.frombuffer(data, dtype=np.float32).reshape(13, 33, 33)``
    /// - ``legal_moves_bytes``: packed i32 pairs (q, r) — the legal moves
    ///   within ``near_radius`` (same ones used for channel 3).
    ///   Use ``np.frombuffer(data, dtype=np.int32).reshape(-1, 2)``
    ///
    /// This avoids two separate Rust→Python calls and all Python loops.
    #[pyo3(signature = (near_radius, constrain_threats=true))]
    fn encode_board_and_legal<'py>(
        &self,
        py: Python<'py>,
        near_radius: i32,
        constrain_threats: bool,
    ) -> (Bound<'py, PyBytes>, i32, i32, Bound<'py, PyBytes>) {
        let game = &self.inner;
        let board = &game.board;

        // ── Compute centroid and offsets ──
        // Python's round() uses banker's rounding (round half to even).
        let (offset_q, offset_r) = if board.is_empty() {
            (-HALF_BOARD, -HALF_BOARD)
        } else {
            let n = board.len() as f64;
            let (mut sq, mut sr) = (0i64, 0i64);
            for &h in board.keys() {
                sq += h.q as i64;
                sr += h.r as i64;
            }
            let cq = bankers_round(sq as f64 / n);
            let cr = bankers_round(sr as f64 / n);
            (cq - HALF_BOARD, cr - HALF_BOARD)
        };

        let current = game.current_player;
        let mc = game.move_count;
        let pr = game.placements_remaining;
        let is_phase_2 = pr == 1 && mc > 0;

        // ── Allocate tensor (all zeros) ──
        let mut tensor = vec![0.0f32; TENSOR_SIZE];

        // Helper: index into flat tensor [ch, gi, gj]
        #[inline(always)]
        fn idx(ch: usize, gi: i32, gj: i32) -> usize {
            ch * BOARD_AREA + (gi as usize) * (BOARD_SIZE as usize) + gj as usize
        }

        // ── Ch 0-1: player stones ──
        for (&h, &player) in board.iter() {
            let gi = h.q - offset_q;
            let gj = h.r - offset_r;
            if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
                if player == current {
                    tensor[idx(0, gi, gj)] = 1.0;
                } else {
                    tensor[idx(1, gi, gj)] = 1.0;
                }
            }
        }

        // ── Ch 7-8: stone recency 1/(1+plies_ago), split by player ──
        for (ply_idx, rec) in game.move_history.iter().enumerate() {
            let gi = rec.cell.q - offset_q;
            let gj = rec.cell.r - offset_r;
            if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
                let plies_ago = mc - ply_idx as u32;
                let recency = 1.0 / (1.0 + plies_ago as f32);
                let ch = if rec.player == current { 7 } else { 8 };
                tensor[idx(ch, gi, gj)] = recency;
            }
        }

        // ── Ch 2: empty cells mask = 1 - ch0 - ch1 ──
        let ch2_start = 2 * BOARD_AREA as usize;
        let ch0_start = 0usize;
        let ch1_start = BOARD_AREA as usize;
        for i in 0..BOARD_AREA as usize {
            tensor[ch2_start + i] = 1.0 - tensor[ch0_start + i] - tensor[ch1_start + i];
        }

        // ── Ch 3: legal moves + build legal_moves output ──
        let mut legal = game.legal_moves_near(near_radius);
        if let Some(constrained) = game.compute_threat_constrained_moves(&legal, constrain_threats)
        {
            legal = constrained;
        }
        let mut legal_buf: Vec<u8> = Vec::with_capacity(legal.len() * 8);
        for h in &legal {
            legal_buf.extend_from_slice(&h.q.to_le_bytes());
            legal_buf.extend_from_slice(&h.r.to_le_bytes());
            let gi = h.q - offset_q;
            let gj = h.r - offset_r;
            if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
                tensor[idx(3, gi, gj)] = 1.0;
            }
        }

        // ── Ch 4: turn phase ──
        if is_phase_2 {
            let start = 4 * BOARD_AREA as usize;
            tensor[start..start + BOARD_AREA as usize].fill(1.0);
        }

        // ── Ch 5: first stone of current turn (phase 2 only) ──
        if is_phase_2 {
            if let Some(last) = game.move_history.last() {
                let gi = last.cell.q - offset_q;
                let gj = last.cell.r - offset_r;
                if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
                    tensor[idx(5, gi, gj)] = 1.0;
                }
            }
        }

        // ── Ch 6: current player color ──
        if current == 0 {
            let start = 6 * BOARD_AREA as usize;
            tensor[start..start + BOARD_AREA as usize].fill(1.0);
        }

        // ── Ch 11: distance from centroid ──
        // hex_distance from each cell to the grid center, normalized by HALF_BOARD.
        // D6-invariant since hex distance is preserved under rotations/reflections.
        {
            let center = Hex::new(offset_q + HALF_BOARD, offset_r + HALF_BOARD);
            for gi in 0..BOARD_SIZE {
                for gj in 0..BOARD_SIZE {
                    let h = Hex::new(gi + offset_q, gj + offset_r);
                    let dist = hex_distance(h, center) as f32 / HALF_BOARD as f32;
                    tensor[idx(11, gi, gj)] = dist;
                }
            }
        }

        // ── Ch 12: opponent's most recent completed turn ──
        for h in game.opponent_last_turn_cells() {
            let gi = h.q - offset_q;
            let gj = h.r - offset_r;
            if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
                tensor[idx(12, gi, gj)] = 1.0;
            }
        }

        // ── Ch 9: opponent's hot cells (empty cells in opp's 4+ windows) ──
        // ── Ch 10: own hot cells (empty cells in own's 4+ windows) ──
        {
            use crate::core::HEX_DIRECTIONS;
            use crate::game::WIN_LENGTH;
            let opp = (1 - current) as usize;
            let own = current as usize;

            for (ch, player_idx) in [(9usize, opp), (10usize, own)] {
                for &(wq, wr, dir) in &game.hot_windows[player_idx] {
                    let (dq, dr) = HEX_DIRECTIONS[dir as usize];
                    for k in 0..WIN_LENGTH {
                        let cq = wq + dq * k;
                        let cr = wr + dr * k;
                        let h = Hex::new(cq, cr);
                        if !board.contains_key(&h) {
                            let gi = cq - offset_q;
                            let gj = cr - offset_r;
                            if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
                                tensor[idx(ch, gi, gj)] = 1.0;
                            }
                        }
                    }
                }
            }
        }

        // ── Pack tensor as bytes (zero-copy on little-endian) ──
        let tensor_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                tensor.as_ptr() as *const u8,
                tensor.len() * std::mem::size_of::<f32>(),
            )
        };

        (
            PyBytes::new(py, &tensor_bytes),
            offset_q,
            offset_r,
            PyBytes::new(py, &legal_buf),
        )
    }

    /// Compute per-cell per-axis influence scores for the current board.
    ///
    /// Returns packed f32 bytes of shape (3, 33, 33) from the current player's
    /// perspective. Values in [-1, 1].
    fn axis_influence<'py>(
        &self,
        py: Python<'py>,
        offset_q: i32,
        offset_r: i32,
    ) -> Bound<'py, PyBytes> {
        let mut out = vec![0.0f32; 3 * BOARD_AREA];
        self.inner.compute_axis_influence(
            offset_q,
            offset_r,
            BOARD_SIZE,
            self.inner.current_player,
            &mut out,
        );
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                out.as_ptr() as *const u8,
                out.len() * std::mem::size_of::<f32>(),
            )
        };
        PyBytes::new(py, bytes)
    }

    /// Classical pattern-based evaluation from `player`'s perspective.
    fn evaluate(&self, player: u8) -> i32 {
        eval::evaluate(&self.inner, player)
    }

    /// Extract the 13-element feature vector (for classical NN).
    fn extract_features(&self) -> Vec<f32> {
        eval::extract_features(&self.inner).to_vec()
    }

    /// Quick heuristic score for a single move (for move ordering).
    fn score_move(&self, q: i32, r: i32) -> i32 {
        eval::score_move(&self.inner, Hex::new(q, r), self.inner.current_player)
    }

    /// Run iterative-deepening alpha-beta search.
    ///
    /// Parameters:
    /// - `time_ms`:     search time budget in milliseconds
    /// - `max_depth`:   hard cap on search depth (turns, not placements)
    /// - `near_radius`: candidate generation radius (ignored, always uses 2)
    /// - `noise_level`: 0.0 = deterministic (default); >0 = sample from top
    ///                  candidates with softmax temperature for variability.
    ///
    /// Returns (best_q, best_r, score, depth_reached, nodes).
    #[pyo3(signature = (time_ms, max_depth, near_radius, noise_level=0.0))]
    fn classical_search(
        &self,
        time_ms: u64,
        max_depth: i32,
        near_radius: i32,
        noise_level: f32,
    ) -> (i32, i32, i32, i32, u64) {
        let result = search::iterative_deepening(
            &self.inner,
            Duration::from_millis(time_ms),
            max_depth,
            near_radius,
            noise_level > 0.0,
            noise_level,
        );

        let chosen = if noise_level > 0.0 {
            epsilon_topk_sample(result.best_move, &result.root_candidates, noise_level)
        } else {
            result.best_move
        };

        (
            chosen.q,
            chosen.r,
            result.score,
            result.depth_reached,
            result.nodes,
        )
    }

    /// Run turn-based search and return the full turn (1 or 2 moves).
    ///
    /// Returns a list of (q, r) tuples representing the full turn.
    /// For the opening move this is [(0, 0)].
    /// For all other turns this is [(q1, r1), (q2, r2)].
    #[pyo3(signature = (time_ms, max_depth, near_radius=2, noise_level=0.0))]
    fn classical_search_turn(
        &self,
        time_ms: u64,
        max_depth: i32,
        near_radius: i32,
        noise_level: f32,
    ) -> (Vec<(i32, i32)>, i32, i32, u64) {
        let result = search::iterative_deepening(
            &self.inner,
            Duration::from_millis(time_ms),
            max_depth,
            near_radius,
            false,
            noise_level,
        );

        let turn = result.best_turn;
        let mut moves = vec![(turn.m1.q, turn.m1.r)];
        if let Some(m2) = turn.m2 {
            moves.push((m2.q, m2.r));
        }

        (moves, result.score, result.depth_reached, result.nodes)
    }

    /// Deep copy of this game state.
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    /// Reset to the initial empty state.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Set a custom board position, bypassing normal turn rules.
    ///
    /// `pieces` is a list of (q, r, player) tuples.
    /// `current_player` (0 or 1) is whose turn it will be after setup.
    /// `placements_remaining` defaults to 2 (or 1 for a fully empty board with P0 to move).
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

    /// Move history as list of (player, q, r).
    fn move_history(&self) -> Vec<(u8, i32, i32)> {
        self.inner
            .move_history
            .iter()
            .map(|r| (r.player, r.cell.q, r.cell.r))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "HexGame(player={}, placements={}, moves={}, over={})",
            self.inner.current_player,
            self.inner.placements_remaining,
            self.inner.move_count,
            self.inner.is_over()
        )
    }
}

// -------------------------------------------------------------------------
// Python-facing MCTS engine wrapper
// -------------------------------------------------------------------------

/// Python-facing MCTS engine that keeps the tree in Rust.
///
/// Usage from Python:
/// ```python
/// engine = MCTSEngine(game, num_sims, c_puct=1.4, near_radius=8)
/// # Get root board tensor for initial GPU eval
/// tensor_bytes, oq, or_, legal_bytes = engine.init_root()
/// # ... run GPU inference to get root_policy, root_value ...
/// engine.expand_root(root_policy_bytes, root_value, oq, or_, legal_bytes)
/// engine.add_dirichlet_noise(noise_array, noise_fraction)
///
/// while not engine.done():
///     tensor_bytes, count = engine.select_leaves(batch_size)
///     # ... GPU inference → policy_bytes, value_bytes ...
///     engine.expand_and_backprop(policy_bytes, value_bytes)
///
/// moves_q, moves_r, visits, root_value = engine.get_results()
/// ```
#[pyclass(name = "MCTSEngine")]
struct PyMCTSEngine {
    inner: MCTSEngine,
}

#[pymethods]
impl PyMCTSEngine {
    #[new]
    #[pyo3(signature = (game, num_simulations, c_puct=1.4, near_radius=8, c_puct_init=19652.0, constrain_threats=true, selector="puct", c1=1.4, c2=3.0, arena_sim_hint=None))]
    fn new(
        game: &PyHexGame,
        num_simulations: u32,
        c_puct: f32,
        near_radius: i32,
        c_puct_init: f32,
        constrain_threats: bool,
        selector: &str,
        c1: f32,
        c2: f32,
        arena_sim_hint: Option<u32>,
    ) -> Self {
        let hint = arena_sim_hint.unwrap_or(num_simulations);
        let mut engine = MCTSEngine::with_arena_sim_hint(
            game.inner.clone(),
            num_simulations,
            hint,
            c_puct,
            near_radius,
            constrain_threats,
        );
        engine.c_puct_init = c_puct_init;
        engine.set_selector(selector, c1, c2);
        Self {
            inner: engine,
        }
    }

    /// Initialize root: encode the board and return tensor bytes for GPU eval.
    /// Returns (tensor_bytes, offset_q, offset_r, legal_moves_bytes) or None.
    fn init_root<'py>(
        &mut self,
        py: Python<'py>,
    ) -> Option<(Bound<'py, PyBytes>, i32, i32, Bound<'py, PyBytes>)> {
        let (tensor, oq, or_, legal) = self.inner.init_root()?;
        let tensor_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                tensor.as_ptr() as *const u8,
                tensor.len() * std::mem::size_of::<f32>(),
            )
        };
        let mut legal_buf: Vec<u8> = Vec::with_capacity(legal.len() * 8);
        for h in &legal {
            legal_buf.extend_from_slice(&h.q.to_le_bytes());
            legal_buf.extend_from_slice(&h.r.to_le_bytes());
        }
        Some((
            PyBytes::new(py, tensor_bytes),
            oq,
            or_,
            PyBytes::new(py, &legal_buf),
        ))
    }

    /// Expand root node with GPU-provided policy and value.
    /// policy_bytes: packed f32 (1089,), value: scalar, legal_bytes: packed i32 pairs.
    fn expand_root(
        &mut self,
        policy_bytes: &[u8],
        value: f32,
        offset_q: i32,
        offset_r: i32,
        legal_bytes: &[u8],
    ) {
        let policy: &[f32] = unsafe {
            std::slice::from_raw_parts(
                policy_bytes.as_ptr() as *const f32,
                policy_bytes.len() / std::mem::size_of::<f32>(),
            )
        };
        let legal_i32: &[i32] = unsafe {
            std::slice::from_raw_parts(
                legal_bytes.as_ptr() as *const i32,
                legal_bytes.len() / std::mem::size_of::<i32>(),
            )
        };
        let legal: Vec<Hex> = legal_i32
            .chunks_exact(2)
            .map(|c| Hex::new(c[0], c[1]))
            .collect();
        self.inner
            .expand_root(policy, value, offset_q, offset_r, &legal);
    }

    /// Add Dirichlet noise to root priors.
    #[pyo3(signature = (noise_bytes, noise_fraction))]
    fn add_dirichlet_noise(
        &mut self,
        noise_bytes: &[u8],
        noise_fraction: f32,
    ) {
        let noise: &[f32] = unsafe {
            std::slice::from_raw_parts(
                noise_bytes.as_ptr() as *const f32,
                noise_bytes.len() / std::mem::size_of::<f32>(),
            )
        };
        self.inner
            .add_dirichlet_noise(noise, noise_fraction);
    }

    /// Initialize Gumbel Sequential Halving mode.
    ///
    /// Replaces Dirichlet noise with Gumbel-based exploration. Call after
    /// `expand_root`. The top `num_considered` actions (by Gumbel + log-prior)
    /// are selected as SH candidates. Simulations are allocated in rounds,
    /// halving the candidate set after each round based on Q-values.
    ///
    /// Do NOT also call `add_dirichlet_noise` when using Gumbel mode.
    #[pyo3(signature = (num_considered=16))]
    fn init_gumbel(&mut self, num_considered: u32) {
        self.inner.init_gumbel(num_considered);
    }

    /// Whether we have done enough simulations.
    fn done(&self) -> bool {
        self.inner.done()
    }

    /// Select leaves and encode their boards.
    /// Returns (tensor_bytes, non_terminal_count).
    /// tensor_bytes is packed f32 of shape (non_terminal_count, 13, 33, 33).
    fn select_leaves<'py>(
        &mut self,
        py: Python<'py>,
        batch_size: u32,
    ) -> (Bound<'py, PyBytes>, u32) {
        let (tensors, count) = self.inner.select_leaves(batch_size);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                tensors.as_ptr() as *const u8,
                tensors.len() * std::mem::size_of::<f32>(),
            )
        };
        (PyBytes::new(py, bytes), count)
    }

    /// Expand and backpropagate using GPU results.
    /// policies_bytes: packed f32 (N, 1089), values_bytes: packed f32 (N,).
    fn expand_and_backprop(&mut self, policies_bytes: &[u8], values_bytes: &[u8]) {
        let policies: &[f32] = unsafe {
            std::slice::from_raw_parts(
                policies_bytes.as_ptr() as *const f32,
                policies_bytes.len() / std::mem::size_of::<f32>(),
            )
        };
        let values: &[f32] = unsafe {
            std::slice::from_raw_parts(
                values_bytes.as_ptr() as *const f32,
                values_bytes.len() / std::mem::size_of::<f32>(),
            )
        };
        self.inner.expand_and_backprop(policies, values);
    }

    /// Pipeline variant of select_leaves: saves current pending → prev_pending
    /// before selecting new leaves. This enables GPU/CPU overlap.
    fn select_leaves_pipeline<'py>(
        &mut self,
        py: Python<'py>,
        batch_size: u32,
    ) -> (Bound<'py, PyBytes>, u32) {
        let (tensors, count) = self.inner.select_leaves_pipeline(batch_size);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                tensors.as_ptr() as *const u8,
                tensors.len() * std::mem::size_of::<f32>(),
            )
        };
        (PyBytes::new(py, bytes), count)
    }

    /// Expand and backprop the PREVIOUS batch (stored in prev_pending).
    /// Used for pipeline overlap: select N+1 → expand N → ...
    fn expand_prev_and_backprop(&mut self, policies_bytes: &[u8], values_bytes: &[u8]) {
        let policies: &[f32] = unsafe {
            std::slice::from_raw_parts(
                policies_bytes.as_ptr() as *const f32,
                policies_bytes.len() / std::mem::size_of::<f32>(),
            )
        };
        let values: &[f32] = unsafe {
            std::slice::from_raw_parts(
                values_bytes.as_ptr() as *const f32,
                values_bytes.len() / std::mem::size_of::<f32>(),
            )
        };
        self.inner.expand_prev_and_backprop(policies, values);
    }

    /// Get results: (moves_q, moves_r, visits, root_value).
    fn get_results(&self) -> (Vec<i32>, Vec<i32>, Vec<u32>, f32) {
        self.inner.get_results()
    }

    /// Get number of root children (for noise array sizing).
    fn root_child_count(&self) -> u16 {
        self.inner.root_child_count()
    }

    /// Get prior probabilities of root children (for shaped Dirichlet noise).
    fn root_child_priors(&self) -> Vec<f32> {
        self.inner.root_child_priors()
    }

    /// Get Q-values of root children from root player's perspective.
    fn root_child_q_values(&self) -> Vec<f32> {
        self.inner.root_child_q_values()
    }

    /// Extract encoded board states and move histories for tree nodes.
    #[pyo3(signature = (min_visits=1))]
    fn extract_tree_node_states<'py>(
        &mut self,
        py: Python<'py>,
        min_visits: u32,
    ) -> (Bound<'py, PyBytes>, Vec<Vec<(i32, i32, i32)>>, usize) {
        let (packed, histories, count) = self.inner.extract_tree_node_states(min_visits);
        let tensor_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                packed.as_ptr() as *const u8,
                packed.len() * std::mem::size_of::<f32>(),
            )
        };
        let py_histories: Vec<Vec<(i32, i32, i32)>> = histories
            .into_iter()
            .map(|history| {
                history
                    .into_iter()
                    .map(|(player, q, r)| (player as i32, q as i32, r as i32))
                    .collect()
            })
            .collect();
        (PyBytes::new(py, tensor_bytes), py_histories, count)
    }

    /// Re-root the tree at the child matching action (q, r) for subtree reuse.
    ///
    /// After placement 1 is selected, call this to advance the tree so that
    /// placement 2's MCTS starts from the surviving subtree. The arena is not
    /// compacted — dead sibling subtrees remain in memory.
    ///
    /// The pipeline must be fully flushed before calling (no pending leaves).
    /// After re-root, call `init_root` + `expand_root` if the new root is not
    /// yet expanded (`root_child_count() == 0`), or go straight to
    /// `add_dirichlet_noise` / `init_gumbel` + search loop if it is.
    #[pyo3(signature = (q, r, new_num_simulations))]
    fn re_root(&mut self, q: i32, r: i32, new_num_simulations: u32) {
        self.inner.re_root(q as i16, r as i16, new_num_simulations);
    }
}

// -------------------------------------------------------------------------
// Python-facing Multi MCTS engine wrapper
// -------------------------------------------------------------------------

/// Manages N MCTSEngines in Rust, merging leaf batches for one GPU call per iteration.
///
/// Usage from Python:
/// ```python
// -------------------------------------------------------------------------
// Bulk classical self-play (for bootstrap training data)
// -------------------------------------------------------------------------

/// Generate self-play data using classical search.
/// Returns a list of (features, outcome) tuples from completed games.
/// This is fast because it uses the Rust alpha-beta engine.
#[pyfunction]
fn classical_self_play(
    num_games: u32,
    time_ms: u64,
    max_depth: i32,
    near_radius: i32,
    max_moves: u32,
) -> Vec<(Vec<f32>, f32, Vec<(i32, i32, u8)>)> {
    let mut results = Vec::new();

    for _ in 0..num_games {
        let mut game = HexGameState::new();
        let mut positions: Vec<(Vec<f32>, u8, Vec<(i32, i32, u8)>)> = Vec::new();
        let mut move_num = 0u32;

        while !game.is_over() && move_num < max_moves {
            let feats = eval::extract_features(&game).to_vec();
            let player = game.current_player;
            let board_snap: Vec<(i32, i32, u8)> =
                game.board.iter().map(|(&h, &p)| (h.q, h.r, p)).collect();
            positions.push((feats, player, board_snap));

            // Use turn-based alpha-beta to pick a turn (1 or 2 moves).
            let result = search::iterative_deepening(
                &game,
                Duration::from_millis(time_ms),
                max_depth,
                near_radius,
                false,
                0.0,
            );
            let turn = result.best_turn;
            game.place(turn.m1.q, turn.m1.r).unwrap_or(true);
            move_num += 1;
            if !game.is_over() {
                if let Some(m2) = turn.m2 {
                    game.place(m2.q, m2.r).unwrap_or(true);
                    move_num += 1;
                }
            }
        }

        let winner = game.winner;
        for (feats, player, board_snap) in positions {
            let outcome = match winner {
                Some(w) if w == player => 1.0f32,
                Some(_) => -1.0f32,
                None => 0.0f32,
            };
            results.push((feats, outcome, board_snap));
        }
    }

    results
}

// -------------------------------------------------------------------------
// Module definition
// -------------------------------------------------------------------------

/// The Python module definition.
#[pymodule]
#[pyo3(name = "_engine")]
fn hexgame(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHexGame>()?;
    m.add_class::<PyMCTSEngine>()?;
    m.add_function(wrap_pyfunction!(classical_self_play, m)?)?;
    m.add("FEATURE_COUNT", eval::FEATURE_COUNT)?;
    m.add("WIN_LENGTH", crate::game::WIN_LENGTH)?;
    m.add("PLACEMENT_RADIUS", crate::game::PLACEMENT_RADIUS)?;
    Ok(())
}
