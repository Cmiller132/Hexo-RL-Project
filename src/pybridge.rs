//! PyO3 bindings — exposes the Rust game engine to Python.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use numpy::{ndarray, PyArray3, PyArray4, PyReadonlyArray1};

use crate::core::Hex;
use crate::threats::{threat_status, ThreatStatus};
use crate::encoder;
use crate::eval;
use crate::board::HexGameState;
use crate::mcts::MCTSEngine;
use crate::search;

use std::cell::Cell;
use std::time::{Duration, SystemTime};

// Re-export encoder constants so shapes stay in sync with the canonical implementation.
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
// Python-facing wrapper for HexGameState
// -------------------------------------------------------------------------

/// Python-facing wrapper around `HexGameState`.
#[pyclass(name = "HexGame")]
pub struct PyHexGame {
    inner: HexGameState,
}

#[pymethods]
impl PyHexGame {
    /// Create a new game in the initial empty state.
    #[new]
    fn new() -> Self {
        Self {
            inner: HexGameState::new(),
        }
    }

    /// Place the current player's tile at (q, r).
    ///
    /// Returns `True` when the turn ends, `False` when the player has another
    /// placement remaining.
    fn place(&mut self, q: i32, r: i32) -> PyResult<bool> {
        self.inner
            .place(q, r)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Undo the last placement.
    fn unplace(&mut self) {
        self.inner.unplace();
    }

    /// Whether the game has ended.
    #[getter]
    fn is_over(&self) -> bool {
        self.inner.is_over()
    }

    /// The winning player (0 or 1), or `None`.
    #[getter]
    fn winner(&self) -> Option<u8> {
        self.inner.winner()
    }

    /// Current player (0 or 1).
    #[getter]
    fn current_player(&self) -> u8 {
        self.inner.current_player()
    }

    /// Placements remaining in the current turn (1 or 2).
    #[getter]
    fn placements_remaining(&self) -> u8 {
        self.inner.placements_remaining()
    }

    /// Total number of individual tile placements so far.
    #[getter]
    fn move_count(&self) -> u32 {
        self.inner.move_count()
    }

    /// Incremental Zobrist hash of the board.
    #[getter]
    fn zobrist_hash(&self) -> u64 {
        self.inner.zobrist()
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
        let me = self.inner.current_player() as usize;
        let opp = 1 - me;
        let own = (self.inner.eval().counts(me as u8).fives > 0
                || self.inner.eval().counts(me as u8).fours > 0) as u8;
        let opp_threat = (self.inner.eval().counts(opp as u8).fives > 0
                       || self.inner.eval().counts(opp as u8).fours > 0) as u8;
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
        self.inner.eval().score()
    }

    /// Number of windows with 4+ pieces for the given player (0 or 1).
    fn window_fours(&self, player: u8) -> i32 {
        self.inner.eval().counts(player).fours as i32
    }

    /// Number of windows with 5+ pieces (one-move wins) for the given player.
    fn window_fives(&self, player: u8) -> i32 {
        self.inner.eval().counts(player).fives as i32
    }

    /// Number of windows with 3+ pieces for the given player.
    fn window_threes(&self, player: u8) -> i32 {
        self.inner.eval().counts(player).threes as i32
    }

    /// Return active threat windows (4+ stones, unblocked) for the given player.
    ///
    /// Each window is a list of 6 `(q, r, occupied)` tuples where `occupied`
    /// is `true` if the player has a stone there, `false` if the cell is empty
    /// (i.e. needs to be filled to complete the line).
    fn get_threat_windows(&self, player: u8) -> Vec<Vec<(i32, i32, bool)>> {
        use crate::core::HEX_DIRECTIONS;
        use crate::core::WIN_LENGTH;
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

    /// Dict of {(q, r): player} for all occupied cells.
    fn board_pieces(&self) -> Vec<(i32, i32, u8)> {
        self.inner
            .stones()
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
    ///
    /// Returns `None` when no threat-based hard constraint applies.
    fn threat_constrained_moves(&self, radius: i32) -> Option<Vec<(i32, i32)>> {
        let legal = self.inner.legal_moves_near(radius);
        let constrained: Vec<Hex> = match threat_status(&self.inner) {
            ThreatStatus::Quiet | ThreatStatus::Unblockable => Vec::new(),
            ThreatStatus::WinningTurn(t) => {
                let mut allowed = vec![t.first()];
                if let Some(s) = t.second() {
                    allowed.push(s);
                }
                legal.into_iter().filter(|h| allowed.contains(h)).collect()
            }
            ThreatStatus::MustBlock(b) => {
                legal.into_iter().filter(|h| b.cells.contains(h)).collect()
            }
        };
        if constrained.is_empty() {
            None
        } else {
            Some(constrained.into_iter().map(|h| (h.q, h.r)).collect())
        }
    }

    /// Legal moves as packed bytes: pairs of little-endian i32 (q, r).
    ///
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
    ///
    /// Use ``np.frombuffer(data, dtype=np.int32).reshape(-1, 3)`` in Python.
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

    /// Move history as packed bytes: triples of little-endian i32 (player, q, r).
    ///
    /// Use ``np.frombuffer(data, dtype=np.int32).reshape(-1, 3)`` in Python.
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

    /// Encode the board as a (13, 33, 33) float32 tensor and return legal moves.
    ///
    /// Delegates to the canonical encoder in `encoder.rs`.
    ///
    /// Returns ``(tensor, offset_q, offset_r, legal_moves_bytes)`` where:
    /// - ``tensor``: a `numpy.ndarray` of shape (13, 33, 33) and dtype float32.
    /// - ``legal_moves_bytes``: packed i32 pairs (q, r) — the legal moves
    ///   within ``near_radius`` (same ones used for channel 3).
    ///   Use ``np.frombuffer(data, dtype=np.int32).reshape(-1, 2)``
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

    /// Extract the 13-element feature vector (for classical NN).
    fn extract_features(&self) -> Vec<f32> {
        eval::extract_features(&self.inner).to_vec()
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
    /// For the opening move this is `[(0, 0)]`.
    /// For all other turns this is `[(q1, r1), (q2, r2)]`.
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
        let mut moves = vec![(turn.first().q, turn.first().r)];
        if let Some(m2) = turn.second() {
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

/// Python-facing MCTS engine that keeps the tree in Rust.
///
/// Usage from Python:
/// ```python
/// engine = MCTSEngine(game, num_sims, c_puct=1.4, near_radius=8)
/// # Get root board tensor for initial GPU eval
/// tensor, oq, or_, legal_bytes = engine.init_root()
/// # ... run GPU inference to get root_policy, root_value ...
/// engine.expand_root(root_policy, root_value, oq, or_, legal_bytes)
/// engine.add_dirichlet_noise(noise_array, noise_fraction)
///
/// while not engine.done():
///     tensor, count = engine.select_leaves(batch_size)
///     # ... GPU inference → policies, values ...
///     engine.expand_and_backprop(policies, values)
///
/// moves_q, moves_r, visits, root_value = engine.get_results()
/// ```
#[pyclass(name = "MCTSEngine")]
struct PyMCTSEngine {
    inner: MCTSEngine,
}

#[pymethods]
impl PyMCTSEngine {
    /// Create a new MCTS engine.
    ///
    /// Parameters:
    /// - `game`: starting board position (`HexGame`)
    /// - `num_simulations`: total MCTS rollouts to perform
    /// - `c_puct`: base exploration constant (default 1.4)
    /// - `near_radius`: legal-move generation radius (default 8)
    /// - `c_puct_init`: dynamic c_puct scaling constant (default 19652.0)
    /// - `constrain_threats`: enable threat-based move filtering at root (default True)
    /// - `arena_sim_hint`: optional arena pre-allocation hint (defaults to `num_simulations`)
    #[new]
    #[pyo3(signature = (game, num_simulations, c_puct=1.4, near_radius=8, c_puct_init=19652.0, constrain_threats=true, arena_sim_hint=None))]
    fn new(
        game: &PyHexGame,
        num_simulations: u32,
        c_puct: f32,
        near_radius: i32,
        c_puct_init: f32,
        constrain_threats: bool,
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
        Self { inner: engine }
    }

    /// Initialize root: encode the board and return a numpy tensor for GPU eval.
    ///
    /// Returns `Some((tensor, offset_q, offset_r, legal_moves_bytes))` or `None`
    /// if the game is already over.
    ///
    /// `tensor` is a `numpy.ndarray` of shape (13, 33, 33) and dtype float32.
    /// `legal_moves_bytes` contains packed i32 pairs (q, r).
    fn init_root<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<(Bound<'py, PyArray3<f32>>, i32, i32, Bound<'py, PyBytes>)>> {
        let Some((tensor, oq, or_, legal)) = self.inner.init_root() else {
            return Ok(None);
        };
        let arr = ndarray::Array3::from_shape_vec(
            (NUM_CHANNELS, BOARD_SIZE as usize, BOARD_SIZE as usize),
            tensor,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let arr = PyArray3::from_owned_array(py, arr);
        let mut legal_buf: Vec<u8> = Vec::with_capacity(legal.len() * 8);
        for h in &legal {
            legal_buf.extend_from_slice(&h.q.to_le_bytes());
            legal_buf.extend_from_slice(&h.r.to_le_bytes());
        }
        Ok(Some((arr, oq, or_, PyBytes::new(py, &legal_buf))))
    }

    /// Expand root node with GPU-provided policy and value.
    ///
    /// `policy` is a 1-D `numpy.ndarray` of length `BOARD_AREA` (1089) containing
    /// the raw policy logits. `value` is a scalar. `legal_bytes` is packed i32 pairs.
    fn expand_root<'py>(
        &mut self,
        policy: PyReadonlyArray1<'py, f32>,
        value: f32,
        offset_q: i32,
        offset_r: i32,
        legal_bytes: &[u8],
    ) {
        let policy_slice = policy.as_slice().expect("policy array must be contiguous");
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
            .expand_root(policy_slice, value, offset_q, offset_r, &legal);
    }

    /// Add Dirichlet noise to root priors.
    ///
    /// `noise` is a 1-D `numpy.ndarray` of the same length as root children.
    /// `noise_fraction` controls the blend (0.0 = no noise, 1.0 = pure noise).
    #[pyo3(signature = (noise, noise_fraction))]
    fn add_dirichlet_noise<'py>(
        &mut self,
        noise: PyReadonlyArray1<'py, f32>,
        noise_fraction: f32,
    ) {
        let noise_slice = noise.as_slice().expect("noise array must be contiguous");
        self.inner.add_dirichlet_noise(noise_slice, noise_fraction);
    }

    /// Whether we have done enough simulations.
    fn done(&self) -> bool {
        self.inner.done()
    }

    /// Select leaves and encode their boards.
    ///
    /// Returns `(tensor, non_terminal_count)` where `tensor` is a
    /// `numpy.ndarray` of shape `(non_terminal_count, 13, 33, 33)`.
    fn select_leaves<'py>(
        &mut self,
        py: Python<'py>,
        batch_size: u32,
    ) -> PyResult<(Bound<'py, PyArray4<f32>>, u32)> {
        let (tensors, count) = self.inner.select_leaves(batch_size);
        let tensor_vec = tensors.to_vec();
        let arr = ndarray::Array4::from_shape_vec(
            (
                count as usize,
                NUM_CHANNELS,
                BOARD_SIZE as usize,
                BOARD_SIZE as usize,
            ),
            tensor_vec,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let arr = PyArray4::from_owned_array(py, arr);
        Ok((arr, count))
    }

    /// Expand and backpropagate using GPU results.
    ///
    /// `policies` is a 1-D `numpy.ndarray` of length `N * BOARD_AREA` containing
    /// the flat policy logits for each leaf. `values` is a 1-D array of length `N`.
    fn expand_and_backprop<'py>(
        &mut self,
        policies: PyReadonlyArray1<'py, f32>,
        values: PyReadonlyArray1<'py, f32>,
    ) {
        let policies_slice = policies.as_slice().expect("policies array must be contiguous");
        let values_slice = values.as_slice().expect("values array must be contiguous");
        self.inner.expand_and_backprop(policies_slice, values_slice);
    }

    /// Get results: `(moves_q, moves_r, visits, root_value)`.
    ///
    /// Returns parallel vectors for all root children.
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
    ///
    /// Returns `(tensor, histories, count)` where:
    /// - `tensor` is a `numpy.ndarray` of shape `(count, 13, 33, 33)`
    /// - `histories` is a parallel vector of move histories as `(player, q, r)` tuples
    /// - `count` is the number of valid candidates extracted
    #[pyo3(signature = (min_visits=1))]
    fn extract_tree_node_states<'py>(
        &mut self,
        py: Python<'py>,
        min_visits: u32,
    ) -> PyResult<(Bound<'py, PyArray4<f32>>, Vec<Vec<(i32, i32, i32)>>, usize)> {
        let (packed, histories, count) = self.inner.extract_tree_node_states(min_visits);
        let arr = ndarray::Array4::from_shape_vec(
            (
                count,
                NUM_CHANNELS,
                BOARD_SIZE as usize,
                BOARD_SIZE as usize,
            ),
            packed,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let arr = PyArray4::from_owned_array(py, arr);
        let py_histories: Vec<Vec<(i32, i32, i32)>> = histories
            .into_iter()
            .map(|history| {
                history
                    .into_iter()
                    .map(|(player, q, r)| (player as i32, q as i32, r as i32))
                    .collect()
            })
            .collect();
        Ok((arr, py_histories, count))
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
    /// `add_dirichlet_noise` + search loop if it is.
    #[pyo3(signature = (q, r, new_num_simulations))]
    fn re_root(&mut self, q: i32, r: i32, new_num_simulations: u32) {
        self.inner.re_root(q as i16, r as i16, new_num_simulations);
    }
}

// -------------------------------------------------------------------------
// Bulk classical self-play (for bootstrap training data)
// -------------------------------------------------------------------------

/// Generate self-play data using classical search.
///
/// Returns a list of `(features, outcome, board_snap)` tuples from completed
/// games. This is fast because it uses the Rust alpha-beta engine.
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
            let player = game.current_player();
            let board_snap: Vec<(i32, i32, u8)> =
                game.stones().iter().map(|(&h, &p)| (h.q, h.r, p)).collect();
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
            game.place(turn.first().q, turn.first().r).unwrap_or(true);
            move_num += 1;
            if !game.is_over() {
                if let Some(m2) = turn.second() {
                    game.place(m2.q, m2.r).unwrap_or(true);
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
    m.add("WIN_LENGTH", crate::core::WIN_LENGTH)?;
    m.add("PLACEMENT_RADIUS", crate::core::PLACEMENT_RADIUS)?;
    Ok(())
}
