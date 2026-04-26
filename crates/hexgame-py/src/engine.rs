use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use numpy::{ndarray, PyArray3, PyArray4, PyReadonlyArray1};

use hexgame_core::board::{GameError, HexGameState};
use hexgame_core::core::HEX_DIRECTIONS;
use hexgame_core::encoder;
use hexgame_core::search;
use hexgame_core::threats::{threat_status, ThreatStatus};
use hexgame_core::Hex;
use hexgame_core::MCTSEngine;
use hexgame_core::WIN_LENGTH;

use std::time::Duration;

use hexgame_core::encoder::{BOARD_SIZE, NUM_CHANNELS};

// -------------------------------------------------------------------------
// Python-facing wrapper for HexGameState
// -------------------------------------------------------------------------

/// Python-facing wrapper around [`HexGameState`].
///
/// Provides board manipulation, threat queries, legal-move generation, and
/// neural-network encoding for Hexo on an infinite board.
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
        if move_count == 0 {
            1
        } else {
            2
        }
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
        let game = &self.inner;
        let mut result = Vec::new();
        for key in game.eval().hot_windows(player) {
            let (wq, wr, dir) = (key.q(), key.r(), key.dir());
            let (dq, dr) = HEX_DIRECTIONS[dir as usize];
            let mut cells = Vec::with_capacity(WIN_LENGTH as usize);
            for k in 0..WIN_LENGTH {
                let cq = wq + dq * k;
                let cr = wr + dr * k;
                let h = Hex::new(cq, cr);
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
            .legal_moves_near_sorted(radius)
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
                if result.is_empty() {
                    None
                } else {
                    Some(result)
                }
            }
            ThreatStatus::MustBlock(b) => {
                let legal = self.inner.legal_moves_near(radius);
                let result: Vec<(i32, i32)> = legal
                    .into_iter()
                    .filter(|h| b.cells().contains(h))
                    .map(|h| (h.q, h.r))
                    .collect();
                if result.is_empty() {
                    None
                } else {
                    Some(result)
                }
            }
        }
    }

    /// Legal moves as packed bytes: pairs of little-endian `i32` `(q, r)`.
    ///
    /// Use `np.frombuffer(data, dtype=np.int32).reshape(-1, 2)` in Python to
    /// decode. This is much faster than returning a list of tuples when
    /// shipping large move lists across the FFI boundary.
    fn legal_moves_near_bytes<'py>(&self, py: Python<'py>, radius: i32) -> Bound<'py, PyBytes> {
        let moves = self.inner.legal_moves_near_sorted(radius);
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
            buf.extend_from_slice(&(r.player() as i32).to_le_bytes());
            buf.extend_from_slice(&r.cell().q.to_le_bytes());
            buf.extend_from_slice(&r.cell().r.to_le_bytes());
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
    #[allow(clippy::type_complexity)]
    fn encode_board_and_legal<'py>(
        &self,
        py: Python<'py>,
        near_radius: i32,
        constrain_threats: bool,
    ) -> PyResult<(Bound<'py, PyArray3<f32>>, i32, i32, Bound<'py, PyBytes>)> {
        let encoded = encoder::encode_board(&self.inner, near_radius, constrain_threats);
        let legal_moves = encoded.legal_moves().to_vec();
        let offset_q = encoded.offset_q;
        let offset_r = encoded.offset_r;
        let arr = ndarray::Array3::from_shape_vec(
            (NUM_CHANNELS, BOARD_SIZE as usize, BOARD_SIZE as usize),
            encoded.tensor,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let arr = PyArray3::from_owned_array(py, arr);
        let mut legal_buf: Vec<u8> = Vec::with_capacity(legal_moves.len() * 8);
        for h in &legal_moves {
            legal_buf.extend_from_slice(&h.q.to_le_bytes());
            legal_buf.extend_from_slice(&h.r.to_le_bytes());
        }
        Ok((arr, offset_q, offset_r, PyBytes::new(py, &legal_buf)))
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
    /// * `noise_level` — `0.0` = deterministic (default); `>0` = injects noise into root
    ///   candidate ordering for training-data variety.
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
                    false,
                    noise_level,
                )
            })
            .map_err(|e: GameError| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
            })?;

        Ok((
            result.best_move.q,
            result.best_move.r,
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
    #[allow(clippy::type_complexity)]
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
            .map_err(|e: GameError| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
            })?;

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
        let pr = placements_remaining.unwrap_or({
            if pieces.is_empty() && current_player == 0 {
                1
            } else {
                2
            }
        });
        self.inner
            .set_position(&pieces, current_player, pr)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))
    }

    /// Move history as a list of `(player, q, r)` tuples.
    fn move_history(&self) -> Vec<(u8, i32, i32)> {
        self.inner
            .move_history()
            .iter()
            .map(|r| (r.player(), r.cell().q, r.cell().r))
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
/// moves_q, moves_r, visit_counts, root_value = engine.get_results()
/// # For priors and Q-values per child:
/// priors = engine.root_child_priors()
/// q_values = engine.root_child_q_values()
/// ```
///
/// The `classical_self_play` function generates bootstrap training data
/// using the alpha-beta engine (no GPU required).

#[pyclass(name = "MCTSEngine")]
pub struct PyMCTSEngine {
    inner: MCTSEngine,
}

#[pymethods]
impl PyMCTSEngine {
    #[new]
    #[pyo3(signature = (game, num_simulations, c_puct=1.4, near_radius=8, c_puct_init=19652.0, constrain_threats=true, arena_sim_hint=None, seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        game: &PyHexGame,
        num_simulations: u32,
        c_puct: f32,
        near_radius: i32,
        c_puct_init: f32,
        constrain_threats: bool,
        arena_sim_hint: Option<u32>,
        seed: u64,
    ) -> Self {
        let hint = arena_sim_hint.unwrap_or(num_simulations);
        let engine = MCTSEngine::with_arena_sim_hint(
            game.inner.clone(),
            num_simulations,
            hint,
            c_puct,
            near_radius,
            constrain_threats,
            c_puct_init,
            seed,
        );
        Self { inner: engine }
    }

    #[allow(clippy::type_complexity)]
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

    fn expand_root<'py>(
        &mut self,
        policy: PyReadonlyArray1<'py, f32>,
        value: f32,
        offset_q: i32,
        offset_r: i32,
        legal_bytes: &[u8],
    ) -> PyResult<()> {
        let policy_slice = policy
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("policy array must be contiguous"))?;
        if !legal_bytes.len().is_multiple_of(8) {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "legal_bytes length {} is not a multiple of 8",
                legal_bytes.len()
            )));
        }
        let mut legal = Vec::with_capacity(legal_bytes.len() / 8);
        for chunk in legal_bytes.chunks_exact(8) {
            let q = i32::from_le_bytes(chunk[0..4].try_into().unwrap());
            let r = i32::from_le_bytes(chunk[4..8].try_into().unwrap());
            legal.push(Hex::new(q, r));
        }
        self.inner
            .expand_root(policy_slice, value, offset_q, offset_r, &legal);
        Ok(())
    }

    #[pyo3(signature = (noise, noise_fraction))]
    fn add_dirichlet_noise<'py>(
        &mut self,
        noise: PyReadonlyArray1<'py, f32>,
        noise_fraction: f32,
    ) -> PyResult<()> {
        let noise_slice = noise
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("noise array must be contiguous"))?;
        self.inner.add_dirichlet_noise(noise_slice, noise_fraction);
        Ok(())
    }

    fn done(&self) -> bool {
        self.inner.done()
    }

    fn select_leaves<'py>(
        &mut self,
        py: Python<'py>,
        batch_size: u32,
    ) -> PyResult<(Bound<'py, PyArray4<f32>>, u32)> {
        let (count, tensor_vec) = py.allow_threads(|| {
            let (tensors, count) = self.inner.select_leaves(batch_size);
            (count, tensors.to_vec())
        });
        let view = ndarray::ArrayView4::from_shape(
            (
                count as usize,
                NUM_CHANNELS,
                BOARD_SIZE as usize,
                BOARD_SIZE as usize,
            ),
            &tensor_vec,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let arr = PyArray4::from_array(py, &view);
        Ok((arr, count))
    }

    fn expand_and_backprop<'py>(
        &mut self,
        policies: PyReadonlyArray1<'py, f32>,
        values: PyReadonlyArray1<'py, f32>,
        py: Python<'py>,
    ) -> PyResult<()> {
        let policies_slice = policies
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("policies array must be contiguous"))?;
        let values_slice = values
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("values array must be contiguous"))?;
        let p = policies_slice.to_vec();
        let v = values_slice.to_vec();
        py.allow_threads(|| {
            self.inner.expand_and_backprop(&p, &v);
        });
        Ok(())
    }

    fn get_results(&self) -> (Vec<i32>, Vec<i32>, Vec<u32>, f32) {
        self.inner.get_results()
    }

    fn root_child_count(&self) -> u16 {
        self.inner.root_child_count()
    }

    fn root_child_priors(&self) -> Vec<f32> {
        self.inner.root_child_priors()
    }

    fn root_child_q_values(&self) -> Vec<f32> {
        self.inner.root_child_q_values()
    }

    #[pyo3(signature = (min_visits=1))]
    #[allow(clippy::type_complexity)]
    fn extract_tree_node_states<'py>(
        &mut self,
        py: Python<'py>,
        min_visits: u32,
    ) -> PyResult<(Bound<'py, PyArray4<f32>>, Vec<Vec<(i32, i32, i32)>>, usize)> {
        let (packed, histories, count) = self
            .inner
            .extract_tree_node_states(min_visits)
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)?;
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

    /// Sample a move from the root's children using temperature-based visit weighting.
    ///
    /// When `temperature == 0.0`, returns the child with the highest visit count.
    /// When `temperature > 0.0`, samples with probability proportional to
    /// `visit_count^(1/temperature)`.
    ///
    /// `rng_state` is used as XOR-shift state for deterministic sampling.
    /// Pass `None` (or omit) to use state `0`.
    #[pyo3(signature = (temperature, rng_state=None))]
    fn sample_action(&self, temperature: f32, rng_state: Option<u64>) -> (i16, i16) {
        let mut state = rng_state.unwrap_or(0);
        self.inner.sample_action(temperature, &mut state)
    }

    /// Returns true if the root Q-value is below the resign threshold.
    ///
    /// Should be called after the MCTS search completes (all sims done).
    fn should_resign(&self, threshold: f32) -> bool {
        self.inner.should_resign(threshold)
    }

    #[pyo3(signature = (q, r, new_num_simulations))]
    fn re_root(&mut self, q: i32, r: i32, new_num_simulations: u32) -> PyResult<()> {
        let q =
            i16::try_from(q).map_err(|_| PyValueError::new_err("q coordinate out of i16 range"))?;
        let r =
            i16::try_from(r).map_err(|_| PyValueError::new_err("r coordinate out of i16 range"))?;
        self.inner
            .re_root(q, r, new_num_simulations)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }
}

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
#[allow(clippy::type_complexity)]
fn classical_self_play(
    py: Python<'_>,
    num_games: u32,
    time_ms: u64,
    max_depth: i32,
    near_radius: i32,
    max_moves: u32,
) -> PyResult<Vec<(Vec<f32>, f32, Vec<(i32, i32, u8)>)>> {
    py.allow_threads(
        || -> Result<Vec<(Vec<f32>, f32, Vec<(i32, i32, u8)>)>, GameError> {
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
        },
    )
    .map_err(|e: GameError| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

// -------------------------------------------------------------------------
// Module registration
// -------------------------------------------------------------------------

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHexGame>()?;
    m.add_class::<PyMCTSEngine>()?;
    m.add("PyHexGame", m.getattr("HexGame")?)?;
    m.add("PyMCTSEngine", m.getattr("MCTSEngine")?)?;
    m.add_function(wrap_pyfunction!(classical_self_play, m)?)?;
    m.add("FEATURE_COUNT", encoder::FEATURE_COUNT)?;
    m.add("WIN_LENGTH", WIN_LENGTH)?;
    m.add("PLACEMENT_RADIUS", hexgame_core::PLACEMENT_RADIUS)?;
    m.add("BOARD_SIZE", encoder::BOARD_SIZE)?;
    m.add("NUM_CHANNELS", encoder::NUM_CHANNELS)?;
    m.add("TENSOR_SIZE", encoder::TENSOR_SIZE)?;
    Ok(())
}
