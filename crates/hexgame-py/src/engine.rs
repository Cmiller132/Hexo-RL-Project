use crate::protocol;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use numpy::{ndarray, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};

use hexgame_core::classical as search;
use hexgame_core::encoding as encoder;
use hexgame_core::mcts::MCTSEngine;
use hexgame_core::rules::{GameError, Hex, HexGameState, HEX_DIRECTIONS, WIN_LENGTH};
use hexgame_core::tactics::{tactical_status, TacticalStatus};

use std::time::Duration;

use hexgame_core::encoding::{BOARD_AREA, BOARD_SIZE, NUM_CHANNELS};

fn hex_to_tuple(h: Hex) -> (i32, i32) {
    (h.q, h.r)
}

fn hex_vec_to_py(cells: Vec<Hex>) -> Vec<(i32, i32)> {
    cells.into_iter().map(hex_to_tuple).collect()
}

fn pair_vec_to_py(pairs: Vec<(Hex, Hex)>) -> Vec<((i32, i32), (i32, i32))> {
    pairs
        .into_iter()
        .map(|(a, b)| (hex_to_tuple(a), hex_to_tuple(b)))
        .collect()
}

#[derive(Clone)]
struct RootSnapshot {
    offset_q: i32,
    offset_r: i32,
    legal: Vec<Hex>,
    root_generation: u64,
}

fn validate_root_snapshot(
    snapshot: &Option<RootSnapshot>,
    offset: Option<(i32, i32)>,
    legal: &[Hex],
    root_generation: u64,
) -> PyResult<()> {
    let Some(snapshot) = snapshot else {
        return Err(PyValueError::new_err(
            "root expansion requires legal_bytes from the most recent init_root() call",
        ));
    };
    if root_generation != snapshot.root_generation {
        return Err(PyValueError::new_err(format!(
            "root token mismatch: got {root_generation}, expected {}",
            snapshot.root_generation
        )));
    }
    if let Some((offset_q, offset_r)) = offset {
        if offset_q != snapshot.offset_q || offset_r != snapshot.offset_r {
            return Err(PyValueError::new_err(format!(
                "root tensor offset mismatch: got ({offset_q}, {offset_r}), expected ({}, {})",
                snapshot.offset_q, snapshot.offset_r
            )));
        }
    }
    if legal.len() != snapshot.legal.len() {
        return Err(PyValueError::new_err(format!(
            "root legal row count mismatch: got {}, expected {}",
            legal.len(),
            snapshot.legal.len()
        )));
    }
    for (idx, (got, expected)) in legal.iter().zip(snapshot.legal.iter()).enumerate() {
        if got != expected {
            return Err(PyValueError::new_err(format!(
                "root legal row mismatch at {idx}: got ({}, {}), expected ({}, {})",
                got.q, got.r, expected.q, expected.r
            )));
        }
    }
    Ok(())
}

fn sort_dedup_hex(cells: &mut Vec<Hex>) {
    cells.sort();
    cells.dedup();
}

fn sort_dedup_pairs(pairs: &mut Vec<(Hex, Hex)>) {
    for (a, b) in pairs.iter_mut() {
        if *b < *a {
            std::mem::swap(a, b);
        }
    }
    pairs.sort();
    pairs.dedup();
}

fn collect_hot_window_empties(
    game: &HexGameState,
    player: u8,
    legal: &std::collections::HashSet<Hex>,
    open_four: &mut Vec<Hex>,
    open_five: &mut Vec<Hex>,
) {
    for key in game.eval().hot_windows(player) {
        let (dq, dr) = HEX_DIRECTIONS[key.dir() as usize];
        let mut empties = Vec::<Hex>::with_capacity(2);
        for k in 0..WIN_LENGTH {
            let h = Hex::new(key.q() + dq * k, key.r() + dr * k);
            if !game.stones().contains_key(&h) && legal.contains(&h) {
                empties.push(h);
            }
        }
        match empties.len() {
            1 => open_five.extend(empties),
            2 => open_four.extend(empties),
            _ => {}
        }
    }
}

fn collect_all_hot_empties(
    game: &HexGameState,
    player: u8,
    legal: &std::collections::HashSet<Hex>,
    out: &mut Vec<Hex>,
) {
    let mut fours = Vec::new();
    let mut fives = Vec::new();
    collect_hot_window_empties(game, player, legal, &mut fours, &mut fives);
    out.extend(fours);
    out.extend(fives);
}

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
    /// If the game is over this resets the winner as well. Returns `False`
    /// when the history is empty.
    fn unplace(&mut self) -> PyResult<bool> {
        if self.inner.move_count() > 0 {
            self.inner
                .unplace()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(true)
        } else {
            Ok(false)
        }
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

    /// Incremental Zobrist hash of the board stones.
    ///
    /// This hash is updated in O(1) on every placement, but it does not include
    /// side-to-move or placements remaining.  Do not use it alone as a full game
    /// state transposition key.
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

    /// Engine-backed tactical oracle using the incremental hot-window index.
    ///
    /// This is intentionally bounded to hot-window empties and exact
    /// `TacticalStatus` constraints instead of rescanning every legal cell.
    #[pyo3(signature = (radius=8))]
    fn tactical_oracle<'py>(&self, py: Python<'py>, radius: i32) -> PyResult<Bound<'py, PyDict>> {
        let legal: std::collections::HashSet<Hex> = self
            .inner
            .legal_moves_near_sorted(radius)
            .into_iter()
            .collect();
        let current = self.inner.current_player();
        let opponent = 1 - current;

        let mut win_now = Vec::<Hex>::new();
        let mut forced = Vec::<Hex>::new();
        let mut cover = Vec::<Hex>::new();
        let mut cover_pairs = Vec::<(Hex, Hex)>::new();
        let mut open_four = Vec::<Hex>::new();
        let mut open_five = Vec::<Hex>::new();

        collect_hot_window_empties(&self.inner, current, &legal, &mut open_four, &mut open_five);
        let status_text = match tactical_status(&self.inner) {
            TacticalStatus::Quiet => "quiet",
            TacticalStatus::Unblockable => {
                collect_all_hot_empties(&self.inner, opponent, &legal, &mut forced);
                cover.extend_from_slice(&forced);
                "unblockable"
            }
            TacticalStatus::WinningTurns(turns) => {
                for turn in turns {
                    win_now.push(turn.first());
                    if let Some(second) = turn.second() {
                        win_now.push(second);
                    }
                }
                "winning_turn"
            }
            TacticalStatus::MustBlock(blocks) => {
                forced.extend_from_slice(blocks.cells());
                cover.extend_from_slice(blocks.cells());
                for &(a, b) in blocks.pairs() {
                    forced.push(a);
                    forced.push(b);
                    cover.push(a);
                    cover.push(b);
                    cover_pairs.push((a, b));
                }
                "must_block"
            }
        };

        sort_dedup_hex(&mut win_now);
        sort_dedup_hex(&mut forced);
        sort_dedup_hex(&mut cover);
        sort_dedup_hex(&mut open_four);
        sort_dedup_hex(&mut open_five);
        sort_dedup_pairs(&mut cover_pairs);

        let out = PyDict::new(py);
        out.set_item("status", status_text)?;
        out.set_item("current_player", current)?;
        out.set_item("placements_remaining", self.inner.placements_remaining())?;
        out.set_item("win_now_cells", hex_vec_to_py(win_now))?;
        out.set_item("forced_block_cells", hex_vec_to_py(forced))?;
        out.set_item("cover_cells", hex_vec_to_py(cover))?;
        out.set_item("cover_pairs", pair_vec_to_py(cover_pairs))?;
        out.set_item("open_four_cells", hex_vec_to_py(open_four))?;
        out.set_item("open_five_cells", hex_vec_to_py(open_five))?;
        Ok(out)
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
        match tactical_status(&self.inner) {
            TacticalStatus::Quiet | TacticalStatus::Unblockable => None,
            TacticalStatus::WinningTurns(turns) => {
                let legal = self.inner.legal_moves_near(radius);
                let mut allowed = Vec::new();
                for turn in turns {
                    allowed.push(turn.first());
                    if let Some(second) = turn.second() {
                        allowed.push(second);
                    }
                }
                let result: Vec<(i32, i32)> = legal
                    .into_iter()
                    .filter(|h| allowed.contains(h))
                    .map(|h| (h.q, h.r))
                    .collect();
                if result.is_empty() {
                    None
                } else {
                    Some(result)
                }
            }
            TacticalStatus::MustBlock(b) => {
                let legal = self.inner.legal_moves_near(radius);
                let mut allowed = b.cells().to_vec();
                for (a, c) in b.pairs() {
                    allowed.push(*a);
                    allowed.push(*c);
                }
                let result: Vec<(i32, i32)> = legal
                    .into_iter()
                    .filter(|h| allowed.contains(h))
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
        let buf = protocol::encode_legal_rows(&moves);
        PyBytes::new(py, &buf)
    }

    /// Board pieces as packed bytes: triples of little-endian `i32` `(q, r, player)`.
    ///
    /// Use `np.frombuffer(data, dtype=np.int32).reshape(-1, 3)` in Python.
    fn board_pieces_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let buf =
            protocol::encode_board_piece_rows(self.inner.stones().iter().map(|(&h, &p)| (h, p)));
        PyBytes::new(py, &buf)
    }

    /// Move history as packed bytes: triples of little-endian `i32` `(player, q, r)`.
    ///
    /// Use `np.frombuffer(data, dtype=np.int32).reshape(-1, 3)` in Python.
    fn move_history_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let buf = protocol::encode_compact_history_rows(self.inner.move_history());
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
        let legal_buf = protocol::encode_legal_rows(&legal_moves);
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

    /// Load a real chronological move history from an empty board.
    ///
    /// `history` is a list of `(q, r, player)` placements.  Each player must
    /// match the legal turn sequence at that point in the replay.  Unlike
    /// `set_position`, this validates that the history could actually occur.
    fn load_history(&mut self, history: Vec<(i32, i32, u8)>) -> PyResult<()> {
        self.inner
            .load_history(&history)
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
/// tensor, oq, or_, legal_bytes, root_token = engine.init_root()
///
/// # 2. Run GPU inference → root_policy, root_value
/// engine.expand_root(root_policy, root_value, oq, or_, legal_bytes, root_token)
/// engine.add_dirichlet_noise(noise_array, noise_fraction)
///
/// # 3. Search loop
/// while not engine.done():
///     tensor, count, batch_token = engine.select_leaves(batch_size)
///     # ... GPU inference → policies, values ...
///     engine.expand_and_backprop(policies, values, batch_token)
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
    last_non_terminal_count: u32,
    last_batch_generation: Option<u64>,
    root_snapshot: Option<RootSnapshot>,
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
        Self {
            inner: engine,
            last_non_terminal_count: 0,
            last_batch_generation: None,
            root_snapshot: None,
        }
    }

    #[allow(clippy::type_complexity)]
    fn init_root<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<
        Option<(
            Bound<'py, PyArray3<f32>>,
            i32,
            i32,
            Bound<'py, PyBytes>,
            u64,
        )>,
    > {
        let Some(init) = self
            .inner
            .init_root()
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?
        else {
            self.root_snapshot = None;
            return Ok(None);
        };
        let arr = ndarray::Array3::from_shape_vec(
            (NUM_CHANNELS, BOARD_SIZE as usize, BOARD_SIZE as usize),
            init.tensor,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let arr = PyArray3::from_owned_array(py, arr);
        let legal_buf = protocol::encode_legal_rows(&init.legal_moves);
        self.root_snapshot = Some(RootSnapshot {
            offset_q: init.offset_q,
            offset_r: init.offset_r,
            legal: init.legal_moves,
            root_generation: init.root_generation,
        });
        Ok(Some((
            arr,
            init.offset_q,
            init.offset_r,
            PyBytes::new(py, &legal_buf),
            init.root_generation,
        )))
    }

    fn expand_root<'py>(
        &mut self,
        policy: PyReadonlyArray1<'py, f32>,
        value: f32,
        offset_q: i32,
        offset_r: i32,
        legal_bytes: &[u8],
        root_generation: u64,
    ) -> PyResult<()> {
        let policy_slice = policy
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("policy array must be contiguous"))?;
        if policy_slice.len() != BOARD_AREA {
            return Err(PyValueError::new_err(format!(
                "policy length {} must equal BOARD_AREA {}",
                policy_slice.len(),
                BOARD_AREA
            )));
        }
        if policy_slice.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "policy logits contain non-finite values",
            ));
        }
        let legal = protocol::decode_legal_rows(legal_bytes)?;
        validate_root_snapshot(
            &self.root_snapshot,
            Some((offset_q, offset_r)),
            &legal,
            root_generation,
        )?;
        self.inner
            .expand_root(
                root_generation,
                policy_slice,
                value,
                offset_q,
                offset_r,
                &legal,
            )
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        self.root_snapshot = None;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_root_with_sparse_priors<'py>(
        &mut self,
        policy: PyReadonlyArray1<'py, f32>,
        value: f32,
        offset_q: i32,
        offset_r: i32,
        legal_bytes: &[u8],
        root_generation: u64,
        sparse_qr: PyReadonlyArray2<'py, i32>,
        sparse_logits: PyReadonlyArray1<'py, f32>,
        stage: u8,
        sparse_mix: f32,
    ) -> PyResult<()> {
        let policy_slice = policy
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("policy array must be contiguous"))?;
        if policy_slice.len() != BOARD_AREA {
            return Err(PyValueError::new_err(format!(
                "policy length {} must equal BOARD_AREA {}",
                policy_slice.len(),
                BOARD_AREA
            )));
        }
        if policy_slice.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "policy logits contain non-finite values",
            ));
        }
        let qr = sparse_qr.as_array();
        if qr.ndim() != 2 || qr.shape()[1] != 2 {
            return Err(PyValueError::new_err("sparse_qr must have shape (N, 2)"));
        }
        let sparse_logits_slice = sparse_logits
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("sparse_logits array must be contiguous"))?;
        if sparse_logits_slice.len() < qr.shape()[0] {
            return Err(PyValueError::new_err(format!(
                "sparse_logits length {} is smaller than sparse_qr rows {}",
                sparse_logits_slice.len(),
                qr.shape()[0]
            )));
        }
        let legal = protocol::decode_legal_rows(legal_bytes)?;
        validate_root_snapshot(
            &self.root_snapshot,
            Some((offset_q, offset_r)),
            &legal,
            root_generation,
        )?;
        let sparse_actions: Vec<(i32, i32)> = qr.outer_iter().map(|row| (row[0], row[1])).collect();
        self.inner
            .expand_root_with_sparse_priors(
                root_generation,
                policy_slice,
                value,
                offset_q,
                offset_r,
                &legal,
                &sparse_actions,
                sparse_logits_slice,
                stage,
                sparse_mix,
            )
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        self.root_snapshot = None;
        Ok(())
    }

    fn expand_root_with_global_priors<'py>(
        &mut self,
        legal_bytes: &[u8],
        root_generation: u64,
        global_qr: PyReadonlyArray2<'py, i32>,
        global_logits: PyReadonlyArray1<'py, f32>,
        value: f32,
    ) -> PyResult<()> {
        let qr = global_qr.as_array();
        if qr.ndim() != 2 || qr.shape()[1] != 2 {
            return Err(PyValueError::new_err("global_qr must have shape (N, 2)"));
        }
        let logits = global_logits
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("global_logits array must be contiguous"))?;
        if logits.len() < qr.shape()[0] {
            return Err(PyValueError::new_err(format!(
                "global_logits length {} is smaller than global_qr rows {}",
                logits.len(),
                qr.shape()[0]
            )));
        }
        let legal = protocol::decode_legal_rows(legal_bytes)?;
        validate_root_snapshot(&self.root_snapshot, None, &legal, root_generation)?;
        let global_actions: Vec<(i32, i32)> = qr.outer_iter().map(|row| (row[0], row[1])).collect();
        self.inner
            .expand_root_with_global_priors(root_generation, &legal, &global_actions, logits, value)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        self.root_snapshot = None;
        Ok(())
    }

    fn apply_root_pair_priors<'py>(
        &mut self,
        pair_qr: PyReadonlyArray2<'py, i32>,
        pair_logits: PyReadonlyArray1<'py, f32>,
        pair_mix: f32,
    ) -> PyResult<()> {
        let qr = pair_qr.as_array();
        let logits = pair_logits
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("pair_logits array must be contiguous"))?;
        if logits.len() < qr.shape()[0] {
            return Err(PyValueError::new_err(format!(
                "pair_logits length {} is smaller than pair_qr rows {}",
                logits.len(),
                qr.shape()[0]
            )));
        }
        let pair_actions = protocol::decode_pair_rows(qr, "pair_qr")?;
        self.inner
            .apply_root_pair_priors(&pair_actions, logits, pair_mix)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    fn apply_root_pair_first_priors<'py>(
        &mut self,
        action_logits: PyReadonlyArray1<'py, f32>,
        pair_mix: f32,
    ) -> PyResult<()> {
        let logits = action_logits.as_slice().map_err(|_| {
            PyErr::new::<PyValueError, _>("pair_first logits array must be contiguous")
        })?;
        self.inner
            .apply_root_pair_first_priors(logits, pair_mix)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    fn apply_root_pair_second_priors<'py>(
        &mut self,
        pair_qr: PyReadonlyArray2<'py, i32>,
        pair_logits: PyReadonlyArray1<'py, f32>,
        pair_mix: f32,
    ) -> PyResult<()> {
        let qr = pair_qr.as_array();
        let logits = pair_logits
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("pair_logits array must be contiguous"))?;
        if logits.len() < qr.shape()[0] {
            return Err(PyValueError::new_err(format!(
                "pair_logits length {} is smaller than pair_qr rows {}",
                logits.len(),
                qr.shape()[0]
            )));
        }
        let pair_actions = protocol::decode_pair_rows(qr, "pair_qr")?;
        self.inner
            .apply_root_pair_second_priors(&pair_actions, logits, pair_mix)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
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
        let child_count = self.inner.root_child_count() as usize;
        if noise_slice.len() < child_count {
            return Err(PyValueError::new_err(format!(
                "noise length {} is smaller than root child count {}",
                noise_slice.len(),
                child_count
            )));
        }
        self.inner
            .add_dirichlet_noise(noise_slice, noise_fraction)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    fn done(&self) -> bool {
        self.inner.done()
    }

    fn select_leaves<'py>(
        &mut self,
        py: Python<'py>,
        batch_size: u32,
    ) -> PyResult<(Bound<'py, PyArray4<f32>>, u32, u64)> {
        let (count, batch_generation, tensor_vec) = py
            .allow_threads(|| {
                self.inner.select_leaves(batch_size).map(|batch| {
                    (
                        batch.non_terminal_count,
                        batch.batch_generation,
                        batch.tensors.to_vec(),
                    )
                })
            })
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        self.last_non_terminal_count = count;
        self.last_batch_generation = Some(batch_generation);
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
        Ok((arr, count, batch_generation))
    }

    fn pending_leaf_metadata<'py>(
        &self,
        py: Python<'py>,
    ) -> Vec<(i32, i32, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        self.inner
            .pending_leaf_metadata()
            .into_iter()
            .map(|(oq, or_, legal, history)| {
                let legal_buf = protocol::encode_legal_rows(&legal);
                (
                    oq,
                    or_,
                    PyBytes::new(py, &legal_buf),
                    PyBytes::new(py, &history),
                )
            })
            .collect()
    }

    fn expand_and_backprop<'py>(
        &mut self,
        policies: PyReadonlyArray1<'py, f32>,
        values: PyReadonlyArray1<'py, f32>,
        batch_generation: u64,
        py: Python<'py>,
    ) -> PyResult<()> {
        let policies_slice = policies
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("policies array must be contiguous"))?;
        let values_slice = values
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("values array must be contiguous"))?;
        let expected_policies = self.last_non_terminal_count as usize * BOARD_AREA;
        if policies_slice.len() != expected_policies {
            return Err(PyValueError::new_err(format!(
                "policies length {} must equal {} ({} leaves * BOARD_AREA)",
                policies_slice.len(),
                expected_policies,
                self.last_non_terminal_count
            )));
        }
        if values_slice.len() != self.last_non_terminal_count as usize {
            return Err(PyValueError::new_err(format!(
                "values length {} must equal selected non-terminal leaf count {}",
                values_slice.len(),
                self.last_non_terminal_count
            )));
        }
        let p = policies_slice.to_vec();
        let v = values_slice.to_vec();
        if p.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err("policies contain non-finite values"));
        }
        if self.last_batch_generation != Some(batch_generation) {
            return Err(PyValueError::new_err(format!(
                "batch token mismatch: got {batch_generation}, expected {:?}",
                self.last_batch_generation
            )));
        }
        py.allow_threads(|| self.inner.expand_and_backprop(batch_generation, &p, &v))
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        self.last_non_terminal_count = 0;
        self.last_batch_generation = None;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_and_backprop_with_sparse<'py>(
        &mut self,
        policies: PyReadonlyArray1<'py, f32>,
        values: PyReadonlyArray1<'py, f32>,
        batch_generation: u64,
        sparse_qr: PyReadonlyArray3<'py, i32>,
        sparse_logits: PyReadonlyArray2<'py, f32>,
        sparse_counts: PyReadonlyArray1<'py, u16>,
        stage: u8,
        sparse_mix: f32,
        py: Python<'py>,
    ) -> PyResult<()> {
        let policies_slice = policies
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("policies array must be contiguous"))?;
        let values_slice = values
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("values array must be contiguous"))?;
        let expected_policies = self.last_non_terminal_count as usize * BOARD_AREA;
        if policies_slice.len() != expected_policies {
            return Err(PyValueError::new_err(format!(
                "policies length {} must equal {}",
                policies_slice.len(),
                expected_policies
            )));
        }
        if values_slice.len() != self.last_non_terminal_count as usize {
            return Err(PyValueError::new_err(format!(
                "values length {} must equal selected non-terminal leaf count {}",
                values_slice.len(),
                self.last_non_terminal_count
            )));
        }
        let qr = sparse_qr.as_array();
        let logits = sparse_logits.as_array();
        let counts = sparse_counts
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("sparse_counts array must be contiguous"))?;
        let n = self.last_non_terminal_count as usize;
        if qr.shape().len() != 3 || qr.shape()[0] < n || qr.shape()[2] != 2 {
            return Err(PyValueError::new_err("sparse_qr must have shape (N, K, 2)"));
        }
        if logits.shape().len() != 2 || logits.shape()[0] < n {
            return Err(PyValueError::new_err(
                "sparse_logits must have shape (N, K)",
            ));
        }
        if counts.len() < n {
            return Err(PyValueError::new_err("sparse_counts length must be >= N"));
        }
        let k = qr.shape()[1].min(logits.shape()[1]);
        let mut sparse_actions: Vec<Vec<(i32, i32)>> = Vec::with_capacity(n);
        let mut sparse_values: Vec<Vec<f32>> = Vec::with_capacity(n);
        for row in 0..n {
            let count = (counts[row] as usize).min(k);
            let mut actions = Vec::with_capacity(count);
            let mut vals = Vec::with_capacity(count);
            for col in 0..count {
                actions.push((qr[[row, col, 0]], qr[[row, col, 1]]));
                vals.push(logits[[row, col]]);
            }
            sparse_actions.push(actions);
            sparse_values.push(vals);
        }
        let p = policies_slice.to_vec();
        let v = values_slice.to_vec();
        if p.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err("policies contain non-finite values"));
        }
        if self.last_batch_generation != Some(batch_generation) {
            return Err(PyValueError::new_err(format!(
                "batch token mismatch: got {batch_generation}, expected {:?}",
                self.last_batch_generation
            )));
        }
        py.allow_threads(|| {
            self.inner.expand_and_backprop_with_sparse(
                batch_generation,
                &p,
                &v,
                &sparse_actions,
                &sparse_values,
                stage,
                sparse_mix,
            )
        })
        .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        self.last_non_terminal_count = 0;
        self.last_batch_generation = None;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_and_backprop_with_sparse_sources<'py>(
        &mut self,
        policies: PyReadonlyArray1<'py, f32>,
        values: PyReadonlyArray1<'py, f32>,
        batch_generation: u64,
        sparse_qr: PyReadonlyArray3<'py, i32>,
        sparse_logits: PyReadonlyArray2<'py, f32>,
        sparse_counts: PyReadonlyArray1<'py, u16>,
        sparse_sources: PyReadonlyArray2<'py, u8>,
        stage: u8,
        sparse_mix: f32,
        py: Python<'py>,
    ) -> PyResult<()> {
        let policies_slice = policies
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("policies array must be contiguous"))?;
        let values_slice = values
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("values array must be contiguous"))?;
        let expected_policies = self.last_non_terminal_count as usize * BOARD_AREA;
        if policies_slice.len() != expected_policies {
            return Err(PyValueError::new_err(format!(
                "policies length {} must equal {}",
                policies_slice.len(),
                expected_policies
            )));
        }
        if values_slice.len() != self.last_non_terminal_count as usize {
            return Err(PyValueError::new_err(format!(
                "values length {} must equal selected non-terminal leaf count {}",
                values_slice.len(),
                self.last_non_terminal_count
            )));
        }
        let qr = sparse_qr.as_array();
        let logits = sparse_logits.as_array();
        let sources = sparse_sources.as_array();
        let counts = sparse_counts
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("sparse_counts array must be contiguous"))?;
        let n = self.last_non_terminal_count as usize;
        if qr.shape().len() != 3 || qr.shape()[0] < n || qr.shape()[2] != 2 {
            return Err(PyValueError::new_err("sparse_qr must have shape (N, K, 2)"));
        }
        if logits.shape().len() != 2 || logits.shape()[0] < n {
            return Err(PyValueError::new_err(
                "sparse_logits must have shape (N, K)",
            ));
        }
        if sources.shape().len() != 2 || sources.shape()[0] < n {
            return Err(PyValueError::new_err(
                "sparse_sources must have shape (N, K)",
            ));
        }
        if counts.len() < n {
            return Err(PyValueError::new_err("sparse_counts length must be >= N"));
        }
        let k = qr.shape()[1].min(logits.shape()[1]).min(sources.shape()[1]);
        let mut sparse_actions: Vec<Vec<(i32, i32)>> = Vec::with_capacity(n);
        let mut sparse_values: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut sparse_source_values: Vec<Vec<u8>> = Vec::with_capacity(n);
        for row in 0..n {
            let count = (counts[row] as usize).min(k);
            let mut actions = Vec::with_capacity(count);
            let mut vals = Vec::with_capacity(count);
            let mut srcs = Vec::with_capacity(count);
            for col in 0..count {
                actions.push((qr[[row, col, 0]], qr[[row, col, 1]]));
                vals.push(logits[[row, col]]);
                srcs.push(sources[[row, col]]);
            }
            sparse_actions.push(actions);
            sparse_values.push(vals);
            sparse_source_values.push(srcs);
        }
        let p = policies_slice.to_vec();
        let v = values_slice.to_vec();
        if p.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err("policies contain non-finite values"));
        }
        if self.last_batch_generation != Some(batch_generation) {
            return Err(PyValueError::new_err(format!(
                "batch token mismatch: got {batch_generation}, expected {:?}",
                self.last_batch_generation
            )));
        }
        py.allow_threads(|| {
            self.inner.expand_and_backprop_with_sparse_sources(
                batch_generation,
                &p,
                &v,
                &sparse_actions,
                &sparse_values,
                &sparse_source_values,
                stage,
                sparse_mix,
            )
        })
        .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        self.last_non_terminal_count = 0;
        self.last_batch_generation = None;
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

    fn root_child_prior_sources(&self) -> Vec<u8> {
        self.inner.root_child_prior_sources()
    }

    fn prior_source_summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let t = self.inner.prior_source_telemetry();
        let dict = PyDict::new(py);
        dict.set_item("root_total_count", t.root_total_count)?;
        dict.set_item("root_sparse_count", t.root_sparse_count)?;
        dict.set_item("root_dense_count", t.root_dense_count)?;
        dict.set_item("root_default_count", t.root_default_count)?;
        dict.set_item("leaf_total_count", t.leaf_total_count)?;
        dict.set_item("leaf_sparse_count", t.leaf_sparse_count)?;
        dict.set_item("leaf_dense_count", t.leaf_dense_count)?;
        dict.set_item("leaf_default_count", t.leaf_default_count)?;
        dict.set_item("root_pair_count", t.root_pair_count)?;
        dict.set_item("leaf_pair_count", t.leaf_pair_count)?;
        dict.set_item("root_sparse_candidate_count", t.root_sparse_candidate_count)?;
        dict.set_item("leaf_sparse_candidate_count", t.leaf_sparse_candidate_count)?;
        dict.set_item("root_pair_candidate_count", t.root_pair_candidate_count)?;
        dict.set_item("leaf_expansion_count", t.leaf_expansion_count)?;
        Ok(dict)
    }

    fn root_child_q_values(&self) -> Vec<f32> {
        self.inner.root_child_q_values()
    }

    fn root_pair_visit_targets(&self) -> Vec<(i32, i32, i32, i32, u32)> {
        self.inner.root_pair_visit_targets()
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
                    .map(|(player, q, r)| (player as i32, q, r))
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
    fn sample_action(&mut self, temperature: f32, rng_state: Option<u64>) -> PyResult<(i32, i32)> {
        if self.inner.root_child_count() == 0 {
            return Err(PyValueError::new_err(
                "sample_action requires an expanded root with at least one child",
            ));
        }
        let mut state = rng_state.unwrap_or(0);
        self.inner
            .sample_action(temperature, &mut state)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
    }

    /// Returns true if the root Q-value is below the resign threshold.
    ///
    /// Should be called after the MCTS search completes (all sims done).
    fn should_resign(&self, threshold: f32) -> bool {
        self.inner.should_resign(threshold)
    }

    #[pyo3(signature = (q, r, new_num_simulations))]
    fn re_root(&mut self, q: i32, r: i32, new_num_simulations: u32) -> PyResult<()> {
        self.inner
            .re_root(q, r, new_num_simulations)
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        self.root_snapshot = None;
        self.last_non_terminal_count = 0;
        self.last_batch_generation = None;
        Ok(())
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
    m.add("PLACEMENT_RADIUS", hexgame_core::rules::PLACEMENT_RADIUS)?;
    m.add("BOARD_SIZE", encoder::BOARD_SIZE)?;
    m.add("NUM_CHANNELS", encoder::NUM_CHANNELS)?;
    m.add("TENSOR_SIZE", encoder::TENSOR_SIZE)?;
    Ok(())
}
