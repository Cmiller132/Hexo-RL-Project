//! Core game state and rules engine for Infinity Hexagonal Tic-Tac-Toe.
//!
//! # Rules
//!
//! 1. Two players: 0 and 1.
//! 2. Player 0 opens with exactly **one** placement at the origin `(0, 0)`.
//! 3. Player 1 then places **two** tiles.
//! 4. Every subsequent turn also has **two** placements.
//! 5. A placement must land on an empty hex within [`PLACEMENT_RADIUS`] (8)
//!    of any previously placed tile.
//! 6. First player to form a contiguous straight line of [`WIN_LENGTH`] (6)
//!    of their own tiles along any of the three hex axes wins.
//! 7. The board is infinite — no draw-by-exhaustion.

use rustc_hash::{FxHashMap, FxHashSet};
use crate::core::{hex_distance, Hex, HEX_DIRECTIONS};
use crate::patterns::{EvalDelta, win_grid_idx, win_grid_in_bounds, WIN_GRID_TOTAL, WIN_LENGTH, PLACEMENT_RADIUS, PATTERN_COUNTS};

// -------------------------------------------------------------------------
// Zobrist hashing (infinite board — mixing function instead of table)
// -------------------------------------------------------------------------

/// Deterministic hash for a (player, cell) pair using bit mixing.
/// XOR this into the board hash on place and unplace for incremental updates.
///
/// # Why a mixing function instead of a precomputed table?
///
/// The board is infinite — we cannot pre-allocate a Zobrist table for every
/// possible coordinate. Instead, we use a deterministic mixing function based
/// on FNV-1a with a final avalanche. This gives us a pseudo-random 64-bit
/// value for any (player, q, r) triple in O(1) time with no memory overhead.
#[inline]
pub fn zobrist_piece(player: u8, cell: Hex) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
    h ^= player as u64;
    h = h.wrapping_mul(0x100_0000_01b3); // FNV prime
    h ^= cell.q as u64;
    h = h.wrapping_mul(0x100_0000_01b3);
    h ^= cell.r as u64;
    h = h.wrapping_mul(0x100_0000_01b3);
    // Final avalanche
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    h
}

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------

/// Powers of 3 for ternary index computation.
/// Used in `unmake_move` to reverse window index updates.
const POW3: [usize; 6] = [1, 3, 9, 27, 81, 243];

// -------------------------------------------------------------------------
// Error type
// -------------------------------------------------------------------------

/// Errors that can occur when attempting a placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GameError {
    /// The game has already been won.
    GameOver,
    /// No placements remaining this turn (internal — should not happen).
    NoPlacements,
    /// The target cell is already occupied.
    CellOccupied(Hex),
    /// The first move must be at the origin.
    MustPlaceAtOrigin,
    /// The target cell is too far from all existing tiles.
    OutOfRadius(Hex),
}

impl std::fmt::Display for GameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GameError::GameOver => write!(f, "Game is already over."),
            GameError::NoPlacements => write!(f, "No placements remaining this turn."),
            GameError::CellOccupied(h) => write!(f, "Cell ({}, {}) is already occupied.", h.q, h.r),
            GameError::MustPlaceAtOrigin => {
                write!(f, "First placement must be at the origin (0, 0).")
            }
            GameError::OutOfRadius(h) => write!(
                f,
                "Cell ({}, {}) is not within {} hexes of any existing tile.",
                h.q, h.r, PLACEMENT_RADIUS
            ),
        }
    }
}

impl std::error::Error for GameError {}

// -------------------------------------------------------------------------
// Move record
// -------------------------------------------------------------------------

/// A single placement in the move history.
///
/// Snapshots the turn state **before** the move was made so that
/// `unmake_move` can restore exactly without fragile `move_count`-based
/// derivation.
#[derive(Debug, Clone, Copy)]
pub struct MoveRecord {
    /// The hex coordinate where the tile was placed.
    pub cell: Hex,
    /// The player who placed this tile (0 or 1).
    pub player: u8,
    /// The player whose turn it was BEFORE this move was made.
    pub current_player_before: u8,
    /// How many placements_remaining BEFORE this move was made.
    pub placements_remaining_before: u8,
    /// Whether there was a winner BEFORE this move was made.
    pub winner_before: Option<u8>,
}

// -------------------------------------------------------------------------
// Game state
// -------------------------------------------------------------------------

/// Complete mutable game state.
///
/// Create with [`HexGameState::new()`], then call [`place()`](HexGameState::place)
/// to advance the game. Query [`is_over()`](HexGameState::is_over) and
/// [`winner`] to check for a win.
///
/// # Example
///
/// ```
/// use hexgame::HexGameState;
///
/// let mut g = HexGameState::new();
/// g.place(0, 0).unwrap();   // Player 0 opens
/// g.place(1, 0).unwrap();   // Player 1, placement 1/2
/// g.place(0, 1).unwrap();   // Player 1, placement 2/2
/// assert_eq!(g.current_player, 0);
/// ```
#[derive(Debug, Clone)]
pub struct HexGameState {
    // ── Rule state ──
    pub board: FxHashMap<Hex, u8>,
    pub current_player: u8,
    pub placements_remaining: u8,
    pub winner: Option<u8>,
    pub winning_line: Option<Vec<Hex>>,
    pub move_count: u32,
    pub move_history: Vec<MoveRecord>,
    pub zobrist_hash: u64,

    // ── Incremental evaluation ──
    pub window_eval: i32,
    pub window_fives: [i32; 2],
    pub window_fours: [i32; 2],
    pub window_threes: [i32; 2],
    pub hot_windows: [FxHashSet<(i32, i32, u8)>; 2],
    pub window_indices: Vec<u16>,

    // ── Internal (pub(crate) so patterns.rs and threats.rs can access) ──
    pub(crate) eval_stack: Vec<EvalDelta>,
    pub(crate) candidate_rc: FxHashMap<Hex, u32>,
    pub(crate) candidate_radius: i32,
}

impl Default for HexGameState {
    fn default() -> Self {
        Self::new()
    }
}

impl HexGameState {
    /// Create a new game in the initial state (empty board, player 0's turn).
    pub fn new() -> Self {
        Self {
            board: FxHashMap::default(),
            current_player: 0,
            placements_remaining: 1,
            winner: None,
            winning_line: None,
            move_count: 0,
            move_history: Vec::new(),
            zobrist_hash: 0,
            window_eval: 0,
            window_fives: [0; 2],
            window_fours: [0; 2],
            window_threes: [0; 2],
            eval_stack: Vec::new(),
            window_indices: vec![0u16; WIN_GRID_TOTAL],
            candidate_rc: FxHashMap::default(),
            candidate_radius: 2,
            hot_windows: [FxHashSet::default(), FxHashSet::default()],
        }
    }

    /// Whether the game has ended (a winner exists).
    #[inline(always)]
    pub fn is_over(&self) -> bool {
        self.winner.is_some()
    }

    /// The opponent's most recent completed turn as an ordered list of cells.
    ///
    /// Returns one cell for Player 0's opening turn, otherwise two cells.
    pub fn opponent_last_turn_cells(&self) -> Vec<Hex> {
        let mut idx = self.move_history.len();

        while idx > 0 && self.move_history[idx - 1].player == self.current_player {
            idx -= 1;
        }

        let mut cells = Vec::with_capacity(2);
        while idx > 0 && self.move_history[idx - 1].player != self.current_player {
            cells.push(self.move_history[idx - 1].cell);
            idx -= 1;
        }
        cells.reverse();
        cells
    }

    /// Place the current player's tile at `(q, r)`.
    ///
    /// Returns `Ok(true)` when this placement ends the current turn,
    /// `Ok(false)` when the player has another placement remaining.
    /// Returns `Err(GameError)` if the move is illegal.
    pub fn place(&mut self, q: i32, r: i32) -> Result<bool, GameError> {
        let cell = Hex::new(q, r);
        self.validate_move(cell)?;

        let player = self.current_player;

        // Snapshot state BEFORE mutating.
        let record = MoveRecord {
            cell,
            player,
            current_player_before: self.current_player,
            placements_remaining_before: self.placements_remaining,
            winner_before: self.winner,
        };

        // Place tile
        self.board.insert(cell, player);
        self.zobrist_hash ^= zobrist_piece(player, cell);
        self.move_count += 1;
        self.move_history.push(record);
        self.placements_remaining -= 1;

        // Update candidate set and incremental eval.
        self.candidate_rc.remove(&cell);
        self.incr_candidate_neighbors(cell);
        let delta = self.compute_eval_delta(cell, player);
        self.push_eval_delta(delta);

        // Check win
        if let Some(line) = self.find_winning_line(cell, player) {
            self.winner = Some(player);
            self.winning_line = Some(line);
            self.placements_remaining = 0;
            return Ok(true);
        }

        // Still has placements?
        if self.placements_remaining > 0 {
            return Ok(false);
        }

        // Advance turn
        self.current_player = 1 - self.current_player;
        self.placements_remaining = 2;
        Ok(true)
    }

    /// Set the board to a custom position, bypassing normal turn rules.
    ///
    /// All pieces in `pieces` are placed directly regardless of who is "current player".
    /// The resulting `current_player` and `placements_remaining` are set explicitly.
    /// Any pre-existing game state is discarded (equivalent to `reset()` first).
    pub fn set_position(
        &mut self,
        pieces: &[(i32, i32, u8)],
        current_player: u8,
        placements_remaining: u8,
    ) -> Result<(), GameError> {
        self.reset();

        // Simulate turn progression so each synthetic MoveRecord captures the
        // correct state before its placement.
        let mut sim_player = self.current_player; // 0
        let mut sim_remaining = self.placements_remaining; // 1

        for &(q, r, player) in pieces {
            let cell = Hex::new(q, r);
            if self.board.contains_key(&cell) {
                return Err(GameError::CellOccupied(cell));
            }

            let record = MoveRecord {
                cell,
                player,
                current_player_before: sim_player,
                placements_remaining_before: sim_remaining,
                winner_before: self.winner,
            };

            self.board.insert(cell, player);
            self.zobrist_hash ^= zobrist_piece(player, cell);
            self.move_count += 1;
            self.move_history.push(record);

            // Update candidate set and incremental eval.
            self.candidate_rc.remove(&cell);
            self.incr_candidate_neighbors(cell);
            let delta = self.compute_eval_delta(cell, player);
            self.push_eval_delta(delta);

            // Detect wins during bulk placement
            if self.winner.is_none() {
                if let Some(line) = self.find_winning_line(cell, player) {
                    self.winner = Some(player);
                    self.winning_line = Some(line);
                }
            }

            // Simulate turn progression
            sim_remaining -= 1;
            if sim_remaining == 0 {
                sim_player = 1 - sim_player;
                sim_remaining = 2;
            }
        }

        self.current_player = current_player & 1;
        self.placements_remaining = placements_remaining.max(1);
        Ok(())
    }

    /// Increment candidate reference counts for empty cells within radius 2 of `cell`.
    /// Must be called **after** the piece is inserted into `self.board`.
    fn incr_candidate_neighbors(&mut self, cell: Hex) {
        let r2 = self.candidate_radius;
        for dq in -r2..=r2 {
            for dr in -r2..=r2 {
                let h = Hex::new(cell.q + dq, cell.r + dr);
                if hex_distance(cell, h) <= r2 && !self.board.contains_key(&h) {
                    *self.candidate_rc.entry(h).or_insert(0) += 1;
                }
            }
        }
    }

    /// Apply an eval delta to the running totals and push it onto the eval stack.
    fn push_eval_delta(&mut self, delta: EvalDelta) {
        self.window_eval += delta.score;
        for i in 0..2 {
            self.window_fives[i] += delta.five_delta[i];
            self.window_fours[i] += delta.four_delta[i];
            self.window_threes[i] += delta.three_delta[i];
        }
        self.eval_stack.push(delta);
    }

    /// Reset to initial empty state.
    pub fn reset(&mut self) {
        self.board.clear();
        self.current_player = 0;
        self.placements_remaining = 1;
        self.winner = None;
        self.winning_line = None;
        self.move_count = 0;
        self.move_history.clear();
        self.zobrist_hash = 0;
        self.window_eval = 0;
        self.window_fives = [0; 2];
        self.window_fours = [0; 2];
        self.window_threes = [0; 2];
        self.eval_stack.clear();
        self.window_indices.fill(0);
        self.candidate_rc.clear();
        self.hot_windows[0].clear();
        self.hot_windows[1].clear();
    }

    /// Undo the last placement. Restores board, turn state, and hash.
    ///
    /// Panics if called on an empty game (no moves to undo).
    pub fn unmake_move(&mut self) {
        // Reverse incremental eval before removing the piece.
        if let Some(delta) = self.eval_stack.pop() {
            self.window_eval -= delta.score;
            for i in 0..2 {
                self.window_fives[i] -= delta.five_delta[i];
                self.window_fours[i] -= delta.four_delta[i];
                self.window_threes[i] -= delta.three_delta[i];
            }
        }

        let rec = self.move_history.pop().expect("no move to undo");

        // Reverse window indices for the removed piece.
        let cell_val = if rec.player == 0 { 1usize } else { 2usize };
        for (dir_idx, &(dq, dr)) in HEX_DIRECTIONS.iter().enumerate() {
            for off in 0..WIN_LENGTH as usize {
                let sq = rec.cell.q - dq * off as i32;
                let sr = rec.cell.r - dr * off as i32;

                if !win_grid_in_bounds(sq, sr) {
                    continue;
                }
                let gi = win_grid_idx(sq, sr, dir_idx as u8);

                let current_idx = self.window_indices[gi] as usize;
                let old_idx = current_idx - cell_val * POW3[off];

                // Update hot windows (reverse transition)
                let (cur_p0, cur_p1) = PATTERN_COUNTS[current_idx];
                let (old_p0, old_p1) = PATTERN_COUNTS[old_idx];
                let wkey = (sq, sr, dir_idx as u8);
                let was_hot_0 = cur_p0 >= 4 && cur_p1 == 0;
                let is_hot_0 = old_p0 >= 4 && old_p1 == 0;
                if was_hot_0 && !is_hot_0 {
                    self.hot_windows[0].remove(&wkey);
                }
                if !was_hot_0 && is_hot_0 {
                    self.hot_windows[0].insert(wkey);
                }
                let was_hot_1 = cur_p1 >= 4 && cur_p0 == 0;
                let is_hot_1 = old_p1 >= 4 && old_p0 == 0;
                if was_hot_1 && !is_hot_1 {
                    self.hot_windows[1].remove(&wkey);
                }
                if !was_hot_1 && is_hot_1 {
                    self.hot_windows[1].insert(wkey);
                }

                self.window_indices[gi] = old_idx as u16;
            }
        }

        self.board.remove(&rec.cell);

        // Reverse candidate set: add this cell back, decrement neighbors.
        let r2 = self.candidate_radius;
        for dq in -r2..=r2 {
            for dr in -r2..=r2 {
                let h = Hex::new(rec.cell.q + dq, rec.cell.r + dr);
                if hex_distance(rec.cell, h) <= r2 && !self.board.contains_key(&h) {
                    if let Some(count) = self.candidate_rc.get_mut(&h) {
                        *count -= 1;
                        if *count == 0 {
                            self.candidate_rc.remove(&h);
                        }
                    }
                }
            }
        }
        // Add the removed cell back as candidate if it has neighbors
        let mut rc = 0u32;
        for dq in -r2..=r2 {
            for dr in -r2..=r2 {
                let h = Hex::new(rec.cell.q + dq, rec.cell.r + dr);
                if hex_distance(rec.cell, h) <= r2 && self.board.contains_key(&h) {
                    rc += 1;
                }
            }
        }
        if rc > 0 {
            self.candidate_rc.insert(rec.cell, rc);
        }

        self.zobrist_hash ^= zobrist_piece(rec.player, rec.cell);
        self.move_count -= 1;

        // Restore turn state from the snapshot.
        self.current_player = rec.current_player_before;
        self.placements_remaining = rec.placements_remaining_before;
        self.winner = rec.winner_before;
        self.winning_line = None;
    }

    /// Return all legal placements (exhaustive radius-8 scan — expensive).
    ///
    /// Returns an empty vec if the game is over.  On an empty board returns
    /// only `[Hex::ORIGIN]`.
    pub fn legal_moves(&self) -> Vec<Hex> {
        if self.is_over() {
            return Vec::new();
        }
        if self.board.is_empty() {
            return vec![Hex::ORIGIN];
        }

        let mut candidates = FxHashSet::default();
        for &cell in self.board.keys() {
            for dq in -PLACEMENT_RADIUS..=PLACEMENT_RADIUS {
                for dr in -PLACEMENT_RADIUS..=PLACEMENT_RADIUS {
                    let cand = Hex::new(cell.q + dq, cell.r + dr);
                    if !self.board.contains_key(&cand)
                        && hex_distance(cell, cand) <= PLACEMENT_RADIUS
                    {
                        candidates.insert(cand);
                    }
                }
            }
        }

        let mut result: Vec<Hex> = candidates.into_iter().collect();
        result.sort();
        result
    }

    /// Return legal placements within `radius` of any occupied cell.
    ///
    /// Faster than [`legal_moves()`](Self::legal_moves) for AI move
    /// generation since most interesting moves are near existing tiles.
    /// The radius is clamped to [`PLACEMENT_RADIUS`].
    pub fn legal_moves_near(&self, radius: i32) -> Vec<Hex> {
        if self.is_over() {
            return Vec::new();
        }
        if self.board.is_empty() {
            return vec![Hex::ORIGIN];
        }

        // Use incremental candidate set if radius matches (fast path).
        if radius == self.candidate_radius && !self.candidate_rc.is_empty() {
            let mut result: Vec<Hex> = self.candidate_rc.keys().copied().collect();
            result.sort();
            return result;
        }

        // Fallback: full scan for different radius.
        let r = radius.min(PLACEMENT_RADIUS);
        let mut candidates = FxHashSet::default();
        for &cell in self.board.keys() {
            for dq in -r..=r {
                for dr in -r..=r {
                    let cand = Hex::new(cell.q + dq, cell.r + dr);
                    if !self.board.contains_key(&cand) && hex_distance(cell, cand) <= r {
                        candidates.insert(cand);
                    }
                }
            }
        }

        let mut result: Vec<Hex> = candidates.into_iter().collect();
        result.sort();
        result
    }

    /// Return the incremental radius-2 candidate set as a Vec.
    /// This is the fast path for the turn-based search which always uses radius 2.
    pub fn candidates_near2(&self) -> Vec<Hex> {
        if self.is_over() {
            return Vec::new();
        }
        if self.board.is_empty() {
            return vec![Hex::ORIGIN];
        }
        let mut result = if !self.candidate_rc.is_empty() {
            self.candidate_rc.keys().copied().collect()
        } else {
            self.legal_moves_near(2)
        };
        result.sort();
        result
    }

    // -----------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------

    pub fn validate_move(&self, cell: Hex) -> Result<(), GameError> {
        if self.is_over() {
            return Err(GameError::GameOver);
        }
        if self.placements_remaining == 0 {
            return Err(GameError::NoPlacements);
        }
        if self.board.contains_key(&cell) {
            return Err(GameError::CellOccupied(cell));
        }
        if self.board.is_empty() && cell != Hex::ORIGIN {
            return Err(GameError::MustPlaceAtOrigin);
        }
        if !self.board.is_empty()
            && !self
                .board
                .keys()
                .any(|&existing| hex_distance(existing, cell) <= PLACEMENT_RADIUS)
        {
            return Err(GameError::OutOfRadius(cell));
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // Win detection
    // -----------------------------------------------------------------

    pub fn find_winning_line(&self, last: Hex, player: u8) -> Option<Vec<Hex>> {
        for &(dq, dr) in &HEX_DIRECTIONS {
            let mut backward = self.collect_run(last, -dq, -dr, player);
            backward.reverse();
            let forward = self.collect_run(last, dq, dr, player);

            let pivot = backward.len();
            let mut line = backward;
            line.push(last);
            line.extend_from_slice(&forward);

            if line.len() >= WIN_LENGTH as usize {
                return Some(Self::select_segment(&line, pivot));
            }
        }
        None
    }

    #[inline]
    fn collect_run(&self, origin: Hex, dq: i32, dr: i32, player: u8) -> Vec<Hex> {
        let mut tiles = Vec::new();
        let mut q = origin.q + dq;
        let mut r = origin.r + dr;
        while self.board.get(&Hex::new(q, r)) == Some(&player) {
            tiles.push(Hex::new(q, r));
            q += dq;
            r += dr;
        }
        tiles
    }

    fn select_segment(line: &[Hex], pivot: usize) -> Vec<Hex> {
        let wl = WIN_LENGTH as usize;
        let lo = pivot.saturating_sub(wl - 1);
        let hi = pivot.min(line.len() - wl);
        let preferred = pivot.saturating_sub((wl - 1) / 2);
        let start = hi.min(lo.max(preferred));
        line[start..start + wl].to_vec()
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Opening rules ---------------------------------------------------

    #[test]
    fn first_move_must_be_origin() {
        let mut g = HexGameState::new();
        assert!(matches!(g.place(1, 0), Err(GameError::MustPlaceAtOrigin)));
    }

    #[test]
    fn opening_turn_has_one_placement() {
        let mut g = HexGameState::new();
        let done = g.place(0, 0).unwrap();
        assert!(done);
        assert_eq!(g.current_player, 1);
        assert_eq!(g.placements_remaining, 2);
    }

    #[test]
    fn second_player_gets_two_placements() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();

        assert_eq!(g.current_player, 1);
        let done = g.place(1, 0).unwrap();
        assert!(!done); // still has one left
        assert_eq!(g.placements_remaining, 1);

        let done = g.place(0, 1).unwrap();
        assert!(done);
        assert_eq!(g.current_player, 0);
        assert_eq!(g.placements_remaining, 2);
    }

    #[test]
    fn opponent_last_turn_cells_handles_opening_turn() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();

        assert_eq!(g.opponent_last_turn_cells(), vec![Hex::new(0, 0)]);
    }

    #[test]
    fn opponent_last_turn_cells_skips_current_partial_turn() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        g.place(1, 0).unwrap();
        g.place(1, 1).unwrap();
        g.place(0, 1).unwrap();

        assert_eq!(
            g.opponent_last_turn_cells(),
            vec![Hex::new(1, 0), Hex::new(1, 1)]
        );
    }

    // -- Placement validation --------------------------------------------

    #[test]
    fn cannot_place_on_occupied_cell() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        assert!(matches!(g.place(0, 0), Err(GameError::CellOccupied(_))));
    }

    #[test]
    fn placement_radius_enforced() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        // Just inside radius is fine.
        g.place(PLACEMENT_RADIUS, 0).unwrap();
        g.place(0, 1).unwrap();
        // Far outside radius must fail.
        assert!(matches!(g.place(100, 100), Err(GameError::OutOfRadius(_))));
    }

    #[test]
    fn cannot_place_at_radius_plus_one() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        // Exactly PLACEMENT_RADIUS + 1 away along q-axis.
        assert!(matches!(
            g.place(PLACEMENT_RADIUS + 1, 0),
            Err(GameError::OutOfRadius(_))
        ));
    }

    #[test]
    fn cannot_move_after_game_over() {
        let g = build_won_game();
        let mut g = g; // need mut
        assert!(g.is_over());
        assert!(matches!(g.place(10, 10), Err(GameError::GameOver)));
    }

    // -- Win detection: all three axes -----------------------------------

    #[test]
    fn horizontal_win_axis_1_0() {
        // Player 0 builds 6-in-a-row along (1, 0): (0,0)..(5,0).
        let g = build_won_game();
        assert_eq!(g.winner, Some(0));
        let wl = g.winning_line.as_ref().unwrap();
        assert_eq!(wl.len(), WIN_LENGTH as usize);
        // Verify the winning line is along axis (1,0).
        for i in 0..wl.len() - 1 {
            assert_eq!(wl[i + 1].q - wl[i].q, 1);
            assert_eq!(wl[i + 1].r - wl[i].r, 0);
        }
    }

    #[test]
    fn diagonal_win_axis_1_neg1() {
        // Player 0 wins along (1, -1): (0,0),(1,-1),(2,-2),(3,-3),(4,-4),(5,-5).
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        // P1 scattered
        g.place(1, 0).unwrap();
        g.place(0, 1).unwrap();
        // P0
        g.place(1, -1).unwrap();
        g.place(2, -2).unwrap();
        // P1 scattered
        g.place(2, 0).unwrap();
        g.place(0, 2).unwrap();
        // P0
        g.place(3, -3).unwrap();
        g.place(4, -4).unwrap();
        // P1 scattered
        g.place(3, 0).unwrap();
        g.place(0, 3).unwrap();
        // P0 completes the line
        g.place(5, -5).unwrap();

        assert_eq!(g.winner, Some(0));
        let wl = g.winning_line.as_ref().unwrap();
        assert_eq!(wl.len(), WIN_LENGTH as usize);
    }

    #[test]
    fn vertical_win_axis_0_1() {
        // Player 0 wins along (0, 1): (0,0),(0,1),(0,2),(0,3),(0,4),(0,5).
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        // P1 scattered (avoid lines on any axis)
        g.place(1, 0).unwrap();
        g.place(-1, 0).unwrap();
        // P0
        g.place(0, 1).unwrap();
        g.place(0, 2).unwrap();
        // P1 scattered
        g.place(2, 0).unwrap();
        g.place(-2, 0).unwrap();
        // P0
        g.place(0, 3).unwrap();
        g.place(0, 4).unwrap();
        // P1 scattered
        g.place(3, 0).unwrap();
        g.place(-3, 0).unwrap();
        // P0 completes the line
        g.place(0, 5).unwrap();

        assert_eq!(g.winner, Some(0));
        let wl = g.winning_line.as_ref().unwrap();
        assert_eq!(wl.len(), WIN_LENGTH as usize);
        // Verify the line is along (0, 1).
        for i in 0..wl.len() - 1 {
            assert_eq!(wl[i + 1].q - wl[i].q, 0);
            assert_eq!(wl[i + 1].r - wl[i].r, 1);
        }
    }

    #[test]
    fn five_in_a_row_does_not_win() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        // P1
        g.place(0, -1).unwrap();
        g.place(0, -2).unwrap();
        // P0
        g.place(1, 0).unwrap();
        g.place(2, 0).unwrap();
        // P1
        g.place(0, -3).unwrap();
        g.place(0, -4).unwrap();
        // P0: 5 in a row (0,0)..(4,0) — not enough
        g.place(3, 0).unwrap();
        g.place(4, 0).unwrap();
        assert!(g.winner.is_none());
    }

    #[test]
    fn player_1_can_win() {
        // Player 1 builds 6-in-a-row along (0, 1).
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        // P1: (0,-1), (0,-2)
        g.place(0, -1).unwrap();
        g.place(0, -2).unwrap();
        // P0 scattered
        g.place(1, 0).unwrap();
        g.place(2, 0).unwrap();
        // P1: (0,-3), (0,-4)
        g.place(0, -3).unwrap();
        g.place(0, -4).unwrap();
        // P0 scattered
        g.place(3, 0).unwrap();
        g.place(4, 0).unwrap();
        // P1: (0,-5), (0,-6) — 6 in a row from (0,-1) to (0,-6)
        g.place(0, -5).unwrap();
        g.place(0, -6).unwrap();

        assert_eq!(g.winner, Some(1));
        assert!(g.is_over());
    }

    #[test]
    fn win_on_first_placement_of_turn() {
        // Set up so player 0 wins on placement 1 of 2 in their turn.
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        // P1 scattered
        g.place(0, -1).unwrap();
        g.place(1, -1).unwrap();
        // P0: (1,0), (2,0)
        g.place(1, 0).unwrap();
        g.place(2, 0).unwrap();
        // P1 scattered
        g.place(-1, 1).unwrap();
        g.place(0, -2).unwrap();
        // P0: (3,0), (4,0) — now has 5 in a row
        g.place(3, 0).unwrap();
        g.place(4, 0).unwrap();
        // P1 scattered
        g.place(-1, 0).unwrap();
        g.place(-2, 1).unwrap();
        // P0: (5,0) wins! This is placement 1 of 2.
        let done = g.place(5, 0).unwrap();
        assert!(done); // turn ends immediately on win
        assert_eq!(g.winner, Some(0));
        assert_eq!(g.placements_remaining, 0);
    }

    #[test]
    fn win_on_second_placement_of_turn() {
        // Make player 0 win on placement 2 of 2.
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        // P1 scattered
        g.place(0, -1).unwrap();
        g.place(1, -1).unwrap();
        // P0: build off-axis first, then line
        g.place(0, 7).unwrap(); // placement 1 (off-axis, not contributing to line)
        g.place(1, 0).unwrap(); // placement 2
                                // P1 scattered
        g.place(-1, 1).unwrap();
        g.place(0, -2).unwrap();
        // P0
        g.place(2, 0).unwrap();
        g.place(3, 0).unwrap();
        // P1 scattered
        g.place(-1, 0).unwrap();
        g.place(-2, 1).unwrap();
        // P0: placement 1 = (4,0), placement 2 = (5,0) wins
        g.place(4, 0).unwrap();
        let done = g.place(5, 0).unwrap();
        assert!(done);
        assert_eq!(g.winner, Some(0));
    }

    // -- Move tracking ---------------------------------------------------

    #[test]
    fn move_count_tracks_placements() {
        let mut g = HexGameState::new();
        assert_eq!(g.move_count, 0);
        g.place(0, 0).unwrap();
        assert_eq!(g.move_count, 1);
        g.place(1, 0).unwrap();
        assert_eq!(g.move_count, 2);
        g.place(0, 1).unwrap();
        assert_eq!(g.move_count, 3);
    }

    #[test]
    fn move_history_records_correctly() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        g.place(1, 0).unwrap();
        g.place(0, 1).unwrap();

        assert_eq!(g.move_history.len(), 3);
        assert_eq!(g.move_history[0].player, 0);
        assert_eq!(g.move_history[0].cell, Hex::ORIGIN);
        assert_eq!(g.move_history[1].player, 1);
        assert_eq!(g.move_history[1].cell, Hex::new(1, 0));
        assert_eq!(g.move_history[2].player, 1);
        assert_eq!(g.move_history[2].cell, Hex::new(0, 1));
    }

    // -- Legal moves -----------------------------------------------------

    #[test]
    fn initial_legal_move_is_origin_only() {
        let g = HexGameState::new();
        assert_eq!(g.legal_moves(), vec![Hex::ORIGIN]);
    }

    #[test]
    fn no_legal_moves_when_game_over() {
        let g = build_won_game();
        assert!(g.legal_moves().is_empty());
    }

    #[test]
    fn legal_moves_exclude_occupied() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        let moves = g.legal_moves();
        assert!(!moves.contains(&Hex::ORIGIN));
    }

    #[test]
    fn legal_moves_near_is_subset_of_full() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        let near: std::collections::HashSet<Hex> = g.legal_moves_near(2).into_iter().collect();
        let full: std::collections::HashSet<Hex> = g.legal_moves().into_iter().collect();
        assert!(near.is_subset(&full));
        assert!(!near.is_empty());
    }

    #[test]
    fn legal_moves_near_clamped_to_placement_radius() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        // radius=100 should behave like radius=PLACEMENT_RADIUS
        let near_huge = g.legal_moves_near(100);
        let full = g.legal_moves();
        assert_eq!(near_huge.len(), full.len());
    }

    // -- Clone / reset ---------------------------------------------------

    #[test]
    fn clone_is_independent() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        let mut c = g.clone();
        c.place(1, 0).unwrap();
        assert!(!g.board.contains_key(&Hex::new(1, 0)));
    }

    #[test]
    fn zobrist_restores_after_unmake() {
        let mut g = HexGameState::new();
        let h0 = g.zobrist_hash;

        g.place(0, 0).unwrap();
        let h1 = g.zobrist_hash;
        assert_ne!(h0, h1);

        g.unmake_move();
        assert_eq!(g.zobrist_hash, h0);
    }

    #[test]
    fn reset_clears_everything() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        g.place(1, 0).unwrap();
        g.reset();
        assert_eq!(g.move_count, 0);
        assert!(g.board.is_empty());
        assert!(g.move_history.is_empty());
        assert!(g.winner.is_none());
        assert!(g.winning_line.is_none());
        assert_eq!(g.current_player, 0);
        assert_eq!(g.placements_remaining, 1);
    }

    // -- Default trait ---------------------------------------------------

    #[test]
    fn default_equals_new() {
        let a = HexGameState::new();
        let b = HexGameState::default();
        assert_eq!(a.current_player, b.current_player);
        assert_eq!(a.placements_remaining, b.placements_remaining);
        assert_eq!(a.move_count, b.move_count);
        assert!(a.board.is_empty() && b.board.is_empty());
    }

    // -- Error display ---------------------------------------------------

    #[test]
    fn error_messages_are_descriptive() {
        assert!(GameError::GameOver.to_string().contains("over"));
        assert!(GameError::CellOccupied(Hex::ORIGIN)
            .to_string()
            .contains("occupied"));
        assert!(GameError::MustPlaceAtOrigin.to_string().contains("origin"));
        assert!(GameError::OutOfRadius(Hex::new(99, 99))
            .to_string()
            .contains("within"));
    }

    // -- Full random game smoke test -------------------------------------

    #[test]
    fn random_game_terminates() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut g = HexGameState::new();
        // Deterministic pseudo-random via hashing the step count.
        let mut step = 0u64;
        g.place(0, 0).unwrap();

        while !g.is_over() && g.move_count < 2000 {
            let moves = g.legal_moves_near(3);
            if moves.is_empty() {
                break;
            }
            let mut h = DefaultHasher::new();
            step.hash(&mut h);
            let idx = h.finish() as usize % moves.len();
            let m = moves[idx];
            g.place(m.q, m.r).unwrap();
            step += 1;
        }
        // Should have terminated with a winner within a reasonable number of moves.
        // (Random games typically end in ~100-900 moves.)
        assert!(g.is_over());
    }

    // -- set_position -----------------------------------------------------

    #[test]
    fn set_position_basic() {
        let mut g = HexGameState::new();
        g.set_position(&[(1, 0, 0), (2, 0, 0), (0, 1, 1)], 0, 2).unwrap();
        assert_eq!(g.board.len(), 3);
        assert_eq!(g.board.get(&Hex::new(1, 0)), Some(&0));
        assert_eq!(g.board.get(&Hex::new(2, 0)), Some(&0));
        assert_eq!(g.board.get(&Hex::new(0, 1)), Some(&1));
        assert_eq!(g.current_player, 0);
        assert_eq!(g.placements_remaining, 2);
    }

    #[test]
    fn set_position_detects_win() {
        let mut g = HexGameState::new();
        g.set_position(
            &[
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (3, 0, 0),
                (4, 0, 0),
                (5, 0, 0),
            ],
            0,
            2,
        )
        .unwrap();
        assert_eq!(g.winner, Some(0));
        assert!(g.is_over());
    }

    #[test]
    fn set_position_rejects_duplicate_cell() {
        let mut g = HexGameState::new();
        let res = g.set_position(&[(1, 0, 0), (1, 0, 1)], 0, 2);
        assert!(matches!(res, Err(GameError::CellOccupied(_))));
    }

    // -- candidates_near2 --------------------------------------------------

    #[test]
    fn candidates_near2_is_sorted() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        g.place(1, 0).unwrap();
        g.place(0, 1).unwrap();
        let cands = g.candidates_near2();
        assert!(!cands.is_empty());
        for i in 1..cands.len() {
            assert!(
                cands[i - 1] < cands[i],
                "candidates_near2 must be sorted: {:?} >= {:?}",
                cands[i - 1],
                cands[i]
            );
        }
    }

    // -- unmake_move eval round-trip ---------------------------------------

    #[test]
    fn unmake_restores_eval_counters() {
        let mut g = HexGameState::new();
        g.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0)],
            0,
            2,
        )
        .unwrap();

        let eval0 = g.window_eval;
        let fives0 = g.window_fives;
        let fours0 = g.window_fours;
        let threes0 = g.window_threes;
        let hot0 = g.hot_windows[0].len();

        g.place(3, 0).unwrap();
        g.unmake_move();

        assert_eq!(g.window_eval, eval0);
        assert_eq!(g.window_fives, fives0);
        assert_eq!(g.window_fours, fours0);
        assert_eq!(g.window_threes, threes0);
        assert_eq!(g.hot_windows[0].len(), hot0);
    }

    // -- Helpers ----------------------------------------------------------

    /// Build a game where player 0 wins with 6-in-a-row along (1, 0).
    /// Line: (0,0) (1,0) (2,0) (3,0) (4,0) (5,0).
    fn build_won_game() -> HexGameState {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        g.place(0, -1).unwrap();
        g.place(1, -1).unwrap();
        g.place(1, 0).unwrap();
        g.place(2, 0).unwrap();
        g.place(-1, 1).unwrap();
        g.place(0, -2).unwrap();
        g.place(3, 0).unwrap();
        g.place(4, 0).unwrap();
        g.place(-1, 0).unwrap();
        g.place(-2, 1).unwrap();
        g.place(5, 0).unwrap();
        g
    }
}
