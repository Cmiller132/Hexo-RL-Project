//! Game state and rules engine for Infinity Hexagonal Tic-Tac-Toe.
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
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;

use crate::core::{hex_distance, Hex, HEX_DIRECTIONS};

// -------------------------------------------------------------------------
// Zobrist hashing (infinite board — mixing function instead of table)
// -------------------------------------------------------------------------

/// Deterministic hash for a (player, cell) pair using bit mixing.
/// XOR this into the board hash on place and unplace for incremental updates.
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

/// Number of tiles in a row required to win.
pub const WIN_LENGTH: i32 = 6;

/// Maximum hex distance from any existing tile for a valid placement.
pub const PLACEMENT_RADIUS: i32 = 8;

// -------------------------------------------------------------------------
// Window evaluation weights (for incremental eval)
// -------------------------------------------------------------------------

/// Powers of 3 for ternary index computation.
const POW3: [usize; 6] = [1, 3, 9, 27, 81, 243];

/// Flat-array grid for window indices (replaces HashMap for speed).
/// Window origins can be up to 5 away from placed pieces (which are within ~20 of origin).
const WIN_GRID_RADIUS: i32 = 30;
const WIN_GRID_SIDE: usize = (2 * WIN_GRID_RADIUS + 1) as usize; // 61
const WIN_GRID_TOTAL: usize = WIN_GRID_SIDE * WIN_GRID_SIDE * 3; // 11163

/// Convert (q, r, dir) to flat array index. Caller must ensure in-bounds.
#[inline(always)]
pub fn win_grid_idx(q: i32, r: i32, dir: u8) -> usize {
    ((q + WIN_GRID_RADIUS) as usize) * WIN_GRID_SIDE * 3
        + ((r + WIN_GRID_RADIUS) as usize) * 3
        + dir as usize
}

/// Check if (q, r) is within the grid bounds.
#[inline(always)]
pub fn win_grid_in_bounds(q: i32, r: i32) -> bool {
    let qi = q + WIN_GRID_RADIUS;
    let ri = r + WIN_GRID_RADIUS;
    qi >= 0 && (qi as usize) < WIN_GRID_SIDE && ri >= 0 && (ri as usize) < WIN_GRID_SIDE
}

/// Precomputed (p0_count, p1_count) for each ternary pattern index (0..728).
/// p0 = count of digits with value 1, p1 = count of digits with value 2.
const fn build_pattern_counts() -> [(u8, u8); 729] {
    let mut table = [(0u8, 0u8); 729];
    let mut i = 0usize;
    while i < 729 {
        let mut idx = i;
        let mut p0 = 0u8;
        let mut p1 = 0u8;
        let mut j = 0;
        while j < 6 {
            let digit = idx % 3;
            if digit == 1 {
                p0 += 1;
            } else if digit == 2 {
                p1 += 1;
            }
            idx /= 3;
            j += 1;
        }
        table[i] = (p0, p1);
        i += 1;
    }
    table
}

pub static PATTERN_COUNTS: [(u8, u8); 729] = build_pattern_counts();

/// Public accessor for pattern counts.
#[inline(always)]
pub fn pattern_counts(idx: usize) -> (u8, u8) {
    PATTERN_COUNTS[idx]
}

/// CMA-ES optimized pattern values for 6-cell windows (ternary encoding).
const PATTERN_VALUES: [i32; 729] = [
    0, -26, 26, -349, -323, -37, 349, 37, 323, -423, -254, -104, -307, 209, -447, -60, -483, 252,
    423, 104, 254, 60, -252, 483, 307, 447, -209, -423, -334, -206, -384, 57, -342, -93, -365, 351,
    -402, 571, -441, 270, 2901, -649, -292, -624, -99, 0, -328, 359, -405, -620, -171, 330, -136,
    469, 423, 206, 334, 93, -351, 365, 384, 342, -57, 0, -359, 328, -330, -469, 136, 405, 171, 620,
    402, 441, -571, 292, 99, 624, -270, 649, -2901, -349, -286, -131, -329, -321, -291, 0, -264,
    591, -384, -412, -599, 182, 1951, -444, -217, -597, 168, 93, -277, 183, -393, -692, 4, 254,
    126, 780, -307, -221, -149, 182, 1702, -203, -254, -695, -126, 270, 1818, -744, 2280, 49414,
    -739, -457, -869, -83, -330, -730, 174, -539, -927, -89, 0, -278, 410, 60, -445, 359, -393,
    -412, -180, 217, 222, 576, -405, -861, 60, -539, -999, -298, 0, -287, 204, 292, -119, 711, -37,
    -465, 243, 457, 474, 1013, 349, 131, 286, 0, -591, 264, 329, 291, 321, -93, -183, 277, -254,
    -780, -126, 393, -4, 692, 384, 599, 412, 217, -168, 597, -182, 444, -1951, -60, -359, 445,
    -217, -576, -222, 393, 180, 412, -292, -711, 119, -457, -1013, -474, 37, -243, 465, 405, -60,
    861, 0, -204, 287, 539, 298, 999, 307, 149, 221, 254, 126, 695, -182, 203, -1702, 330, -174,
    730, 0, -410, 278, 539, 89, 927, -270, 744, -1818, 457, 83, 869, -2280, 739, -49414, -26, -428,
    0, -286, -125, -448, 131, -234, 104, -334, -489, -458, -221, 2688, -232, -359, -474, 69, 206,
    -453, -79, -445, -327, 228, 149, 339, 384, -254, -489, 79, -412, 2510, -280, -183, -1171, -132,
    571, 2308, -505, 1818, 48109, -210, -711, -583, -74, -359, -658, 0, -861, -812, -224, -174,
    -555, 608, 104, -453, 458, -277, -948, -457, 599, 21, 866, -328, -658, 0, -730, -865, -398,
    -60, -318, 404, 441, 67, 505, -119, -300, 344, 744, 548, 748, -323, -125, -104, -321, 3001,
    -436, -591, -278, 0, 57, 2510, -866, 1702, 49588, -964, -576, -992, -430, -351, -948, 132,
    -412, -1270, -188, 126, -337, 212, 209, 2688, -384, 1951, 49588, -1837, -780, -1042, -212,
    2901, 48109, -748, 49414, 0, -1010, -1013, -1181, -742, -469, -865, -608, -999, -952, -642,
    -410, -568, 0, -252, -327, -69, -692, -1270, -28, -168, -136, 430, -620, -812, -404, -927,
    -952, -761, -204, -258, 0, 99, -300, 74, -465, -131, -123, 83, 156, 742, 37, -234, 448, -264,
    -278, 0, 291, 18, 436, -365, -1171, 457, -695, -1042, -135, 180, -209, 28, 342, 21, 280, 222,
    -136, 312, 203, -41, 1837, -483, -474, -228, -597, -992, -312, -4, -209, 188, -624, -583, -344,
    -869, -1181, -548, -243, -332, 123, 171, -318, 224, -287, -258, 0, 89, -267, 642, 447, 339,
    232, 126, -337, 135, 444, -41, 964, -136, -555, 398, -278, -568, 0, 298, -267, 761, 649, 548,
    211, 474, 156, 548, 739, 598, 1010, 26, 0, 428, -131, -104, 234, 286, 448, 125, -206, 79, 453,
    -149, -384, -339, 445, -228, 327, 334, 458, 489, 359, -69, 474, 221, 232, -2688, -104, -458,
    453, -599, -866, -21, 277, 457, 948, -441, -505, -67, -744, -748, -548, 119, -344, 300, 328, 0,
    658, 60, -404, 318, 730, 398, 865, 254, -79, 489, 183, 132, 1171, 412, 280, -2510, 359, 0, 658,
    174, -608, 555, 861, 224, 812, -571, 505, -2308, 711, 74, 583, -1818, 211, -48109, -37, -448,
    234, -291, -436, -18, 264, 0, 278, -342, -280, -21, -203, -1837, 41, -222, -312, 136, 365,
    -457, 1171, -180, -28, 209, 695, 135, 1042, -447, -232, -339, -444, -964, 41, -126, -135, 337,
    -649, -211, -548, -739, -1010, -598, -474, -548, -156, 136, -398, 555, -298, -761, 267, 278, 0,
    568, 483, 228, 474, 4, -188, 209, 597, 312, 992, -171, -224, 318, -89, -642, 267, 287, 0, 258,
    624, 344, 583, 243, -123, 332, 869, 548, 1181, 323, 104, 125, 591, 0, 278, 321, 436, -3001,
    351, -132, 948, -126, -212, 337, 412, 188, 1270, -57, 866, -2510, 576, 430, 992, -1702, 964,
    -49588, 252, 69, 327, 168, -430, 136, 692, 28, 1270, -99, -74, 300, -83, -742, -156, 465, 123,
    131, 620, 404, 812, 204, 0, 258, 927, 761, 952, -209, 384, -2688, 780, 212, 1042, -1951, 1837,
    -49588, 469, 608, 865, 410, 0, 568, 999, 642, 952, -2901, 748, -48109, 1013, 742, 1181, -49414,
    1010, 0,
];

/// Saved per-move incremental eval delta for unmake.
#[derive(Debug, Clone, Copy)]
struct EvalDelta {
    score: i32,
    five_delta: [i32; 2],
    four_delta: [i32; 2],
    three_delta: [i32; 2],
}

// -------------------------------------------------------------------------
// Error type
// -------------------------------------------------------------------------

/// Errors that can occur when attempting a placement.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone, Copy)]
pub struct MoveRecord {
    /// The player who placed this tile (0 or 1).
    pub player: u8,
    /// The hex coordinate where the tile was placed.
    pub cell: Hex,
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
    /// Board: maps hex coordinate to player index (0 or 1).
    pub board: FxHashMap<Hex, u8>,
    /// Whose turn it is (0 or 1).
    pub current_player: u8,
    /// How many placements remain in the current turn (1 or 2).
    pub placements_remaining: u8,
    /// The winning player, if any.
    pub winner: Option<u8>,
    /// The coordinates forming the winning line, if any.
    pub winning_line: Option<Vec<Hex>>,
    /// Total number of individual tile placements so far.
    pub move_count: u32,
    /// Chronological record of every placement.
    pub move_history: Vec<MoveRecord>,
    /// Incremental Zobrist hash of the board position.
    pub zobrist_hash: u64,
    /// Running window-based eval score from player 0's perspective.
    pub window_eval: i32,
    /// Number of windows with 5+ pieces per player [p0, p1] (near-win threats).
    pub window_fives: [i32; 2],
    /// Number of windows with exactly 4 pieces per player [p0, p1].
    pub window_fours: [i32; 2],
    /// Number of windows with exactly 3 pieces per player [p0, p1].
    pub window_threes: [i32; 2],
    /// Stack of eval deltas for unmake_move.
    eval_stack: Vec<EvalDelta>,
    /// Stored ternary pattern index per window: flat array indexed by (q+30, r+30, dir).
    pub window_indices: Vec<u16>,
    /// Incremental candidate set: empty cells near existing pieces.
    /// Key = hex cell, Value = reference count (how many placed pieces are within near_radius).
    candidate_rc: FxHashMap<Hex, u32>,
    /// The near radius for candidate tracking.
    candidate_radius: i32,
    /// Hot windows: (wq, wr, dir) with 4+ of a player's stones and 0 opponent.
    /// Updated incrementally during place/unmake for O(hot_set) threat detection.
    pub hot_windows: [FxHashSet<(i32, i32, u8)>; 2],
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

        // Place tile
        self.board.insert(cell, player);
        self.zobrist_hash ^= zobrist_piece(player, cell);
        self.move_count += 1;
        self.move_history.push(MoveRecord { player, cell });
        self.placements_remaining -= 1;

        // Update candidate set: remove this cell, add neighbors.
        self.candidate_rc.remove(&cell);
        let r2 = self.candidate_radius;
        for dq in -r2..=r2 {
            for dr in -r2..=r2 {
                let h = Hex::new(cell.q + dq, cell.r + dr);
                if hex_distance(cell, h) <= r2 && !self.board.contains_key(&h) {
                    *self.candidate_rc.entry(h).or_insert(0) += 1;
                }
            }
        }

        // Incremental eval update (piece is now in board).
        let delta = self.compute_eval_delta(cell, player);
        self.window_eval += delta.score;
        for i in 0..2 {
            self.window_fives[i] += delta.five_delta[i];
            self.window_fours[i] += delta.four_delta[i];
            self.window_threes[i] += delta.three_delta[i];
        }
        self.eval_stack.push(delta);

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
        for &(q, r, player) in pieces {
            let cell = Hex::new(q, r);
            if self.board.contains_key(&cell) {
                return Err(GameError::CellOccupied(cell));
            }
            self.board.insert(cell, player);
            self.zobrist_hash ^= zobrist_piece(player, cell);
            self.move_count += 1;
            self.move_history.push(MoveRecord { player, cell });

            // Update candidate set: remove this cell, add neighbors.
            self.candidate_rc.remove(&cell);
            let r2 = self.candidate_radius;
            for dq in -r2..=r2 {
                for dr in -r2..=r2 {
                    let h = Hex::new(cell.q + dq, cell.r + dr);
                    if hex_distance(cell, h) <= r2 && !self.board.contains_key(&h) {
                        *self.candidate_rc.entry(h).or_insert(0) += 1;
                    }
                }
            }

            // Incremental eval update; also updates hot_windows and window_indices.
            let delta = self.compute_eval_delta(cell, player);
            self.window_eval += delta.score;
            for i in 0..2 {
                self.window_fives[i] += delta.five_delta[i];
                self.window_fours[i] += delta.four_delta[i];
                self.window_threes[i] += delta.three_delta[i];
            }
            self.eval_stack.push(delta);
        }
        self.current_player = current_player & 1;
        self.placements_remaining = placements_remaining.max(1);
        Ok(())
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

    /// Compute the incremental eval change from placing `player`'s piece at `cell`.
    /// Called AFTER the piece is already inserted into `self.board`.
    /// Uses stored window indices for O(18) lookups instead of O(108).
    fn compute_eval_delta(&mut self, cell: Hex, player: u8) -> EvalDelta {
        let cell_val = if player == 0 { 1usize } else { 2usize };
        let mut score = 0i32;
        let mut five_delta = [0i32; 2];
        let mut four_delta = [0i32; 2];
        let mut three_delta = [0i32; 2];

        for (dir_idx, &(dq, dr)) in HEX_DIRECTIONS.iter().enumerate() {
            for off in 0..WIN_LENGTH as usize {
                let sq = cell.q - dq * off as i32;
                let sr = cell.r - dr * off as i32;

                if !win_grid_in_bounds(sq, sr) {
                    continue;
                }
                let gi = win_grid_idx(sq, sr, dir_idx as u8);

                let old_idx = self.window_indices[gi] as usize;
                let new_idx = old_idx + cell_val * POW3[off];

                // Score delta from pattern table.
                score += PATTERN_VALUES[new_idx] - PATTERN_VALUES[old_idx];

                // Threat count deltas from precomputed count table.
                let (old_p0, old_p1) = PATTERN_COUNTS[old_idx];
                let (new_p0, new_p1) = PATTERN_COUNTS[new_idx];

                #[inline]
                fn classify(
                    own: u8,
                    other: u8,
                    fives: &mut i32,
                    fours: &mut i32,
                    threes: &mut i32,
                    sign: i32,
                ) {
                    if other == 0 {
                        match own {
                            5..=6 => *fives += sign,
                            4 => *fours += sign,
                            3 => *threes += sign,
                            _ => {}
                        }
                    }
                }
                classify(
                    old_p0,
                    old_p1,
                    &mut five_delta[0],
                    &mut four_delta[0],
                    &mut three_delta[0],
                    -1,
                );
                classify(
                    old_p1,
                    old_p0,
                    &mut five_delta[1],
                    &mut four_delta[1],
                    &mut three_delta[1],
                    -1,
                );
                classify(
                    new_p0,
                    new_p1,
                    &mut five_delta[0],
                    &mut four_delta[0],
                    &mut three_delta[0],
                    1,
                );
                classify(
                    new_p1,
                    new_p0,
                    &mut five_delta[1],
                    &mut four_delta[1],
                    &mut three_delta[1],
                    1,
                );

                // Update hot windows (transition tracking)
                let wkey = (sq, sr, dir_idx as u8);
                let was_hot_0 = old_p0 >= 4 && old_p1 == 0;
                let is_hot_0 = new_p0 >= 4 && new_p1 == 0;
                if was_hot_0 && !is_hot_0 {
                    self.hot_windows[0].remove(&wkey);
                }
                if !was_hot_0 && is_hot_0 {
                    self.hot_windows[0].insert(wkey);
                }
                let was_hot_1 = old_p1 >= 4 && old_p0 == 0;
                let is_hot_1 = new_p1 >= 4 && new_p0 == 0;
                if was_hot_1 && !is_hot_1 {
                    self.hot_windows[1].remove(&wkey);
                }
                if !was_hot_1 && is_hot_1 {
                    self.hot_windows[1].insert(wkey);
                }

                // Update stored index.
                self.window_indices[gi] = new_idx as u16;
            }
        }

        EvalDelta {
            score,
            five_delta,
            four_delta,
            three_delta,
        }
    }

    /// Compute the eval delta for a hypothetical placement without modifying state.
    /// Returns the score change from player 0's perspective.
    /// Used for move ordering — much faster than full score_move.
    #[inline]
    pub fn move_eval_delta(&self, cell: Hex, player: u8) -> i32 {
        let cell_val = if player == 0 { 1usize } else { 2usize };
        let mut delta = 0i32;
        for (dir_idx, &(dq, dr)) in HEX_DIRECTIONS.iter().enumerate() {
            for off in 0..WIN_LENGTH as usize {
                let sq = cell.q - dq * off as i32;
                let sr = cell.r - dr * off as i32;
                if !win_grid_in_bounds(sq, sr) {
                    continue;
                }
                let gi = win_grid_idx(sq, sr, dir_idx as u8);
                let old_idx = self.window_indices[gi] as usize;
                let new_idx = old_idx + cell_val * POW3[off];
                delta += PATTERN_VALUES[new_idx] - PATTERN_VALUES[old_idx];
            }
        }
        delta
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
        self.winner = None;
        self.winning_line = None;

        // Derive turn state from move_count.
        // move_count 0 → player 0, remaining 1 (opening).
        // move_count 1 → player 1, remaining 2.
        // move_count >= 2: turns of 2 placements each, starting with player 1.
        //   offset = move_count - 1  (subtract the 1-placement opening)
        //   turn_index = offset / 2  (which 2-placement turn we're in)
        //   placement_in_turn = offset % 2
        //   player = (turn_index % 2 == 0) → 1, else → 0  (player 1 goes first after opening)
        if self.move_count == 0 {
            self.current_player = 0;
            self.placements_remaining = 1;
        } else if self.move_count == 1 {
            self.current_player = 1;
            self.placements_remaining = 2;
        } else {
            let offset = (self.move_count - 1) as usize;
            let turn_index = offset / 2;
            let placement_in_turn = offset % 2;
            self.current_player = if turn_index % 2 == 0 { 1 } else { 0 };
            self.placements_remaining = (2 - placement_in_turn) as u8;
        }
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

    /// Collect empty cells from a player's hot windows.
    ///
    /// Each returned inner vec corresponds to one hot window and contains the
    /// empty cells that would need to be filled or blocked in that window.
    pub fn collect_threat_window_empties(&self, player: u8) -> Vec<Vec<Hex>> {
        let p = player as usize;
        if self.window_fours[p] == 0 && self.window_fives[p] == 0 {
            return Vec::new();
        }

        let mut result = Vec::new();
        for &(wq, wr, dir) in &self.hot_windows[p] {
            let (dq, dr) = HEX_DIRECTIONS[dir as usize];
            let mut empties = Vec::new();
            for k in 0..WIN_LENGTH {
                let h = Hex::new(wq + dq * k, wr + dr * k);
                if !self.board.contains_key(&h) {
                    empties.push(h);
                }
            }
            if !empties.is_empty() && empties.len() <= 2 {
                result.push(empties);
            }
        }
        result
    }

    /// Collect the union of empty cells from a player's hot windows.
    pub fn collect_threat_cells(&self, player: u8) -> Vec<Hex> {
        let mut cells: Vec<Hex> = self
            .collect_threat_window_empties(player)
            .into_iter()
            .flatten()
            .collect();
        cells.sort();
        cells.dedup();
        cells
    }

    fn collect_winning_threat_cells(&self, player: u8, available_placements: u8) -> Vec<Hex> {
        let p = player as usize;
        if self.window_fours[p] == 0 && self.window_fives[p] == 0 {
            return Vec::new();
        }

        let mut cells = Vec::new();
        for empties in self.collect_threat_window_empties(player) {
            match empties.len() {
                1 if available_placements >= 1 => cells.extend(empties),
                2 if available_placements >= 2 => cells.extend(empties),
                _ => {}
            }
        }
        cells.sort();
        cells.dedup();
        cells
    }

    fn collect_blocking_threat_cells(&self, player: u8, available_placements: u8) -> Vec<Hex> {
        let must_hit = self.collect_threat_window_empties(player);
        if must_hit.is_empty() {
            return Vec::new();
        }

        // Deduplicated union of all block-candidate cells.
        let mut all_block_cells: Vec<Hex> =
            must_hit.iter().flat_map(|s| s.iter().copied()).collect();
        all_block_cells.sort();
        all_block_cells.dedup();

        if available_placements <= 1 {
            // Must find a single cell that hits every threat window.
            let cells: Vec<Hex> = all_block_cells
                .into_iter()
                .filter(|cell| must_hit.iter().all(|set| set.contains(cell)))
                .collect();
            return cells;
        }

        // With 2 placements: only return cells that participate in at least
        // one covering pair (a pair that hits every threat window).
        let mut valid = FxHashSet::default();
        for i in 0..all_block_cells.len() {
            for j in i..all_block_cells.len() {
                let c1 = all_block_cells[i];
                let c2 = all_block_cells[j];
                let covers_all = must_hit
                    .iter()
                    .all(|set| set.contains(&c1) || set.contains(&c2));
                if covers_all {
                    valid.insert(c1);
                    valid.insert(c2);
                }
            }
        }
        let mut cells: Vec<Hex> = valid.into_iter().collect();
        cells.sort();
        cells
    }

    /// Returns whether `player` has hot windows that cannot all be covered by
    /// the given number of available placements.
    pub fn is_player_win_unblockable(&self, player: u8, available_placements: u8) -> bool {
        let must_hit = self.collect_threat_window_empties(player);
        if must_hit.is_empty() {
            return false;
        }

        let mut all_block_cells: Vec<Hex> = must_hit.iter().flat_map(|s| s.iter().copied()).collect();
        all_block_cells.sort();
        all_block_cells.dedup();

        match available_placements {
            0 => true,
            1 => !all_block_cells
                .iter()
                .copied()
                .any(|cell| must_hit.iter().all(|set| set.contains(&cell))),
            _ => {
                for i in 0..all_block_cells.len() {
                    for j in i..all_block_cells.len() {
                        let c1 = all_block_cells[i];
                        let c2 = all_block_cells[j];
                        let covers_all = must_hit
                            .iter()
                            .all(|set| set.contains(&c1) || set.contains(&c2));
                        if covers_all {
                            return false;
                        }
                    }
                }
                true
            }
        }
    }

    /// Returns whether the opponent's hot windows cannot all be blocked with
    /// the current player's remaining placements.
    pub fn is_opponent_win_unblockable(&self, available_placements: u8) -> bool {
        self.is_player_win_unblockable(1 - self.current_player, available_placements)
    }

    /// Returns a threat-filtered subset of `legal_moves`, or `None` when no
    /// hard constraint applies.
    pub fn compute_threat_constrained_moves(
        &self,
        legal_moves: &[Hex],
        constrain_threats: bool,
    ) -> Option<Vec<Hex>> {
        if !constrain_threats || legal_moves.is_empty() {
            return None;
        }

        let current = self.current_player;
        let current_idx = current as usize;
        let opp = 1 - current;
        let opp_idx = opp as usize;
        let available_placements = self.placements_remaining;

        if self.window_fives[current_idx] > 0 || self.window_fours[current_idx] > 0 {
            let winning_cells = self.collect_winning_threat_cells(current, available_placements);
            if !winning_cells.is_empty() {
                let winning_set: FxHashSet<Hex> = winning_cells.into_iter().collect();
                let filtered: Vec<Hex> = legal_moves
                    .iter()
                    .copied()
                    .filter(|h| winning_set.contains(h))
                    .collect();
                if !filtered.is_empty() {
                    return Some(filtered);
                }
            }
        }

        if self.window_fives[opp_idx] > 0 || self.window_fours[opp_idx] > 0 {
            if self.is_opponent_win_unblockable(available_placements) {
                return None;
            }

            let blocking_cells = self.collect_blocking_threat_cells(opp, available_placements);
            if !blocking_cells.is_empty() {
                let blocking_set: FxHashSet<Hex> = blocking_cells.into_iter().collect();
                let filtered: Vec<Hex> = legal_moves
                    .iter()
                    .copied()
                    .filter(|h| blocking_set.contains(h))
                    .collect();
                if !filtered.is_empty() {
                    return Some(filtered);
                }
            }
        }

        None
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
        if !self.candidate_rc.is_empty() {
            return self.candidate_rc.keys().copied().collect();
        }
        // Fallback: generate from scratch
        self.legal_moves_near(2)
    }

    // -----------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------

    fn validate_move(&self, cell: Hex) -> Result<(), GameError> {
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

    fn find_winning_line(&self, last: Hex, player: u8) -> Option<Vec<Hex>> {
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

    /// Compute per-cell, per-axis influence scores for the board encoding tensor.
    ///
    /// For each cell within the 33×33 window and each of the 3 hex axes,
    /// sums nonlinear URGENCY weights over all 6-cell windows along that axis
    /// containing the cell.  URGENCY maps stone count → threat value:
    /// [0, 0.02, 0.06, 0.15, 0.45, 1.0, 1.0].  Per window: URGENCY[own] − URGENCY[opp].
    ///
    /// Writes into `out` which has layout [dir][gi * BOARD_SIZE + gj] (3 × 1089).
    /// Only unblocked (pure) windows contribute: a window containing both players'
    /// pieces is "contested" and scores zero.  The sum over overlapping windows
    /// produces values that strongly emphasize 4-5 stone threats.
    pub fn compute_axis_influence(
        &self,
        offset_q: i32,
        offset_r: i32,
        board_size: i32,
        perspective: u8,
        out: &mut [f32],
    ) {
        // Nonlinear urgency weights: maps stone count in a pure window
        // to threat urgency.  Exponential-ish so 4-5 stones dominate.
        // Index = number of stones (0..=6).
        // Positive for own stones, negative for opponent stones.
        const URGENCY: [f32; 7] = [
            0.00, // 0 stones — empty window
            0.02, // 1 stone  — trivial
            0.06, // 2 stones — minor
            0.15, // 3 stones — developing threat
            0.45, // 4 stones — hot window (matches ch10/11)
            1.00, // 5 stones — one move from win, MUST respond
            1.00, // 6 stones — already won (shouldn't happen)
        ];

        // out has length 3 * board_size * board_size
        let area = (board_size * board_size) as usize;
        debug_assert!(out.len() >= 3 * area);
        out.iter_mut().for_each(|v| *v = 0.0);

        for dir in 0u8..3 {
            let (dq, dr) = HEX_DIRECTIONS[dir as usize];
            for gi in 0..board_size {
                for gj in 0..board_size {
                    let cq = gi + offset_q;
                    let cr = gj + offset_r;

                    let mut total = 0.0f32;
                    for off in 0..WIN_LENGTH {
                        let wq = cq - dq * off;
                        let wr = cr - dr * off;

                        if !win_grid_in_bounds(wq, wr) {
                            continue;
                        }
                        let idx = self.window_indices[win_grid_idx(wq, wr, dir)] as usize;
                        if idx == 0 {
                            continue;
                        }
                        let (p0, p1) = PATTERN_COUNTS[idx];
                        let (own, opp) = if perspective == 0 {
                            (p0 as f32, p1 as f32)
                        } else {
                            (p1 as f32, p0 as f32)
                        };
                        // Skip contested windows
                        if own > 0.0 && opp > 0.0 {
                            continue;
                        }
                        let own_i = own as usize;
                        let opp_i = opp as usize;
                        total += URGENCY[own_i] - URGENCY[opp_i];
                    }
                    out[dir as usize * area + (gi as usize) * board_size as usize + gj as usize] =
                        total;
                }
            }
        }
    }

    /// Compute per-cell tactical affordance targets for the training signal.
    ///
    /// 1-turn (2-ply) solver: checks forced wins/losses within a single
    /// turn, respecting `placements_remaining` so that cells requiring more
    /// stones than available are never marked.
    ///
    /// Three binary channels (packed as `[ch][gi * board_size + gj]`):
    ///   - Ch 0 — **win_now**: cells that contribute to a forced win this
    ///     turn.  With 1 stone: cells completing a 5-own pure window.
    ///     With 2 stones: also cells in a 4-own pure window (place one,
    ///     complete with the second).
    ///   - Ch 1 — **block_win_now**: cells blocking the opponent's forced
    ///     win on their next 2-stone turn.  Only marked when the total
    ///     number of independent blocks required ≤ `placements_remaining`.
    ///   - Ch 2 — **create_hot**: empty cells in own 3-stone pure windows
    ///     (placing here creates a new 4-stone hot window / threat).
    ///
    /// Writes into `out` which has layout `[ch][gi * board_size + gj]` (3 × area).
    pub fn compute_tactical_targets(
        &self,
        offset_q: i32,
        offset_r: i32,
        board_size: i32,
        perspective: u8,
        out: &mut [f32],
    ) {
        let area = (board_size * board_size) as usize;
        debug_assert!(out.len() >= 3 * area);
        out[..3 * area].iter_mut().for_each(|v| *v = 0.0);

        let remaining = self.placements_remaining;
        let opp_idx = (1 - perspective) as usize;

        // ── Block feasibility ──────────────────────────────────────────
        // Opponent gets 2 stones on their next turn.  We check whether
        // all of their 1-turn forced wins can be blocked with our
        // remaining placements.
        //
        // 5-own opp windows → immediate 1-stone win: every empty cell
        //   in such a window is a must-block.
        // 4-own opp windows → 2-stone win: filling either empty
        //   neutralises the window (1 block per independent window).
        //
        // If total blocks needed > remaining → mark nothing (unblockable).
        let mut must_block: FxHashSet<Hex> = FxHashSet::default();
        let mut four_win_empties: Vec<[Hex; 2]> = Vec::new();

        for &(wq, wr, dir) in &self.hot_windows[opp_idx] {
            let idx = self.window_indices[win_grid_idx(wq, wr, dir)] as usize;
            let (p0, p1) = PATTERN_COUNTS[idx];
            let opp_count = if perspective == 0 { p1 } else { p0 };
            let (dq, dr) = HEX_DIRECTIONS[dir as usize];

            if opp_count >= 5 {
                for k in 0..WIN_LENGTH {
                    let h = Hex::new(wq + dq * k, wr + dr * k);
                    if !self.board.contains_key(&h) {
                        must_block.insert(h);
                    }
                }
            } else if opp_count == 4 {
                let mut e = [Hex::ORIGIN; 2];
                let mut ei = 0;
                for k in 0..WIN_LENGTH {
                    let h = Hex::new(wq + dq * k, wr + dr * k);
                    if !self.board.contains_key(&h) && ei < 2 {
                        e[ei] = h;
                        ei += 1;
                    }
                }
                if ei == 2 {
                    four_win_empties.push(e);
                }
            }
        }

        // Greedy set-cover: assign one blocker per independent 4-own window.
        // Insert both empties into covered since blocking either suffices,
        // and sharing maximises coverage of overlapping windows.
        let mut covered = must_block.clone();
        let mut extra = 0usize;
        for [e1, e2] in &four_win_empties {
            if covered.contains(e1) || covered.contains(e2) {
                continue;
            }
            extra += 1;
            covered.insert(*e1);
            covered.insert(*e2);
        }
        let block_feasible = must_block.len() + extra <= remaining as usize;

        // ── Per-cell scan ──────────────────────────────────────────────
        for gi in 0..board_size {
            for gj in 0..board_size {
                let cq = gi + offset_q;
                let cr = gj + offset_r;
                let h = Hex::new(cq, cr);
                if self.board.contains_key(&h) {
                    continue;
                }
                let flat = (gi as usize) * board_size as usize + gj as usize;

                for dir in 0u8..3 {
                    let (dq, dr) = HEX_DIRECTIONS[dir as usize];
                    for off in 0..WIN_LENGTH {
                        let wq = cq - dq * off;
                        let wr = cr - dr * off;
                        if !win_grid_in_bounds(wq, wr) {
                            continue;
                        }
                        let idx =
                            self.window_indices[win_grid_idx(wq, wr, dir)] as usize;
                        if idx == 0 {
                            continue;
                        }
                        let (p0, p1) = PATTERN_COUNTS[idx];
                        let (own, opp_cnt) = if perspective == 0 {
                            (p0, p1)
                        } else {
                            (p1, p0)
                        };

                        // Ch 0 — win_now
                        if opp_cnt == 0
                            && ((own >= 5 && remaining >= 1)
                                || (own == 4 && remaining >= 2))
                        {
                            out[flat] = 1.0;
                        }

                        // Ch 1 — block_win_now
                        if block_feasible
                            && own == 0
                            && (opp_cnt >= 5 || opp_cnt == 4)
                        {
                            out[area + flat] = 1.0;
                        }

                        // Ch 2 — create_hot
                        if opp_cnt == 0 && own == 3 {
                            out[2 * area + flat] = 1.0;
                        }
                    }
                }
            }
        }
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
