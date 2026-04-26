//! Core game state and rules engine for Hexo.
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

use crate::core::{hex_distance, Hex, HEX_DIRECTIONS};
use crate::core::{PLACEMENT_RADIUS, WIN_LENGTH};
use crate::eval::state::EvalState;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

// -------------------------------------------------------------------------
// Zobrist hashing (infinite board — mixing function instead of table)
// -------------------------------------------------------------------------

/// Deterministic hash for a (player, cell) pair using bit mixing.
///
/// XOR this value into the board hash on place and again on unplace for
/// incremental updates.
///
/// # Why a mixing function instead of a precomputed table?
///
/// The board is infinite — we cannot pre-allocate a Zobrist table for every
/// possible coordinate. Instead, we use a deterministic mixing function based
/// on FNV-1a with a final avalanche. This gives us a pseudo-random 64-bit
/// value for any (player, q, r) triple in O(1) time with no memory overhead.
#[inline]
pub(crate) fn zobrist_piece(player: u8, cell: Hex) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
    h ^= player as u64;
    h = h.wrapping_mul(0x100_0000_01b3); // FNV prime
    h ^= cell.q as u64;
    h = h.wrapping_mul(0x100_0000_01b3);
    h ^= cell.r as u64;
    h = h.wrapping_mul(0x100_0000_01b3);
    // Final avalanche: spread high-bit information into low bits.
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    h
}

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
    /// The player argument must be 0 or 1.
    InvalidPlayer(u8),
    /// The remaining argument must be 1 or 2.
    InvalidRemaining(u8),
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
            GameError::InvalidPlayer(p) => write!(f, "Invalid player: {}. Must be 0 or 1.", p),
            GameError::InvalidRemaining(r) => {
                write!(f, "Invalid remaining: {}. Must be 1 or 2.", r)
            }
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
/// [`HexGameState::unplace`] can restore exactly without fragile
/// `move_count`-based derivation.
///
/// # Invariant
///
/// Every field reflects the state *before* this placement occurred.  After
/// `place` / `unplace` round-trips the game state must be bit-identical to
/// what it was before the placement, including `winning_line`.
#[derive(Debug, Clone)]
pub struct MoveRecord {
    pub(crate) cell: Hex,
    pub(crate) player: u8,
    pub(crate) current_player_before: u8,
    pub(crate) placements_remaining_before: u8,
    pub(crate) winner_before: Option<u8>,
    pub(crate) winning_line_before: Option<[Hex; WIN_LENGTH as usize]>,
}

impl MoveRecord {
    /// The hex coordinate where the tile was placed.
    pub fn cell(&self) -> Hex {
        self.cell
    }
    /// The player who placed this tile (0 or 1).
    pub fn player(&self) -> u8 {
        self.player
    }
    /// The player whose turn it was BEFORE this move was made.
    pub fn current_player_before(&self) -> u8 {
        self.current_player_before
    }
    /// How many placements_remaining BEFORE this move was made.
    pub fn placements_remaining_before(&self) -> u8 {
        self.placements_remaining_before
    }
    /// Whether there was a winner BEFORE this move was made.
    pub fn winner_before(&self) -> Option<u8> {
        self.winner_before
    }
    /// The winning line BEFORE this move was made.
    pub fn winning_line_before(&self) -> Option<&[Hex]> {
        self.winning_line_before.as_ref().map(|a| a.as_slice())
    }
}

// -------------------------------------------------------------------------
// Candidate set
// -------------------------------------------------------------------------

/// Incremental reference-counted candidate set for fast move generation.
///
/// Instead of scanning the entire board on every move query, the engine
/// maintains a set of empty cells that are within `radius` hexes of at least
/// one occupied cell.  Each candidate stores a reference count equal to the
/// number of occupied cells within `radius` of it.
///
/// # Invariants
///
/// * Every key in `rc` is an empty hex (not in `HexGameState::stones`).
/// * `rc[h] > 0` for every stored key.
/// * A hex is present iff at least one stone lies within `radius` of it.
#[derive(Debug, Clone)]
pub(crate) struct CandidateSet {
    /// Reference counts: how many stones are within `radius` of this empty hex.
    rc: FxHashMap<Hex, u32>,
    /// The neighbour radius used when incrementing / decrementing counts.
    radius: i32,
}

impl CandidateSet {
    pub(crate) fn new(radius: i32) -> Self {
        Self {
            rc: FxHashMap::default(),
            radius,
        }
    }

    pub(crate) fn contains(&self, cell: Hex) -> bool {
        self.rc.contains_key(&cell)
    }

    pub(crate) fn remove(&mut self, cell: Hex) {
        self.rc.remove(&cell);
    }

    /// Reverse the candidate-set updates for a stone removal.
    ///
    /// Decrements reference counts for empty cells within `radius` of `cell`
    /// and re-inserts `cell` itself if any stones are still within range.
    pub(crate) fn on_unplace(&mut self, cell: Hex, stones: &Stones) {
        let r2 = self.radius;
        for dq in -r2..=r2 {
            for dr in -r2..=r2 {
                let h = Hex::new(cell.q + dq, cell.r + dr);
                if hex_distance(cell, h) <= r2 && !stones.contains_key(&h) {
                    if let Some(count) = self.rc.get_mut(&h) {
                        *count -= 1;
                        if *count == 0 {
                            self.rc.remove(&h);
                        }
                    }
                }
            }
        }

        let mut rc = 0u32;
        for dq in -r2..=r2 {
            for dr in -r2..=r2 {
                let h = Hex::new(cell.q + dq, cell.r + dr);
                if hex_distance(cell, h) <= r2 && stones.contains_key(&h) {
                    rc += 1;
                }
            }
        }
        if rc > 0 {
            self.rc.insert(cell, rc);
        }
    }
}

// -------------------------------------------------------------------------
// Game state
// -------------------------------------------------------------------------

/// Alias for the stone map: each occupied hex maps to the player who owns it.
/// Alias for the stone storage type.
///
/// This is intentionally a transparent type alias rather than a newtype.
/// `FxHashMap<Hex, u8>` is the definitive representation; if it ever changes,
/// every internal caller will need updating anyway, so the alias adds clarity
/// without pretending to hide implementation details.
pub(crate) type Stones = FxHashMap<Hex, u8>;

/// Complete mutable game state.
///
/// Create with [`HexGameState::new()`], then call [`place()`](HexGameState::place)
/// to advance the game. Query [`is_over()`](HexGameState::is_over) and
/// [`winner`](HexGameState::winner) to check for a win.
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
/// assert_eq!(g.current_player(), 0);
/// ```
#[derive(Debug, Clone)]
pub struct HexGameState {
    /// Map of all occupied cells → owning player (0 or 1).
    stones: Stones,
    /// Player to move (0 or 1).
    current_player: u8,
    /// How many stones the current player may still place this turn.
    ///
    /// * Opening: 1 for Player 0.
    /// * Normal turns: 2.
    /// * Immediately after a win: 0 (game over).
    placements_remaining: u8,
    /// `Some(winner)` when the game has ended; `None` while ongoing.
    winner: Option<u8>,
    /// The exact 6-in-a-row that produced the win, if any.
    winning_line: Option<[Hex; WIN_LENGTH as usize]>,
    /// Total number of individual stone placements so far.
    move_count: u32,
    /// Stack of snapshots for undo.  One entry per placement (not per turn).
    move_history: Vec<MoveRecord>,
    /// Incremental Zobrist hash.  XORed with `zobrist_piece` on every place/unplace.
    zobrist: u64,
    /// Incremental evaluation state (threat counts, hot windows, etc.).
    eval: EvalState,
    /// Reference-counted empty cells near stones (radius 2).
    candidates: CandidateSet,
    /// Reference-counted empty cells within PLACEMENT_RADIUS of any stone for O(1) radius check.
    placement_candidates: CandidateSet,
}

impl Default for HexGameState {
    fn default() -> Self {
        Self::new()
    }
}

impl HexGameState {
    // ── Public accessors ────────────────────────────────────────────────

    /// The map of all placed stones.
    pub fn stones(&self) -> &Stones {
        &self.stones
    }

    /// The incremental evaluation state.
    pub fn eval(&self) -> &EvalState {
        &self.eval
    }

    /// The player whose turn it is (0 or 1).
    pub fn current_player(&self) -> u8 {
        self.current_player
    }

    /// How many placements the current player still has this turn.
    pub fn placements_remaining(&self) -> u8 {
        self.placements_remaining
    }

    /// The winner, if any.
    pub fn winner(&self) -> Option<u8> {
        self.winner
    }

    /// The winning line of 6 cells, if the game is over.
    pub fn winning_line(&self) -> Option<&[Hex]> {
        self.winning_line.as_ref().map(|a| &a[..])
    }

    /// Total number of individual tile placements so far.
    pub fn move_count(&self) -> u32 {
        self.move_count
    }

    /// The history of all placements made so far.
    pub fn move_history(&self) -> &[MoveRecord] {
        &self.move_history
    }

    /// The incremental Zobrist hash of the current position.
    pub fn zobrist(&self) -> u64 {
        self.zobrist
    }

    /// Whether the game has ended (a winner exists).
    #[inline(always)]
    pub fn is_over(&self) -> bool {
        self.winner.is_some()
    }

    // ── Construction ────────────────────────────────────────────────────

    /// Create a new game in the initial empty state.
    pub fn new() -> Self {
        Self {
            stones: FxHashMap::default(),
            current_player: 0,
            placements_remaining: 1,
            winner: None,
            winning_line: None,
            move_count: 0,
            move_history: Vec::new(),
            zobrist: 0,
            eval: EvalState::new(),
            candidates: CandidateSet::new(2),
            placement_candidates: CandidateSet::new(PLACEMENT_RADIUS),
        }
    }

    // ── Placement ───────────────────────────────────────────────────────

    /// Place the current player's tile at `(q, r)`.
    ///
    /// Returns `Ok(true)` when this placement ends the current turn,
    /// `Ok(false)` when the player has another placement remaining.
    /// Returns `Err(GameError)` if the move is illegal.
    ///
    /// # Side effects
    ///
    /// * Inserts the stone into `self.stones`.
    /// * Updates the incremental Zobrist hash.
    /// * Pushes a [`MoveRecord`] onto `move_history`.
    /// * Decrements `placements_remaining`.
    /// * Updates the candidate set and incremental evaluation.
    /// * Checks for a win; if found, sets `winner`, `winning_line`, and
    ///   `placements_remaining = 0`.
    /// * If the turn is complete and no win occurred, switches player and
    ///   resets `placements_remaining` to 2.
    pub fn place(&mut self, q: i32, r: i32) -> Result<bool, GameError> {
        let cell = Hex::new(q, r);
        self.validate_move(cell)?;
        Ok(self.commit_placement(cell))
    }

    /// Undo the last placement. Restores board, turn state, and hash.
    ///
    /// # Panics
    ///
    /// Panics if called on an empty game (no moves to undo).
    pub fn unplace(&mut self) {
        self.eval.unplace();

        let rec = self.move_history.pop().expect("no move to undo");

        self.stones.remove(&rec.cell);
        self.zobrist ^= zobrist_piece(rec.player, rec.cell);
        self.move_count -= 1;

        self.candidates.on_unplace(rec.cell, &self.stones);
        self.placement_candidates.on_unplace(rec.cell, &self.stones);

        self.current_player = rec.current_player_before;
        self.placements_remaining = rec.placements_remaining_before;
        self.winner = rec.winner_before;
        self.winning_line = rec.winning_line_before;
    }

    /// Set the board to a custom position, bypassing normal turn rules.
    ///
    /// All pieces in `stones` are placed directly regardless of who is
    /// "current player".  The resulting `current_player` and
    /// `placements_remaining` are set explicitly.  Any pre-existing game
    /// state is discarded (equivalent to [`reset()`](Self::reset) first).
    ///
    /// # Use cases
    ///
    /// * Loading positions from test fixtures or databases.
    /// * Setting up synthetic board states for oracle / property tests.
    ///
    /// # Arguments
    ///
    /// * `stones` — slice of `(q, r, player)` tuples to place.
    /// * `player` — who is to move after setup.
    /// * `remaining` — how many placements that player has left this turn.
    ///
    /// # Errors
    ///
    /// Returns [`GameError::CellOccupied`] if any two entries in `stones`
    /// refer to the same hex.
    pub fn set_position(
        &mut self,
        stones: &[(i32, i32, u8)],
        player: u8,
        remaining: u8,
    ) -> Result<(), GameError> {
        if player > 1 {
            return Err(GameError::InvalidPlayer(player));
        }
        if !(1..=2).contains(&remaining) {
            return Err(GameError::InvalidRemaining(remaining));
        }

        self.reset();

        let mut sim_player = self.current_player;
        let mut sim_remaining = self.placements_remaining;

        for &(q, r, stone_player) in stones {
            let cell = Hex::new(q, r);
            if self.stones.contains_key(&cell) {
                return Err(GameError::CellOccupied(cell));
            }
            if stone_player > 1 {
                return Err(GameError::InvalidPlayer(stone_player));
            }
            if self.stones.is_empty() && cell != Hex::ORIGIN {
                return Err(GameError::MustPlaceAtOrigin);
            }
            if !self.stones.is_empty()
                && !self
                    .stones
                    .keys()
                    .any(|&e| hex_distance(e, cell) <= PLACEMENT_RADIUS)
            {
                return Err(GameError::OutOfRadius(cell));
            }

            let record = MoveRecord {
                cell,
                player: stone_player,
                current_player_before: sim_player,
                placements_remaining_before: sim_remaining,
                winner_before: self.winner,
                winning_line_before: self.winning_line,
            };

            self.stones.insert(cell, stone_player);
            self.zobrist ^= zobrist_piece(stone_player, cell);
            self.move_count += 1;
            self.move_history.push(record);

            self.candidates.remove(cell);
            self.placement_candidates.remove(cell);
            self.incr_candidate_neighbors(cell);
            self.incr_placement_candidate_neighbors(cell);
            self.eval.place(cell, stone_player);

            if self.winner.is_none() {
                if let Some(line) = self.find_winning_line(cell, stone_player) {
                    self.winner = Some(stone_player);
                    self.winning_line = Some(line);
                }
            }

            sim_remaining -= 1;
            if sim_remaining == 0 {
                sim_player = 1 - sim_player;
                sim_remaining = 2;
            }
        }

        self.current_player = player;
        self.placements_remaining = remaining;
        Ok(())
    }

    /// Reset to initial empty state.
    pub fn reset(&mut self) {
        self.stones.clear();
        self.current_player = 0;
        self.placements_remaining = 1;
        self.winner = None;
        self.winning_line = None;
        self.move_count = 0;
        self.move_history.clear();
        self.zobrist = 0;
        self.eval.clear();
        self.candidates = CandidateSet::new(2);
        self.placement_candidates = CandidateSet::new(PLACEMENT_RADIUS);
    }

    #[cfg(test)]
    pub(crate) fn place_unchecked(&mut self, cell: Hex) {
        self.commit_placement(cell);
    }

    /// Shared placement logic used by both `place` and `place_unchecked`.
    fn commit_placement(&mut self, cell: Hex) -> bool {
        let player = self.current_player;

        let record = MoveRecord {
            cell,
            player,
            current_player_before: self.current_player,
            placements_remaining_before: self.placements_remaining,
            winner_before: self.winner,
            winning_line_before: self.winning_line,
        };

        self.stones.insert(cell, player);
        self.zobrist ^= zobrist_piece(player, cell);
        self.move_count += 1;
        self.move_history.push(record);
        self.placements_remaining -= 1;

        self.candidates.remove(cell);
        self.placement_candidates.remove(cell);
        self.incr_candidate_neighbors(cell);
        self.incr_placement_candidate_neighbors(cell);
        self.eval.place(cell, player);

        if let Some(line) = self.find_winning_line(cell, player) {
            self.winner = Some(player);
            self.winning_line = Some(line);
            self.placements_remaining = 0;
            return true;
        }

        if self.placements_remaining > 0 {
            return false;
        }

        self.current_player = 1 - self.current_player;
        self.placements_remaining = 2;
        true
    }

    // ── Validation ──────────────────────────────────────────────────────

    /// Validate whether `cell` is a legal placement in the current position.
    ///
    /// Returns `Ok(())` if the move is legal, otherwise the specific error
    /// that would occur.
    pub(crate) fn validate_move(&self, cell: Hex) -> Result<(), GameError> {
        if self.is_over() {
            return Err(GameError::GameOver);
        }
        if self.placements_remaining == 0 {
            return Err(GameError::NoPlacements);
        }
        if self.stones.contains_key(&cell) {
            return Err(GameError::CellOccupied(cell));
        }
        // Opening invariant: the very first stone must be at the origin.
        if self.stones.is_empty() && cell != Hex::ORIGIN {
            return Err(GameError::MustPlaceAtOrigin);
        }
        // Radius invariant: every non-opening stone must be within
        // PLACEMENT_RADIUS of at least one existing stone.
        if !self.stones.is_empty() && !self.placement_candidates.contains(cell) {
            return Err(GameError::OutOfRadius(cell));
        }
        Ok(())
    }

    // ── Legal moves ─────────────────────────────────────────────────────

    /// Return all legal placements (exhaustive radius-8 scan — expensive).
    ///
    /// Returns an empty vec if the game is over.  On an empty board returns
    /// only [`Hex::ORIGIN`].
    pub fn legal_moves(&self) -> Vec<Hex> {
        if self.is_over() {
            return Vec::new();
        }
        if self.stones.is_empty() {
            return vec![Hex::ORIGIN];
        }

        let mut candidates = FxHashSet::default();
        for &cell in self.stones.keys() {
            for dq in -PLACEMENT_RADIUS..=PLACEMENT_RADIUS {
                for dr in -PLACEMENT_RADIUS..=PLACEMENT_RADIUS {
                    let cand = Hex::new(cell.q + dq, cell.r + dr);
                    if !self.stones.contains_key(&cand)
                        && hex_distance(cell, cand) <= PLACEMENT_RADIUS
                    {
                        candidates.insert(cand);
                    }
                }
            }
        }

        let result: Vec<Hex> = candidates.into_iter().collect();
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
        if self.stones.is_empty() {
            return vec![Hex::ORIGIN];
        }

        // Use incremental candidate set if radius matches (fast path).
        if radius == self.candidates.radius && !self.candidates.rc.is_empty() {
            return self.candidates.rc.keys().copied().collect();
        }

        // Fallback: full scan for different radius.
        let r = radius.min(PLACEMENT_RADIUS);
        let mut candidates = FxHashSet::default();
        for &cell in self.stones.keys() {
            for dq in -r..=r {
                for dr in -r..=r {
                    let cand = Hex::new(cell.q + dq, cell.r + dr);
                    if !self.stones.contains_key(&cand) && hex_distance(cell, cand) <= r {
                        candidates.insert(cand);
                    }
                }
            }
        }

        let result: Vec<Hex> = candidates.into_iter().collect();
        result
    }

    /// Return the incremental radius-2 candidate set as a Vec.
    ///
    /// This is the fast path for the turn-based search which always uses
    /// radius 2.  Falls back to [`legal_moves_near(2)`](Self::legal_moves_near)
    /// if the incremental set is empty (e.g. after a reset).
    pub fn candidates_near2(&self) -> Vec<Hex> {
        if self.is_over() {
            return Vec::new();
        }
        if self.stones.is_empty() {
            return vec![Hex::ORIGIN];
        }
        let result = if !self.candidates.rc.is_empty() {
            self.candidates.rc.keys().copied().collect()
        } else {
            self.legal_moves_near(2)
        };
        result
    }

    /// Sorted version of legal_moves_near — for tests and Python export only.
    pub fn legal_moves_near_sorted(&self, radius: i32) -> Vec<Hex> {
        let mut v = self.legal_moves_near(radius);
        v.sort();
        v
    }

    pub fn candidates_near2_sorted(&self) -> Vec<Hex> {
        let mut v = self.candidates_near2();
        v.sort();
        v
    }

    /// The opponent's most recent completed turn as an ordered list of cells.
    ///
    /// Returns one cell for Player 0's opening turn, otherwise two cells.
    pub fn opponent_last_turn_cells(&self) -> smallvec::SmallVec<[Hex; 2]> {
        let mut idx = self.move_history.len();

        // Skip the current player's partial turn, if any.
        while idx > 0 && self.move_history[idx - 1].player == self.current_player {
            idx -= 1;
        }

        // Collect the opponent's last completed turn.
        let mut cells = smallvec::SmallVec::new();
        while idx > 0 && self.move_history[idx - 1].player != self.current_player {
            cells.push(self.move_history[idx - 1].cell);
            idx -= 1;
        }
        cells.reverse();
        cells
    }

    // ── Win detection ───────────────────────────────────────────────────

    /// Check whether `last` (just placed by `player`) completed a winning line.
    ///
    /// Scans in all three principal directions.  For each direction it collects
    /// the contiguous run of `player` stones that includes `last`, then checks
    /// whether the run is at least [`WIN_LENGTH`] long.
    ///
    /// # Returns
    ///
    /// `Some(winning_line)` — a vector of exactly 6 [`Hex`] cells forming the
    /// winning segment, preferring one centered on `last` if possible.
    pub(crate) fn find_winning_line(&self, last: Hex, player: u8) -> Option<[Hex; WIN_LENGTH as usize]> {
        for &(dq, dr) in &HEX_DIRECTIONS {
            // Collect the contiguous run on both sides of `last` along this axis.
            let mut backward = self.collect_run(last, -dq, -dr, player);
            backward.reverse();
            let forward = self.collect_run(last, dq, dr, player);

            // Assemble the full run with `last` in the middle.
            let pivot = backward.len();
            let mut line = backward;
            line.push(last);
            line.extend_from_slice(&forward);

            // If the run is long enough, extract the best 6-stone segment.
            if line.len() >= WIN_LENGTH as usize {
                return Some(Self::select_segment(&line, pivot));
            }
        }
        None
    }

    /// Collect consecutive stones belonging to `player` starting from the
    /// neighbour of `origin` in direction `(dq, dr)`.
    ///
    /// Does NOT include `origin` itself; the caller appends it separately.
    #[inline]
    fn collect_run(&self, origin: Hex, dq: i32, dr: i32, player: u8) -> SmallVec<[Hex; 6]> {
        let mut tiles = SmallVec::new();
        let mut q = origin.q + dq;
        let mut r = origin.r + dr;
        while self.stones.get(&Hex::new(q, r)) == Some(&player) {
            tiles.push(Hex::new(q, r));
            q += dq;
            r += dr;
        }
        tiles
    }

    /// Extract a [`WIN_LENGTH`]-long segment from a longer contiguous run.
    ///
    /// `pivot` is the index of the most recently placed stone within `line`.
    /// The algorithm prefers a segment centred on `pivot`, but clamps the
    /// start so the window stays entirely inside the run.
    ///
    /// # Example
    ///
    /// If `line` has 8 stones and `pivot` is 3, the preferred segment starts
    /// at `3 - 2 = 1` (zero-based), giving indices 1..7 — a 6-stone window
    /// centred roughly on the pivot.
    fn select_segment(line: &[Hex], pivot: usize) -> [Hex; WIN_LENGTH as usize] {
        let wl = WIN_LENGTH as usize;
        let lo = pivot.saturating_sub(wl - 1);
        let hi = pivot.min(line.len() - wl);
        let preferred = pivot.saturating_sub((wl - 1) / 2);
        let start = hi.min(lo.max(preferred));
        line[start..start + wl]
            .try_into()
            .expect("select_segment: slice length != WIN_LENGTH")
    }

    // ── Candidate helpers ───────────────────────────────────────────────

    /// Increment candidate reference counts for empty cells within radius 2 of `cell`.
    ///
    /// Must be called **after** the piece is inserted into `self.stones`,
    /// otherwise the newly occupied cell would incorrectly be counted as a
    /// candidate.
    fn incr_candidate_neighbors(&mut self, cell: Hex) {
        let r2 = self.candidates.radius;
        for dq in -r2..=r2 {
            for dr in -r2..=r2 {
                let h = Hex::new(cell.q + dq, cell.r + dr);
                if hex_distance(cell, h) <= r2 && !self.stones.contains_key(&h) {
                    *self.candidates.rc.entry(h).or_insert(0) += 1;
                }
            }
        }
    }

    /// Increment placement-candidate reference counts for empty cells within PLACEMENT_RADIUS.
    fn incr_placement_candidate_neighbors(&mut self, cell: Hex) {
        let r = PLACEMENT_RADIUS;
        for dq in -r..=r {
            for dr in -r..=r {
                let h = Hex::new(cell.q + dq, cell.r + dr);
                if hex_distance(cell, h) <= r && !self.stones.contains_key(&h) {
                    *self.placement_candidates.rc.entry(h).or_insert(0) += 1;
                }
            }
        }
    }
}
