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
use crate::eval::state::EvalState;
use crate::core::{PLACEMENT_RADIUS, WIN_LENGTH};

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
pub fn zobrist_piece(player: u8, cell: Hex) -> u64 {
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
    /// The winning line BEFORE this move was made.
    ///
    /// Snapshotted so that `unplace` can restore a won position exactly
    /// rather than unconditionally clearing the line.
    pub winning_line_before: Option<Vec<Hex>>,
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
pub struct CandidateSet {
    /// Reference counts: how many stones are within `radius` of this empty hex.
    rc: FxHashMap<Hex, u32>,
    /// The neighbour radius used when incrementing / decrementing counts.
    radius: i32,
}

impl CandidateSet {
    /// Create a new empty candidate set with the given radius.
    pub fn new(radius: i32) -> Self {
        Self {
            rc: FxHashMap::default(),
            radius,
        }
    }

    /// Remove a cell from the candidate set.
    ///
    /// Called when a stone is placed on this cell; it is no longer empty and
    /// therefore cannot be a legal move.
    pub fn remove(&mut self, cell: Hex) {
        self.rc.remove(&cell);
    }

    /// Clear all candidates.
    pub fn clear(&mut self) {
        self.rc.clear();
    }
}

// -------------------------------------------------------------------------
// Game state
// -------------------------------------------------------------------------

/// Alias for the stone map: each occupied hex maps to the player who owns it.
pub type Stones = FxHashMap<Hex, u8>;

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
    winning_line: Option<Vec<Hex>>,
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
        self.winning_line.as_deref()
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

        let player = self.current_player;

        // Snapshot state BEFORE mutating so unplace can restore exactly.
        let record = MoveRecord {
            cell,
            player,
            current_player_before: self.current_player,
            placements_remaining_before: self.placements_remaining,
            winner_before: self.winner,
            winning_line_before: self.winning_line.clone(),
        };

        // Commit the stone to the board.
        self.stones.insert(cell, player);
        self.zobrist ^= zobrist_piece(player, cell);
        self.move_count += 1;
        self.move_history.push(record);
        self.placements_remaining -= 1;

        // Update incremental structures: candidate set and eval state.
        self.candidates.remove(cell);
        self.incr_candidate_neighbors(cell);
        self.eval.place(&self.stones, cell, player);

        // Check whether this placement completed a winning line.
        if let Some(line) = self.find_winning_line(cell, player) {
            self.winner = Some(player);
            self.winning_line = Some(line);
            self.placements_remaining = 0;
            return Ok(true);
        }

        // If the player still has placements left, the turn continues.
        if self.placements_remaining > 0 {
            return Ok(false);
        }

        // Turn complete — swap player and reset to two placements.
        self.current_player = 1 - self.current_player;
        self.placements_remaining = 2;
        Ok(true)
    }

    /// Undo the last placement. Restores board, turn state, and hash.
    ///
    /// # Panics
    ///
    /// Panics if called on an empty game (no moves to undo).
    ///
    /// # Algorithm
    ///
    /// 1. Reverse the incremental evaluation (`eval.unplace`).
    /// 2. Pop the [`MoveRecord`] for the move being undone.
    /// 3. Remove the stone from `stones` and XOR its Zobrist contribution.
    /// 4. Decrement reference counts for all empty neighbours within radius 2
    ///    of the removed stone; remove any count that reaches zero.
    /// 5. Re-add the removed cell to the candidate set if any stones are
    ///    still within radius 2 of it.
    /// 6. Restore `current_player`, `placements_remaining`, `winner`, and
    ///    `winning_line` from the snapshot.
    pub fn unplace(&mut self) {
        // Reverse incremental eval BEFORE we know which move is undone.
        // EvalState maintains its own parallel stack, so the order is safe.
        self.eval.unplace();

        let rec = self.move_history.pop().expect("no move to undo");

        // Remove the stone from the board.
        self.stones.remove(&rec.cell);
        self.zobrist ^= zobrist_piece(rec.player, rec.cell);
        self.move_count -= 1;

        // ---- Restore candidate set --------------------------------------
        let r2 = self.candidates.radius;

        // 4a. Decrement reference counts for every empty cell within radius 2
        //     of the removed stone.  If a count drops to zero, the cell is no
        //     longer adjacent to any stone and is removed from candidates.
        for dq in -r2..=r2 {
            for dr in -r2..=r2 {
                let h = Hex::new(rec.cell.q + dq, rec.cell.r + dr);
                if hex_distance(rec.cell, h) <= r2 && !self.stones.contains_key(&h) {
                    if let Some(count) = self.candidates.rc.get_mut(&h) {
                        *count -= 1;
                        if *count == 0 {
                            self.candidates.rc.remove(&h);
                        }
                    }
                }
            }
        }

        // 4b. Re-insert the removed cell into the candidate set if there are
        //     still stones within radius 2 of it.  The new count is exactly
        //     the number of such stones.
        let mut rc = 0u32;
        for dq in -r2..=r2 {
            for dr in -r2..=r2 {
                let h = Hex::new(rec.cell.q + dq, rec.cell.r + dr);
                if hex_distance(rec.cell, h) <= r2 && self.stones.contains_key(&h) {
                    rc += 1;
                }
            }
        }
        if rc > 0 {
            self.candidates.rc.insert(rec.cell, rc);
        }

        // ---- Restore turn state from snapshot ---------------------------
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
        self.reset();

        // Simulate turn progression so each synthetic MoveRecord captures the
        // correct state before its placement.
        let mut sim_player = self.current_player; // 0
        let mut sim_remaining = self.placements_remaining; // 1

        for &(q, r, player) in stones {
            let cell = Hex::new(q, r);
            if self.stones.contains_key(&cell) {
                return Err(GameError::CellOccupied(cell));
            }

            // Snapshot the pre-placement state.
            let record = MoveRecord {
                cell,
                player,
                current_player_before: sim_player,
                placements_remaining_before: sim_remaining,
                winner_before: self.winner,
                winning_line_before: self.winning_line.clone(),
            };

            // Place the stone.
            self.stones.insert(cell, player);
            self.zobrist ^= zobrist_piece(player, cell);
            self.move_count += 1;
            self.move_history.push(record);

            // Update incremental structures.
            self.candidates.remove(cell);
            self.incr_candidate_neighbors(cell);
            self.eval.place(&self.stones, cell, player);

            // Detect wins during bulk placement.
            if self.winner.is_none() {
                if let Some(line) = self.find_winning_line(cell, player) {
                    self.winner = Some(player);
                    self.winning_line = Some(line);
                }
            }

            // Advance the simulated turn counter.
            sim_remaining -= 1;
            if sim_remaining == 0 {
                sim_player = 1 - sim_player;
                sim_remaining = 2;
            }
        }

        self.current_player = player & 1;
        self.placements_remaining = remaining.max(1);
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
        self.eval = EvalState::new();
        self.candidates = CandidateSet::new(2);
    }

    /// Places without validation, used by test oracle and internal modules.
    ///
    /// # Safety
    ///
    /// Callers must guarantee that `cell` is empty and legal.  Violating this
    /// may corrupt `EvalState`, the candidate set, or the Zobrist hash.
    #[allow(dead_code)]
    pub(crate) fn place_unchecked(&mut self, cell: Hex) {
        let player = self.current_player;

        // Snapshot state before mutation.
        let record = MoveRecord {
            cell,
            player,
            current_player_before: self.current_player,
            placements_remaining_before: self.placements_remaining,
            winner_before: self.winner,
            winning_line_before: self.winning_line.clone(),
        };

        // Commit the stone.
        self.stones.insert(cell, player);
        self.zobrist ^= zobrist_piece(player, cell);
        self.move_count += 1;
        self.move_history.push(record);
        self.placements_remaining -= 1;

        // Update incremental structures.
        self.candidates.remove(cell);
        self.incr_candidate_neighbors(cell);
        self.eval.place(&self.stones, cell, player);

        // Check for win.
        if let Some(line) = self.find_winning_line(cell, player) {
            self.winner = Some(player);
            self.winning_line = Some(line);
            self.placements_remaining = 0;
            return;
        }

        // Advance turn if completed.
        if self.placements_remaining == 0 {
            self.current_player = 1 - self.current_player;
            self.placements_remaining = 2;
        }
    }

    // ── Validation ──────────────────────────────────────────────────────

    /// Validate whether `cell` is a legal placement in the current position.
    ///
    /// Returns `Ok(())` if the move is legal, otherwise the specific error
    /// that would occur.
    pub fn validate_move(&self, cell: Hex) -> Result<(), GameError> {
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
        if !self.stones.is_empty()
            && !self
                .stones
                .keys()
                .any(|&existing| hex_distance(existing, cell) <= PLACEMENT_RADIUS)
        {
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
        if self.stones.is_empty() {
            return vec![Hex::ORIGIN];
        }

        // Use incremental candidate set if radius matches (fast path).
        if radius == self.candidates.radius && !self.candidates.rc.is_empty() {
            let mut result: Vec<Hex> = self.candidates.rc.keys().copied().collect();
            result.sort();
            return result;
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

        let mut result: Vec<Hex> = candidates.into_iter().collect();
        result.sort();
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
        let mut result = if !self.candidates.rc.is_empty() {
            self.candidates.rc.keys().copied().collect()
        } else {
            self.legal_moves_near(2)
        };
        result.sort();
        result
    }

    /// The opponent's most recent completed turn as an ordered list of cells.
    ///
    /// Returns one cell for Player 0's opening turn, otherwise two cells.
    ///
    /// # Algorithm
    ///
    /// 1. Walk backward from the end of `move_history`, skipping any
    ///    placements belonging to `current_player` (these are from an
    ///    in-progress turn).
    /// 2. Collect consecutive placements belonging to the opponent.
    /// 3. Reverse so the cells are in chronological order.
    pub fn opponent_last_turn_cells(&self) -> Vec<Hex> {
        let mut idx = self.move_history.len();

        // Skip the current player's partial turn, if any.
        while idx > 0 && self.move_history[idx - 1].player == self.current_player {
            idx -= 1;
        }

        // Collect the opponent's last completed turn.
        let mut cells = Vec::with_capacity(2);
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
    pub fn find_winning_line(&self, last: Hex, player: u8) -> Option<Vec<Hex>> {
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
    fn collect_run(&self, origin: Hex, dq: i32, dr: i32, player: u8) -> Vec<Hex> {
        let mut tiles = Vec::new();
        let mut q = origin.q + dq;
        let mut r = origin.r + dr;
        // Step outward one cell at a time while each cell belongs to `player`.
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
    fn select_segment(line: &[Hex], pivot: usize) -> Vec<Hex> {
        let wl = WIN_LENGTH as usize;
        // Earliest start that still includes the pivot in the 6-window.
        let lo = pivot.saturating_sub(wl - 1);
        // Latest start such that the window fits inside the line.
        let hi = pivot.min(line.len() - wl);
        // Ideal start: centre the window on the pivot.
        let preferred = pivot.saturating_sub((wl - 1) / 2);
        // Clamp preferred start to the valid [lo, hi] range.
        let start = hi.min(lo.max(preferred));
        line[start..start + wl].to_vec()
    }

    // ── Eval helpers ────────────────────────────────────────────────────

    /// Compute the eval delta for a hypothetical placement without modifying state.
    pub fn move_eval_delta(&self, cell: Hex, player: u8) -> i32 {
        self.eval.hypothetical_score_delta(cell, player)
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
                // Only count empty cells within the circular radius.
                if hex_distance(cell, h) <= r2 && !self.stones.contains_key(&h) {
                    *self.candidates.rc.entry(h).or_insert(0) += 1;
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
        assert!(!g.stones.contains_key(&Hex::new(1, 0)));
    }

    #[test]
    fn zobrist_restores_after_unmake() {
        let mut g = HexGameState::new();
        let h0 = g.zobrist;

        g.place(0, 0).unwrap();
        let h1 = g.zobrist;
        assert_ne!(h0, h1);

        g.unplace();
        assert_eq!(g.zobrist, h0);
    }

    #[test]
    fn reset_clears_everything() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        g.place(1, 0).unwrap();
        g.reset();
        assert_eq!(g.move_count, 0);
        assert!(g.stones.is_empty());
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
        assert!(a.stones.is_empty() && b.stones.is_empty());
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
        assert_eq!(g.stones.len(), 3);
        assert_eq!(g.stones.get(&Hex::new(1, 0)), Some(&0));
        assert_eq!(g.stones.get(&Hex::new(2, 0)), Some(&0));
        assert_eq!(g.stones.get(&Hex::new(0, 1)), Some(&1));
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

    // -- unplace eval round-trip -------------------------------------------

    #[test]
    fn unmake_restores_eval_counters() {
        let mut g = HexGameState::new();
        g.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0)],
            0,
            2,
        )
        .unwrap();

        let eval0 = g.eval.score();
        let fives0 = [g.eval.counts(0).fives, g.eval.counts(1).fives];
        let fours0 = [g.eval.counts(0).fours, g.eval.counts(1).fours];
        let threes0 = [g.eval.counts(0).threes, g.eval.counts(1).threes];
        let hot0 = g.eval.hot_len(0);

        g.place(3, 0).unwrap();
        g.unplace();

        assert_eq!(g.eval.score(), eval0);
        assert_eq!([g.eval.counts(0).fives, g.eval.counts(1).fives], fives0);
        assert_eq!([g.eval.counts(0).fours, g.eval.counts(1).fours], fours0);
        assert_eq!([g.eval.counts(0).threes, g.eval.counts(1).threes], threes0);
        assert_eq!(g.eval.hot_len(0), hot0);
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
