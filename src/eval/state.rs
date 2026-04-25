use crate::core::{Hex, WindowKey, HEX_DIRECTIONS, WIN_LENGTH};
use crate::eval::grid::{win_grid_idx, win_grid_in_bounds, WIN_GRID_TOTAL};
use crate::eval::hot::HotWindows;
use crate::eval::patterns::{PATTERN_COUNTS, PATTERN_VALUES, POW3};

/// Aggregate threat statistics for a single player.
///
/// These counts are used by the search engine to quickly detect forcing
/// lines (e.g. "does the opponent already have a five threat?").
///
/// # Invariants
/// * All fields are monotonically non-decreasing across the lifetime of a
///   game (they only decrease during `unplace`, which is the inverse of
///   `place`).
/// * `fives` > 0 implies an immediate win is available on the next turn.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct ThreatCounts {
    fives: u32,
    fours: u32,
    threes: u32,
}

impl ThreatCounts {
    /// Apply a signed delta to the threat counts.
    ///
    /// # Panics (debug builds only)
    ///
    /// `debug_assert!` fires if any field would underflow.  Underflow is a
    /// logic bug: it means `unplace` was called more times than `place`, or
    /// the incremental update tables are inconsistent.
    pub(crate) fn apply(&mut self, delta: &ThreatCountsDelta) {
        debug_assert!(
            (self.fives as i32 + delta.fives) >= 0,
            "fives underflow: {} + {}",
            self.fives,
            delta.fives
        );
        debug_assert!(
            (self.fours as i32 + delta.fours) >= 0,
            "fours underflow: {} + {}",
            self.fours,
            delta.fours
        );
        debug_assert!(
            (self.threes as i32 + delta.threes) >= 0,
            "threes underflow: {} + {}",
            self.threes,
            delta.threes
        );
        self.fives = (self.fives as i32 + delta.fives) as u32;
        self.fours = (self.fours as i32 + delta.fours) as u32;
        self.threes = (self.threes as i32 + delta.threes) as u32;
    }

    /// Number of immediate win threats (≥5 own stones, 0 opponent).
    #[inline]
    pub fn fives(&self) -> u32 {
        self.fives
    }

    /// Number of one-away win threats (exactly 4 own stones, 0 opponent).
    #[inline]
    pub fn fours(&self) -> u32 {
        self.fours
    }

    /// Number of two-away win threats (exactly 3 own stones, 0 opponent).
    #[inline]
    pub fn threes(&self) -> u32 {
        self.threes
    }
}

/// Signed change in threat counts produced by a single stone placement.
///
/// A positive value means the corresponding threat category increased;
/// negative means it decreased.  Used inside [`EvalDelta`] so that `unplace`
/// can subtract the exact same delta back out.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub(crate) struct ThreatCountsDelta {
    fives: i32,
    fours: i32,
    threes: i32,
}

impl std::ops::Neg for ThreatCountsDelta {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            fives: -self.fives,
            fours: -self.fours,
            threes: -self.threes,
        }
    }
}

/// Complete incremental delta for a single stone placement.
///
/// `EvalState::unplace` consumes the mirrored delta from the internal stack
/// to restore the previous state.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct EvalDelta {
    cell: Hex,
    cell_val: u8,
    score: i32,
    counts: [ThreatCountsDelta; 2],
}

/// Incremental evaluation state.
///
/// Instead of re-scanning the entire board after every move, `EvalState`
/// maintains a dense win-grid of pattern indices and updates only the
/// windows that touch the newly placed (or removed) stone.  This makes
/// `place` and `unplace` `O(1)` — specifically they touch at most
/// `3 directions × 6 offsets = 18` windows.
///
/// # Core data structures
///
/// * `score` — accumulated static evaluation from P0's perspective.
/// * `counts` — per-player [`ThreatCounts`] (fives, fours, threes).
/// * `hot` — a [`HotWindows`] cache of windows with ≥ 4 own stones and 0
///   opponent stones.
/// * `indices` — flattened win grid mapping every `(q, r, dir)` inside
///   [`WIN_GRID_RADIUS`](crate::eval::grid::WIN_GRID_RADIUS) to a pattern
///   index `0..728`.  Boxed so the large array lives on the heap but the
///   `EvalState` struct itself remains small and cheap to move.
/// * `delta_stack` — stack of [`EvalDelta`]s supporting `unplace`.  One entry
///   per stone placed.
///
/// NOTE: `Clone` copies the entire `Box<[u16; WIN_GRID_TOTAL]>` (~22 KB).
/// Acceptable for search-invocation clones (once per `iterative_deepening`
/// root) but would be costly inside MCTS node expansion.
#[derive(Clone, Debug)]
pub struct EvalState {
    score: i32,
    counts: [ThreatCounts; 2],
    hot: HotWindows,
    indices: Box<[u16; WIN_GRID_TOTAL]>,
    delta_stack: Vec<EvalDelta>,
}

/// Update the threat-count delta for one player based on a window's old and
/// new stone counts.
///
/// # Arguments
/// * `delta`      — the accumulator to mutate.
/// * `old_own`    — how many of the player's stones were in the window before.
/// * `old_other`  — how many opponent stones were in the window before.
/// * `new_own`    — how many of the player's stones are in the window now.
/// * `new_other`  — how many opponent stones are in the window now.
///
/// # Logic
/// A "five" threat is a window with ≥ 5 own stones and 0 opponent stones.
/// A "four"  threat is a window with exactly 4 own stones and 0 opponent.
/// A "three" threat is a window with exactly 3 own stones and 0 opponent.
///
/// The delta records `+1` when a window enters one of those categories and
/// `-1` when it leaves.
#[inline]
fn classify_delta(
    delta: &mut ThreatCountsDelta,
    old_own: u8,
    old_other: u8,
    new_own: u8,
    new_other: u8,
) {
    let old_five = (old_own >= 5 && old_other == 0) as i32;
    let new_five = (new_own >= 5 && new_other == 0) as i32;
    delta.fives += new_five - old_five;

    let old_four = (old_own == 4 && old_other == 0) as i32;
    let new_four = (new_own == 4 && new_other == 0) as i32;
    delta.fours += new_four - old_four;

    let old_three = (old_own == 3 && old_other == 0) as i32;
    let new_three = (new_own == 3 && new_other == 0) as i32;
    delta.threes += new_three - old_three;
}

/// Synchronously update the [`HotWindows`] cache for a single window.
///
/// A window becomes "hot" when it crosses the threshold
/// `own >= 4 && other == 0`, and ceases to be hot when it drops below that
/// threshold.  This helper ensures `hot` stays consistent with the win grid.
#[inline]
fn update_hot(
    hot: &mut HotWindows,
    key: WindowKey,
    player: u8,
    old_own: u8,
    old_other: u8,
    new_own: u8,
    new_other: u8,
) {
    let was_hot = old_own >= 4 && old_other == 0;
    let is_hot = new_own >= 4 && new_other == 0;
    if was_hot && !is_hot {
        hot.remove(player, key);
    } else if !was_hot && is_hot {
        hot.insert(player, key);
    }
}

/// Iterate the 18 windows that touch `cell`.
///
/// Passes `dir` and `off` directly to the callback to avoid re-deriving
/// them from a flat index inside `place`, `unplace`, and
/// `hypothetical_score_delta`.
#[inline]
fn visit_windows(cell: Hex, mut cb: impl FnMut(i32, i32, u8, u8)) {
    for (dir, &(dq, dr)) in HEX_DIRECTIONS.iter().enumerate() {
        for off in 0..WIN_LENGTH as usize {
            let sq = cell.q - dq * off as i32;
            let sr = cell.r - dr * off as i32;
            cb(sq, sr, dir as u8, off as u8);
        }
    }
}

impl Default for EvalState {
    fn default() -> Self {
        Self::new()
    }
}

impl EvalState {
    /// Create a fresh evaluation state with an empty board.
    ///
    /// All pattern indices start at `0` (the "all empty" pattern), the score
    /// is `0`, and there are no hot windows or threat counts.
    pub fn new() -> Self {
        Self {
            score: 0,
            counts: [ThreatCounts::default(); 2],
            hot: HotWindows::new(),
            indices: Box::new([0; WIN_GRID_TOTAL]),
            delta_stack: Vec::new(),
        }
    }

    /// Zero all state in place, avoiding reallocation.
    pub fn clear(&mut self) {
        self.indices.fill(0);
        self.score = 0;
        self.counts = [ThreatCounts::default(); 2];
        self.hot = HotWindows::new();
        self.delta_stack.clear();
    }

    /// Incrementally evaluate a stone placement.
    ///
    /// # Arguments
    /// * `cell`    — the coordinate where the stone is placed.
    /// * `player`  — `0` or `1`.
    #[inline]
    pub fn place(&mut self, cell: Hex, player: u8) {
        let cell_val = player + 1; // 1 or 2
        let mut delta = EvalDelta {
            cell,
            cell_val,
            score: 0,
            counts: [ThreatCountsDelta::default(); 2],
        };

        visit_windows(cell, |sq, sr, dir, off| {
            if !win_grid_in_bounds(sq, sr) {
                return;
            }

            let gi = win_grid_idx(sq, sr, dir);
            let old_idx = self.indices[gi] as usize;
            let new_idx = old_idx + (cell_val as usize) * POW3[off as usize];
            assert!(new_idx < 729, "pattern index out of range: {} (cell_val={}, off={})", new_idx, cell_val, off);

            delta.score += PATTERN_VALUES[new_idx] - PATTERN_VALUES[old_idx];

            let (old_p0, old_p1) = PATTERN_COUNTS[old_idx];
            let (new_p0, new_p1) = PATTERN_COUNTS[new_idx];

            classify_delta(&mut delta.counts[0], old_p0, old_p1, new_p0, new_p1);
            classify_delta(&mut delta.counts[1], old_p1, old_p0, new_p1, new_p0);

            let key = WindowKey::new(sq, sr, dir);
            update_hot(&mut self.hot, key, 0, old_p0, old_p1, new_p0, new_p1);
            update_hot(&mut self.hot, key, 1, old_p1, old_p0, new_p1, new_p0);

            self.indices[gi] = new_idx as u16;
        });

        self.score += delta.score;
        self.counts[0].apply(&delta.counts[0]);
        self.counts[1].apply(&delta.counts[1]);
        self.delta_stack.push(delta);
    }

    /// Undo the most recent stone placement, restoring the previous evaluation.
    ///
    /// # Panics
    ///
    /// Panics if `unplace` is called when no stone has been placed (i.e. the
    /// delta stack is empty).
    #[inline]
    pub fn unplace(&mut self) {
        let delta = self
            .delta_stack
            .pop()
            .expect("unplace called with empty stack");

        self.score -= delta.score;
        self.counts[0].apply(&(-delta.counts[0]));
        self.counts[1].apply(&(-delta.counts[1]));

        let cell = delta.cell;
        let cell_val = delta.cell_val as usize;

        visit_windows(cell, |sq, sr, dir, off| {
            if !win_grid_in_bounds(sq, sr) {
                return;
            }

            let gi = win_grid_idx(sq, sr, dir);
            let new_idx = self.indices[gi] as usize;
            debug_assert!(
                new_idx >= cell_val * POW3[off as usize],
                "unplace: index underflow at gi={gi}"
            );
            let old_idx = new_idx - cell_val * POW3[off as usize];
            assert!(old_idx < 729, "pattern index out of range on unplace: {}", old_idx);

            let (old_p0, old_p1) = PATTERN_COUNTS[old_idx];
            let (new_p0, new_p1) = PATTERN_COUNTS[new_idx];

            let key = WindowKey::new(sq, sr, dir);
            update_hot(&mut self.hot, key, 0, new_p0, new_p1, old_p0, old_p1);
            update_hot(&mut self.hot, key, 1, new_p1, new_p0, old_p1, old_p0);

            self.indices[gi] = old_idx as u16;
        });

        #[cfg(debug_assertions)]
        self.assert_invariants();
    }

    /// Debug-only consistency check.
    ///
    /// Recomputes the hot-window sets from scratch by scanning the entire
    /// win grid and comparing with the incremental cache.  Expensive, but
    /// invaluable for catching incremental-update bugs.
    #[cfg(debug_assertions)]
    fn assert_invariants(&self) {
        use crate::eval::grid::WIN_GRID_RADIUS;

        let mut expected = [
            std::collections::HashSet::new(),
            std::collections::HashSet::new(),
        ];
        for q in -WIN_GRID_RADIUS..=WIN_GRID_RADIUS {
            for r in -WIN_GRID_RADIUS..=WIN_GRID_RADIUS {
                for dir in 0..3u8 {
                    let gi = win_grid_idx(q, r, dir);
                    let idx = self.indices[gi] as usize;
                    let (p0, p1) = PATTERN_COUNTS[idx];
                    if p0 >= 4 && p1 == 0 {
                        expected[0].insert(WindowKey::new(q, r, dir));
                    }
                    if p1 >= 4 && p0 == 0 {
                        expected[1].insert(WindowKey::new(q, r, dir));
                    }
                }
            }
        }

        // Compare recomputed with actual (order-independent because swap_remove
        // may reorder the internal SmallVec).
        for player in 0..2 {
            let actual: std::collections::HashSet<_> = self.hot.iter(player).collect();
            assert_eq!(
                actual.len(),
                expected[player as usize].len(),
                "hot window count mismatch for player {}",
                player
            );
            assert_eq!(
                actual, expected[player as usize],
                "hot window mismatch for player {}",
                player
            );
        }
    }

    /// Current static evaluation score from P0's perspective.
    ///
    /// Positive values favour P0; negative values favour P1.
    #[inline]
    pub fn score(&self) -> i32 {
        self.score
    }

    /// Threat counts for `player` (0 or 1).
    #[inline]
    pub fn counts(&self, player: u8) -> ThreatCounts {
        self.counts[player as usize]
    }

    /// Returns `true` if `player` has an active four or five threat.
    #[inline]
    pub fn has_threats(&self, player: u8) -> bool {
        let c = self.counts[player as usize];
        c.fives() > 0 || c.fours() > 0
    }

    /// Returns `true` if either player has an active four or five threat.
    #[inline]
    pub fn has_any_threats(&self) -> bool {
        self.has_threats(0) || self.has_threats(1)
    }

    /// Iterate over the hot windows for `player`.
    ///
    /// See [`HotWindows`](crate::eval::hot::HotWindows) for the definition of
    /// "hot".
    #[inline]
    pub fn hot_windows(&self, player: u8) -> impl Iterator<Item = WindowKey> + '_ {
        self.hot.iter(player)
    }

    /// Returns `true` if `player` has no hot windows.
    #[inline]
    pub fn hot_is_empty(&self, player: u8) -> bool {
        self.hot.is_empty(player)
    }

    /// Number of hot windows currently tracked for `player`.
    #[inline]
    pub fn hot_len(&self, player: u8) -> usize {
        self.hot.len(player)
    }

    /// Compute the score delta that would result from placing a stone at
    /// `cell` for `player`, **without mutating state**.
    ///
    /// This is useful for move ordering and quiescence search: the engine
    /// can rank candidate moves by their immediate material impact before
    /// committing to a full `place` / `unplace` pair.
    ///
    /// # Algorithm
    /// Identical to the score-accumulation half of `place`, but without
    /// writing back to `indices`, updating `counts`, or touching `hot`.
    pub fn hypothetical_score_delta(&self, cell: Hex, player: u8) -> i32 {
        let cell_val = (player + 1) as usize;
        let mut delta = 0i32;

        visit_windows(cell, |sq, sr, dir, off| {
            if !win_grid_in_bounds(sq, sr) {
                return;
            }

            let gi = win_grid_idx(sq, sr, dir);
            let old_idx = self.indices[gi] as usize;
            let new_idx = old_idx + cell_val * POW3[off as usize];
            assert!(new_idx < 729, "pattern index out of range in score_delta: {}", new_idx);

            delta += PATTERN_VALUES[new_idx] - PATTERN_VALUES[old_idx];
        });

        delta
    }
}
