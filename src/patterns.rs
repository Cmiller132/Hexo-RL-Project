//! Ternary 6-cell window pattern evaluation.
//!
//! This module implements the incremental evaluation system for Infinity Hexagonal
//! Tic-Tac-Toe. Every 6-cell sliding window along the three hex axes is encoded as
//! a base-3 number (0=empty, 1=player0, 2=player1), producing 3^6 = 729 possible
//! patterns. Each pattern has a pre-computed static evaluation weight (CMA-ES optimized).
//!
//! When a stone is placed, only the windows containing that cell need updating —
//! at most 18 windows (3 directions × 6 origins). This makes evaluation O(18) per
//! placement instead of O(board_size × windows).

use rustc_hash::FxHashSet;
use crate::core::{Hex, HEX_DIRECTIONS};
use crate::board::HexGameState;

// ── Constants ──

/// Number of cells in a win-checking window.
///
/// A player wins by forming a contiguous straight line of exactly this many
/// stones along any of the three hex principal axes.
pub const WIN_LENGTH: i32 = 6;

/// Maximum distance from any existing tile for a valid placement.
///
/// Every new stone must be placed within this hex distance of at least one
/// already-placed stone. This keeps the game locally connected and prevents
/// players from playing infinitely far away.
pub const PLACEMENT_RADIUS: i32 = 8;

/// Total number of distinct ternary patterns for a 6-cell window (3^6).
///
/// Each cell in a window can be empty (0), player 0 (1), or player 1 (2),
/// giving 3^6 = 729 unique configurations.
pub const PATTERN_COUNT: usize = 729;

/// Powers of 3 for ternary index computation.
///
/// `POW3[i]` = 3^i. When a stone is placed at offset `i` within a 6-cell window,
/// the ternary digit contributed by that stone is `cell_val * POW3[i]`, where
/// `cell_val` is 1 for player 0 and 2 for player 1.
const POW3: [usize; 6] = [1, 3, 9, 27, 81, 243];

/// Flat-array grid radius for storing window indices.
///
/// Window origins can be up to 5 cells away from any placed piece. Since pieces
/// are constrained to be within [`PLACEMENT_RADIUS`] (8) of existing pieces, a
/// radius of 30 comfortably covers all reachable window origins without overflow.
pub const WIN_GRID_RADIUS: i32 = 30;

/// Side length of the square win-grid (`2 * WIN_GRID_RADIUS + 1`).
///
/// The grid spans `[-30, 30]` in both axial coordinates, giving 61 steps per axis.
const WIN_GRID_SIDE: usize = (2 * WIN_GRID_RADIUS + 1) as usize; // 61

/// Total number of entries in the flat win-grid array.
///
/// Layout: `(q + 30) * 61 * 3 + (r + 30) * 3 + dir`.
/// This equals 61 × 61 × 3 = 11_163 entries.
pub const WIN_GRID_TOTAL: usize = WIN_GRID_SIDE * WIN_GRID_SIDE * 3; // 11163

// ── Types ──

/// Saved per-move incremental eval delta for unmake.
///
/// When a stone is placed, [`HexGameState::compute_eval_delta`] returns this
/// struct containing all changes that were applied to the running evaluation
/// state. The values are stored on `eval_stack` so that `unmake_move` can
/// reverse them exactly.
#[derive(Debug, Clone, Copy)]
pub struct EvalDelta {
    /// Change in the overall window-based evaluation score (player 0's perspective).
    pub score: i32,
    /// Change in the count of near-win (5+ stones) windows per player `[p0, p1]`.
    pub five_delta: [i32; 2],
    /// Change in the count of hot (exactly 4 stones) windows per player `[p0, p1]`.
    pub four_delta: [i32; 2],
    /// Change in the count of developing (exactly 3 stones) windows per player `[p0, p1]`.
    pub three_delta: [i32; 2],
}

// ── Functions ──

/// Convert `(q, r, dir)` to a flat array index into `window_indices`.
///
/// # Panics
/// Caller must ensure `win_grid_in_bounds(q, r)` returns `true`; otherwise the
/// computed index will be out of bounds.
#[inline(always)]
pub fn win_grid_idx(q: i32, r: i32, dir: u8) -> usize {
    ((q + WIN_GRID_RADIUS) as usize) * WIN_GRID_SIDE * 3
        + ((r + WIN_GRID_RADIUS) as usize) * 3
        + dir as usize
}

/// Check if `(q, r)` is within the pre-allocated win-grid bounds.
///
/// Returns `true` when the coordinate can safely be passed to [`win_grid_idx`].
#[inline(always)]
pub fn win_grid_in_bounds(q: i32, r: i32) -> bool {
    let qi = q + WIN_GRID_RADIUS;
    let ri = r + WIN_GRID_RADIUS;
    qi >= 0 && (qi as usize) < WIN_GRID_SIDE && ri >= 0 && (ri as usize) < WIN_GRID_SIDE
}

/// Precomputed `(p0_count, p1_count)` for each ternary pattern index (0..728).
///
/// `p0` = count of digits with value 1, `p1` = count of digits with value 2.
/// This is evaluated at compile time and stored in [`PATTERN_COUNTS`].
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

/// Static lookup table: for each of the 729 ternary patterns, the number of
/// player-0 stones and player-1 stones it contains.
///
/// Generated by [`build_pattern_counts`] at compile time.
pub static PATTERN_COUNTS: [(u8, u8); 729] = build_pattern_counts();

/// Public accessor for pattern counts.
///
/// Returns `(p0_count, p1_count)` for the given pattern index.
#[inline(always)]
pub fn pattern_counts(idx: usize) -> (u8, u8) {
    PATTERN_COUNTS[idx]
}

/// CMA-ES optimized pattern values for 6-cell windows (ternary encoding).
///
/// Index = base-3 number where digit `i` (0..5) is:
/// * `0` — empty cell
/// * `1` — player 0 stone
/// * `2` — player 1 stone
///
/// The value is the static evaluation contribution of that window from player 0's
/// perspective (positive = good for player 0).
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

/// Classify a pattern by its stone counts for threat-level tracking.
///
/// If `other == 0` (the window is uncontested — no opponent stones), then `own`
/// stones are classified into threat tiers:
/// * `5..=6` → `fives`   (one move from a win, or already won)
/// * `4`     → `fours`   (hot window — two moves from a win)
/// * `3`     → `threes`  (developing threat)
///
/// `sign` is `+1` when adding a pattern (new stone placed) or `-1` when removing
/// one (during unmake).
#[inline]
pub fn classify(
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

// ── impl HexGameState ──

impl HexGameState {
    /// Compute the incremental eval change from placing `player`'s piece at `cell`.
    ///
    /// **Must be called AFTER the piece is already inserted into `self.board`.**
    ///
    /// Iterates over every 6-cell window that contains `cell`. There are at most
    /// 18 such windows: 3 directions × 6 possible origin offsets. For each window:
    ///
    /// 1. **Read the old ternary pattern index** from `self.window_indices`.
    /// 2. **Compute the new index** by adding the ternary digit for the placed
    ///    stone. The digit is `cell_val * POW3[off]`, where `cell_val` is 1 for
    ///    player 0 and 2 for player 1, and `POW3[off]` is the base-3 place value
    ///    of offset `off` within the 6-cell window.
    /// 3. **Score delta** — look up the old and new pattern values in
    ///    [`PATTERN_VALUES`] and add `new_value - old_value` to the running score.
    /// 4. **Threat count deltas** — use [`PATTERN_COUNTS`] to get the old and new
    ///    `(p0_count, p1_count)` pairs. Call [`classify`] with `sign = -1` to
    ///    remove the old pattern's contribution, then `sign = +1` to add the new
    ///    pattern's contribution. This updates `five_delta`, `four_delta`, and
    ///    `three_delta` for both players.
    /// 5. **Update `hot_windows`** — a window is "hot" for a player when it
    ///    contains 4+ of that player's stones and 0 opponent stones. If the window
    ///    transitions into or out of this state, insert or remove the window key
    ///    `(sq, sr, dir)` from the appropriate `hot_windows` set.
    /// 6. **Write the new index back** to `self.window_indices` so future
    ///    placements see the updated pattern.
    ///
    /// Returns an [`EvalDelta`] summarising all changes so the caller can apply
    /// them to the running totals (`window_eval`, `window_fives`, etc.) and push
    /// the delta onto `eval_stack` for unmake.
    pub fn compute_eval_delta(&mut self, cell: Hex, player: u8) -> EvalDelta {
        // Ternary digit for this player: 1 for player 0, 2 for player 1.
        let cell_val = if player == 0 { 1usize } else { 2usize };
        let mut score = 0i32;
        let mut five_delta = [0i32; 2];
        let mut four_delta = [0i32; 2];
        let mut three_delta = [0i32; 2];

        for (dir_idx, &(dq, dr)) in HEX_DIRECTIONS.iter().enumerate() {
            for off in 0..WIN_LENGTH as usize {
                // Window origin such that `cell` is at offset `off` along direction `dir_idx`.
                let sq = cell.q - dq * off as i32;
                let sr = cell.r - dr * off as i32;

                if !win_grid_in_bounds(sq, sr) {
                    continue;
                }
                let gi = win_grid_idx(sq, sr, dir_idx as u8);

                // 1. Read the old ternary pattern index for this window.
                let old_idx = self.window_indices[gi] as usize;

                // 2. Compute the new index by adding the ternary digit for the placed stone.
                //    POW3[off] is the place-value of offset `off` in base 3.
                let new_idx = old_idx + cell_val * POW3[off];

                // 3. Score delta = new pattern value - old pattern value.
                score += PATTERN_VALUES[new_idx] - PATTERN_VALUES[old_idx];

                // 4. Threat count deltas from precomputed count table.
                let (old_p0, old_p1) = PATTERN_COUNTS[old_idx];
                let (new_p0, new_p1) = PATTERN_COUNTS[new_idx];

                // Subtract old pattern's contribution.
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
                // Add new pattern's contribution.
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

                // 5. Update hot_windows (transition tracking).
                //    A window is "hot" for a player when it has 4+ of that player's stones
                //    and 0 opponent stones.
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

                // 6. Update stored index so future placements see the new pattern.
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
    ///
    /// Returns the score change from player 0's perspective.
    ///
    /// This is a read-only version of [`compute_eval_delta`]: it reads the current
    /// `window_indices` to get each window's old pattern index, computes what the
    /// new index would be after placing `player`'s stone at `cell`, and sums the
    /// [`PATTERN_VALUES`] deltas. No threat counts, hot-window sets, or stored
    /// indices are updated.
    ///
    /// Used for move ordering — much faster than a full evaluation.
    #[inline]
    pub fn move_eval_delta(&self, cell: Hex, player: u8) -> i32 {
        // Ternary digit for this player: 1 for player 0, 2 for player 1.
        let cell_val = if player == 0 { 1usize } else { 2usize };
        let mut delta = 0i32;

        for (dir_idx, &(dq, dr)) in HEX_DIRECTIONS.iter().enumerate() {
            for off in 0..WIN_LENGTH as usize {
                // Window origin such that `cell` is at offset `off` along direction `dir_idx`.
                let sq = cell.q - dq * off as i32;
                let sr = cell.r - dr * off as i32;

                if !win_grid_in_bounds(sq, sr) {
                    continue;
                }
                let gi = win_grid_idx(sq, sr, dir_idx as u8);

                // Read the current ternary pattern index for this window.
                let old_idx = self.window_indices[gi] as usize;

                // Compute what the new index would be after placing the stone.
                let new_idx = old_idx + cell_val * POW3[off];

                // Add the pattern-value delta to the running total.
                delta += PATTERN_VALUES[new_idx] - PATTERN_VALUES[old_idx];
            }
        }
        delta
    }
}
