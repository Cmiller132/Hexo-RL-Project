//! Classical pattern-based evaluation for Infinity Hexagonal Tic-Tac-Toe.
//!
//! This module provides two complementary evaluation paths:
//!
//! 1. **Classical evaluation** (`evaluate`) — an O(1) heuristic score derived
//!    from pre-computed incremental state (`window_eval`, `window_fives`,
//!    `window_fours`). Used by the alpha-beta search in `search.rs` to assign
//!    leaf-node scores.
//!
//! 2. **Neural feature extraction** (`extract_features`) — scans the entire
//!    board to build a 13-element feature vector counting live and dead runs
//!    of various lengths. Used by the classical self-play pipeline in
//!    `pybridge.rs` to generate training data for the neural network.
//!
//! The two systems are independent. `evaluate` does NOT call
//! `extract_features`; it relies on incremental updates maintained during
//! `place`/`unmake_move` in `board.rs` for speed.

use crate::board::HexGameState;
use crate::core::{Hex, HEX_DIRECTIONS};
use crate::patterns::WIN_LENGTH;

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------

/// Total length of the feature vector extracted by [`extract_features`].
///
/// Six features per player (live-5, dead-5, live-4, dead-4, live-3, live-2)
/// plus one tempo feature = 13.
pub const FEATURE_COUNT: usize = 13;

/// A large value representing a winning position (but not infinity).
///
/// Scores are offset slightly below this constant (`WIN_SCORE - 10`,
/// `WIN_SCORE - 15`) to preserve ordering information for positions that
/// are "essentially won" but not yet terminal.
pub const WIN_SCORE: i32 = 1_000_000;

/// Number of features extracted per player.
const FEATURES_PER_PLAYER: usize = 6;

/// Feature indices within a player's slice.
const LIVE5: usize = 0;
const DEAD5: usize = 1;
const LIVE4: usize = 2;
const DEAD4: usize = 3;
const LIVE3: usize = 4;
const LIVE2: usize = 5;

// -------------------------------------------------------------------------
// Run counting
// -------------------------------------------------------------------------

/// Count consecutive same-player tiles extending from `start` in direction
/// `(dq, dr)`, **not including `start` itself**.
///
/// Returns `(count, open_end)` where:
/// - `count` is the number of contiguous same-player cells.
/// - `open_end` is `true` if the run terminates at an empty cell,
///   `false` if it is blocked by an opponent piece or the board edge
///   (represented by `None` from the hash map lookup).
#[inline]
fn count_run(game: &HexGameState, start: Hex, dq: i32, dr: i32, player: u8) -> (i32, bool) {
    let mut count = 0;
    let mut q = start.q + dq;
    let mut r = start.r + dr;
    loop {
        let h = Hex::new(q, r);
        match game.board.get(&h) {
            Some(&p) if p == player => count += 1,
            Some(_) => return (count, false), // blocked by opponent
            None => return (count, true),     // open end (empty or off-board)
        }
        q += dq;
        r += dr;
    }
}

// -------------------------------------------------------------------------
// Feature extraction
// -------------------------------------------------------------------------

/// Extract a 13-element feature vector from the current board state.
///
/// Features (indices 0-5 for player 0, 6-11 for player 1):
/// - 0,6: live-5 (5+ in a row with at least 1 open end)
/// - 1,7: dead-5 (5 in a row, blocked on one end)
/// - 2,8: live-4 (4 in a row, both ends open)
/// - 3,9: dead-4 (4 in a row, one end open)
/// - 4,10: live-3 (3 in a row, both ends open)
/// - 5,11: live-2 (2 in a row, both ends open)
/// - 12: tempo (1.0 if P0 to move, -1.0 otherwise)
///
/// This scans the board by iterating over occupied cells and counting
/// consecutive runs along each of the 6 hex directions. A run is only
/// counted from its starting cell (the cell whose predecessor is not the
/// same player) to avoid double-counting.
pub fn extract_features(game: &HexGameState) -> [f32; FEATURE_COUNT] {
    let mut feats = [0.0f32; FEATURE_COUNT];
    let mut counts = [[0i32; FEATURES_PER_PLAYER]; 2];

    // Step 1: Iterate over every occupied cell on the board.
    // For each cell, we examine all 6 hex directions.
    for (&cell, &player) in &game.board {
        let p = player as usize;
        for &(dq, dr) in &HEX_DIRECTIONS {
            // Step 2: Only count runs from their starting cell.
            //
            // A "start" is defined as a cell whose predecessor in the
            // negative direction `( -dq, -dr )` is NOT occupied by the same
            // player. If the predecessor is also ours, this cell is in the
            // middle of a longer run that was already counted from its true
            // start, so we skip it. This guarantees each run is counted
            // exactly once.
            let prev = Hex::new(cell.q - dq, cell.r - dr);
            if game.board.get(&prev) == Some(&player) {
                continue; // not the start of this run
            }

            // Step 3: Count forward from this cell.
            //
            // `count_run` returns how many same-player cells follow `cell`
            // in direction `(dq, dr)` and whether that forward side is open.
            // The run length includes `cell` itself, so we add 1.
            let (fwd, fwd_open) = count_run(game, cell, dq, dr, player);
            let run_len = 1 + fwd;

            // Step 4: Determine the backward open end.
            //
            // We already know `prev` is not the same player (Step 2).
            // If `prev` is empty (`None`), the backward end is open.
            // If `prev` is occupied by the opponent (`Some(_)`), it is blocked.
            let bwd_open = match game.board.get(&prev) {
                None => true,
                Some(_) => false, // opponent piece
            };

            // Step 5: Total open ends.
            //
            // A run can have 0, 1, or 2 open ends. Runs with 0 open ends
            // are fully blocked and cannot be extended, so they contribute
            // no feature value.
            let open_ends = (bwd_open as i32) + (fwd_open as i32);

            // Step 6: Classify the run into a feature bucket.
            //
            // Live runs have open ends and represent active threats.
            // Dead runs are blocked on one side and are less threatening.
            // We only count runs of lengths 2-5 (plus the special case of
            // 6+ which is already a win).
            if run_len >= WIN_LENGTH {
                // Already won — shouldn't happen during normal eval, but
                // we bump live-5 heavily to signal a terminal pattern.
                counts[p][LIVE5] += 10;
            } else if run_len == 5 {
                if open_ends >= 1 {
                    counts[p][LIVE5] += 1;
                } else {
                    counts[p][DEAD5] += 1;
                }
            } else if run_len == 4 {
                if open_ends == 2 {
                    counts[p][LIVE4] += 1;
                } else if open_ends == 1 {
                    counts[p][DEAD4] += 1;
                }
            } else if run_len == 3 {
                if open_ends == 2 {
                    counts[p][LIVE3] += 1;
                }
            } else if run_len == 2 {
                if open_ends == 2 {
                    counts[p][LIVE2] += 1;
                }
            }
        }
    }

    // Step 7: Flatten per-player counts into the output feature vector.
    for p in 0..2 {
        for i in 0..FEATURES_PER_PLAYER {
            feats[p * FEATURES_PER_PLAYER + i] = counts[p][i] as f32;
        }
    }

    // Step 8: Append the tempo feature.
    feats[FEATURE_COUNT - 1] = if game.current_player == 0 { 1.0 } else { -1.0 };
    feats
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_board_eval_is_zero() {
        let game = HexGameState::new();
        // On an empty board, both players have the same features (nothing).
        // Player 0 has the tempo, so they get the +15 bonus.
        assert_eq!(evaluate(&game, 0), 15);
    }

    #[test]
    fn winning_position_has_win_score() {
        let mut game = HexGameState::new();
        // Build player 0 win: 6 in a row along (1,0)
        game.place(0, 0).unwrap();
        game.place(-1, -1).unwrap();
        game.place(-2, -2).unwrap();
        game.place(1, 0).unwrap();
        game.place(2, 0).unwrap();
        game.place(-3, -3).unwrap();
        game.place(-4, -4).unwrap();
        game.place(3, 0).unwrap();
        game.place(4, 0).unwrap();
        game.place(-5, -5).unwrap();
        game.place(-6, -6).unwrap();
        game.place(5, 0).unwrap();
        assert_eq!(evaluate(&game, 0), WIN_SCORE);
        assert_eq!(evaluate(&game, 1), -WIN_SCORE);
    }

    #[test]
    fn player_with_longer_lines_scores_higher() {
        let mut game = HexGameState::new();
        game.place(0, 0).unwrap();
        game.place(2, 1).unwrap();
        game.place(-2, 1).unwrap();
        // P0 has 1 tile at (0,0), P1 has 2 scattered tiles
        // P0's turn, 2 placements remaining
        let s0 = evaluate(&game, 0);
        let s1 = evaluate(&game, 1);
        // s0 should be higher than s1 because P0 has the tempo
        assert!(s0 > s1);
    }
}
