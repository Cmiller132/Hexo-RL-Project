//! Classical pattern-based feature extraction for Infinity Hexagonal Tic-Tac-Toe.
//!
//! This module provides the **neural feature extraction** path (`extract_features`)
//! — it scans the entire board to build a 13-element feature vector counting live
//! and dead runs of various lengths. It is used by the classical self-play pipeline
//! in `pybridge.rs` to generate training data for the neural network.
//!
//! The alpha-beta search in `search.rs` uses its own minimal O(1) evaluator that
//! reads `window_eval` directly; it does not call `extract_features`.

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
    fn empty_board_features() {
        let game = HexGameState::new();
        let feats = extract_features(&game);
        // All counts are zero; tempo is +1.0 because P0 to move.
        for i in 0..FEATURE_COUNT - 1 {
            assert_eq!(feats[i], 0.0, "feature {} should be zero on empty board", i);
        }
        assert_eq!(feats[FEATURE_COUNT - 1], 1.0);
    }

    #[test]
    fn tempo_feature_flips_for_player_1() {
        let mut game = HexGameState::new();
        game.place(0, 0).unwrap(); // P0 opens
        game.place(1, 0).unwrap(); // P1 first placement
        game.place(0, 1).unwrap(); // P1 second placement, turn switches to P0
        // After the opening, current player is P0 again (move_count=3)
        // Let's get to P1's turn.
        game.place(2, 0).unwrap(); // P0 first
        game.place(3, 0).unwrap(); // P0 second, turn switches to P1
        assert_eq!(game.current_player, 1);
        let feats = extract_features(&game);
        assert_eq!(feats[FEATURE_COUNT - 1], -1.0);
    }

    #[test]
    fn live_five_is_counted() {
        let mut game = HexGameState::new();
        // P0 live-5 along (1,0): (0,0)..(4,0), both ends open.
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)],
            0,
            2,
        )
        .unwrap();
        let feats = extract_features(&game);
        assert_eq!(feats[0], 1.0); // P0 live-5
    }

    #[test]
    fn dead_five_is_counted() {
        let mut game = HexGameState::new();
        // P0 dead-5: blocked on both ends by P1 at (-1,0) and (5,0).
        game.set_position(
            &[
                (-1, 0, 1), // P1 blocker left
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (3, 0, 0),
                (4, 0, 0),
                (5, 0, 1), // P1 blocker right
            ],
            0,
            2,
        )
        .unwrap();
        let feats = extract_features(&game);
        assert_eq!(feats[1], 1.0); // P0 dead-5
    }

    #[test]
    fn six_in_a_row_bumps_live_five() {
        let mut game = HexGameState::new();
        // 6 in a row for P0 along (1,0).
        game.set_position(
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
        let feats = extract_features(&game);
        // Terminal pattern adds 10 to live-5 bucket
        assert!(feats[0] >= 10.0);
    }

    #[test]
    fn live_four_is_counted() {
        let mut game = HexGameState::new();
        // P0 live-4: (0,0)..(3,0), both ends open.
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
            0,
            2,
        )
        .unwrap();
        let feats = extract_features(&game);
        assert_eq!(feats[2], 1.0); // P0 live-4
    }

    #[test]
    fn live_three_and_live_two() {
        let mut game = HexGameState::new();
        // P0 live-3: (0,0),(1,0),(2,0).
        game.set_position(&[(0, 0, 0), (1, 0, 0), (2, 0, 0)], 0, 2).unwrap();
        let feats = extract_features(&game);
        assert_eq!(feats[4], 1.0); // P0 live-3
        assert_eq!(feats[5], 0.0); // no live-2 yet
    }

    #[test]
    fn opponent_features_are_separate() {
        let mut game = HexGameState::new();
        // P0 live-3 along (1,0), P1 live-3 along (0,1) — far apart so they don't interfere.
        game.set_position(
            &[
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0), // P0 live-3
                (5, 0, 1),
                (5, 1, 1),
                (5, 2, 1), // P1 live-3
            ],
            0,
            2,
        )
        .unwrap();
        let feats = extract_features(&game);
        assert_eq!(feats[4], 1.0);  // P0 live-3
        assert_eq!(feats[10], 1.0); // P1 live-3
    }
}
