//! Classical pattern-based feature extraction for Infinity Hexagonal Tic-Tac-Toe.
//!
//! This module provides two complementary evaluation mechanisms:
//!
//! 1. **Incremental evaluation** ([`EvalState`]) — used during search.
//!    Updates only the windows touched by a newly placed stone, making
//!    `place` and `unplace` `O(1)`.
//! 2. **Classical feature extraction** ([`extract_features`]) — used by the
//!    neural-network encoder to generate a 13-dimensional feature vector
//!    suitable for machine-learning pipelines.
//!
//! # Sub-modules
//!
//! | Module      | Purpose                                               |
//! |-------------|-------------------------------------------------------|
//! | `grid`      | Win-grid spatial indexing and bounds checks           |
//! | `hot`       | Zero-alloc cache of urgent (4+ stone) threat windows  |
//! | `patterns`  | Pre-computed ternary pattern tables (729 entries)     |
//! | `state`     | [`EvalState`], [`ThreatCounts`], incremental update    |

pub mod grid;
pub mod hot;
pub mod patterns;
pub mod state;

use crate::board::HexGameState;
use crate::core::{Hex, HEX_DIRECTIONS, WIN_LENGTH};

pub use state::{EvalDelta, EvalState, ThreatCounts, ThreatCountsDelta};

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------

/// Total number of scalar features emitted by [`extract_features`].
///
/// The vector is structured as:
/// ```text
/// [P0_live5, P0_dead5, P0_live4, P0_dead4, P0_live3, P0_live2,
///  P1_live5, P1_dead5, P1_live4, P1_dead4, P1_live3, P1_live2,
///  tempo]
/// ```
/// where `tempo` is `+1.0` when P0 is to move and `-1.0` when P1 is to move.
pub const FEATURE_COUNT: usize = 13;

/// Static evaluation bonus for an immediate win.
pub const WIN_SCORE: i32 = 1_000_000;

const FEATURES_PER_PLAYER: usize = 6;
const LIVE5: usize = 0;
const DEAD5: usize = 1;
const LIVE4: usize = 2;
const DEAD4: usize = 3;
const LIVE3: usize = 4;
const LIVE2: usize = 5;

// -------------------------------------------------------------------------
// Run counting
// -------------------------------------------------------------------------

/// Count contiguous stones of `player` starting from `start` along `(dq, dr)`.
///
/// # Arguments
/// * `game`   — the board state.
/// * `start`  — the first cell of the run (already known to hold `player`).
/// * `dq`     — q-step per cell along the line.
/// * `dr`     — r-step per cell along the line.
/// * `player` — the player whose stones we are counting.
///
/// # Returns
/// A tuple `(count, open_end)` where:
/// * `count`    — number of *additional* consecutive `player` stones after
///   `start` (not counting `start` itself).
/// * `open_end` — `true` if the run ended because the next cell is empty;
///   `false` if it ended because the next cell holds an opponent stone.
#[inline]
fn count_run(game: &HexGameState, start: Hex, dq: i32, dr: i32, player: u8) -> (i32, bool) {
    let mut count = 0;
    let mut q = start.q + dq;
    let mut r = start.r + dr;
    loop {
        let h = Hex::new(q, r);
        match game.stones().get(&h) {
            Some(&p) if p == player => count += 1,
            Some(_) => return (count, false), // blocked by opponent
            None => return (count, true),     // open end
        }
        q += dq;
        r += dr;
    }
}

// -------------------------------------------------------------------------
// Feature extraction
// -------------------------------------------------------------------------

/// Extract a 13-dimensional classical feature vector from the board.
///
/// The features count "runs" of contiguous stones for each player along all
/// three principal hex directions.  A run is only counted if the cell
/// immediately before its start is **not** the same player's stone; this
/// prevents double-counting the same line segment.
///
/// # Feature semantics
///
/// | Index | Name    | Meaning                                           |
/// |-------|---------|---------------------------------------------------|
/// | 0     | P0 live5| P0 has ≥5 in a row with at least one open end     |
/// | 1     | P0 dead5| P0 has exactly 5 in a row with zero open ends     |
/// | 2     | P0 live4| P0 has exactly 4 in a row with **two** open ends  |
/// | 3     | P0 dead4| P0 has exactly 4 in a row with **one** open end   |
/// | 4     | P0 live3| P0 has exactly 3 in a row with two open ends      |
/// | 5     | P0 live2| P0 has exactly 2 in a row with two open ends      |
/// | 6–11  | P1 …    | Same six features mirrored for player 1           |
/// | 12    | tempo   | `+1.0` if P0 to move, `-1.0` otherwise            |
///
/// "Open end" means the adjacent cell past the run is empty.  A run of 6
/// or more stones is treated as a `live5` and multiplied by 10 to emphasise
/// its decisiveness.
pub fn extract_features(game: &HexGameState) -> [f32; FEATURE_COUNT] {
    let mut feats = [0.0f32; FEATURE_COUNT];
    // counts[p][i] accumulates the raw integer counts for player p, feature i.
    let mut counts = [[0i32; FEATURES_PER_PLAYER]; 2];

    // Iterate over every stone on the board.
    for (&cell, &player) in game.stones() {
        let p = player as usize;
        for &(dq, dr) in &HEX_DIRECTIONS {
            // Only start a run if the previous cell in this direction is
            // NOT the same player's stone.  This guarantees each maximal
            // contiguous run is counted exactly once.
            let prev = Hex::new(cell.q - dq, cell.r - dr);
            if game.stones().get(&prev) == Some(&player) {
                continue;
            }

            // Count how far the run extends forward from `cell`.
            let (fwd, fwd_open) = count_run(game, cell, dq, dr, player);
            let run_len = 1 + fwd; // include `cell` itself

            // Determine whether the cell just before `cell` is open.
            let bwd_open = match game.stones().get(&prev) {
                None => true,
                Some(_) => false,
            };

            let open_ends = (bwd_open as i32) + (fwd_open as i32);

            // Classify the run into one of the threat categories.
            if run_len >= WIN_LENGTH {
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

    // Copy the integer counts into the float feature vector.
    for p in 0..2 {
        for i in 0..FEATURES_PER_PLAYER {
            feats[p * FEATURES_PER_PLAYER + i] = counts[p][i] as f32;
        }
    }

    // Final feature: tempo (whose turn it is).
    feats[FEATURE_COUNT - 1] = if game.current_player() == 0 { 1.0 } else { -1.0 };
    feats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_board_features() {
        let game = HexGameState::new();
        let feats = extract_features(&game);
        for i in 0..FEATURE_COUNT - 1 {
            assert_eq!(feats[i], 0.0, "feature {} should be zero on empty board", i);
        }
        assert_eq!(feats[FEATURE_COUNT - 1], 1.0);
    }

    #[test]
    fn tempo_feature_flips_for_player_1() {
        let mut game = HexGameState::new();
        game.place(0, 0).unwrap();
        game.place(1, 0).unwrap();
        game.place(0, 1).unwrap();
        game.place(2, 0).unwrap();
        game.place(3, 0).unwrap();
        assert_eq!(game.current_player(), 1);
        let feats = extract_features(&game);
        assert_eq!(feats[FEATURE_COUNT - 1], -1.0);
    }

    #[test]
    fn live_five_is_counted() {
        let mut game = HexGameState::new();
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)],
            0,
            2,
        )
        .unwrap();
        let feats = extract_features(&game);
        assert_eq!(feats[0], 1.0);
    }

    #[test]
    fn dead_five_is_counted() {
        let mut game = HexGameState::new();
        game.set_position(
            &[
                (-1, 0, 1),
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (3, 0, 0),
                (4, 0, 0),
                (5, 0, 1),
            ],
            0,
            2,
        )
        .unwrap();
        let feats = extract_features(&game);
        assert_eq!(feats[1], 1.0);
    }

    #[test]
    fn six_in_a_row_bumps_live_five() {
        let mut game = HexGameState::new();
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
        assert!(feats[0] >= 10.0);
    }

    #[test]
    fn live_four_is_counted() {
        let mut game = HexGameState::new();
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
            0,
            2,
        )
        .unwrap();
        let feats = extract_features(&game);
        assert_eq!(feats[2], 1.0);
    }

    #[test]
    fn live_three_and_live_two() {
        let mut game = HexGameState::new();
        game.set_position(&[(0, 0, 0), (1, 0, 0), (2, 0, 0)], 0, 2).unwrap();
        let feats = extract_features(&game);
        assert_eq!(feats[4], 1.0);
        assert_eq!(feats[5], 0.0);
    }

    #[test]
    fn opponent_features_are_separate() {
        let mut game = HexGameState::new();
        game.set_position(
            &[
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (5, 0, 1),
                (5, 1, 1),
                (5, 2, 1),
            ],
            0,
            2,
        )
        .unwrap();
        let feats = extract_features(&game);
        assert_eq!(feats[4], 1.0);
        assert_eq!(feats[10], 1.0);
    }
}
