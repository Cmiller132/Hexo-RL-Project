//! Classical pattern-based feature extraction for Infinity Hexagonal Tic-Tac-Toe.

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

pub const FEATURE_COUNT: usize = 13;
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

#[inline]
fn count_run(game: &HexGameState, start: Hex, dq: i32, dr: i32, player: u8) -> (i32, bool) {
    let mut count = 0;
    let mut q = start.q + dq;
    let mut r = start.r + dr;
    loop {
        let h = Hex::new(q, r);
        match game.stones().get(&h) {
            Some(&p) if p == player => count += 1,
            Some(_) => return (count, false),
            None => return (count, true),
        }
        q += dq;
        r += dr;
    }
}

// -------------------------------------------------------------------------
// Feature extraction
// -------------------------------------------------------------------------

pub fn extract_features(game: &HexGameState) -> [f32; FEATURE_COUNT] {
    let mut feats = [0.0f32; FEATURE_COUNT];
    let mut counts = [[0i32; FEATURES_PER_PLAYER]; 2];

    for (&cell, &player) in game.stones() {
        let p = player as usize;
        for &(dq, dr) in &HEX_DIRECTIONS {
            let prev = Hex::new(cell.q - dq, cell.r - dr);
            if game.stones().get(&prev) == Some(&player) {
                continue;
            }

            let (fwd, fwd_open) = count_run(game, cell, dq, dr, player);
            let run_len = 1 + fwd;

            let bwd_open = match game.stones().get(&prev) {
                None => true,
                Some(_) => false,
            };

            let open_ends = (bwd_open as i32) + (fwd_open as i32);

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

    for p in 0..2 {
        for i in 0..FEATURES_PER_PLAYER {
            feats[p * FEATURES_PER_PLAYER + i] = counts[p][i] as f32;
        }
    }

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
