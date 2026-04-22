//! Pattern-based position evaluation for Hex Tic-Tac-Toe.
//!
//! Scans all occupied cells along the 3 hex axes and classifies line patterns
//! (live-5, live-4, dead-4, live-3, etc.) to produce a heuristic score.
//! Also extracts a feature vector for use by the neural network.

use crate::core::{Hex, HEX_DIRECTIONS};
use crate::game::{HexGameState, WIN_LENGTH};

// -------------------------------------------------------------------------
// Feature indices (for NN input)
// -------------------------------------------------------------------------

/// Number of features extracted per player (×2 players = total feature vec).
pub const FEATURES_PER_PLAYER: usize = 6;
/// Total feature vector length: per-player features × 2 + 1 (tempo).
pub const FEATURE_COUNT: usize = FEATURES_PER_PLAYER * 2 + 1;

/// Feature indices within a player's slice:
/// 0 = live-5 count, 1 = dead-5 count, 2 = live-4, 3 = dead-4,
/// 4 = live-3, 5 = live-2
const LIVE5: usize = 0;
const DEAD5: usize = 1;
const LIVE4: usize = 2;
const DEAD4: usize = 3;
const LIVE3: usize = 4;
const LIVE2: usize = 5;

/// A large value representing a winning position (but not infinity).
pub const WIN_SCORE: i32 = 1_000_000;

// -------------------------------------------------------------------------
// Run counting
// -------------------------------------------------------------------------

/// Count consecutive same-player tiles extending from `start` in direction
/// (dq, dr). Returns (count, open_end) where open_end is true if the run
/// terminates at an empty cell (false = blocked by opponent or board edge).
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
            None => return (count, true),     // open end
        }
        q += dq;
        r += dr;
    }
}

// -------------------------------------------------------------------------
// Feature extraction
// -------------------------------------------------------------------------

/// Extract pattern features for both players.
///
/// Returns `[f32; FEATURE_COUNT]`:
/// - `[0..6]`  = player 0 features (live5, dead5, live4, dead4, live3, live2)
/// - `[6..12]` = player 1 features
/// - `[12]`    = tempo (1.0 if player 0 to move, -1.0 if player 1)
pub fn extract_features(game: &HexGameState) -> [f32; FEATURE_COUNT] {
    let mut feats = [0.0f32; FEATURE_COUNT];
    let mut counts = [[0i32; FEATURES_PER_PLAYER]; 2];

    // We iterate over all occupied cells but only count each run once.
    // A run is counted from the cell that is the "start" of the run
    // (the cell in the negative direction is NOT the same player).
    for (&cell, &player) in &game.board {
        let p = player as usize;
        for &(dq, dr) in &HEX_DIRECTIONS {
            // Only count from the start of a run: check that the cell
            // before us (in the negative direction) is not the same player.
            let prev = Hex::new(cell.q - dq, cell.r - dr);
            if game.board.get(&prev) == Some(&player) {
                continue; // not the start of this run
            }

            // Count forward from this cell (including this cell).
            let (fwd, fwd_open) = count_run(game, cell, dq, dr, player);
            let run_len = 1 + fwd; // includes this cell

            // Check backward end (the cell before `cell` in -direction).
            let bwd_open = match game.board.get(&prev) {
                None => true,
                Some(_) => false, // opponent piece (we already excluded same-player above)
            };

            let open_ends = (bwd_open as i32) + (fwd_open as i32);

            // Classify the run.
            if run_len >= WIN_LENGTH {
                // Already won — shouldn't happen during eval normally, but handle it.
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
    feats[FEATURE_COUNT - 1] = if game.current_player == 0 { 1.0 } else { -1.0 };
    feats
}

// -------------------------------------------------------------------------
// Classical evaluation
// -------------------------------------------------------------------------

/// Evaluate the position from the perspective of `player`.
///
/// Reads pre-computed incremental window evaluation from the game state.
/// O(1) per call — all window scanning is done incrementally during place/unmake.
pub fn evaluate(game: &HexGameState, player: u8) -> i32 {
    if let Some(w) = game.winner {
        return if w == player { WIN_SCORE } else { -WIN_SCORE };
    }

    let our_turn = game.current_player == player;
    let placements = game.placements_remaining as i32;

    // Base score from pattern table (stored from player-0 perspective).
    let mut score = if player == 0 {
        game.window_eval
    } else {
        -game.window_eval
    };

    let my_fives = game.window_fives[player as usize];
    let opp_fives = game.window_fives[1 - player as usize];
    let my_fours = game.window_fours[player as usize];
    let opp_fours = game.window_fours[1 - player as usize];

    // ── Threat analysis ──────────────────────────────────────────────

    // Near-win (5-in-a-row in window): essentially won/lost.
    if my_fives > 0 {
        if our_turn {
            return WIN_SCORE - 10;
        }
        if my_fives > 1 {
            return WIN_SCORE - 15;
        }
        score += 50_000;
    }

    if opp_fives > 0 {
        if !our_turn {
            return -(WIN_SCORE - 10);
        }
        if opp_fives > placements {
            return -(WIN_SCORE - 15);
        }
        score -= 50_000 * opp_fives;
    }

    // Opponent four-threats: dangerous regardless of turn.
    if opp_fours >= 2 {
        if !our_turn {
            score -= 35_000;
        } else {
            score -= 20_000;
        }
    }
    if opp_fours >= 1 {
        if !our_turn {
            score -= 15_000;
        } else {
            score -= 6_000;
        }
    }

    // Our four-threats: only valuable if we can act.
    if my_fours >= 2 && our_turn && placements >= 2 {
        score += 30_000;
    }
    if my_fours >= 1 && our_turn && placements >= 2 {
        score += 12_000;
    }

    if our_turn {
        score += 15;
    }

    score
}

// -------------------------------------------------------------------------
// Forcing move detection (for quiescence search)
// -------------------------------------------------------------------------

/// Returns true if placing at `cell` is a "forcing" move — one that creates
/// or blocks a significant threat.
///
/// Specifically, returns true when the placement:
/// - Wins the game immediately (creates a run of WIN_LENGTH)
/// - Creates a live-5 (run of WIN_LENGTH-1, one placement from winning)
/// - Creates a live-4 (run of WIN_LENGTH-2 with both ends open)
/// - Blocks the opponent from completing a run of WIN_LENGTH
/// - Blocks the opponent's live-5 threat (run of WIN_LENGTH-1)
pub fn is_forcing_move(game: &HexGameState, cell: Hex, player: u8) -> bool {
    let opp = 1 - player;
    for &(dq, dr) in &HEX_DIRECTIONS {
        let (fwd, fwd_open) = count_run(game, cell, dq, dr, player);
        let (bwd, bwd_open) = count_run(game, cell, -dq, -dr, player);
        let run = 1 + fwd + bwd;
        let open_ends = (fwd_open as i32) + (bwd_open as i32);

        if run >= WIN_LENGTH {
            return true;
        } // immediate win
        if run >= WIN_LENGTH - 1 {
            return true;
        } // creates live-5
        if run == WIN_LENGTH - 2 && open_ends == 2 {
            return true;
        } // creates live-4

        let (ofwd, _) = count_run(game, cell, dq, dr, opp);
        let (obwd, _) = count_run(game, cell, -dq, -dr, opp);
        let opp_run = 1 + ofwd + obwd;

        if opp_run >= WIN_LENGTH {
            return true;
        } // blocks opponent win
        if opp_run >= WIN_LENGTH - 1 {
            return true;
        } // blocks opponent live-5
    }
    false
}

// -------------------------------------------------------------------------
// Move scoring (for move ordering in search)
// -------------------------------------------------------------------------

/// Quick heuristic score for placing `player`'s tile at `cell`.
///
/// Used for move ordering — does NOT do a full eval. Higher is better.
pub fn score_move(game: &HexGameState, cell: Hex, player: u8) -> i32 {
    let mut score = 0i32;

    for &(dq, dr) in &HEX_DIRECTIONS {
        let (fwd, _fwd_open) = count_run(game, cell, dq, dr, player);
        let (bwd, _bwd_open) = count_run(game, cell, -dq, -dr, player);
        let run = 1 + fwd + bwd;

        if run >= WIN_LENGTH {
            return 1_000_000; // winning move
        }
        score += run * run * 10; // prefer extending longer lines

        // Check opponent threat at this cell.
        let opp = 1 - player;
        let (ofwd, _) = count_run(game, cell, dq, dr, opp);
        let (obwd, _) = count_run(game, cell, -dq, -dr, opp);
        let opp_run = 1 + ofwd + obwd;
        if opp_run >= WIN_LENGTH {
            score += 500_000; // blocking opponent win
        } else if opp_run >= 4 {
            score += opp_run as i32 * 200; // blocking a threat
        }
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_board_eval_is_zero() {
        let game = HexGameState::new();
        // On an empty board, both players have the same features (nothing).
        assert_eq!(evaluate(&game, 0), 15); // just tempo bonus
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

    #[test]
    fn is_forcing_detects_win() {
        let mut game = HexGameState::new();
        // Build player 0 with 5-in-a-row (live-5): (0,0)..(4,0)
        game.place(0, 0).unwrap();
        game.place(0, 2).unwrap();
        game.place(1, 2).unwrap(); // P1 filler
        game.place(1, 0).unwrap();
        game.place(2, 0).unwrap();
        game.place(0, 3).unwrap();
        game.place(1, 3).unwrap(); // P1 filler
        game.place(3, 0).unwrap();
        game.place(4, 0).unwrap();
        // Now P0 has (0,0),(1,0),(2,0),(3,0),(4,0) = 5 in a row
        // Placing at (5,0) would win
        assert!(is_forcing_move(&game, Hex::new(5, 0), 0));
        // Placing far away should not be forcing
        assert!(!is_forcing_move(&game, Hex::new(0, 5), 0));
    }
}
