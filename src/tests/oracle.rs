use crate::board::HexGameState;
use crate::core::{Hex, Turn};
use crate::threats::live_cells;

#[derive(Debug, PartialEq, Eq, Default)]
pub struct TurnAnalysis {
    pub legal: Vec<Turn>,
    pub winning: Vec<Turn>,
    pub blocking_single: Vec<Hex>,   // only if placements_remaining == 1
    pub blocking_pairs: Vec<Turn>,   // only if placements_remaining == 2
}

/// THE oracle. Brute-force enumerate all legal turns, classify each.
/// Test-only — never called from production.
pub fn analyse(game: &mut HexGameState) -> TurnAnalysis {
    let me = game.current_player();
    let opp = 1 - me;
    let candidates: Vec<Hex> = game.candidates_near2().into_iter().collect();

    let mut analysis = TurnAnalysis::default();

    for &c1 in &candidates {
        let had_one = game.placements_remaining() == 1;
        game.place_unchecked(c1);
        if game.winner() == Some(me) {
            analysis.winning.push(Turn::single(c1));
            analysis.legal.push(Turn::single(c1));
            game.unplace();
            continue;
        }
        if had_one {
            // Turn over, not a win.
            let is_block = !any_winning_turn_for(game, opp);
            if is_block { analysis.blocking_single.push(c1); }
            analysis.legal.push(Turn::single(c1));
            game.unplace();
            continue;
        }
        // Second placement.
        for &c2 in &candidates {
            if c2 <= c1 { continue; }  // canonical ordering
            game.place_unchecked(c2);
            let turn = Turn::pair(c1, c2);
            if game.winner() == Some(me) {
                analysis.winning.push(turn);
                analysis.legal.push(turn);
            } else {
                let is_block = !any_winning_turn_for(game, opp);
                if is_block { analysis.blocking_pairs.push(turn); }
                analysis.legal.push(turn);
            }
            game.unplace();
        }
        game.unplace();
    }

    analysis
}

/// Check if `player` has any winning turn from the current position.
/// Only checks cells in the player's hot windows — a player can only win by
/// filling empty cells in windows that already have 4+ of their stones.
fn any_winning_turn_for(game: &mut HexGameState, player: u8) -> bool {
    let mut candidates: Vec<Hex> = Vec::new();
    live_cells(game, player, &mut candidates);
    // If there are no hot windows the player cannot win this turn.
    if candidates.is_empty() {
        return false;
    }
    for &c1 in &candidates {
        let had_one = game.placements_remaining() == 1;
        game.place_unchecked(c1);
        if game.winner() == Some(player) {
            game.unplace();
            return true;
        }
        if !had_one {
            for &c2 in &candidates {
                if c2 <= c1 { continue; }
                game.place_unchecked(c2);
                let wins = game.winner() == Some(player);
                game.unplace();
                if wins {
                    game.unplace();
                    return true;
                }
            }
        }
        game.unplace();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_finds_winning_single() {
        let mut g = HexGameState::new();
        let stones = &[
            (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0),
            (0, 1, 1), (1, 1, 1),
        ];
        g.set_position(stones, 0, 2).unwrap();
        let analysis = analyse(&mut g);
        assert!(analysis.winning.contains(&Turn::single(Hex::new(5, 0))));
        assert!(analysis.winning.contains(&Turn::single(Hex::new(-1, 0))));
    }

    #[test]
    fn oracle_finds_blocking_single() {
        let mut g = HexGameState::new();
        let stones = &[
            (-1, 0, 0),
            (0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1), (4, 0, 1),
            (0, 1, 0), (1, 1, 0),
        ];
        g.set_position(stones, 0, 1).unwrap();
        let analysis = analyse(&mut g);
        assert!(analysis.blocking_single.contains(&Hex::new(5, 0)));
    }

    #[test]
    fn oracle_finds_blocking_pair() {
        let mut g = HexGameState::new();
        let stones = &[
            (0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1),
            (10, 0, 0), (10, 1, 0),
        ];
        g.set_position(stones, 0, 2).unwrap();
        let analysis = analyse(&mut g);
        let block = Turn::pair(Hex::new(-1, 0), Hex::new(4, 0));
        assert!(analysis.blocking_pairs.contains(&block));
    }

    #[test]
    fn oracle_unblockable_single() {
        let mut g = HexGameState::new();
        let stones = &[
            (-1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1), (4, 0, 1),
            (-1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1), (4, 1, 1),
        ];
        g.set_position(stones, 0, 1).unwrap();
        let analysis = analyse(&mut g);
        assert!(analysis.blocking_single.is_empty());
        assert!(analysis.winning.is_empty());
    }

    #[test]
    fn oracle_unblockable_pair() {
        let mut g = HexGameState::new();
        let stones = &[
            (0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1),
            (0, 2, 1), (1, 2, 1), (2, 2, 1), (3, 2, 1),
            (10, 0, 0), (10, 1, 0),
        ];
        g.set_position(stones, 0, 2).unwrap();
        let analysis = analyse(&mut g);
        assert!(analysis.blocking_pairs.is_empty());
        assert!(analysis.winning.is_empty());
    }

    #[test]
    fn oracle_restores_game_state() {
        let mut g = HexGameState::new();
        let stones = &[
            (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0),
            (0, 1, 1), (1, 1, 1),
        ];
        g.set_position(stones, 0, 2).unwrap();

        let before_player = g.current_player();
        let before_remaining = g.placements_remaining();
        let before_winner = g.winner();
        let before_count = g.move_count();
        let before_zobrist = g.zobrist();

        let _ = analyse(&mut g);

        assert_eq!(g.current_player(), before_player);
        assert_eq!(g.placements_remaining(), before_remaining);
        assert_eq!(g.winner(), before_winner);
        assert_eq!(g.move_count(), before_count);
        assert_eq!(g.zobrist(), before_zobrist);
    }
}
