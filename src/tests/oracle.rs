//! Brute-force oracle for threat-analysis verification.
//!
//! The oracle is a **test-only** reference implementation. It exhaustively
//! enumerates every legal turn for a given board position, simulates each
//! turn to completion, and classifies the result. This produces a ground-truth
//! [`TurnAnalysis`] that can be compared against the engine's fast incremental
//! threat path to detect bugs or semantic drift.
//!
//! ## Why brute-force?
//!
//! The fast path ([`threat_status`](crate::threats::threat_status)) uses
//! incremental pattern counts and carefully pruned logic to classify positions
//! in microseconds. The oracle does none of that optimization — it simply tries
//! every move. Because the two implementations share almost no code, any
//! agreement between them is strong evidence of correctness.
//!
//! ## Performance characteristics
//!
//! - **Time complexity:** `O(C² · W)` where `C` is the number of candidate
//!   cells (typically ≤ 50) and `W` is the cost of win detection.
//! - **Space complexity:** `O(C²)` for the output vectors.
//! - A single `analyse` call on a mid-game position takes **milliseconds**,
//!   which is fine for tests but unusable in a search loop.
//! - The oracle internally clones and mutates the game state, then fully
//!   restores it before returning, so it is safe to call on a live position.
//!
//! ## Candidate sets
//!
//! To stay fast while remaining correct, the oracle uses three different
//! candidate sets for the three kinds of analysis:
//!
//! - **Winning turns** — only cells returned by [`live_cells`](crate::threats::live_cells)
//!   for the player to move. A player can only win by completing a hot window,
//!   and every empty cell in a hot window is automatically a legal move (it is
//!   within `PLACEMENT_RADIUS` of the stones that make the window hot).
//! - **Blocking turns** — only cells from the opponent's `live_cells` set are
//!   relevant for blocking. The oracle also checks every *other* legal cell as
//!   the companion placement, because a pair that contains one intersection cell
//!   and one irrelevant cell is still a valid block.
//! - **`legal`** — a representative sample built from `candidates_near2()`
//!   plus all winning/blocking turns, so the property tests exercise every
//!   must-play turn.

use crate::board::HexGameState;
use crate::core::{Hex, Turn};
use crate::threats::live_cells;

/// Exhaustive classification of all legal turns from a single position.
///
/// Populated by [`analyse`].
#[derive(Debug, PartialEq, Eq, Default)]
pub struct TurnAnalysis {
    /// A representative set of legal turns that includes all radius-2
    /// candidates plus every winning or blocking turn discovered by the oracle.
    pub legal: Vec<Turn>,
    /// Turns that win immediately for the player to move.
    pub winning: Vec<Turn>,
    /// Single cells that block the opponent from winning on their next turn.
    /// Only meaningful when `placements_remaining == 1`.
    pub blocking_single: Vec<Hex>,
    /// Two-cell turns that block the opponent from winning on their next turn.
    /// Only meaningful when `placements_remaining == 2`.
    pub blocking_pairs: Vec<Turn>,
}

/// Run the brute-force oracle on `game`.
///
/// The function simulates **every** relevant turn, checks whether the turn
/// wins for the current player, and then checks whether the *opponent* has
/// any winning response. A turn is classified as:
///
/// - `winning` — the current player wins immediately.
/// - `blocking_single` / `blocking_pairs` — the opponent has no winning
///   response after this turn.
///
/// # Invariant
///
/// `game` is returned in exactly the same state it had on entry
/// (player, placements remaining, winner, move count, zobrist hash).
pub fn analyse(game: &mut HexGameState) -> TurnAnalysis {
    let me = game.current_player();
    let opp = 1 - me;

    // Relevant cells for winning and blocking analysis.
    let mut live_me = Vec::new();
    live_cells(game, me, &mut live_me);

    let mut live_opp = Vec::new();
    live_cells(game, opp, &mut live_opp);

    // Representative legal sample (radius-2 candidates).
    let near2: Vec<Hex> = game.candidates_near2().into_iter().collect();

    // Full legal set — used for the companion placement in blocking pairs.
    let legal_all: Vec<Hex> = game.legal_moves().into_iter().collect();

    let mut analysis = TurnAnalysis::default();

    // ── 1. Winning turns ──
    // A player can only win by filling empty cells in their own hot windows.
    for &c1 in &live_me {
        let had_one = game.placements_remaining() == 1;
        game.place_unchecked(c1);
        let c1_wins = game.winner() == Some(me);
        if c1_wins {
            analysis.winning.push(Turn::single(c1));
        }
        if had_one || c1_wins {
            // Turn is over, or the game is already won — no second placement.
            game.unplace();
            continue;
        }
        for &c2 in &live_me {
            if c2 <= c1 { continue; }
            game.place_unchecked(c2);
            if game.winner() == Some(me) {
                analysis.winning.push(Turn::pair(c1, c2));
            }
            game.unplace();
        }
        game.unplace();
    }

    // ── 2. Legal representative sample ──
    for &c1 in &near2 {
        let had_one = game.placements_remaining() == 1;
        if had_one {
            analysis.legal.push(Turn::single(c1));
        } else {
            for &c2 in &near2 {
                if c2 <= c1 { continue; }
                analysis.legal.push(Turn::pair(c1, c2));
            }
        }
    }

    // ── 3. Blocking turns ──
    if game.placements_remaining() == 1 {
        // With one placement, only cells in opponent hot windows can block.
        for &c1 in &live_opp {
            game.place_unchecked(c1);
            let is_block = !any_winning_turn_for(game, opp);
            if is_block {
                analysis.blocking_single.push(c1);
            }
            game.unplace();
        }
    } else {
        // With two placements, at least one cell must be in an opponent hot
        // window to intersect a threat. We iterate over live_opp for the first
        // cell and legal_all for the second, canonicalising each pair so we
        // never simulate the same unordered pair twice.
        let mut seen_pairs = std::collections::HashSet::new();
        for &c1 in &live_opp {
            for &c2 in &legal_all {
                if c1 == c2 { continue; }
                let turn = Turn::pair(c1, c2);
                if !seen_pairs.insert(turn) {
                    continue;
                }
                // Simulate in canonical order.
                game.place_unchecked(turn.first());
                if game.winner() == Some(me) {
                    game.unplace();
                    continue;
                }
                game.place_unchecked(turn.second().unwrap());
                if game.winner() == Some(me) {
                    game.unplace();
                    game.unplace();
                    continue;
                }
                let is_block = !any_winning_turn_for(game, opp);
                if is_block {
                    analysis.blocking_pairs.push(turn);
                }
                game.unplace();
                game.unplace();
            }
        }
    }

    // Merge winning and blocking turns into `legal` so the property tests
    // exercise every must-play turn.
    for turn in &analysis.winning {
        if !analysis.legal.contains(turn) {
            analysis.legal.push(*turn);
        }
    }
    for turn in &analysis.blocking_pairs {
        if !analysis.legal.contains(turn) {
            analysis.legal.push(*turn);
        }
    }
    if game.placements_remaining() == 1 {
        for &cell in &analysis.blocking_single {
            let turn = Turn::single(cell);
            if !analysis.legal.contains(&turn) {
                analysis.legal.push(turn);
            }
        }
    }

    analysis
}

/// Check whether `player` has any winning turn from the current position.
///
/// This is a helper for the oracle. It only examines cells that appear in
/// the player's hot windows (`live_cells`), because a player can only win
/// by filling empty cells in windows that already contain 4+ of their stones.
///
/// The search space is therefore much smaller than a full board scan,
/// although the logic is still brute-force.
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

    /// Player 0 has five stones in a row on the q-axis with two open ends.
    /// The oracle must find both instant-winning single placements.
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

    /// Player 1 has five in a row and it is player 0's turn with one
    /// placement remaining. The oracle must find the single blocking cell.
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

    /// Player 1 has four in a row with open ends and player 0 has two
    /// placements remaining. The oracle must find the pair of cells that
    /// together block player 1 from winning next turn.
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

    /// Player 1 has two independent five-in-a-row threats and player 0 has
    /// only one placement remaining. No single cell can block both threats,
    /// so the oracle must report an empty `blocking_single` list.
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

    /// Player 1 has two independent four-in-a-row threats on separate rows.
    /// Player 0 has two placements remaining, but the threats are far enough
    /// apart that no pair of cells can block both. The oracle must report
    /// an empty `blocking_pairs` list.
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

    /// The oracle must leave the game state untouched after analysis.
    ///
    /// Because `analyse` mutates the game internally (placing and unplacing
    /// stones), a bug in the restore logic would corrupt the caller's state.
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
