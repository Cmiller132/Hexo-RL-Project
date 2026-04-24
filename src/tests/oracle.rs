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
//! candidate sets for the three kinds of analysis.  **All three sets are
//! independent of the fast-path `live_cells` function** so that the oracle
//! remains a true ground-truth verifier.
//!
//! - **Winning turns** — every cell in `game.candidates_near2()` is examined.
//!   Any winning placement must be adjacent to an existing stone (it fills an
//!   empty cell in a hot window), so the radius-2 superset is guaranteed to
//!   contain every possible winning cell.
//! - **Blocking turns** — the first placement is drawn from the same
//!   radius-2 superset (`candidates_near2()`), because a block must intersect
//!   an opponent hot window.  The companion placement in a pair is drawn from
//!   the full legal set (`legal_moves()`), since an irrelevant second cell can
//!   still form a valid blocking pair together with an intersecting first cell.
//! - **`legal`** — a representative sample built from `candidates_near2()`
//!   plus all winning/blocking turns, so the property tests exercise every
//!   must-play turn.

use crate::board::HexGameState;
use crate::core::{hex_distance, Hex, Turn};

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

/// Return the empty cells within distance 2 of any stone owned by `player`.
///
/// This is a **player-specific** subset of `game.candidates_near2()`.  Any
/// cell that can complete a winning line for `player` must lie in a hot window,
/// and every empty cell in a hot window is within distance 1 (hence ≤ 2) of a
/// player stone in that window.  Therefore this set is a safe superset for
/// winning-turn enumeration and is independent of the fast-path `live_cells`.
fn player_candidates_near2(game: &HexGameState, player: u8) -> Vec<Hex> {
    let all = game.candidates_near2();
    let player_stones: Vec<Hex> = game
        .stones()
        .iter()
        .filter(|(_, &p)| p == player)
        .map(|(&h, _)| h)
        .collect();

    all.into_iter()
        .filter(|c| player_stones.iter().any(|s| hex_distance(*s, *c) <= 2))
        .collect()
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

    // Player-specific radius-2 supersets for winning / blocking analysis.
    // Using player-specific sets instead of the full `candidates_near2()`
    // reduces the search space by ~2× while remaining independent of
    // `live_cells`.
    let near2_me = player_candidates_near2(game, me);
    let near2_opp = player_candidates_near2(game, opp);

    // Full legal set — used for the companion placement in blocking pairs.
    let legal_all: Vec<Hex> = game.legal_moves().into_iter().collect();

    let mut analysis = TurnAnalysis::default();

    // ── 1. Winning turns ──
    // A player can only win by filling empty cells in their own hot windows.
    // `player_candidates_near2` is a safe superset because every empty cell in
    // a hot window is within distance 1 of an existing player stone.
    for &c1 in &near2_me {
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
        for &c2 in &near2_me {
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
    let near2_all: Vec<Hex> = game.candidates_near2();
    for &c1 in &near2_all {
        let had_one = game.placements_remaining() == 1;
        if had_one {
            analysis.legal.push(Turn::single(c1));
        } else {
            for &c2 in &near2_all {
                if c2 <= c1 { continue; }
                analysis.legal.push(Turn::pair(c1, c2));
            }
        }
    }

    // ── 3. Blocking turns ──
    // Compute all opponent winning turns from the state AFTER the current
    // player completes their turn (opponent to move with 2 placements).
    // Adding current-player stones can only *destroy* opponent winning turns
    // (by occupying cells the opponent would need), never create new ones.
    // Therefore a block is valid iff it intersects every pre-block opponent
    // winning turn.
    let opp_winning = opp_winning_turns_after_turn(game, opp);

    if game.placements_remaining() == 1 {
        // With one placement, only cells in opponent hot windows can block.
        for &c1 in &near2_opp {
            // A winning single for the current player is not a "block".
            if analysis.winning.contains(&Turn::single(c1)) {
                continue;
            }
            // c1 blocks iff it lies in every opponent winning turn.
            let is_block = opp_winning.iter().all(|turn| {
                c1 == turn.first() || turn.second() == Some(c1)
            });
            if is_block {
                analysis.blocking_single.push(c1);
            }
        }
    } else {
        // With two placements, at least one cell must be in an opponent hot
        // window to intersect a threat. We iterate over the opponent-specific
        // radius-2 superset for the first cell and legal_all for the second,
        // canonicalising each pair so we never simulate the same unordered
        // pair twice.
        let mut seen_pairs = std::collections::HashSet::new();
        for &c1 in &near2_opp {
            for &c2 in &legal_all {
                if c1 == c2 { continue; }
                let turn = Turn::pair(c1, c2);
                if !seen_pairs.insert(turn) {
                    continue;
                }
                // Skip if either cell is a winning single for the current player
                // or if the pair itself is a winning pair.
                if analysis.winning.contains(&Turn::single(c1))
                    || analysis.winning.contains(&Turn::single(c2))
                    || analysis.winning.contains(&turn)
                {
                    continue;
                }
                // The pair blocks iff it intersects every opponent winning turn.
                let is_block = opp_winning.iter().all(|ow| {
                    let a = ow.first();
                    if let Some(b) = ow.second() {
                        c1 == a || c2 == a || c1 == b || c2 == b
                    } else {
                        c1 == a || c2 == a
                    }
                });
                if is_block {
                    analysis.blocking_pairs.push(turn);
                }
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

/// Compute all winning turns for `player` from a game state that is identical
/// to `game` except the player to move is `player` with 2 placements remaining.
///
/// This captures the opponent's perspective after the current player finishes
/// their turn, which is what blocking analysis needs.
fn opp_winning_turns_after_turn(game: &HexGameState, player: u8) -> Vec<Turn> {
    let stones: Vec<(i32, i32, u8)> = game
        .stones()
        .iter()
        .map(|(&h, &p)| (h.q, h.r, p))
        .collect();
    let mut g = HexGameState::new();
    g.set_position(&stones, player, 2).expect("oracle: invalid test position");
    all_winning_turns_for(&mut g, player)
}

/// Return every winning turn for `player` from the current position.
///
/// The function examines cells in the player-specific radius-2 superset
/// (`player_candidates_near2`) — a set that is independent of the fast-path
/// `live_cells` function.  Any winning placement must be adjacent to an
/// existing player stone, so this superset is guaranteed to contain every
/// possible winning cell.
///
/// `game` is returned in exactly the same state it had on entry.
fn all_winning_turns_for(game: &mut HexGameState, player: u8) -> Vec<Turn> {
    let candidates = player_candidates_near2(game, player);
    let mut winning = Vec::new();
    for &c1 in &candidates {
        let had_one = game.placements_remaining() == 1;
        game.place_unchecked(c1);
        if game.winner() == Some(player) {
            winning.push(Turn::single(c1));
            game.unplace();
            continue;
        }
        if !had_one {
            for &c2 in &candidates {
                if c2 <= c1 { continue; }
                game.place_unchecked(c2);
                if game.winner() == Some(player) {
                    winning.push(Turn::pair(c1, c2));
                }
                game.unplace();
            }
        }
        game.unplace();
    }
    winning
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
