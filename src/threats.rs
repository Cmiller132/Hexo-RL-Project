//! Threat analysis for Infinity Hexagonal Tic-Tac-Toe.
//!
//! This module provides free functions for classifying the tactical situation,
//! checking whether a turn satisfies threat constraints, and enumerating live
//! cells.  It replaces the old `impl HexGameState` threat methods.

use smallvec::SmallVec;
use crate::board::HexGameState;
use crate::core::{Hex, Turn, HEX_DIRECTIONS, WIN_LENGTH};

/// Classification of the current tactical situation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThreatStatus {
    /// No immediate threats for either side.
    Quiet,
    /// The current player can force a win this turn.
    WinningTurn(Turn),
    /// The current player must block opponent threats.
    MustBlock(BlockConstraint),
    /// Opponent threats cannot be blocked with the remaining placements.
    Unblockable,
}

/// Exact blocking constraint when the opponent has immediate threats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockConstraint {
    /// Single cells that block every threat window (used when 1 placement remains).
    pub cells: SmallVec<[Hex; 16]>,
    /// Valid pairs of cells that together block every threat window (used when
    /// 2 placements remain).  Pairs are stored in canonical order (`c1 <= c2`).
    pub pairs: SmallVec<[(Hex, Hex); 32]>,
    /// Cells that appear in at least one valid pair. Used for the single-
    /// placement-is-acceptable-when-two-remain heuristic.
    pub union_cells: SmallVec<[Hex; 16]>,
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

/// Collect opponent hot windows together with their empty cells.
///
/// Only windows with at least one empty cell are returned (fully-occupied
/// 6-windows are filtered out).
fn opponent_threat_windows(game: &HexGameState) -> SmallVec<[SmallVec<[Hex; 2]>; 16]> {
    let opp = 1 - game.current_player();
    let counts = game.eval().counts(opp);
    if counts.fours == 0 && counts.fives == 0 {
        return SmallVec::new();
    }

    let mut result = SmallVec::new();
    for key in game.eval().hot_windows(opp) {
        let (dq, dr) = HEX_DIRECTIONS[key.dir() as usize];
        let mut empties = SmallVec::<[Hex; 2]>::new();
        for k in 0..WIN_LENGTH {
            let h = Hex::new(key.q() + dq * k, key.r() + dr * k);
            if !game.stones().contains_key(&h) {
                empties.push(h);
            }
        }
        if !empties.is_empty() {
            result.push(empties);
        }
    }
    result
}

// -------------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------------

/// Classify the current tactical situation.
pub fn threat_status(game: &HexGameState) -> ThreatStatus {
    // 1. Game over?
    if game.winner().is_some() {
        return ThreatStatus::Quiet;
    }

    // 2. Fast exit: no fives or fours for either player.
    let current = game.current_player();
    let curr_counts = game.eval().counts(current);
    let opp = 1 - current;
    let opp_counts = game.eval().counts(opp);
    if curr_counts.fives == 0
        && curr_counts.fours == 0
        && opp_counts.fives == 0
        && opp_counts.fours == 0
    {
        return ThreatStatus::Quiet;
    }

    // 3. Can the current player win immediately?
    let remaining = game.placements_remaining();
    for key in game.eval().hot_windows(current) {
        let (dq, dr) = HEX_DIRECTIONS[key.dir() as usize];
        let mut empties = SmallVec::<[Hex; 2]>::new();
        for k in 0..WIN_LENGTH {
            let h = Hex::new(key.q() + dq * k, key.r() + dr * k);
            if !game.stones().contains_key(&h) {
                empties.push(h);
            }
        }
        match empties.len() {
            1 => return ThreatStatus::WinningTurn(Turn::single(empties[0])),
            2 if remaining >= 2 => {
                return ThreatStatus::WinningTurn(Turn::pair(empties[0], empties[1]));
            }
            _ => {}
        }
    }

    // 4. Opponent threats?
    let must_hit = opponent_threat_windows(game);
    if must_hit.is_empty() {
        return ThreatStatus::Quiet;
    }

    // 5. Build exact BlockConstraint.
    let placements = game.placements_remaining();

    // Collect all unique empty cells across threat windows.
    let mut all_cells: Vec<Hex> =
        must_hit.iter().flat_map(|w| w.iter().copied()).collect();
    all_cells.sort();
    all_cells.dedup();

    // Intersection: cells that appear in EVERY threat window.
    // These single-handedly block all threats, so any turn containing one
    // of them is valid regardless of the other placement(s).
    let mut cells = SmallVec::<[Hex; 16]>::new();
    for &cell in &all_cells {
        if must_hit.iter().all(|w| w.contains(&cell)) {
            cells.push(cell);
        }
    }

    if placements <= 1 {
        if cells.is_empty() {
            return ThreatStatus::Unblockable;
        }
        let pairs = SmallVec::<[(Hex, Hex); 32]>::new();
        let union_cells = SmallVec::<[Hex; 16]>::new();
        return ThreatStatus::MustBlock(BlockConstraint { cells, pairs, union_cells });
    }

    // placements >= 2
    let mut pairs = SmallVec::<[(Hex, Hex); 32]>::new();
    for i in 0..all_cells.len() {
        for j in (i + 1)..all_cells.len() {
            let c1 = all_cells[i];
            let c2 = all_cells[j];
            debug_assert_ne!(c1, c2, "self-pair detected in BlockConstraint enumeration");
            if must_hit.iter().all(|w| w.contains(&c1) || w.contains(&c2)) {
                pairs.push((c1, c2));
            }
        }
    }

    if pairs.is_empty() && cells.is_empty() {
        return ThreatStatus::Unblockable;
    }

    // Union of all cells appearing in any valid pair.
    let mut union_cells = SmallVec::<[Hex; 16]>::new();
    for &(c1, c2) in &pairs {
        if !union_cells.contains(&c1) {
            union_cells.push(c1);
        }
        if !union_cells.contains(&c2) {
            union_cells.push(c2);
        }
    }

    ThreatStatus::MustBlock(BlockConstraint { cells, pairs, union_cells })
}

/// Check whether a single turn is legal under threat constraints,
/// given a pre-computed threat status.
pub fn turn_satisfies_status(status: &ThreatStatus, turn: Turn) -> bool {
    match status {
        ThreatStatus::Quiet => true,
        ThreatStatus::WinningTurn(w) => turn == *w,
        ThreatStatus::MustBlock(b) => {
            if turn.placements() == 1 {
                // When 2 placements remain, a single placement is acceptable if
                // it is part of at least one valid blocking pair.
                b.cells.contains(&turn.first()) || b.union_cells.contains(&turn.first())
            } else {
                let second = turn.second().unwrap();
                // If either cell alone blocks every threat window, any pair
                // containing it is valid.
                if b.cells.contains(&turn.first()) || b.cells.contains(&second) {
                    return true;
                }
                b.pairs.iter().any(|&(a, b)| {
                    (a == turn.first() && b == second) || (a == second && b == turn.first())
                })
            }
        }
        ThreatStatus::Unblockable => true,
    }
}

/// Check whether a single turn is legal under threat constraints.
pub fn turn_satisfies_threats(game: &HexGameState, turn: Turn) -> bool {
    if game.winner().is_some() {
        return true;
    }
    turn_satisfies_status(&threat_status(game), turn)
}

/// Cells that are tactically live for the given player.
///
/// Clears `out` and pushes every empty cell that appears in at least one of
/// the player's hot windows.
pub fn live_cells(game: &HexGameState, player: u8, out: &mut Vec<Hex>) {
    out.clear();
    let counts = game.eval().counts(player);
    if counts.fives == 0 && counts.fours == 0 {
        return;
    }
    for key in game.eval().hot_windows(player) {
        let (dq, dr) = HEX_DIRECTIONS[key.dir() as usize];
        for k in 0..WIN_LENGTH {
            let h = Hex::new(key.q() + dq * k, key.r() + dr * k);
            if !game.stones().contains_key(&h) && !out.contains(&h) {
                out.push(h);
            }
        }
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::HexGameState;
    use crate::core::Hex;

    // ── Winning threat cells (5-window and 4-window) ──────────────────────

    #[test]
    fn winning_turn_five_window() {
        let mut game = HexGameState::new();
        // P0 has a 5-stone run along (1,0): (0,0)..(4,0).
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)],
            0,
            2,
        )
        .unwrap();

        match threat_status(&game) {
            ThreatStatus::WinningTurn(t) => {
                // With remaining=2 the first hot window is the 5-window at
                // origin (0,0) whose only empty is (5,0).
                assert_eq!(t.placements(), 1);
                assert_eq!(t.first(), Hex::new(5, 0));
            }
            other => panic!("expected WinningTurn, got {:?}", other),
        }
    }

    #[test]
    fn winning_turn_four_window_with_two_placements() {
        let mut game = HexGameState::new();
        // P0 has a 4-stone run along (1,0): (0,0)..(3,0).
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
            0,
            2,
        )
        .unwrap();

        match threat_status(&game) {
            ThreatStatus::WinningTurn(t) => {
                // The first hot window is at origin (0,0) with empties (4,0),(5,0).
                assert_eq!(t.placements(), 2);
                assert_eq!(t.first(), Hex::new(4, 0));
                assert_eq!(t.second(), Some(Hex::new(5, 0)));
            }
            other => panic!("expected WinningTurn, got {:?}", other),
        }
    }

    #[test]
    fn no_winning_turn_with_one_placement_on_four_window() {
        let mut game = HexGameState::new();
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
            0,
            1,
        )
        .unwrap();
        assert!(matches!(threat_status(&game), ThreatStatus::Quiet));
    }

    // ── Blocking single placement ─────────────────────────────────────────

    #[test]
    fn block_constraint_single_placement_intersection() {
        let mut game = HexGameState::new();
        // P1 has a 4-stone run (0,0)..(3,0) and P0 already blocked one end at (-2,0).
        game.set_position(
            &[
                (-2, 0, 0), // P0 blocker
                (0, 0, 1),
                (1, 0, 1),
                (2, 0, 1),
                (3, 0, 1),
            ],
            0,
            1,
        )
        .unwrap();

        match threat_status(&game) {
            ThreatStatus::MustBlock(b) => {
                assert_eq!(b.cells.len(), 1);
                assert_eq!(b.cells[0], Hex::new(4, 0));
                assert!(b.pairs.is_empty());
            }
            other => panic!("expected MustBlock, got {:?}", other),
        }
    }

    // ── Blocking with two placements (exact pair enumeration) ─────────────

    #[test]
    fn block_constraint_two_placements_exact_pairs() {
        let mut game = HexGameState::new();
        // P1 bare 4-run (0,0)..(3,0). P0 has 2 placements.
        game.set_position(
            &[(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)],
            0,
            2,
        )
        .unwrap();

        match threat_status(&game) {
            ThreatStatus::MustBlock(b) => {
                // Valid covering pairs for the three hot windows.
                assert!(b.pairs.contains(&(Hex::new(-2, 0), Hex::new(4, 0))));
                assert!(b.pairs.contains(&(Hex::new(-1, 0), Hex::new(4, 0))));
                assert!(b.pairs.contains(&(Hex::new(-1, 0), Hex::new(5, 0))));

                // Invalid pairs must not be present.
                assert!(!b.pairs.contains(&(Hex::new(-2, 0), Hex::new(5, 0))));
                assert!(!b.pairs.contains(&(Hex::new(4, 0), Hex::new(5, 0))));

                // Union of all cells that appear in any valid pair.
                assert_eq!(b.union_cells.len(), 4);
                assert!(b.union_cells.contains(&Hex::new(-2, 0)));
                assert!(b.union_cells.contains(&Hex::new(-1, 0)));
                assert!(b.union_cells.contains(&Hex::new(4, 0)));
                assert!(b.union_cells.contains(&Hex::new(5, 0)));
            }
            other => panic!("expected MustBlock, got {:?}", other),
        }
    }

    // ── Unblockable detection ─────────────────────────────────────────────

    #[test]
    fn unblockable_single_placement_disjoint_threats() {
        let mut game = HexGameState::new();
        // P1 has two disjoint 5-stone runs. P0 has only 1 placement.
        game.set_position(
            &[
                (0, 0, 1),
                (1, 0, 1),
                (2, 0, 1),
                (3, 0, 1),
                (4, 0, 1),
                (10, 0, 1),
                (11, 0, 1),
                (12, 0, 1),
                (13, 0, 1),
                (14, 0, 1),
            ],
            0,
            1,
        )
        .unwrap();

        assert!(matches!(threat_status(&game), ThreatStatus::Unblockable));
    }

    #[test]
    fn unblockable_two_placements_disjoint_five_windows() {
        let mut game = HexGameState::new();
        // P1 has two disjoint 5-runs. P0 has 2 placements.
        game.set_position(
            &[
                (0, 0, 1),
                (1, 0, 1),
                (2, 0, 1),
                (3, 0, 1),
                (4, 0, 1),
                (10, 0, 1),
                (11, 0, 1),
                (12, 0, 1),
                (13, 0, 1),
                (14, 0, 1),
            ],
            0,
            2,
        )
        .unwrap();

        assert!(matches!(threat_status(&game), ThreatStatus::Unblockable));
    }

    #[test]
    fn not_unblockable_when_common_cell_exists() {
        let mut game = HexGameState::new();
        // P1 has a 4-run (0,0)..(3,0) and P0 (current player) already blocked one end at (-2,0).
        game.set_position(
            &[
                (-2, 0, 0), // P0 blocker
                (0, 0, 1),
                (1, 0, 1),
                (2, 0, 1),
                (3, 0, 1),
            ],
            0,
            1,
        )
        .unwrap();

        match threat_status(&game) {
            ThreatStatus::MustBlock(b) => {
                assert!(b.cells.contains(&Hex::new(4, 0)));
            }
            other => panic!("expected MustBlock, got {:?}", other),
        }
    }

    // ── turn_satisfies_threats ────────────────────────────────────────────

    #[test]
    fn turn_satisfies_threats_own_win() {
        let mut game = HexGameState::new();
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)],
            0,
            2,
        )
        .unwrap();

        let status = threat_status(&game);
        let winning = match status {
            ThreatStatus::WinningTurn(t) => t,
            _ => panic!("expected winning turn"),
        };

        assert!(turn_satisfies_threats(&game, winning));
        assert!(!turn_satisfies_threats(&game, Turn::single(Hex::new(100, 0))));
    }

    #[test]
    fn turn_satisfies_threats_must_block_single() {
        let mut game = HexGameState::new();
        game.set_position(
            &[
                (-2, 0, 0), // P0 blocker
                (0, 0, 1),
                (1, 0, 1),
                (2, 0, 1),
                (3, 0, 1),
            ],
            0,
            1,
        )
        .unwrap();

        assert!(turn_satisfies_threats(&game, Turn::single(Hex::new(4, 0))));
        assert!(!turn_satisfies_threats(&game, Turn::single(Hex::new(-1, 0))));
    }

    #[test]
    fn turn_satisfies_threats_must_block_pair() {
        let mut game = HexGameState::new();
        game.set_position(
            &[(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)],
            0,
            2,
        )
        .unwrap();

        // Valid blocking pairs
        assert!(turn_satisfies_threats(
            &game,
            Turn::pair(Hex::new(-1, 0), Hex::new(4, 0))
        ));
        assert!(turn_satisfies_threats(
            &game,
            Turn::pair(Hex::new(-1, 0), Hex::new(5, 0))
        ));
        assert!(turn_satisfies_threats(
            &game,
            Turn::pair(Hex::new(-2, 0), Hex::new(4, 0))
        ));

        // Invalid pair
        assert!(!turn_satisfies_threats(
            &game,
            Turn::pair(Hex::new(-2, 0), Hex::new(5, 0))
        ));

        // Single placement at a cell in the union is accepted (per API spec).
        assert!(turn_satisfies_threats(&game, Turn::single(Hex::new(-1, 0))));
        // Single placement outside the union is rejected.
        assert!(!turn_satisfies_threats(&game, Turn::single(Hex::new(100, 0))));
    }

    #[test]
    fn turn_satisfies_threats_unblockable_returns_true() {
        let mut game = HexGameState::new();
        game.set_position(
            &[
                (0, 0, 1),
                (1, 0, 1),
                (2, 0, 1),
                (3, 0, 1),
                (4, 0, 1),
                (10, 0, 1),
                (11, 0, 1),
                (12, 0, 1),
                (13, 0, 1),
                (14, 0, 1),
            ],
            0,
            1,
        )
        .unwrap();

        // Unblockable means the threat filter does not constrain moves.
        assert!(turn_satisfies_threats(&game, Turn::single(Hex::new(5, 0))));
        assert!(turn_satisfies_threats(&game, Turn::single(Hex::new(100, 0))));
    }

    // ── live_cells ────────────────────────────────────────────────────────

    #[test]
    fn live_cells_five_window() {
        let mut game = HexGameState::new();
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)],
            0,
            2,
        )
        .unwrap();

        let mut cells = Vec::new();
        live_cells(&game, 0, &mut cells);
        assert_eq!(cells.len(), 4);
        assert!(cells.contains(&Hex::new(-2, 0)));
        assert!(cells.contains(&Hex::new(-1, 0)));
        assert!(cells.contains(&Hex::new(5, 0)));
        assert!(cells.contains(&Hex::new(6, 0)));
    }

    #[test]
    fn live_cells_four_window() {
        let mut game = HexGameState::new();
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
            0,
            2,
        )
        .unwrap();

        let mut cells = Vec::new();
        live_cells(&game, 0, &mut cells);
        assert_eq!(cells.len(), 4);
        assert!(cells.contains(&Hex::new(-2, 0)));
        assert!(cells.contains(&Hex::new(-1, 0)));
        assert!(cells.contains(&Hex::new(4, 0)));
        assert!(cells.contains(&Hex::new(5, 0)));
    }

    #[test]
    fn live_cells_empty_when_no_threats() {
        let mut game = HexGameState::new();
        game.set_position(&[(0, 0, 0), (1, 0, 0), (2, 0, 0)], 0, 2).unwrap();

        let mut cells = Vec::new();
        live_cells(&game, 0, &mut cells);
        assert!(cells.is_empty());
    }

    // ── Edge cases ────────────────────────────────────────────────────────

    #[test]
    fn blocked_window_is_not_hot() {
        let mut game = HexGameState::new();
        game.set_position(
            &[
                (-1, 0, 0),
                (0, 0, 0),
                (2, 0, 1), // P1 blocker inside
                (3, 0, 0),
                (4, 0, 0),
            ],
            0,
            2,
        )
        .unwrap();

        let mut cells = Vec::new();
        live_cells(&game, 0, &mut cells);
        // No hot window should contain the opponent stone.
        assert!(!cells.contains(&Hex::new(2, 0)));
        // With the block there are no hot windows for P0.
        assert!(cells.is_empty());
    }

    #[test]
    fn three_window_is_not_hot() {
        let mut game = HexGameState::new();
        game.set_position(&[(0, 0, 0), (1, 0, 0), (2, 0, 0)], 0, 2).unwrap();

        assert!(game.eval().hot_is_empty(0));
        assert!(matches!(threat_status(&game), ThreatStatus::Quiet));
    }

    #[test]
    fn game_over_is_quiet() {
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

        assert!(game.winner().is_some());
        assert!(matches!(threat_status(&game), ThreatStatus::Quiet));
        assert!(turn_satisfies_threats(&game, Turn::single(Hex::new(0, 0))));
    }

    #[test]
    fn overlapping_hot_windows_share_empties() {
        let mut game = HexGameState::new();
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)],
            0,
            1,
        )
        .unwrap();

        let mut cells = Vec::new();
        live_cells(&game, 0, &mut cells);
        assert!(cells.contains(&Hex::new(-1, 0)));
        assert!(cells.contains(&Hex::new(5, 0)));
    }
}
