use crate::board::HexGameState;
use crate::core::{Hex, Turn};
use crate::encoder::encode_board;
use crate::threats::*;

#[cfg(test)]
mod tests {
    use super::*;

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

        match tactical_status(&game) {
            TacticalStatus::WinningTurns(turns) => {
                assert!(turns.contains(&Turn::single(Hex::new(-1, 0))));
                assert!(turns.contains(&Turn::single(Hex::new(5, 0))));
            }
            other => panic!("expected WinningTurns, got {:?}", other),
        }
    }

    #[test]
    fn winning_turn_four_window_with_two_placements() {
        let mut game = HexGameState::new();
        // P0 has a 4-stone run along (1,0): (0,0)..(3,0).
        game.set_position(&[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], 0, 2)
            .unwrap();

        match tactical_status(&game) {
            TacticalStatus::WinningTurns(turns) => {
                assert!(turns.contains(&Turn::pair(Hex::new(-2, 0), Hex::new(-1, 0))));
                assert!(turns.contains(&Turn::pair(Hex::new(-1, 0), Hex::new(4, 0))));
                assert!(turns.contains(&Turn::pair(Hex::new(4, 0), Hex::new(5, 0))));
            }
            other => panic!("expected WinningTurns, got {:?}", other),
        }
    }

    #[test]
    fn no_winning_turn_with_one_placement_on_four_window() {
        let mut game = HexGameState::new();
        game.set_position(&[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], 0, 1)
            .unwrap();
        assert!(matches!(threat_status(&game), ThreatStatus::Quiet));
    }

    // ── Blocking single placement ─────────────────────────────────────────

    #[test]
    fn block_constraint_single_placement_intersection() {
        let mut game = HexGameState::new();
        // P1 has a 4-stone run (3,0)..(6,0). P0 already blocked the left window
        // with stones at (0,0) [origin] and (1,0).
        game.set_position(
            &[
                (0, 0, 0),
                (1, 0, 0),
                (3, 0, 1),
                (4, 0, 1),
                (5, 0, 1),
                (6, 0, 1),
            ],
            0,
            1,
        )
        .unwrap();

        match threat_status(&game) {
            ThreatStatus::MustBlock(b) => {
                assert_eq!(b.cells().len(), 1);
                assert_eq!(b.cells()[0], Hex::new(7, 0));
                assert!(b.pairs().is_empty());
            }
            other => panic!("expected MustBlock, got {:?}", other),
        }
    }

    // ── Blocking with two placements (exact pair enumeration) ─────────────

    #[test]
    fn block_constraint_two_placements_exact_pairs() {
        let mut game = HexGameState::new();
        // P1 bare 4-run (0,0)..(3,0). P0 has 2 placements.
        game.set_position(&[(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)], 0, 2)
            .unwrap();

        match threat_status(&game) {
            ThreatStatus::MustBlock(b) => {
                // Valid covering pairs for the three hot windows.
                assert!(b.pairs().contains(&(Hex::new(-2, 0), Hex::new(4, 0))));
                assert!(b.pairs().contains(&(Hex::new(-1, 0), Hex::new(4, 0))));
                assert!(b.pairs().contains(&(Hex::new(-1, 0), Hex::new(5, 0))));

                // Invalid pairs must not be present.
                assert!(!b.pairs().contains(&(Hex::new(-2, 0), Hex::new(5, 0))));
                assert!(!b.pairs().contains(&(Hex::new(4, 0), Hex::new(5, 0))));
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
        // P1 has a 4-run (3,0)..(6,0) and P0 already blocked the left window
        // with stones at (0,0) [origin] and (1,0). A single blocking cell exists.
        game.set_position(
            &[
                (0, 0, 0),
                (1, 0, 0),
                (3, 0, 1),
                (4, 0, 1),
                (5, 0, 1),
                (6, 0, 1),
            ],
            0,
            1,
        )
        .unwrap();

        match threat_status(&game) {
            ThreatStatus::MustBlock(b) => {
                assert!(b.cells().contains(&Hex::new(7, 0)));
            }
            other => panic!("expected MustBlock, got {:?}", other),
        }
    }

    // ── turn_satisfies_status ────────────────────────────────────────────

    #[test]
    fn turn_satisfies_status_own_win() {
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

        assert!(turn_satisfies_status(&status, winning));
        assert!(!turn_satisfies_status(
            &status,
            Turn::single(Hex::new(100, 0))
        ));
    }

    #[test]
    fn turn_satisfies_status_must_block_single() {
        let mut game = HexGameState::new();
        game.set_position(
            &[
                (0, 0, 0),
                (1, 0, 0),
                (3, 0, 1),
                (4, 0, 1),
                (5, 0, 1),
                (6, 0, 1),
            ],
            0,
            1,
        )
        .unwrap();

        let status = threat_status(&game);
        assert!(turn_satisfies_status(&status, Turn::single(Hex::new(7, 0))));
        assert!(!turn_satisfies_status(
            &status,
            Turn::single(Hex::new(2, 0))
        ));
    }

    #[test]
    fn turn_satisfies_status_must_block_pair() {
        let mut game = HexGameState::new();
        game.set_position(&[(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)], 0, 2)
            .unwrap();

        let status = threat_status(&game);

        // Valid blocking pairs
        assert!(turn_satisfies_status(
            &status,
            Turn::pair(Hex::new(-1, 0), Hex::new(4, 0))
        ));
        assert!(turn_satisfies_status(
            &status,
            Turn::pair(Hex::new(-1, 0), Hex::new(5, 0))
        ));
        assert!(turn_satisfies_status(
            &status,
            Turn::pair(Hex::new(-2, 0), Hex::new(4, 0))
        ));

        // Invalid pair
        assert!(!turn_satisfies_status(
            &status,
            Turn::pair(Hex::new(-2, 0), Hex::new(5, 0))
        ));

        // Single placement cannot block all threats when 2 placements are required.
        assert!(!turn_satisfies_status(
            &status,
            Turn::single(Hex::new(-1, 0))
        ));
        assert!(!turn_satisfies_status(
            &status,
            Turn::single(Hex::new(100, 0))
        ));
    }

    #[test]
    fn turn_satisfies_status_unblockable_returns_true() {
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

        let status = threat_status(&game);
        // Unblockable means the threat filter does not constrain moves.
        assert!(turn_satisfies_status(&status, Turn::single(Hex::new(5, 0))));
        assert!(turn_satisfies_status(
            &status,
            Turn::single(Hex::new(100, 0))
        ));
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
        game.set_position(&[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], 0, 2)
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
        game.set_position(&[(0, 0, 0), (1, 0, 0), (2, 0, 0)], 0, 2)
            .unwrap();

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
                (0, 0, 0),
                (-1, 0, 0),
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
        game.set_position(&[(0, 0, 0), (1, 0, 0), (2, 0, 0)], 0, 2)
            .unwrap();

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
        let status = threat_status(&game);
        assert!(turn_satisfies_status(&status, Turn::single(Hex::new(0, 0))));
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

    #[test]
    fn tactical_status_retains_multiple_winning_singles_for_masks() {
        let mut game = HexGameState::new();
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)],
            0,
            2,
        )
        .unwrap();

        let status = tactical_status(&game);
        let TacticalStatus::WinningTurns(turns) = &status else {
            panic!("expected complete winning turns, got {:?}", status);
        };
        assert!(turns.contains(&Turn::single(Hex::new(-1, 0))));
        assert!(turns.contains(&Turn::single(Hex::new(5, 0))));

        let encoded = encode_board(&game, 2, true);
        assert!(encoded.legal_moves().contains(&Hex::new(-1, 0)));
        assert!(encoded.legal_moves().contains(&Hex::new(5, 0)));
    }

    #[test]
    fn far_grid_four_and_five_threats_use_sparse_scanner() {
        let bridge = [(0, 0, 0), (8, 0, 0), (16, 0, 0), (24, 0, 0), (32, 0, 0)];

        let mut four = HexGameState::new();
        four.set_position(
            &[
                bridge[0],
                bridge[1],
                bridge[2],
                bridge[3],
                bridge[4],
                (40, 0, 1),
                (41, 0, 1),
                (42, 0, 1),
                (43, 0, 1),
            ],
            0,
            2,
        )
        .unwrap();
        assert_eq!(
            four.eval().hot_len(1),
            0,
            "fixture should be outside eval grid"
        );
        match tactical_status(&four) {
            TacticalStatus::MustBlock(block) => {
                assert!(block.pairs().contains(&(Hex::new(38, 0), Hex::new(44, 0))));
                assert!(block.pairs().contains(&(Hex::new(39, 0), Hex::new(44, 0))));
                assert!(block.pairs().contains(&(Hex::new(39, 0), Hex::new(45, 0))));
            }
            other => panic!("expected far-grid MustBlock for four, got {:?}", other),
        }

        let mut five = HexGameState::new();
        five.set_position(
            &[
                bridge[0],
                bridge[1],
                bridge[2],
                bridge[3],
                bridge[4],
                (39, 0, 0),
                (40, 0, 1),
                (41, 0, 1),
                (42, 0, 1),
                (43, 0, 1),
                (44, 0, 1),
            ],
            0,
            1,
        )
        .unwrap();
        assert_eq!(
            five.eval().hot_len(1),
            0,
            "fixture should be outside eval grid"
        );
        match tactical_status(&five) {
            TacticalStatus::MustBlock(block) => {
                assert!(block.cells().contains(&Hex::new(45, 0)));
            }
            other => panic!("expected far-grid MustBlock for five, got {:?}", other),
        }
    }

    #[test]
    fn far_grid_six_is_terminal_before_tactical_filtering() {
        let mut game = HexGameState::new();
        game.set_position(
            &[
                (0, 0, 0),
                (8, 0, 0),
                (16, 0, 0),
                (24, 0, 0),
                (32, 0, 0),
                (40, 0, 1),
                (41, 0, 1),
                (42, 0, 1),
                (43, 0, 1),
                (44, 0, 1),
                (45, 0, 1),
            ],
            0,
            2,
        )
        .unwrap();

        assert_eq!(game.winner(), Some(1));
        assert!(matches!(tactical_status(&game), TacticalStatus::Quiet));
    }

    #[test]
    fn tactical_mask_cells_reports_complete_must_block_and_unblockable_semantics() {
        let mut block = HexGameState::new();
        block
            .set_position(&[(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)], 0, 2)
            .unwrap();
        let status = tactical_status(&block);
        let mut cells = Vec::new();
        assert!(tactical_mask_cells(&status, &mut cells));
        for expected in [
            Hex::new(-2, 0),
            Hex::new(-1, 0),
            Hex::new(4, 0),
            Hex::new(5, 0),
        ] {
            assert!(cells.contains(&expected), "missing mask cell {expected}");
        }

        let mut unblockable = HexGameState::new();
        unblockable
            .set_position(
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
        let status = tactical_status(&unblockable);
        assert!(matches!(status, TacticalStatus::Unblockable));
        assert!(!tactical_mask_cells(&status, &mut cells));
        assert!(cells.is_empty());
    }
}
