use hexgame_core::board::*;
use hexgame_core::core::{Hex, PLACEMENT_RADIUS, WIN_LENGTH};

#[cfg(test)]
mod tests {
    use super::*;

    // -- Opening rules ---------------------------------------------------

    #[test]
    fn first_move_must_be_origin() {
        let mut g = HexGameState::new();
        assert!(matches!(g.place(1, 0), Err(GameError::MustPlaceAtOrigin)));
    }

    #[test]
    fn opening_turn_has_one_placement() {
        let mut g = HexGameState::new();
        let done = g.place(0, 0).unwrap();
        assert!(done);
        assert_eq!(g.current_player(), 1);
        assert_eq!(g.placements_remaining(), 2);
    }

    #[test]
    fn second_player_gets_two_placements() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();

        assert_eq!(g.current_player(), 1);
        let done = g.place(1, 0).unwrap();
        assert!(!done); // still has one left
        assert_eq!(g.placements_remaining(), 1);

        let done = g.place(0, 1).unwrap();
        assert!(done);
        assert_eq!(g.current_player(), 0);
        assert_eq!(g.placements_remaining(), 2);
    }

    #[test]
    fn opponent_last_turn_cells_handles_opening_turn() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();

        assert_eq!(g.opponent_last_turn_cells().as_slice(), &[Hex::new(0, 0)]);
    }

    #[test]
    fn opponent_last_turn_cells_skips_current_partial_turn() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        g.place(1, 0).unwrap();
        g.place(1, 1).unwrap();
        g.place(0, 1).unwrap();

        let cells = g.opponent_last_turn_cells();
        assert_eq!(cells.as_slice(), &[Hex::new(1, 0), Hex::new(1, 1)]);
    }

    // -- Placement validation --------------------------------------------

    #[test]
    fn cannot_place_on_occupied_cell() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        assert!(matches!(g.place(0, 0), Err(GameError::CellOccupied(_))));
    }

    #[test]
    fn placement_radius_enforced() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        // Just inside radius is fine.
        g.place(PLACEMENT_RADIUS, 0).unwrap();
        g.place(0, 1).unwrap();
        // Far outside radius must fail.
        assert!(matches!(g.place(100, 100), Err(GameError::OutOfRadius(_))));
    }

    #[test]
    fn cannot_place_at_radius_plus_one() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        // Exactly PLACEMENT_RADIUS + 1 away along q-axis.
        assert!(matches!(
            g.place(PLACEMENT_RADIUS + 1, 0),
            Err(GameError::OutOfRadius(_))
        ));
    }

    #[test]
    fn cannot_move_after_game_over() {
        let g = build_won_game();
        let mut g = g; // need mut
        assert!(g.is_over());
        assert!(matches!(g.place(10, 10), Err(GameError::GameOver)));
    }

    // -- Win detection: all three axes -----------------------------------

    #[test]
    fn horizontal_win_axis_1_0() {
        // Player 0 builds 6-in-a-row along (1, 0): (0,0)..(5,0).
        let g = build_won_game();
        assert_eq!(g.winner(), Some(0));
        let wl = g.winning_line().unwrap();
        assert_eq!(wl.len(), WIN_LENGTH as usize);
        // Verify the winning line is along axis (1,0).
        for i in 0..wl.len() - 1 {
            assert_eq!(wl[i + 1].q - wl[i].q, 1);
            assert_eq!(wl[i + 1].r - wl[i].r, 0);
        }
    }

    #[test]
    fn diagonal_win_axis_1_neg1() {
        // Player 0 wins along (1, -1): (0,0),(1,-1),(2,-2),(3,-3),(4,-4),(5,-5).
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        // P1 scattered
        g.place(1, 0).unwrap();
        g.place(0, 1).unwrap();
        // P0
        g.place(1, -1).unwrap();
        g.place(2, -2).unwrap();
        // P1 scattered
        g.place(2, 0).unwrap();
        g.place(0, 2).unwrap();
        // P0
        g.place(3, -3).unwrap();
        g.place(4, -4).unwrap();
        // P1 scattered
        g.place(3, 0).unwrap();
        g.place(0, 3).unwrap();
        // P0 completes the line
        g.place(5, -5).unwrap();

        assert_eq!(g.winner(), Some(0));
        let wl = g.winning_line().unwrap();
        assert_eq!(wl.len(), WIN_LENGTH as usize);
    }

    #[test]
    fn vertical_win_axis_0_1() {
        // Player 0 wins along (0, 1): (0,0),(0,1),(0,2),(0,3),(0,4),(0,5).
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        // P1 scattered (avoid lines on any axis)
        g.place(1, 0).unwrap();
        g.place(-1, 0).unwrap();
        // P0
        g.place(0, 1).unwrap();
        g.place(0, 2).unwrap();
        // P1 scattered
        g.place(2, 0).unwrap();
        g.place(-2, 0).unwrap();
        // P0
        g.place(0, 3).unwrap();
        g.place(0, 4).unwrap();
        // P1 scattered
        g.place(3, 0).unwrap();
        g.place(-3, 0).unwrap();
        // P0 completes the line
        g.place(0, 5).unwrap();

        assert_eq!(g.winner(), Some(0));
        let wl = g.winning_line().unwrap();
        assert_eq!(wl.len(), WIN_LENGTH as usize);
        // Verify the line is along (0, 1).
        for i in 0..wl.len() - 1 {
            assert_eq!(wl[i + 1].q - wl[i].q, 0);
            assert_eq!(wl[i + 1].r - wl[i].r, 1);
        }
    }

    #[test]
    fn five_in_a_row_does_not_win() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        // P1
        g.place(0, -1).unwrap();
        g.place(0, -2).unwrap();
        // P0
        g.place(1, 0).unwrap();
        g.place(2, 0).unwrap();
        // P1
        g.place(0, -3).unwrap();
        g.place(0, -4).unwrap();
        // P0: 5 in a row (0,0)..(4,0) — not enough
        g.place(3, 0).unwrap();
        g.place(4, 0).unwrap();
        assert!(g.winner().is_none());
    }

    #[test]
    fn player_1_can_win() {
        // Player 1 builds 6-in-a-row along (0, 1).
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        // P1: (0,-1), (0,-2)
        g.place(0, -1).unwrap();
        g.place(0, -2).unwrap();
        // P0 scattered
        g.place(1, 0).unwrap();
        g.place(2, 0).unwrap();
        // P1: (0,-3), (0,-4)
        g.place(0, -3).unwrap();
        g.place(0, -4).unwrap();
        // P0 scattered
        g.place(3, 0).unwrap();
        g.place(4, 0).unwrap();
        // P1: (0,-5), (0,-6) — 6 in a row from (0,-1) to (0,-6)
        g.place(0, -5).unwrap();
        g.place(0, -6).unwrap();

        assert_eq!(g.winner(), Some(1));
        assert!(g.is_over());
    }

    #[test]
    fn win_on_first_placement_of_turn() {
        // Set up so player 0 wins on placement 1 of 2 in their turn.
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        // P1 scattered
        g.place(0, -1).unwrap();
        g.place(1, -1).unwrap();
        // P0: (1,0), (2,0)
        g.place(1, 0).unwrap();
        g.place(2, 0).unwrap();
        // P1 scattered
        g.place(-1, 1).unwrap();
        g.place(0, -2).unwrap();
        // P0: (3,0), (4,0) — now has 5 in a row
        g.place(3, 0).unwrap();
        g.place(4, 0).unwrap();
        // P1 scattered
        g.place(-1, 0).unwrap();
        g.place(-2, 1).unwrap();
        // P0: (5,0) wins! This is placement 1 of 2.
        let done = g.place(5, 0).unwrap();
        assert!(done); // turn ends immediately on win
        assert_eq!(g.winner(), Some(0));
        assert_eq!(g.placements_remaining(), 0);
    }

    #[test]
    fn win_on_second_placement_of_turn() {
        // Make player 0 win on placement 2 of 2.
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        // P1 scattered
        g.place(0, -1).unwrap();
        g.place(1, -1).unwrap();
        // P0: build off-axis first, then line
        g.place(0, 7).unwrap(); // placement 1 (off-axis, not contributing to line)
        g.place(1, 0).unwrap(); // placement 2
                                // P1 scattered
        g.place(-1, 1).unwrap();
        g.place(0, -2).unwrap();
        // P0
        g.place(2, 0).unwrap();
        g.place(3, 0).unwrap();
        // P1 scattered
        g.place(-1, 0).unwrap();
        g.place(-2, 1).unwrap();
        // P0: placement 1 = (4,0), placement 2 = (5,0) wins
        g.place(4, 0).unwrap();
        let done = g.place(5, 0).unwrap();
        assert!(done);
        assert_eq!(g.winner(), Some(0));
    }

    // -- Move tracking ---------------------------------------------------

    #[test]
    fn move_count_tracks_placements() {
        let mut g = HexGameState::new();
        assert_eq!(g.move_count(), 0);
        g.place(0, 0).unwrap();
        assert_eq!(g.move_count(), 1);
        g.place(1, 0).unwrap();
        assert_eq!(g.move_count(), 2);
        g.place(0, 1).unwrap();
        assert_eq!(g.move_count(), 3);
    }

    #[test]
    fn move_history_records_correctly() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        g.place(1, 0).unwrap();
        g.place(0, 1).unwrap();

        assert_eq!(g.move_history().len(), 3);
        assert_eq!(g.move_history()[0].player(), 0);
        assert_eq!(g.move_history()[0].cell(), Hex::ORIGIN);
        assert_eq!(g.move_history()[1].player(), 1);
        assert_eq!(g.move_history()[1].cell(), Hex::new(1, 0));
        assert_eq!(g.move_history()[2].player(), 1);
        assert_eq!(g.move_history()[2].cell(), Hex::new(0, 1));
    }

    // -- Legal moves -----------------------------------------------------

    #[test]
    fn initial_legal_move_is_origin_only() {
        let g = HexGameState::new();
        assert_eq!(g.legal_moves(), vec![Hex::ORIGIN]);
    }

    #[test]
    fn no_legal_moves_when_game_over() {
        let g = build_won_game();
        assert!(g.legal_moves().is_empty());
    }

    #[test]
    fn legal_moves_exclude_occupied() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        let moves = g.legal_moves();
        assert!(!moves.contains(&Hex::ORIGIN));
    }

    #[test]
    fn legal_moves_near_is_subset_of_full() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        let near: std::collections::HashSet<Hex> = g.legal_moves_near(2).into_iter().collect();
        let full: std::collections::HashSet<Hex> = g.legal_moves().into_iter().collect();
        assert!(near.is_subset(&full));
        assert!(!near.is_empty());
    }

    #[test]
    fn legal_moves_near_clamped_to_placement_radius() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        // radius=100 should behave like radius=PLACEMENT_RADIUS
        let near_huge = g.legal_moves_near(100);
        let full = g.legal_moves();
        assert_eq!(near_huge.len(), full.len());
    }

    // -- Clone / reset ---------------------------------------------------

    #[test]
    fn clone_is_independent() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        let mut c = g.clone();
        c.place(1, 0).unwrap();
        assert!(c.stones().contains_key(&Hex::new(1, 0)));
        assert!(!g.stones().contains_key(&Hex::new(1, 0)));
    }

    #[test]
    fn zobrist_restores_after_unmake() {
        let mut g = HexGameState::new();
        let h0 = g.zobrist();

        g.place(0, 0).unwrap();
        let h1 = g.zobrist();
        assert_ne!(h0, h1);

        g.unplace();
        assert_eq!(g.zobrist(), h0);
    }

    #[test]
    fn reset_clears_everything() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        g.place(1, 0).unwrap();
        g.reset();
        assert_eq!(g.move_count(), 0);
        assert!(g.stones().is_empty());
        assert!(g.move_history().is_empty());
        assert!(g.winner().is_none());
        assert!(g.winning_line().is_none());
        assert_eq!(g.current_player(), 0);
        assert_eq!(g.placements_remaining(), 1);
    }

    // -- Default trait ---------------------------------------------------

    #[test]
    fn default_equals_new() {
        let a = HexGameState::new();
        let b = HexGameState::default();
        assert_eq!(a.current_player(), b.current_player());
        assert_eq!(a.placements_remaining(), b.placements_remaining());
        assert_eq!(a.move_count(), b.move_count());
        assert!(a.stones().is_empty() && b.stones().is_empty());
    }

    // -- Error display ---------------------------------------------------

    #[test]
    fn error_messages_are_descriptive() {
        assert!(GameError::GameOver.to_string().contains("over"));
        assert!(GameError::CellOccupied(Hex::ORIGIN)
            .to_string()
            .contains("occupied"));
        assert!(GameError::MustPlaceAtOrigin.to_string().contains("origin"));
        assert!(GameError::OutOfRadius(Hex::new(99, 99))
            .to_string()
            .contains("within"));
    }

    // -- Full random game smoke test -------------------------------------

    #[test]
    fn random_game_terminates() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut g = HexGameState::new();
        // Deterministic pseudo-random via hashing the step count.
        let mut step = 0u64;
        g.place(0, 0).unwrap();

        while !g.is_over() && g.move_count() < 2000 {
            let moves = g.legal_moves_near(3);
            if moves.is_empty() {
                break;
            }
            let mut h = DefaultHasher::new();
            step.hash(&mut h);
            let idx = h.finish() as usize % moves.len();
            let m = moves[idx];
            g.place(m.q, m.r).unwrap();
            step += 1;
        }
        // Should have terminated with a winner within a reasonable number of moves.
        // (Random games typically end in ~100-900 moves.)
        assert!(g.is_over());
    }

    // -- set_position -----------------------------------------------------

    #[test]
    fn set_position_basic() {
        let mut g = HexGameState::new();
        g.set_position(&[(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 1)], 0, 2)
            .unwrap();
        assert_eq!(g.stones().len(), 4);
        assert_eq!(g.stones().get(&Hex::new(0, 0)), Some(&0));
        assert_eq!(g.stones().get(&Hex::new(1, 0)), Some(&0));
        assert_eq!(g.stones().get(&Hex::new(2, 0)), Some(&0));
        assert_eq!(g.stones().get(&Hex::new(0, 1)), Some(&1));
        assert_eq!(g.current_player(), 0);
        assert_eq!(g.placements_remaining(), 2);
    }

    #[test]
    fn set_position_detects_win() {
        let mut g = HexGameState::new();
        g.set_position(
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
        assert_eq!(g.winner(), Some(0));
        assert!(g.is_over());
    }

    #[test]
    fn set_position_rejects_duplicate_cell() {
        let mut g = HexGameState::new();
        let res = g.set_position(&[(0, 0, 0), (1, 0, 0), (1, 0, 1)], 0, 2);
        assert!(matches!(res, Err(GameError::CellOccupied(_))));
    }

    // -- candidates_near2 --------------------------------------------------

    #[test]
    fn candidates_near2_is_sorted() {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        g.place(1, 0).unwrap();
        g.place(0, 1).unwrap();
        let cands = g.candidates_near2_sorted();
        assert!(!cands.is_empty());
        for i in 1..cands.len() {
            assert!(
                cands[i - 1] < cands[i],
                "candidates_near2 must be sorted: {:?} >= {:?}",
                cands[i - 1],
                cands[i]
            );
        }
    }

    // -- unplace eval round-trip -------------------------------------------

    #[test]
    fn unmake_restores_eval_counters() {
        let mut g = HexGameState::new();
        g.set_position(&[(0, 0, 0), (1, 0, 0), (2, 0, 0)], 0, 2)
            .unwrap();

        let eval0 = g.eval().score();
        let fives0 = [g.eval().counts(0).fives(), g.eval().counts(1).fives()];
        let fours0 = [g.eval().counts(0).fours(), g.eval().counts(1).fours()];
        let threes0 = [g.eval().counts(0).threes(), g.eval().counts(1).threes()];
        let hot0 = g.eval().hot_len(0);

        g.place(3, 0).unwrap();
        g.unplace();

        assert_eq!(g.eval().score(), eval0);
        assert_eq!(
            [g.eval().counts(0).fives(), g.eval().counts(1).fives()],
            fives0
        );
        assert_eq!(
            [g.eval().counts(0).fours(), g.eval().counts(1).fours()],
            fours0
        );
        assert_eq!(
            [g.eval().counts(0).threes(), g.eval().counts(1).threes()],
            threes0
        );
        assert_eq!(g.eval().hot_len(0), hot0);
    }

    // -- Proptests ----------------------------------------------------------

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

        #[test]
        fn place_unplace_is_identity(
            moves in prop::collection::vec((-8i32..=8, -8i32..=8), 1..15)
        ) {
            let mut game = HexGameState::new();
            let _ = game.place(0, 0);
            let mut placed = 0usize;
            for (q, r) in moves {
                if game.place(q, r).is_ok() {
                    placed += 1;
                }
            }
            let _hash_before = game.zobrist();
            for _ in 0..placed + 1 {
                game.unplace();
            }
            assert_eq!(game.zobrist(), 0, "zobrist after full unplace must be zero");
            assert_eq!(game.move_count(), 0);
        }

        #[test]
        fn zobrist_changes_on_every_valid_placement(
            q in -8i32..=8,
            r in -8i32..=8,
        ) {
            let mut game = HexGameState::new();
            let _ = game.place(0, 0);
            let before = game.zobrist();
            if game.place(q, r).is_ok() {
                assert_ne!(game.zobrist(), before, "zobrist must change on placement");
            }
        }
    }

    // -- Helpers ----------------------------------------------------------

    /// Build a game where player 0 wins with 6-in-a-row along (1, 0).
    /// Line: (0,0) (1,0) (2,0) (3,0) (4,0) (5,0).
    fn build_won_game() -> HexGameState {
        let mut g = HexGameState::new();
        g.place(0, 0).unwrap();
        g.place(0, -1).unwrap();
        g.place(1, -1).unwrap();
        g.place(1, 0).unwrap();
        g.place(2, 0).unwrap();
        g.place(-1, 1).unwrap();
        g.place(0, -2).unwrap();
        g.place(3, 0).unwrap();
        g.place(4, 0).unwrap();
        g.place(-1, 0).unwrap();
        g.place(-2, 1).unwrap();
        g.place(5, 0).unwrap();
        g
    }
}
