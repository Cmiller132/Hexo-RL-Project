use crate::board::HexGameState;
use crate::core::{hex_distance, Hex, PLACEMENT_RADIUS};
use rustc_hash::FxHashSet;

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn legal_moves_near_into_matches_bruteforce() {
        let mut game = populated_game();

        assert_legal_matches_bruteforce(&game, 2);
        assert_legal_matches_bruteforce(&game, 3);
        assert_legal_matches_bruteforce(&game, PLACEMENT_RADIUS);
        assert_legal_matches_bruteforce(&game, PLACEMENT_RADIUS + 10);

        game.unplace().unwrap();
        game.unplace().unwrap();

        assert_legal_matches_bruteforce(&game, 2);
        assert_legal_matches_bruteforce(&game, PLACEMENT_RADIUS);
    }

    #[test]
    fn legal_moves_uses_full_placement_radius() {
        let game = populated_game();
        let mut legal = game.legal_moves();
        let mut brute = brute_legal_moves_near(&game, PLACEMENT_RADIUS);
        legal.sort();
        brute.sort();
        assert_eq!(legal, brute);
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

        #[test]
        fn legal_moves_match_bruteforce_after_place_and_unplace(
            ops in prop::collection::vec((-8i32..=8, -8i32..=8, any::<bool>()), 1..48)
        ) {
            let mut game = HexGameState::new();
            assert_legal_matches_bruteforce(&game, 2);
            assert_legal_matches_bruteforce(&game, PLACEMENT_RADIUS);

            for (q, r, undo) in ops {
                if undo && game.move_count() > 0 {
                    game.unplace().unwrap();
                } else {
                    let _ = game.place(q, r);
                }
                assert_legal_matches_bruteforce(&game, 2);
                assert_legal_matches_bruteforce(&game, PLACEMENT_RADIUS);
            }
        }
    }

    fn populated_game() -> HexGameState {
        let mut game = HexGameState::new();
        for (q, r) in [(0, 0), (1, 0), (0, 1), (-1, 1), (2, -1), (-2, 0), (1, -2)] {
            game.place(q, r).unwrap();
        }
        game
    }

    fn assert_legal_matches_bruteforce(game: &HexGameState, radius: i32) {
        let mut actual = Vec::with_capacity(8);
        game.legal_moves_near_into(radius, &mut actual);
        let mut expected = brute_legal_moves_near(game, radius);
        actual.sort();
        expected.sort();
        assert_eq!(actual, expected, "radius={radius}");
    }

    fn brute_legal_moves_near(game: &HexGameState, radius: i32) -> Vec<Hex> {
        if game.is_over() {
            return Vec::new();
        }
        if game.stones().is_empty() {
            return vec![Hex::ORIGIN];
        }

        let r = radius.clamp(0, PLACEMENT_RADIUS);
        let mut candidates = FxHashSet::default();
        for &cell in game.stones().keys() {
            for dq in -r..=r {
                for dr in -r..=r {
                    let cand = Hex::new(cell.q + dq, cell.r + dr);
                    if !game.stones().contains_key(&cand) && hex_distance(cell, cand) <= r {
                        candidates.insert(cand);
                    }
                }
            }
        }
        candidates.into_iter().collect()
    }
}
