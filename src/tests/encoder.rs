use crate::encoder::*;
use crate::board::HexGameState;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_board_features() {
        let game = HexGameState::new();
        let feats = extract_features(&game);
        for (i, &feat) in feats.iter().enumerate().take(FEATURE_COUNT - 1) {
            assert_eq!(feat, 0.0, "feature {} should be zero on empty board", i);
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
