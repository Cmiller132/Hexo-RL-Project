use crate::eval::state::*;
use crate::core::Hex;
use rustc_hash::FxHashMap;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_is_default() {
        let state = EvalState::new();
        assert_eq!(state.score(), 0);
        assert_eq!(state.counts(0), ThreatCounts::default());
        assert_eq!(state.counts(1), ThreatCounts::default());
        assert_eq!(state.hot_windows(0).count(), 0);
        assert_eq!(state.hot_windows(1).count(), 0);
    }

    #[test]
    fn place_unplace_roundtrip() {
        let mut state = EvalState::new();
        let mut stones = FxHashMap::default();

        let cell = Hex::new(0, 0);
        state.place(cell, 0);
        stones.insert(cell, 0);

        // Placing on an empty board must change the score.
        assert_ne!(state.score(), 0);
        assert_eq!(state.counts(0), ThreatCounts::default());

        state.unplace();

        // After unplace, state should be identical to new.
        assert_eq!(state.score(), 0);
        assert_eq!(state.counts(0), ThreatCounts::default());
        assert_eq!(state.counts(1), ThreatCounts::default());
        assert_eq!(state.hot_windows(0).count(), 0);
        assert_eq!(state.hot_windows(1).count(), 0);
    }

    #[test]
    fn place_unplace_multiple_roundtrip() {
        let mut state = EvalState::new();
        let mut stones = FxHashMap::default();

        let cells = vec![(0, 0, 0), (1, 0, 1), (2, 0, 0), (0, 1, 1)];
        for &(q, r, p) in &cells {
            let cell = Hex::new(q, r);
            state.place(cell, p);
            stones.insert(cell, p);
        }

        for _ in 0..cells.len() {
            state.unplace();
        }

        assert_eq!(state.score(), 0);
        assert_eq!(state.counts(0), ThreatCounts::default());
        assert_eq!(state.counts(1), ThreatCounts::default());
        assert_eq!(state.hot_windows(0).count(), 0);
        assert_eq!(state.hot_windows(1).count(), 0);
    }

    #[test]
    fn counts_update_for_simple_patterns() {
        let mut state = EvalState::new();
        let mut stones = FxHashMap::default();

        // Place 5 P0 stones in a row along direction 0: (0,0) through (4,0).
        // This creates at least one five threat (5 in a 6-cell window, uncontested).
        for i in 0..5 {
            let cell = Hex::new(i, 0);
            state.place(cell, 0);
            stones.insert(cell, 0);
        }

        let counts = state.counts(0);
        assert!(counts.fives() >= 1, "expected at least one five threat, got {:?}", counts);
    }

    #[test]
    fn hot_windows_tracked_correctly() {
        let mut state = EvalState::new();
        let mut stones = FxHashMap::default();

        // Place 4 P0 stones in a row along direction 0: (0,0) through (3,0).
        // This should create hot windows for P0 (own >= 4 && other == 0).
        for i in 0..4 {
            let cell = Hex::new(i, 0);
            state.place(cell, 0);
            stones.insert(cell, 0);
        }

        let hot: Vec<_> = state.hot_windows(0).collect();
        assert!(!hot.is_empty(), "expected hot windows for P0 after 4-in-a-row");

        // P1 should have no hot windows
        assert_eq!(state.hot_windows(1).count(), 0);
    }

    #[test]
    fn hypothetical_matches_actual_delta() {
        let mut state = EvalState::new();

        let cell = Hex::new(0, 0);
        let hypo = state.hypothetical_score_delta(cell, 0);
        let score_before = state.score();
        state.place(cell, 0);
        let actual_delta = state.score() - score_before;

        assert_eq!(hypo, actual_delta, "hypothetical score delta should match actual");
    }

    #[test]
    fn invariant_check_runs_on_roundtrip() {
        let mut state = EvalState::new();
        let mut stones = FxHashMap::default();

        // Place and unplace several stones to exercise the invariant check.
        let cells = vec![(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1), (0, 1, 0), (1, 1, 1)];
        for &(q, r, p) in &cells {
            let cell = Hex::new(q, r);
            state.place(cell, p);
            stones.insert(cell, p);
        }

        for _ in 0..cells.len() {
            state.unplace();
        }

        // If we get here without panicking, the invariant checks passed.
        assert_eq!(state.score(), 0);
        assert_eq!(state.counts(0), ThreatCounts::default());
        assert_eq!(state.counts(1), ThreatCounts::default());
        assert_eq!(state.hot_windows(0).count(), 0);
        assert_eq!(state.hot_windows(1).count(), 0);
    }

    #[test]
    fn opponent_blocks_removes_hot() {
        let mut state = EvalState::new();
        let mut stones = FxHashMap::default();

        // P0 places 5 in a row
        for i in 0..5 {
            let cell = Hex::new(i, 0);
            state.place(cell, 0);
            stones.insert(cell, 0);
        }

        let before_block = state.counts(0);
        assert!(before_block.fives() >= 1, "expected fives before blocking");

        // P1 blocks at one end
        let block = Hex::new(-1, 0);
        state.place(block, 1);
        stones.insert(block, 1);

        let after_block = state.counts(0);
        // Blocking should reduce the number of five threats
        assert!(
            after_block.fives() < before_block.fives(),
            "blocking should reduce fives: before={:?}, after={:?}",
            before_block, after_block
        );

        // Unplace should restore everything back to default
        state.unplace(); // undo P1 block
        for _ in 0..5 {
            state.unplace(); // undo P0 placements
        }

        assert_eq!(state.score(), 0);
        assert_eq!(state.counts(0), ThreatCounts::default());
        assert_eq!(state.counts(1), ThreatCounts::default());
        assert_eq!(state.hot_windows(0).count(), 0);
        assert_eq!(state.hot_windows(1).count(), 0);
    }
}
