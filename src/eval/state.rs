use rustc_hash::FxHashMap;
use crate::core::{Hex, HEX_DIRECTIONS, WIN_LENGTH, WindowKey};
use crate::eval::grid::{win_grid_idx, win_grid_in_bounds, WIN_GRID_TOTAL};
use crate::eval::hot::HotWindows;
use crate::eval::patterns::{PATTERN_VALUES, PATTERN_COUNTS, POW3};

#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct ThreatCounts {
    pub fives: u32,
    pub fours: u32,
    pub threes: u32,
}

impl ThreatCounts {
    pub fn apply(&mut self, delta: &ThreatCountsDelta) {
        debug_assert!((self.fives as i32 + delta.fives) >= 0,
            "fives underflow: {} + {}", self.fives, delta.fives);
        debug_assert!((self.fours as i32 + delta.fours) >= 0,
            "fours underflow: {} + {}", self.fours, delta.fours);
        debug_assert!((self.threes as i32 + delta.threes) >= 0,
            "threes underflow: {} + {}", self.threes, delta.threes);
        self.fives = (self.fives as i32 + delta.fives) as u32;
        self.fours = (self.fours as i32 + delta.fours) as u32;
        self.threes = (self.threes as i32 + delta.threes) as u32;
    }
}

#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct ThreatCountsDelta {
    pub fives: i32,
    pub fours: i32,
    pub threes: i32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EvalDelta {
    pub cell: Hex,
    pub player: u8,
    pub score: i32,
    pub counts: [ThreatCountsDelta; 2],
}

#[derive(Clone, Debug)]
pub struct EvalState {
    score: i32,
    counts: [ThreatCounts; 2],
    hot: HotWindows,
    indices: Box<[u16; WIN_GRID_TOTAL]>,
    delta_stack: Vec<EvalDelta>,
}

#[inline]
fn classify_delta(delta: &mut ThreatCountsDelta, old_own: u8, old_other: u8, new_own: u8, new_other: u8) {
    let old_five = (old_own >= 5 && old_other == 0) as i32;
    let new_five = (new_own >= 5 && new_other == 0) as i32;
    delta.fives += new_five - old_five;

    let old_four = (old_own == 4 && old_other == 0) as i32;
    let new_four = (new_own == 4 && new_other == 0) as i32;
    delta.fours += new_four - old_four;

    let old_three = (old_own == 3 && old_other == 0) as i32;
    let new_three = (new_own == 3 && new_other == 0) as i32;
    delta.threes += new_three - old_three;
}

#[inline]
fn update_hot(hot: &mut HotWindows, key: WindowKey, player: u8, old_own: u8, old_other: u8, new_own: u8, new_other: u8) {
    let was_hot = old_own >= 4 && old_other == 0;
    let is_hot = new_own >= 4 && new_other == 0;
    if was_hot && !is_hot {
        hot.remove(player, key);
    } else if !was_hot && is_hot {
        hot.insert(player, key);
    }
}

impl Default for EvalState {
    fn default() -> Self {
        Self::new()
    }
}

impl EvalState {
    pub fn new() -> Self {
        Self {
            score: 0,
            counts: [ThreatCounts::default(); 2],
            hot: HotWindows::new(),
            indices: Box::new([0; WIN_GRID_TOTAL]),
            delta_stack: Vec::new(),
        }
    }

    pub fn place(&mut self, _stones: &FxHashMap<Hex, u8>, cell: Hex, player: u8) -> EvalDelta {
        let cell_val = (player + 1) as usize; // 1 or 2
        let mut delta = EvalDelta {
            cell,
            player,
            score: 0,
            counts: [ThreatCountsDelta::default(); 2],
        };

        for dir in 0..3 {
            let (dq, dr) = HEX_DIRECTIONS[dir];
            for off in 0..WIN_LENGTH as usize {
                let sq = cell.q - dq * off as i32;
                let sr = cell.r - dr * off as i32;
                // WIN_GRID_RADIUS (30) caps evaluation at ~3–4 moves from origin per axis.
                // For stones near the grid boundary, some of the 18 windows extend outside
                // the grid; those windows simply don't contribute to evaluation. This is a
                // known approximation, not a bug.
                if !win_grid_in_bounds(sq, sr) {
                    continue;
                }

                let gi = win_grid_idx(sq, sr, dir as u8);
                let old_idx = self.indices[gi] as usize;
                let new_idx = old_idx + cell_val * POW3[off];
                debug_assert!(new_idx < 729);

                delta.score += PATTERN_VALUES[new_idx] - PATTERN_VALUES[old_idx];

                let (old_p0, old_p1) = PATTERN_COUNTS[old_idx];
                let (new_p0, new_p1) = PATTERN_COUNTS[new_idx];

                classify_delta(&mut delta.counts[0], old_p0, old_p1, new_p0, new_p1);
                classify_delta(&mut delta.counts[1], old_p1, old_p0, new_p1, new_p0);

                let key = WindowKey::new(sq, sr, dir as u8);
                update_hot(&mut self.hot, key, 0, old_p0, old_p1, new_p0, new_p1);
                update_hot(&mut self.hot, key, 1, old_p1, old_p0, new_p1, new_p0);

                self.indices[gi] = new_idx as u16;
            }
        }

        self.score += delta.score;
        self.counts[0].apply(&delta.counts[0]);
        self.counts[1].apply(&delta.counts[1]);
        self.delta_stack.push(delta);
        delta
    }

    pub fn unplace(&mut self) {
        let delta = self.delta_stack.pop().expect("unplace called with empty stack");

        self.score -= delta.score;
        self.counts[0].apply(&ThreatCountsDelta {
            fives: -delta.counts[0].fives,
            fours: -delta.counts[0].fours,
            threes: -delta.counts[0].threes,
        });
        self.counts[1].apply(&ThreatCountsDelta {
            fives: -delta.counts[1].fives,
            fours: -delta.counts[1].fours,
            threes: -delta.counts[1].threes,
        });

        let cell = delta.cell;
        let player = delta.player;
        let cell_val = (player + 1) as usize;

        for dir in 0..3 {
            let (dq, dr) = HEX_DIRECTIONS[dir];
            for off in 0..WIN_LENGTH as usize {
                let sq = cell.q - dq * off as i32;
                let sr = cell.r - dr * off as i32;
                // WIN_GRID_RADIUS (30) caps evaluation at ~3–4 moves from origin per axis.
                // For stones near the grid boundary, some of the 18 windows extend outside
                // the grid; those windows simply don't contribute to evaluation. This is a
                // known approximation, not a bug.
                if !win_grid_in_bounds(sq, sr) {
                    continue;
                }

                let gi = win_grid_idx(sq, sr, dir as u8);
                let new_idx = self.indices[gi] as usize;
                let old_idx = new_idx - cell_val * POW3[off];
                debug_assert!(old_idx < 729);

                let (old_p0, old_p1) = PATTERN_COUNTS[old_idx];
                let (new_p0, new_p1) = PATTERN_COUNTS[new_idx];

                let key = WindowKey::new(sq, sr, dir as u8);
                // Reverse hot transitions by swapping old/new.
                update_hot(&mut self.hot, key, 0, new_p0, new_p1, old_p0, old_p1);
                update_hot(&mut self.hot, key, 1, new_p1, new_p0, old_p1, old_p0);

                self.indices[gi] = old_idx as u16;
            }
        }

        #[cfg(debug_assertions)]
        {
            self.assert_invariants();
        }
    }

    #[cfg(debug_assertions)]
    fn assert_invariants(&self) {
        use crate::eval::grid::WIN_GRID_RADIUS;
        // Recompute hot windows from scratch from indices + PATTERN_COUNTS
        let mut expected = [
            std::collections::HashSet::new(),
            std::collections::HashSet::new(),
        ];
        for q in -WIN_GRID_RADIUS..=WIN_GRID_RADIUS {
            for r in -WIN_GRID_RADIUS..=WIN_GRID_RADIUS {
                for dir in 0..3u8 {
                    let gi = win_grid_idx(q, r, dir);
                    let idx = self.indices[gi] as usize;
                    let (p0, p1) = PATTERN_COUNTS[idx];
                    if p0 >= 4 && p1 == 0 {
                        expected[0].insert(WindowKey::new(q, r, dir));
                    }
                    if p1 >= 4 && p0 == 0 {
                        expected[1].insert(WindowKey::new(q, r, dir));
                    }
                }
            }
        }

        // Compare recomputed with actual (order-independent because swap_remove
        // may reorder the internal SmallVec).
        for player in 0..2 {
            let actual: std::collections::HashSet<_> = self.hot.iter(player).collect();
            assert_eq!(
                actual.len(),
                expected[player as usize].len(),
                "hot window count mismatch for player {}", player
            );
            assert_eq!(
                actual, expected[player as usize],
                "hot window mismatch for player {}", player
            );
        }
    }

    pub fn score(&self) -> i32 {
        self.score
    }

    pub fn counts(&self, player: u8) -> ThreatCounts {
        self.counts[player as usize]
    }

    pub fn hot_windows(&self, player: u8) -> impl Iterator<Item = WindowKey> + '_ {
        self.hot.iter(player)
    }

    pub fn hot_is_empty(&self, player: u8) -> bool {
        self.hot.is_empty(player)
    }

    pub fn hot_len(&self, player: u8) -> usize {
        self.hot.len(player)
    }

    pub fn hypothetical_score_delta(&self, cell: Hex, player: u8) -> i32 {
        let cell_val = (player + 1) as usize;
        let mut delta = 0i32;

        for dir in 0..3 {
            let (dq, dr) = HEX_DIRECTIONS[dir];
            for off in 0..WIN_LENGTH as usize {
                let sq = cell.q - dq * off as i32;
                let sr = cell.r - dr * off as i32;
                // WIN_GRID_RADIUS (30) caps evaluation at ~3–4 moves from origin per axis.
                // For stones near the grid boundary, some of the 18 windows extend outside
                // the grid; those windows simply don't contribute to evaluation. This is a
                // known approximation, not a bug.
                if !win_grid_in_bounds(sq, sr) {
                    continue;
                }

                let gi = win_grid_idx(sq, sr, dir as u8);
                let old_idx = self.indices[gi] as usize;
                let new_idx = old_idx + cell_val * POW3[off];
                debug_assert!(new_idx < 729);

                delta += PATTERN_VALUES[new_idx] - PATTERN_VALUES[old_idx];
            }
        }

        delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;

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
        let delta = state.place(&stones, cell, 0);
        stones.insert(cell, 0);

        assert_ne!(delta.score, 0); // placing on empty board should change score
        assert_eq!(state.counts(0), ThreatCounts { fives: 0, fours: 0, threes: 0 });

        state.unplace();

        // After unplace, state should be identical to new
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
            state.place(&stones, cell, p);
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
            state.place(&stones, cell, 0);
            stones.insert(cell, 0);
        }

        let counts = state.counts(0);
        assert!(counts.fives >= 1, "expected at least one five threat, got {:?}", counts);
    }

    #[test]
    fn hot_windows_tracked_correctly() {
        let mut state = EvalState::new();
        let mut stones = FxHashMap::default();

        // Place 4 P0 stones in a row along direction 0: (0,0) through (3,0).
        // This should create hot windows for P0 (own >= 4 && other == 0).
        for i in 0..4 {
            let cell = Hex::new(i, 0);
            state.place(&stones, cell, 0);
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
        let stones = FxHashMap::default();

        let cell = Hex::new(0, 0);
        let hypo = state.hypothetical_score_delta(cell, 0);
        let delta = state.place(&stones, cell, 0);

        assert_eq!(hypo, delta.score, "hypothetical score delta should match actual");
    }

    #[test]
    fn invariant_check_runs_on_roundtrip() {
        let mut state = EvalState::new();
        let mut stones = FxHashMap::default();

        // Place and unplace several stones to exercise the invariant check
        let cells = vec![(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1), (0, 1, 0), (1, 1, 1)];
        for &(q, r, p) in &cells {
            let cell = Hex::new(q, r);
            state.place(&stones, cell, p);
            stones.insert(cell, p);
        }

        for _ in 0..cells.len() {
            state.unplace();
        }

        // If we get here without panicking, the invariant checks passed
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
            state.place(&stones, cell, 0);
            stones.insert(cell, 0);
        }

        let before_block = state.counts(0);
        assert!(before_block.fives >= 1, "expected fives before blocking");

        // P1 blocks at one end
        let block = Hex::new(-1, 0);
        state.place(&stones, block, 1);
        stones.insert(block, 1);

        let after_block = state.counts(0);
        // Blocking should reduce the number of five threats
        assert!(
            after_block.fives < before_block.fives,
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
