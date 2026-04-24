//! Consistency tests for incremental pattern evaluation.
//!
//! The engine's evaluation is incremental: instead of re-scanning the whole
//! board after every stone, [`EvalState`](crate::eval::state::EvalState) updates
//! only the windows that touch the changed cell. This module verifies that the
//! incremental result matches a full brute-force recompute on every test
//! position.

use crate::core::{Hex, HEX_DIRECTIONS, WIN_LENGTH};
use crate::eval::grid::{WIN_GRID_RADIUS, win_grid_in_bounds};
use crate::eval::patterns::{PATTERN_COUNTS, PATTERN_VALUES, POW3};
use crate::eval::state::{EvalState, ThreatCounts};
use rustc_hash::FxHashMap;

/// Recompute the total pattern score from scratch by scanning every window
/// in a fixed radius and summing `PATTERN_VALUES`.
///
/// This is `O(N²)` and only used as a reference in tests.
fn recompute_score(stones: &FxHashMap<Hex, u8>) -> i32 {
    let mut total = 0i32;
    for q in -10..=10 {
        for r in -10..=10 {
            for dir in 0..3u8 {
                let (dq, dr) = HEX_DIRECTIONS[dir as usize];
                let mut idx = 0usize;
                for off in 0..WIN_LENGTH as usize {
                    let h = Hex::new(q + dq * off as i32, r + dr * off as i32);
                    let val = match stones.get(&h) {
                        Some(&0) => 1,
                        Some(&1) => 2,
                        _ => 0,
                    };
                    idx += val * POW3[off];
                }
                total += PATTERN_VALUES[idx];
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `PATTERN_COUNTS` (precomputed stone counts per pattern)
    /// matches a manual base-3 decomposition.
    ///
    /// Every pattern index is a 6-digit base-3 number where:
    /// - `0` = empty
    /// - `1` = player 0 stone
    /// - `2` = player 1 stone
    #[test]
    fn ternary_index_roundtrip() {
        for idx in 0..729usize {
            let (expected_p0, expected_p1) = PATTERN_COUNTS[idx];
            let mut manual_p0 = 0u8;
            let mut manual_p1 = 0u8;
            let mut n = idx;
            for _ in 0..6 {
                let digit = (n % 3) as u8;
                if digit == 1 {
                    manual_p0 += 1;
                } else if digit == 2 {
                    manual_p1 += 1;
                }
                n /= 3;
            }
            assert_eq!(
                (manual_p0, manual_p1),
                (expected_p0, expected_p1),
                "PATTERN_COUNTS[{}] mismatch",
                idx
            );
        }
    }

    /// Sanity-check that `POW3` is actually the powers of three.
    #[test]
    fn pow3_values() {
        let mut pow = 1usize;
        for i in 0..6 {
            assert_eq!(POW3[i], pow, "POW3[{}] should be {}", i, pow);
            pow *= 3;
        }
    }

    /// Ensure the pattern-value table is non-trivial.
    ///
    /// A table full of zeros would pass many other tests while being useless.
    #[test]
    fn pattern_values_not_all_zero() {
        let mut all_zero = true;
        for &v in PATTERN_VALUES.iter() {
            if v != 0 {
                all_zero = false;
                break;
            }
        }
        assert!(!all_zero, "PATTERN_VALUES should not all be zero");
    }

    /// After each incremental `place`, the score must equal the brute-force
    /// recomputed score for the same stone set.
    #[test]
    fn incremental_place_consistency() {
        let mut eval = EvalState::new();
        let mut stones = FxHashMap::default();

        let placements = [
            (Hex::new(0, 0), 0u8),
            (Hex::new(1, 0), 1u8),
            (Hex::new(2, 0), 0u8),
            (Hex::new(0, 1), 1u8),
            (Hex::new(-1, -1), 0u8),
        ];

        for &(cell, player) in &placements {
            eval.place(cell, player);
            stones.insert(cell, player);
            let recomputed = recompute_score(&stones);
            assert_eq!(
                eval.score(), recomputed,
                "score mismatch after placing at {:?} for player {}",
                cell, player
            );
        }
    }

    /// After placing several stones and then unplacing all of them, the
    /// `EvalState` must return to its pristine default (score 0, no threats,
    /// no hot windows).
    #[test]
    fn incremental_unplace_restores_score() {
        let mut eval = EvalState::new();
        let mut stones = FxHashMap::default();

        let placements = [
            (Hex::new(0, 0), 0u8),
            (Hex::new(1, 0), 1u8),
            (Hex::new(2, 0), 0u8),
            (Hex::new(0, 1), 1u8),
        ];

        for &(cell, player) in &placements {
            eval.place(cell, player);
            stones.insert(cell, player);
        }

        for _ in 0..placements.len() {
            eval.unplace();
        }

        assert_eq!(eval.score(), 0, "score should be 0 after unplacing all");
        assert_eq!(eval.counts(0), ThreatCounts::default());
        assert_eq!(eval.counts(1), ThreatCounts::default());
        assert_eq!(eval.hot_windows(0).count(), 0);
        assert_eq!(eval.hot_windows(1).count(), 0);
    }

    /// The set of hot windows reported by `EvalState` must exactly match the
    /// set obtained by brute-force scanning the win grid and recomputing
    /// pattern indices from the stone map.
    #[test]
    fn hot_windows_recomputed_match() {
        let mut eval = EvalState::new();
        let mut stones = FxHashMap::default();

        let placements = [
            (Hex::new(0, 0), 0u8),
            (Hex::new(1, 0), 1u8),
            (Hex::new(2, 0), 0u8),
            (Hex::new(3, 0), 0u8),
            (Hex::new(4, 0), 0u8),
            (Hex::new(0, 1), 1u8),
        ];

        for &(cell, player) in &placements {
            eval.place(cell, player);
            stones.insert(cell, player);
        }

        // Recompute hot windows by scanning all board windows and computing indices from stones.
        let mut recomputed_0 = Vec::new();
        let mut recomputed_1 = Vec::new();

        for q in -WIN_GRID_RADIUS..=WIN_GRID_RADIUS {
            for r in -WIN_GRID_RADIUS..=WIN_GRID_RADIUS {
                if !win_grid_in_bounds(q, r) {
                    continue;
                }
                for dir in 0..3u8 {
                    let (dq, dr) = HEX_DIRECTIONS[dir as usize];
                    let mut idx = 0usize;
                    for off in 0..WIN_LENGTH as usize {
                        let h = Hex::new(q + dq * off as i32, r + dr * off as i32);
                        let val = match stones.get(&h) {
                            Some(&0) => 1,
                            Some(&1) => 2,
                            _ => 0,
                        };
                        idx += val * POW3[off];
                    }
                    let (p0, p1) = PATTERN_COUNTS[idx];
                    if p0 >= 4 && p1 == 0 {
                        recomputed_0.push(crate::core::WindowKey::new(q, r, dir));
                    }
                    if p1 >= 4 && p0 == 0 {
                        recomputed_1.push(crate::core::WindowKey::new(q, r, dir));
                    }
                }
            }
        }

        for (player, mut expected) in [(0u8, recomputed_0), (1u8, recomputed_1)] {
            let mut actual: Vec<_> = eval.hot_windows(player).collect();
            expected.sort_by_key(|k| (k.q(), k.r(), k.dir()));
            actual.sort_by_key(|k| (k.q(), k.r(), k.dir()));
            assert_eq!(
                actual, expected,
                "hot windows mismatch for player {}",
                player
            );
        }
    }

    // -- Pattern table integrity (moved from eval/patterns.rs) ------------

    #[test]
    fn test_pattern_values_len() {
        assert_eq!(PATTERN_VALUES.len(), 729);
    }

    #[test]
    fn test_pattern_counts_len() {
        assert_eq!(PATTERN_COUNTS.len(), 729);
    }

    #[test]
    fn test_pattern_counts_known_patterns() {
        // All-empty window → (0, 0).
        assert_eq!(PATTERN_COUNTS[0], (0, 0));

        // Single P0 stone at offset 0 → digit 1 at position 0 → index 1.
        assert_eq!(PATTERN_COUNTS[1], (1, 0));

        // Single P1 stone at offset 0 → digit 2 at position 0 → index 2.
        assert_eq!(PATTERN_COUNTS[2], (0, 1));

        // idx=3: base-3 digits [0,1,0,0,0,0] → one P0 stone at offset 1.
        assert_eq!(PATTERN_COUNTS[3], (1, 0));
        // idx=5: base-3 digits [2,1,0,0,0,0] → one P0 at offset 0, one P1 at offset 1.
        assert_eq!(PATTERN_COUNTS[5], (1, 1));

        // Six P0 stones: digits all 1.
        // idx = 1 + 3 + 9 + 27 + 81 + 243 = 364.
        assert_eq!(PATTERN_COUNTS[364], (6, 0));

        // Six P1 stones: digits all 2.
        // idx = 2·(1 + 3 + 9 + 27 + 81 + 243) = 728.
        assert_eq!(PATTERN_COUNTS[728], (0, 6));
    }

    /// Verify that `PATTERN_VALUES` has not been accidentally corrupted.
    ///
    /// Plan Invariant 1 requires the table to be bit-identical to the tuned
    /// CMA-ES weights. This checksum catches silent corruption (editor
    /// accidents, merge conflicts, etc.). If the weights are intentionally
    /// retuned, update the expected digest.
    #[test]
    fn pattern_values_checksum() {
        // FNV-1a 64-bit over the raw i32 values (stable across platforms).
        let mut hash: u64 = 0xcbf29ce484222325;
        for &v in PATTERN_VALUES.iter() {
            hash ^= (v as u32) as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        // If this fails after an intentional retune, replace the literal.
        assert_eq!(hash, 0x9f5d14a209044de4,
            "PATTERN_VALUES checksum mismatch — table may be corrupted");
    }
}
