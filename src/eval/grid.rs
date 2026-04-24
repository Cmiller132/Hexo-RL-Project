pub const WIN_GRID_RADIUS: i32 = 30;
pub const WIN_GRID_SIDE: usize = 61; // 2 * 30 + 1
pub const WIN_GRID_TOTAL: usize = 61 * 61 * 3; // 11163

#[inline]
pub fn win_grid_idx(q: i32, r: i32, dir: u8) -> usize {
    ((q + WIN_GRID_RADIUS) as usize) * WIN_GRID_SIDE * 3
        + ((r + WIN_GRID_RADIUS) as usize) * 3
        + dir as usize
}

/// Debug-only bounds check. Release builds assume PLACEMENT_RADIUS < WIN_GRID_RADIUS.
#[inline]
pub fn win_grid_in_bounds(q: i32, r: i32) -> bool {
    let qi = q + WIN_GRID_RADIUS;
    let ri = r + WIN_GRID_RADIUS;
    qi >= 0 && (qi as usize) < WIN_GRID_SIDE && ri >= 0 && (ri as usize) < WIN_GRID_SIDE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn win_grid_total_is_correct() {
        assert_eq!(WIN_GRID_TOTAL, 61 * 61 * 3);
        assert_eq!(WIN_GRID_TOTAL, 11163);
    }

    #[test]
    fn win_grid_idx_bijection_within_bounds() {
        let mut seen = std::collections::HashSet::new();

        for q in -WIN_GRID_RADIUS..=WIN_GRID_RADIUS {
            for r in -WIN_GRID_RADIUS..=WIN_GRID_RADIUS {
                for dir in 0..3u8 {
                    let idx = win_grid_idx(q, r, dir);
                    assert!(
                        idx < WIN_GRID_TOTAL,
                        "index {} out of bounds for q={}, r={}, dir={}",
                        idx,
                        q,
                        r,
                        dir
                    );
                    assert!(
                        seen.insert(idx),
                        "duplicate index {} for q={}, r={}, dir={}",
                        idx,
                        q,
                        r,
                        dir
                    );
                }
            }
        }

        assert_eq!(seen.len(), WIN_GRID_TOTAL);
    }

    #[test]
    fn win_grid_idx_specific_values() {
        assert_eq!(
            win_grid_idx(0, 0, 0),
            (WIN_GRID_RADIUS as usize) * WIN_GRID_SIDE * 3 + (WIN_GRID_RADIUS as usize) * 3
        );
        assert_eq!(win_grid_idx(-WIN_GRID_RADIUS, -WIN_GRID_RADIUS, 0), 0);
        assert_eq!(win_grid_idx(-WIN_GRID_RADIUS, -WIN_GRID_RADIUS, 1), 1);
        assert_eq!(win_grid_idx(-WIN_GRID_RADIUS, -WIN_GRID_RADIUS, 2), 2);
        assert_eq!(
            win_grid_idx(WIN_GRID_RADIUS, WIN_GRID_RADIUS, 2),
            WIN_GRID_TOTAL - 1
        );
    }

    #[test]
    fn win_grid_in_bounds_behavior() {
        assert!(win_grid_in_bounds(0, 0));
        assert!(win_grid_in_bounds(-WIN_GRID_RADIUS, -WIN_GRID_RADIUS));
        assert!(win_grid_in_bounds(WIN_GRID_RADIUS, WIN_GRID_RADIUS));
        assert!(!win_grid_in_bounds(-WIN_GRID_RADIUS - 1, 0));
        assert!(!win_grid_in_bounds(WIN_GRID_RADIUS + 1, 0));
    }
}
