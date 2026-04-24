use crate::eval::grid::*;

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
