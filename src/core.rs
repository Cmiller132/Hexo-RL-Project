//! Axial hex coordinates and distance for a hexagonal tic-tac-toe engine.
//!
//! This module provides the fundamental spatial primitive [`Hex`] using **axial
//! coordinates** `(q, r)`.  Axial coordinates are a compact two-number
//! representation of hex-grid positions.  They are equivalent to cube
//! coordinates `(x, y, z)` with the constraint `x + y + z = 0`; here `x = q`,
//! `z = r`, and `y = -q - r` is implicit.
//!
//! # Principal axes for win detection
//!
//! In hexagonal tic-tac-toe a win is a straight line of stones.  On a hex grid
//! there are exactly three unique directions through opposite sides of a hex,
//! so we only need three principal vectors to detect every possible line:
//!
//! | Axis | Direction `(dq, dr)` | Geometric meaning                     |
//! |------|----------------------|---------------------------------------|
//! | A    | `(1, 0)`             | Move east (column / "q" axis)         |
//! | B    | `(0, 1)`             | Move south-east (row / "r" axis)      |
//! | C    | `(1, -1)`            | Move north-east (diagonal axis)       |
//!
//! ## Why three directions are sufficient
//!
//! A regular hexagon has three pairs of parallel sides.  Any straight line on
//! the grid must be parallel to one of those three side-pair orientations.
//! Consequently every possible winning line aligns with one of the vectors
//! above or its exact negative (e.g. `(-1, 0)`).  Win-detection logic can
//! therefore scan forward and backward along just these three axes and is
//! guaranteed to find every contiguous run of stones.

use std::fmt;

/// An axial hex coordinate `(q, r)`.
///
/// `q` is the column (pointing east) and `r` is the row (pointing south-east).
/// Together they uniquely identify a single hex cell on the board.
///
/// # Why these derives matter
///
/// * `Copy` – `Hex` is small (two `i32`s) and is passed around by value
///   thousands of times during search; copying avoids reference overhead.
/// * `Eq` + `Hash` – required so `Hex` can be used as a key in `HashMap` and
///   `HashSet`, which the engine uses for fast stone lookups.
/// * `Ord` – allows sorting and enables deterministic ordering when `Hex`
///   values are stored in ordered collections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hex {
    pub q: i32,
    pub r: i32,
}

impl Hex {
    /// Create a new axial coordinate from `q` and `r`.
    ///
    /// This is a `const fn`, so it can be used in constant contexts such as
    /// static board definitions or `const` arrays.
    #[inline(always)]
    pub const fn new(q: i32, r: i32) -> Self {
        Self { q, r }
    }

    /// The axial origin `(0, 0)`.
    ///
    /// Provided as a convenient named constant so callers do not have to
    /// construct it manually every time.
    pub const ORIGIN: Self = Self { q: 0, r: 0 };
}

impl fmt::Display for Hex {
    /// Render a hex as `(q, r)`.
    ///
    /// This format is used in debug output, logs, and UI text so that human
    /// readers can immediately see the two axial components.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.q, self.r)
    }
}

impl PartialOrd for Hex {
    /// Delegates to [`Ord::cmp`] so that `Hex` has a total ordering compatible
    /// with `Ord`.  This makes `Hex` usable in `BTreeMap` / `BTreeSet`.
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hex {
    /// Lexicographic ordering: first by `q`, then by `r`.
    ///
    /// A deterministic order is important when the engine stores move lists
    /// in sorted containers or when reproducibility across runs is required.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.q.cmp(&other.q).then(self.r.cmp(&other.r))
    }
}

/// The three principal line-checking directions on a hex grid.
///
/// Each tuple is a delta `(dq, dr)` that moves one step along a unique axis.
/// Because hex lines have 180° symmetry, checking the negated versions of
/// these three vectors (`(-1, 0)`, `(0, -1)`, `(-1, 1)`) covers the opposite
/// direction on the same line.  Together they form a complete basis for
/// straight-line win detection on a hexagonal board.
pub const HEX_DIRECTIONS: [(i32, i32); 3] = [(1, 0), (0, 1), (1, -1)];

/// Compute the hex distance between two axial coordinates.
///
/// Hex distance is the minimum number of steps required to move from `a` to
/// `b` on the grid, where each step travels to any of the six neighbours.
///
/// # Cube-distance formula
///
/// In cube coordinates `(x, y, z)` with `x + y + z = 0`, distance is simply
/// `max(|dx|, |dy|, |dz|)`.  When working in axial `(q, r)` we recover the
/// implicit third cube component as `s = -q - r`.  Substituting and
/// simplifying yields the equivalent axial form:
///
/// ```text
/// distance = (|dq| + |dr| + |dq + dr|) / 2
/// ```
///
/// # Examples
///
/// ```
/// use hexgame::core::{Hex, hex_distance};
///
/// assert_eq!(hex_distance(Hex::ORIGIN, Hex::ORIGIN), 0);
/// assert_eq!(hex_distance(Hex::new(0, 0), Hex::new(3, 0)), 3);
/// assert_eq!(hex_distance(Hex::new(1, 1), Hex::new(-1, -1)), 4);
/// ```
#[inline(always)]
pub fn hex_distance(a: Hex, b: Hex) -> i32 {
    // 1. Compute the delta in axial space.
    let dq = a.q - b.q; // difference along the q (column) axis
    let dr = a.r - b.r; // difference along the r (row) axis

    // 2. Recover the implicit third cube component.
    //    In cube coordinates (x, y, z) we have x = q, z = r, and y = -q - r.
    //    The delta in y is therefore dy = -(dq) - (dr) = -(dq + dr).
    //    Because we only need the absolute value, |dy| == |dq + dr|.
    let ds = dq + dr; // this is the negative of the third cube delta

    // 3. Apply the cube-distance formula adapted for axial coordinates.
    //    In cube form distance = max(|dx|, |dy|, |dz|).
    //    An equivalent expression that avoids the branch inside max() is:
    //    (|dx| + |dy| + |dz|) / 2.
    //    Substituting dx = dq, dz = dr, dy = -ds gives:
    //    (|dq| + |dr| + |ds|) / 2.
    (dq.abs() + dr.abs() + ds.abs()) / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_same_cell() {
        assert_eq!(hex_distance(Hex::ORIGIN, Hex::ORIGIN), 0);
    }

    #[test]
    fn distance_adjacent_along_each_direction() {
        for &(dq, dr) in &HEX_DIRECTIONS {
            assert_eq!(hex_distance(Hex::ORIGIN, Hex::new(dq, dr)), 1);
        }
    }

    #[test]
    fn distance_is_symmetric() {
        let a = Hex::new(3, -1);
        let b = Hex::new(-2, 4);
        assert_eq!(hex_distance(a, b), hex_distance(b, a));
    }

    #[test]
    fn distance_known_values() {
        assert_eq!(hex_distance(Hex::ORIGIN, Hex::new(3, 0)), 3);
        assert_eq!(hex_distance(Hex::ORIGIN, Hex::new(0, 5)), 5);
        assert_eq!(hex_distance(Hex::ORIGIN, Hex::new(2, -2)), 2);
        assert_eq!(hex_distance(Hex::new(1, 1), Hex::new(-1, -1)), 4);
    }

    #[test]
    fn distance_negative_coordinates() {
        assert_eq!(hex_distance(Hex::new(-3, -2), Hex::new(-3, -2)), 0);
        assert_eq!(hex_distance(Hex::new(-1, 0), Hex::new(1, 0)), 2);
    }

    #[test]
    fn hex_display() {
        assert_eq!(format!("{}", Hex::new(3, -1)), "(3, -1)");
        assert_eq!(format!("{}", Hex::ORIGIN), "(0, 0)");
    }

    #[test]
    fn hex_ordering() {
        let mut hexes = vec![Hex::new(1, 0), Hex::new(0, 1), Hex::new(0, 0)];
        hexes.sort();
        assert_eq!(hexes, vec![Hex::new(0, 0), Hex::new(0, 1), Hex::new(1, 0)]);
    }

    #[test]
    fn hex_equality_and_hashing() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Hex::new(0, 0));
        set.insert(Hex::new(1, 0));
        set.insert(Hex::new(0, 0)); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn directions_count() {
        assert_eq!(HEX_DIRECTIONS.len(), 3);
    }
}
