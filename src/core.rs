//! Hexagonal coordinate type and distance computation.
//!
//! Uses **axial coordinates** `(q, r)`. The three principal hex axes
//! checked for win detection are:
//!
//! | Axis | Direction `(dq, dr)` |
//! |------|----------------------|
//! | A    | `(1, 0)`             |
//! | B    | `(0, 1)`             |
//! | C    | `(1, -1)`            |

use std::fmt;

/// An immutable axial hex coordinate.
///
/// Stores two `i32` fields `q` (column) and `r` (row). Implements `Copy`,
/// `Eq`, `Hash`, and `Ord` so it can be used as a hash-map key or sorted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hex {
    pub q: i32,
    pub r: i32,
}

impl Hex {
    /// Create a new coordinate.
    #[inline(always)]
    pub const fn new(q: i32, r: i32) -> Self {
        Self { q, r }
    }

    /// The origin `(0, 0)`.
    pub const ORIGIN: Self = Self { q: 0, r: 0 };
}

impl fmt::Display for Hex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.q, self.r)
    }
}

impl PartialOrd for Hex {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.q.cmp(&other.q).then(self.r.cmp(&other.r))
    }
}

/// The three line-checking directions: `(1,0)`, `(0,1)`, `(1,-1)`.
///
/// Win detection scans along each of these axes (and their negatives).
pub const HEX_DIRECTIONS: [(i32, i32); 3] = [(1, 0), (0, 1), (1, -1)];

/// Compute the hex distance between two axial coordinates.
///
/// Uses the cube-distance formula:
/// `max(|dq|, |dr|, |dq + dr|)` which equals `(|dq| + |dr| + |dq+dr|) / 2`.
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
    let dq = a.q - b.q;
    let dr = a.r - b.r;
    (dq.abs() + dr.abs() + (dq + dr).abs()) / 2
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
