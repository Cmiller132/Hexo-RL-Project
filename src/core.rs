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

/// A turn consists of 1 or 2 placements.
///
/// In Infinity Hexagonal Tic-Tac-Toe the opening turn has exactly one
/// placement; every subsequent turn has two.  [`Turn`] captures either case
/// in a single compact value.
///
/// # Canonical ordering invariant
///
/// When a turn contains two placements they are stored in **sorted order**
/// (`first <= second`).  This guarantees that `Turn::pair(a, b) ==
/// Turn::pair(b, a)`, which is essential for transposition-table consistency
/// — the same physical move must always hash to the same key regardless of
/// the order in which the two cells were supplied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Turn {
    /// The first (or only) cell of this turn.  For a two-placement turn this
    /// is the lexicographically smaller cell.
    first: Hex,
    /// The second cell, if any.  When present, `first <= second` is guaranteed.
    second: Option<Hex>,
}

impl Turn {
    /// Create a single-placement turn.
    ///
    /// Used for the opening move (Player 0 places exactly one stone at the
    /// origin) and for one-stone test positions.
    #[inline]
    pub const fn single(h: Hex) -> Self {
        Turn { first: h, second: None }
    }

    /// Create a two-placement turn with canonical ordering.
    ///
    /// The two cells are reordered so that the smaller [`Hex`] is stored in
    /// `first`.  This ensures hash and equality consistency: swapping the
    /// arguments produces an identical `Turn`.
    #[inline]
    pub fn pair(a: Hex, b: Hex) -> Self {
        // Canonicalize: smaller Hex first.  This makes Turn equality and
        // hashing independent of argument order.
        if a <= b {
            Turn { first: a, second: Some(b) }
        } else {
            Turn { first: b, second: Some(a) }
        }
    }

    /// The first (or only) placement of this turn.
    #[inline]
    pub const fn first(self) -> Hex {
        self.first
    }

    /// The second placement, if this is a two-stone turn.
    #[inline]
    pub const fn second(self) -> Option<Hex> {
        self.second
    }

    /// How many individual placements this turn contains (1 or 2).
    #[inline]
    pub const fn placements(self) -> u8 {
        if self.second.is_some() { 2 } else { 1 }
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

/// Number of stones in a row required to win.
pub const WIN_LENGTH: i32 = 6;

/// Maximum distance from the origin at which a stone may be placed.
pub const PLACEMENT_RADIUS: i32 = 8;

/// A compact key representing a sliding "window" of cells along one of the
/// three principal hex directions.
///
/// `WindowKey` replaces the old `(i32, i32, u8)` tuple for hot-window lookups.
/// It packs `q`, `r` and `dir` into a single `u32` so that it can be used
/// directly as a cheap `HashMap` / `HashSet` key without heap allocation.
///
/// # Bit layout
///
/// ```text
/// [ dir:2 | r:15 (signed, -16384..16383) | q:15 (signed, -16384..16383) ]
/// ```
///
/// Both coordinates are stored in two's-complement form and sign-extended on
/// extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowKey(u32);

impl WindowKey {
    /// Create a new `WindowKey` from axial coordinates and a direction index.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `q` or `r` are outside the range
    /// `-16384..=16383` or if `dir` is larger than `3`.
    #[inline(always)]
    pub const fn new(q: i32, r: i32, dir: u8) -> Self {
        debug_assert!(
            q >= -16384 && q <= 16383,
            "q coordinate out of 15-bit signed range"
        );
        debug_assert!(
            r >= -16384 && r <= 16383,
            "r coordinate out of 15-bit signed range"
        );
        debug_assert!(dir < 4, "dir must fit in 2 bits (0..3)");

        let q_bits = (q as u32) & 0x7FFF;
        let r_bits = (r as u32) & 0x7FFF;
        let dir_bits = (dir as u32) & 0x3;
        Self((dir_bits << 30) | (r_bits << 15) | q_bits)
    }

    /// Extract the `q` coordinate (15-bit signed, sign-extended to `i32`).
    #[inline(always)]
    pub const fn q(self) -> i32 {
        sign_extend_15(self.0 & 0x7FFF)
    }

    /// Extract the `r` coordinate (15-bit signed, sign-extended to `i32`).
    #[inline(always)]
    pub const fn r(self) -> i32 {
        sign_extend_15((self.0 >> 15) & 0x7FFF)
    }

    /// Extract the direction index (`0..3`).
    #[inline(always)]
    pub const fn dir(self) -> u8 {
        ((self.0 >> 30) & 0x3) as u8
    }

    /// Return the [`Hex`] at `offset` steps along this window's direction.
    ///
    /// `offset` may be negative to step backward.  The direction vector is
    /// taken from [`HEX_DIRECTIONS`].
    #[inline(always)]
    pub fn cell_at(self, offset: i32) -> Hex {
        let (dq, dr) = HEX_DIRECTIONS[self.dir() as usize];
        Hex::new(self.q() + dq * offset, self.r() + dr * offset)
    }
}

/// Sign-extend a 15-bit two's-complement value to a full `i32`.
const fn sign_extend_15(raw: u32) -> i32 {
    // The sign bit for a 15-bit value is bit 14 (0x4000).
    if raw & 0x4000 != 0 {
        // Set all upper bits to 1 so the i32 interpretation becomes negative.
        (raw | 0xFFFF8000) as i32
    } else {
        raw as i32
    }
}

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

    // ------------------------------------------------------------------
    // WindowKey round-trip and behaviour tests
    // ------------------------------------------------------------------

    #[test]
    fn windowkey_roundtrip_positive() {
        let key = WindowKey::new(42, 100, 2);
        assert_eq!(key.q(), 42);
        assert_eq!(key.r(), 100);
        assert_eq!(key.dir(), 2);
    }

    #[test]
    fn windowkey_roundtrip_negative() {
        let key = WindowKey::new(-100, -42, 1);
        assert_eq!(key.q(), -100);
        assert_eq!(key.r(), -42);
        assert_eq!(key.dir(), 1);
    }

    #[test]
    fn windowkey_roundtrip_mixed() {
        let key = WindowKey::new(-16384, 16383, 3);
        assert_eq!(key.q(), -16384);
        assert_eq!(key.r(), 16383);
        assert_eq!(key.dir(), 3);
    }

    #[test]
    fn windowkey_roundtrip_zero() {
        let key = WindowKey::new(0, 0, 0);
        assert_eq!(key.q(), 0);
        assert_eq!(key.r(), 0);
        assert_eq!(key.dir(), 0);
    }

    #[test]
    fn windowkey_max_positive_coords() {
        let key = WindowKey::new(16383, 16383, 0);
        assert_eq!(key.q(), 16383);
        assert_eq!(key.r(), 16383);
    }

    #[test]
    fn windowkey_equality_and_hashing() {
        use std::collections::HashSet;
        let a = WindowKey::new(1, 2, 0);
        let b = WindowKey::new(1, 2, 0);
        let c = WindowKey::new(1, 2, 1);
        assert_eq!(a, b);
        assert_ne!(a, c);

        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b); // duplicate
        set.insert(c);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn windowkey_copy_trait() {
        let a = WindowKey::new(5, -5, 2);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn windowkey_cell_at_along_axis_a() {
        // Direction 0 is (1, 0).
        let key = WindowKey::new(0, 0, 0);
        assert_eq!(key.cell_at(0), Hex::new(0, 0));
        assert_eq!(key.cell_at(1), Hex::new(1, 0));
        assert_eq!(key.cell_at(5), Hex::new(5, 0));
        assert_eq!(key.cell_at(-1), Hex::new(-1, 0));
    }

    #[test]
    fn windowkey_cell_at_along_axis_b() {
        // Direction 1 is (0, 1).
        let key = WindowKey::new(3, -2, 1);
        assert_eq!(key.cell_at(0), Hex::new(3, -2));
        assert_eq!(key.cell_at(2), Hex::new(3, 0));
        assert_eq!(key.cell_at(-2), Hex::new(3, -4));
    }

    #[test]
    fn windowkey_cell_at_along_axis_c() {
        // Direction 2 is (1, -1).
        let key = WindowKey::new(1, 1, 2);
        assert_eq!(key.cell_at(0), Hex::new(1, 1));
        assert_eq!(key.cell_at(1), Hex::new(2, 0));
        assert_eq!(key.cell_at(-1), Hex::new(0, 2));
    }

    #[test]
    fn windowkey_size_is_u32() {
        assert_eq!(std::mem::size_of::<WindowKey>(), 4);
    }

    // ------------------------------------------------------------------
    // Turn tests
    // ------------------------------------------------------------------

    #[test]
    fn turn_single() {
        let t = Turn::single(Hex::new(1, 2));
        assert_eq!(t.first(), Hex::new(1, 2));
        assert_eq!(t.second(), None);
        assert_eq!(t.placements(), 1);
    }

    #[test]
    fn turn_pair_ordered() {
        let t = Turn::pair(Hex::new(1, 2), Hex::new(3, 4));
        assert_eq!(t.first(), Hex::new(1, 2));
        assert_eq!(t.second(), Some(Hex::new(3, 4)));
        assert_eq!(t.placements(), 2);
    }

    #[test]
    fn turn_pair_canonical_ordering() {
        // When passed in reverse order, pair should canonicalize (smaller first).
        let t = Turn::pair(Hex::new(3, 4), Hex::new(1, 2));
        assert_eq!(t.first(), Hex::new(1, 2));
        assert_eq!(t.second(), Some(Hex::new(3, 4)));
        assert_eq!(t.placements(), 2);
    }

    #[test]
    fn turn_pair_equal_hexes() {
        let t = Turn::pair(Hex::new(2, 2), Hex::new(2, 2));
        assert_eq!(t.first(), Hex::new(2, 2));
        assert_eq!(t.second(), Some(Hex::new(2, 2)));
        assert_eq!(t.placements(), 2);
    }

    #[test]
    fn turn_copy_trait() {
        let a = Turn::single(Hex::new(0, 0));
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn turn_equality_and_hashing() {
        use std::collections::HashSet;
        let a = Turn::pair(Hex::new(1, 0), Hex::new(0, 1));
        let b = Turn::pair(Hex::new(0, 1), Hex::new(1, 0));
        assert_eq!(a, b);

        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b); // duplicate after canonicalization
        assert_eq!(set.len(), 1);
    }
}
