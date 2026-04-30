//! Axial hex coordinates and distance for the Hexo engine.
//!
//! This module provides the fundamental spatial primitive [`Hex`] using **axial
//! coordinates** `(q, r)`.  Axial coordinates are a compact two-number
//! representation of hex-grid positions.  They are equivalent to cube
//! coordinates `(x, y, z)` with the constraint `x + y + z = 0`; here `x = q`,
//! `z = r`, and `y = -q - r` is implicit.
//!
//! # Principal axes for win detection
//!
//! In Hexo a win is a straight line of 6 stones.  On a hex grid
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
    /// A total order is required so that [`Turn::pair`] can canonicalise
    /// two cells (smaller first).  A deterministic lexicographic order on
    /// [`Hex`] guarantees that the same physical move always hashes to the
    /// same key, which keeps transposition tables and `HashSet`s consistent
    /// regardless of argument order.  It also makes `Hex` usable in
    /// `BTreeMap` / `BTreeSet`.
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hex {
    /// Lexicographic ordering: first by `q`, then by `r`.
    ///
    /// This order is the foundation of canonical pair storage in
    /// [`Turn::pair`].  Without a deterministic total order, swapping the
    /// arguments to `pair` would produce distinct values, breaking equality
    /// and hashing invariants used by transposition tables and `HashSet`s.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.q.cmp(&other.q).then(self.r.cmp(&other.r))
    }
}

/// A turn consists of 1 or 2 placements.
///
/// In Hexo the opening turn has exactly one placement; every subsequent turn
/// has two.  [`Turn`] captures either case in a single compact value.
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
        Turn {
            first: h,
            second: None,
        }
    }

    /// Create a two-placement turn with canonical ordering.
    ///
    /// The two cells are reordered so that the smaller [`Hex`] is stored in
    /// `first`.  This ensures hash and equality consistency: swapping the
    /// arguments produces an identical `Turn`.
    #[inline]
    pub fn pair(a: Hex, b: Hex) -> Self {
        // A turn must place two distinct stones; self-pairs are a logic error
        // in every caller and would corrupt transposition-table keys.
        debug_assert_ne!(a, b, "Turn::pair requires two distinct cells");
        // Canonicalize: smaller Hex first.  This makes Turn equality and
        // hashing independent of argument order.
        if a <= b {
            Turn {
                first: a,
                second: Some(b),
            }
        } else {
            Turn {
                first: b,
                second: Some(a),
            }
        }
    }

    /// The first (or only) placement of this turn.
    ///
    /// For a two-stone turn this is the lexicographically smaller cell,
    /// which callers rely on when applying or undoing moves.
    #[inline]
    pub const fn first(self) -> Hex {
        self.first
    }

    /// The second placement, if this is a two-stone turn.
    ///
    /// When present, `first() <= second()` is guaranteed so that search
    /// and encoding layers can treat moves as canonical values.
    #[inline]
    pub const fn second(self) -> Option<Hex> {
        self.second
    }

    /// How many individual placements this turn contains (1 or 2).
    ///
    /// Search and encoding layers need this count to know how many stones
    /// to commit or undo in a single turn.
    #[inline]
    pub const fn placements(self) -> u8 {
        if self.second.is_some() {
            2
        } else {
            1
        }
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

/// Maximum distance from any existing stone at which a non-opening stone may be placed.
pub const PLACEMENT_RADIUS: i32 = 8;

/// A compact key representing a sliding "window" of cells along one of the
/// three principal hex directions.
///
/// `WindowKey` replaces the old `(i32, i32, u8)` tuple for hot-window lookups.
/// It stores full `i32` coordinates plus a validated direction index. This is
/// intentionally wider than the bounded eval grid: release builds must never
/// alias distant windows by truncating coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowKey {
    q: i32,
    r: i32,
    dir: u8,
}

impl WindowKey {
    /// Create a new `WindowKey` from axial coordinates and a direction index.
    ///
    /// # Panics
    ///
    /// Panics if `dir` is not a valid `HEX_DIRECTIONS` index (0..3).  Use
    /// [`WindowKey::try_new`] for runtime direction values that should be
    /// rejected without panicking.
    #[inline(always)]
    pub const fn new(q: i32, r: i32, dir: u8) -> Self {
        assert!(dir < 3, "dir must be a valid HEX_DIRECTIONS index (0..3)");
        Self { q, r, dir }
    }

    /// Fallibly create a `WindowKey` from axial coordinates and a runtime
    /// direction index.
    #[inline(always)]
    pub const fn try_new(q: i32, r: i32, dir: u8) -> Option<Self> {
        if dir < 3 {
            Some(Self { q, r, dir })
        } else {
            None
        }
    }

    /// Extract the `q` coordinate.
    #[inline(always)]
    pub const fn q(self) -> i32 {
        self.q
    }

    /// Extract the `r` coordinate.
    #[inline(always)]
    pub const fn r(self) -> i32 {
        self.r
    }

    /// Extract the direction index (`0..3`).
    #[inline(always)]
    pub const fn dir(self) -> u8 {
        self.dir
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

/// Compute the hex distance between two axial coordinates.
///
/// Returns the minimum number of steps required to move from `a` to `b`,
/// using the axial form of the cube-distance formula:
///
/// ```text
/// distance = (|dq| + |dr| + |dq + dr|) / 2
/// ```
///
/// # Examples
///
/// ```
/// use hexgame_core::rules::{hex_distance, Hex};
///
/// assert_eq!(hex_distance(Hex::ORIGIN, Hex::ORIGIN), 0);
/// assert_eq!(hex_distance(Hex::new(0, 0), Hex::new(3, 0)), 3);
/// assert_eq!(hex_distance(Hex::new(1, 1), Hex::new(-1, -1)), 4);
/// ```
#[inline(always)]
pub fn hex_distance(a: Hex, b: Hex) -> i32 {
    let dq = a.q - b.q;
    let dr = a.r - b.r;
    let ds = dq + dr;
    (dq.abs() + dr.abs() + ds.abs()) / 2
}
