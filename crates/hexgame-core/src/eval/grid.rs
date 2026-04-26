//! Win-grid spatial indexing for incremental evaluation.
//!
//! # What is the win grid?
//!
//! A win grid is a dense, finite lookup table that stores the current pattern
//! index for every possible 6-cell "window" (a potential winning line) near
//! the board origin.  Instead of scanning the board repeatedly to count
//! threats, the engine **incrementally updates** the handful of windows that
//! touch a newly-placed stone.
//!
//! Each window is identified by:
//! * its **origin** `(q, r)` — the first cell of the 6-cell line,
//! * its **direction** `dir ∈ {0,1,2}` — one of the three principal hex axes.
//!
//! The grid is a 3D array conceptually shaped:
//! ```text
//! [WIN_GRID_SIDE][WIN_GRID_SIDE][3]
//! ```
//! flattened to a 1D `Box<[u16; WIN_GRID_TOTAL]>` inside [`EvalState`](crate::eval::state::EvalState).
//!
//! # Why `WIN_GRID_RADIUS = 30`?
//!
//! `PLACEMENT_RADIUS` is 8 (the farthest a single stone may be placed from the
//! origin in one move).  After `k` moves along one axis a stone can reach
//! coordinate `8·k`.  A stone at `(q, r)` spawns windows with origins as far as
//! `(q + 5, r)` in the forward direction.  For `k = 3` moves the worst-case
//! origin is `8·3 + 5 = 29`, which fits inside radius 30.  For `k = 4` the
//! origin would be `37 > 30`, so windows are clipped.
//!
//! Therefore radius 30 covers roughly **3–4 full moves** from the origin along
//! any axis.  Games that extend farther simply do not get incremental
//! evaluation for the out-of-bounds windows; this is a known and acceptable
//! approximation that keeps the table tiny (≈ 11 k entries).

/// Maximum axial distance from the origin stored in the win grid.
///
/// The grid covers coordinates `[-30, 30]` inclusive on both the `q` and `r`
/// axes.  See the module-level documentation for the geometric justification.
pub const WIN_GRID_RADIUS: i32 = 30;

/// Side length of the square win-grid footprint.
///
/// `2 * WIN_GRID_RADIUS + 1 = 61` cells per axis.
pub const WIN_GRID_SIDE: usize = 2 * WIN_GRID_RADIUS as usize + 1;

/// Total number of slots in the flattened win grid.
///
/// `61 * 61 * 3 = 11_163`.  Each slot holds a 16-bit pattern index (0..728).
pub const WIN_GRID_TOTAL: usize = WIN_GRID_SIDE * WIN_GRID_SIDE * 3;

/// Map a 3D win-grid coordinate `(q, r, dir)` to a 1D array index.
///
/// # Arguments
/// * `q`   — window origin axial column.  Must satisfy `|q| ≤ WIN_GRID_RADIUS`.
/// * `r`   — window origin axial row.     Must satisfy `|r| ≤ WIN_GRID_RADIUS`.
/// * `dir` — direction index `0..2` (see [`HEX_DIRECTIONS`](crate::core::HEX_DIRECTIONS)).
///
/// # Panics
///
/// The function does **not** perform runtime bounds checking; callers must
/// guarantee the coordinate is inside the grid, usually by calling
/// [`win_grid_in_bounds`] first.
///
/// # Layout
///
/// Indices are laid out in row-major order with direction as the fastest
/// varying dimension:
/// ```text
/// idx = ((q + R) * SIDE * 3) + ((r + R) * 3) + dir
/// ```
/// where `R = WIN_GRID_RADIUS`.
#[inline]
pub fn win_grid_idx(q: i32, r: i32, dir: u8) -> usize {
    ((q + WIN_GRID_RADIUS) as usize) * WIN_GRID_SIDE * 3
        + ((r + WIN_GRID_RADIUS) as usize) * 3
        + dir as usize
}

/// Runtime guard: is the window origin `(q, r)` inside the finite win grid?
///
/// This check **must remain a runtime guard** (not a `debug_assert!`).  See
/// Fix 5 in `Docs/CODE_REVIEW_FIXES.md` for the full analysis: after roughly
/// four moves along a single axis, window origins can exceed radius 30, so the
/// branch is exercised in long games.  Windows that fall outside the grid are
/// simply skipped; they do not contribute to evaluation.  This is a known
/// approximation, not a bug.
#[inline]
pub fn win_grid_in_bounds(q: i32, r: i32) -> bool {
    let qi = q + WIN_GRID_RADIUS;
    let ri = r + WIN_GRID_RADIUS;
    qi >= 0 && (qi as usize) < WIN_GRID_SIDE && ri >= 0 && (ri as usize) < WIN_GRID_SIDE
}
