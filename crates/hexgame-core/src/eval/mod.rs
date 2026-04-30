//! Classical pattern-based feature extraction for Hexo.
//!
//! This module provides one complementary evaluation mechanism:
//!
//! * **Incremental evaluation** ([`EvalState`]) — used during search.
//!   Updates only the windows touched by a newly placed stone, making
//!   `place` and `unplace` `O(1)`.
//!
//! # Sub-modules
//!
//! | Module      | Purpose                                               |
//! |-------------|-------------------------------------------------------|
//! | `grid`      | Win-grid spatial indexing and bounds checks           |
//! | `hot`       | Zero-alloc cache of urgent (4+ stone) threat windows  |
//! | `patterns`  | Pre-computed ternary pattern tables (729 entries)     |
//! | `state`     | [`EvalState`], [`ThreatCounts`], incremental update    |

pub mod grid;
pub mod hot;
pub mod patterns;
pub mod state;
