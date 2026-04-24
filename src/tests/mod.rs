//! Test-only modules for the Hexgame engine.
//!
//! This directory contains three complementary test suites:
//!
//! - **`patterns`** — Low-level consistency checks for the incremental
//!   evaluation system: pattern table round-trips, hot-window recomputation,
//!   and score restore after `unplace`.
//! - **`threats`** — Property-based tests (`proptest`) that compare the fast
//!   threat-analysis path against a brute-force oracle over hundreds of
//!   randomly-generated board positions.
//! - **`oracle`** — The brute-force reference implementation used by the
//!   threat tests. Exhaustively enumerates all legal turns to classify winning
//!   moves, blocking singles, and blocking pairs.
//!
//! All tests are gated behind `#[cfg(test)]` and do not appear in release
//! builds.

#[cfg(test)]
pub mod oracle;
#[cfg(test)]
pub mod threats;
#[cfg(test)]
pub mod patterns;
