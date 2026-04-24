//! Test-only modules for the Hexgame engine.
//!
//! This directory contains ten complementary test modules:
//!
//! | Module             | Purpose |
//! |--------------------|---------|
//! | `board`            | Board rules and win detection |
//! | `core`             | Hex coordinates, turns, and window keys |
//! | `encoder`          | Neural-network tensor encoding |
//! | `eval_state`       | Incremental evaluation consistency |
//! | `grid`             | Win-grid spatial indexing |
//! | `hot`              | Hot-window cache correctness |
//! | `oracle`           | Brute-force threat-analysis verifier |
//! | `patterns`         | Pattern table integrity |
//! | `threats`          | Property-based threat-analysis verification |
//! | `threats_internal` | Low-level threat constraint logic |
//!
//! All tests are gated behind `#[cfg(test)]` and do not appear in release
//! builds.

#[cfg(test)]
pub mod board;
#[cfg(test)]
pub mod core;
#[cfg(test)]
pub mod encoder;
#[cfg(test)]
pub mod eval_state;
#[cfg(test)]
pub mod grid;
#[cfg(test)]
pub mod hot;
#[cfg(test)]
pub mod oracle;
#[cfg(test)]
pub mod patterns;
#[cfg(test)]
pub mod threats;
#[cfg(test)]
pub mod threats_internal;
