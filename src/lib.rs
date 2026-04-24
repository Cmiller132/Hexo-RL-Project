//! # Hexgame
//!
//! A game engine for **Infinity Hexagonal Tic-Tac-Toe** with PyO3 Python bindings
//! for neural-network-based training.
//!
//! ## Module Architecture
//!
//! | Module | Responsibility |
//! |--------|---------------|
//! | [`core`] | Hex coordinates, distances, directions, `Turn`, `WindowKey` |
//! | [`board`] | Game state, rules, placement/undo, win detection, legal moves |
//! | [`eval`] | Pattern tables, incremental evaluation (`EvalState`), feature extraction |
//! | [`threats`] | Threat detection, forced-move constraints (`ThreatStatus`) |
//! | [`encoder`] | Unified 13-channel NN tensor encoder |
//! | [`search`] | Turn-based alpha-beta search (classical engine) |
//! | [`mcts`] | Neural MCTS with PUCT |
//! | [`pybridge`] | PyO3 bindings exposing engine to Python |
//!
//! ## Layer dependency
//!
//! ```text
//! core → eval → board → threats → {search, mcts, encoder} → py
//!                              ↑
//!                          tests/oracle  (test-only)
//! ```
//!
//! ## Rules
//!
//! - Two players (0 and 1) take turns on an infinite hexagonal grid using
//!   axial coordinates `(q, r)`.
//! - Player 0 opens with **one** tile at the origin `(0, 0)`.
//! - Every subsequent turn consists of **two** placements.
//! - Each placement must land on an empty hex within [`PLACEMENT_RADIUS`]
//!   (8) of any existing tile.
//! - The first player to form [`WIN_LENGTH`] (6) tiles in a contiguous
//!   straight line along any of the three hex axes wins.
//! - The board is infinite — there is no draw.

pub mod board;
pub mod core;
pub mod encoder;
pub mod eval;
pub mod mcts;
pub mod search;
pub mod threats;

#[cfg(feature = "python")]
mod pybridge;

#[cfg(test)]
mod tests;

// Re-exports for convenient access
pub use core::{hex_distance, Hex, Turn, HEX_DIRECTIONS, PLACEMENT_RADIUS, WIN_LENGTH};
pub use eval::{extract_features, FEATURE_COUNT};
pub use board::{GameError, HexGameState, MoveRecord};
