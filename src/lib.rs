//! # Hexgame
//!
//! A game engine for **Infinity Hexagonal Tic-Tac-Toe** with PyO3 Python bindings
//! for neural-network-based training.
//!
//! ## Module Architecture
//!
//! | Module | Responsibility |
//! |--------|---------------|
//! | [`core`] | Hex coordinates, distances, directions |
//! | [`board`] | Game state, rules, placement/undo, win detection, legal moves |
//! | [`patterns`] | Ternary 6-cell window encoding, incremental evaluation |
//! | [`threats`] | Hot windows, threat detection, forced-move constraints |
//! | [`encoder`] | Unified 13-channel NN tensor encoder |
//! | [`eval`] | Classical evaluation and feature extraction |
//! | [`search`] | Turn-based alpha-beta search (classical engine) |
//! | [`mcts`] | Neural MCTS with PUCT (kept minimal) |
//! | [`pybridge`] | PyO3 bindings exposing engine to Python |
//!
//! The old `game` module is now a thin re-export for backward compatibility.
//! Its contents have been split into [`board`] (state), [`patterns`] (evaluation),
//! and [`threats`] (tactics).
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
pub mod game;
pub mod mcts;
pub mod patterns;
pub mod search;
pub mod threats;

#[cfg(feature = "python")]
mod pybridge;

// Re-exports for convenient access
pub use core::{hex_distance, Hex, HEX_DIRECTIONS};
pub use eval::{evaluate, extract_features, FEATURE_COUNT};
pub use game::{GameError, HexGameState, MoveRecord, PLACEMENT_RADIUS, WIN_LENGTH};
