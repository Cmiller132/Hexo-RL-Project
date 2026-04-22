//! # Hexgame
//!
//! A game engine for **Infinity Hexagonal Tic-Tac-Toe** with a Python ML
//! training pipeline.
//!
//! ## Architecture
//!
//! - **Rust** — fast game engine, classical alpha-beta search, feature
//!   extraction, and PyO3 bindings for Python interop.
//! - **Python** — PyTorch neural network, MCTS, training loop, and
//!   evaluation (in the `python/hexgame/` package).
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

pub mod core;
pub mod eval;
pub mod game;
pub mod mcts;
pub mod search;

#[cfg(feature = "python")]
mod pybridge;

pub use core::{hex_distance, Hex, HEX_DIRECTIONS};
pub use eval::{evaluate, extract_features, score_move, FEATURE_COUNT};
pub use game::{GameError, HexGameState, MoveRecord, PLACEMENT_RADIUS, WIN_LENGTH};
