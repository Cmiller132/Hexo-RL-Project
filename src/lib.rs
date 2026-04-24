//! # Hexgame
//!
//! A high-performance game engine for **Infinity Hexagonal Tic-Tac-Toe**,
//! a variant of Hex played on an infinite hexagonal grid.
//!
//! ## Game Variant
//!
//! - Two players (0 and 1) alternate turns on an infinite hexagonal board using
//!   axial coordinates `(q, r)`.
//! - Player 0 opens with **one** tile at the origin `(0, 0)`.
//! - Every subsequent turn consists of **two** placements (except the opening).
//! - Each placement must land on an empty hex within [`PLACEMENT_RADIUS`]
//!   (8 cells) of any existing tile.
//! - The first player to form [`WIN_LENGTH`] (6) tiles in a contiguous straight
//!   line along any of the three hex axes wins.
//! - The board is infinite — there is no draw condition.
//!
//! This variant is sometimes called "6-in-a-row Hex" or "Infinity Hex".
//! The two-placements-per-turn rule dramatically changes tactical and strategic
//! considerations compared to standard single-placement Hex.
//!
//! ## Module Architecture
//!
//! | Module | Responsibility |
//! |--------|---------------|
//! | [`core`] | Hex coordinates, distances, directions, [`Turn`], [`WindowKey`] |
//! | [`eval`] | Pattern tables, incremental evaluation ([`EvalState`]), threat counts |
//! | [`board`] | Game state, rules, placement/undo, win detection, legal moves |
//! | [`threats`] | Threat detection, forced-move constraints ([`ThreatStatus`]) |
//! | [`encoder`] | Unified 13-channel neural-network tensor encoder |
//! | [`search`] | Turn-based alpha-beta search (classical engine) |
//! | [`mcts`] | Neural MCTS with PUCT |
//! | [`pybridge`] | PyO3 bindings exposing the engine to Python (optional `python` feature) |
//!
//! ## Dependency Graph
//!
//! ```text
//! core → eval → board → threats → {search, mcts, encoder} → pybridge
//!                              ↑
//!                          tests/oracle  (test-only brute-force verifier)
//! ```
//!
//! ## Key Types
//!
//! - [`HexGameState`] — the main game state; supports incremental place/unplace.
//! - [`Turn`] — a single placement or a pair of placements.
//! - [`ThreatStatus`](threats::ThreatStatus) — the threat classification of a position (winning, must-block, etc.).
//! - [`EvalState`](eval::state::EvalState) — incremental pattern evaluation with `O(1)` updates per stone.
//!
//! ## Testing Strategy
//!
//! Correctness is validated by a brute-force **oracle** (`tests/oracle`) that
//! exhaustively enumerates every legal turn for small-to-medium positions and
//! compares the result against the incremental fast paths. Property-based tests
//! (`proptest`) run this comparison over hundreds of randomly-generated game
//! positions.

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
