//! # Hexgame
//!
//! A high-performance game engine for **Hexo**, a game most similar to Connect 6
//! played on an infinite hexagonal grid.
//!
//! ## Game Rules
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
//! Hexo is a new game; it shares the hexagonal grid geometry with Hex but has
//! no relation to Hex as a game. The two-placements-per-turn rule (like Connect 6)
//! dramatically changes tactical and strategic considerations compared to
//! single-placement games.
//!
//! ## Module Architecture
//!
//! | Facade | Responsibility |
//! |--------|---------------|
//! | crate root / [`rules`] | Hex coordinates, rules, turns, and game state |
//! | [`encoding`] | Unified 13-channel neural-network tensor encoder |
//! | [`tactics`] | Complete tactical status and mask helpers |
//! | [`classical`] | Turn-based alpha-beta search |
//! | [`mcts`] | Neural MCTS with PUCT, exposed for the Python FFI crate |
//!
//! ## Dependency Graph
//!
//! ```text
//! core → eval → board → threats → {search, mcts, encoder}
//!                              ↑
//!                          tests/oracle  (test-only brute-force verifier)
//! ```
//!
//! ## Key Types
//!
//! - [`HexGameState`](rules::HexGameState) — the main game state; supports incremental place/unplace.
//! - [`Turn`](rules::Turn) — a single placement or a pair of placements.
//! - [`TacticalStatus`](tactics::TacticalStatus) — the tactical classification of a position (winning, must-block, etc.).
//!
//! ## Testing Strategy
//!
//! Correctness is validated by a brute-force **oracle** (`tests/oracle`) that
//! exhaustively enumerates every legal turn for small-to-medium positions and
//! compares the result against the incremental fast paths. Property-based tests
//! (`proptest`) run this comparison over hundreds of randomly-generated game
//! positions.

mod board;
mod core;
mod encoder;
mod eval;
pub mod mcts;
mod search;
mod threats;
pub mod v1;
pub mod v1_pair_search;

#[cfg(test)]
mod tests;

/// Stable rules and board-state facade.
pub mod rules {
    pub use crate::board::{GameError, HexGameState, MoveRecord};
    pub use crate::core::{
        hex_distance, Hex, Turn, WindowKey, HEX_DIRECTIONS, PLACEMENT_RADIUS, WIN_LENGTH,
    };
}

/// Stable tensor-encoding facade.
pub mod encoding {
    pub use crate::encoder::{
        encode_board, encode_board_into, extract_features, EncodedBoard, BOARD_AREA, BOARD_SIZE,
        FEATURE_COUNT, HALF_BOARD, NUM_CHANNELS, TENSOR_SIZE, WIN_SCORE,
    };
}

/// Stable tactical-analysis facade.
pub mod tactics {
    pub use crate::threats::{
        live_cells, tactical_mask_cells, tactical_status, turn_satisfies_tactical, BlockConstraint,
        TacticalStatus,
    };
}

/// Stable classical-search facade.
pub mod classical {
    pub use crate::search::{iterative_deepening, SearchResult};
}
