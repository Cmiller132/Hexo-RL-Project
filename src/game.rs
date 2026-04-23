//! Re-export module for backward compatibility.
//!
//! The core game state lives in [`crate::board`] and the threat analysis
//! methods live in [`crate::threats`]. This module re-exports the public
//! API so that existing code using `crate::game` continues to work.

pub use crate::board::{GameError, HexGameState, MoveRecord};
pub use crate::patterns::{PLACEMENT_RADIUS, WIN_LENGTH};
