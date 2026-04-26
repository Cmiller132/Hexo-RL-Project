//! Hot-window tracking — a zero-alloc cache of urgent win threats.
//!
//! # What is a "hot" window?
//!
//! A window (a 6-cell line) is **hot** for player `P` when it contains at
//! least 4 of `P`'s stones and **zero** opponent stones.  Hot windows are the
//! most dangerous threats on the board: one more stone by `P` creates a
//! five-in-six (an immediate win on the next turn), and two more stones create
//! an unavoidable six-in-a-row.
//!
//! # Why track them separately?
//!
//! During MCTS / search the engine needs to answer two questions quickly:
//! 1. Does the current player have any urgent threats?  (fast win detection)
//! 2. Does the opponent have threats that must be blocked?  (defensive pruning)
//!
//! Re-scanning the entire board to find these windows would be `O(n)`.  By
//! incrementally maintaining a small set of hot windows inside
//! [`EvalState`](crate::eval::state::EvalState), the query becomes `O(1)`
//! (just check whether the set is empty) and iteration is `O(k)` where `k` is
//! the number of hot windows (typically 0–10 in normal play, rarely above 20).
//!
//! # Zero-allocation design
//!
//! `HotWindows` stores keys in a [`SmallVec<[WindowKey; 32]>`](smallvec::SmallVec)
//! per player.  The inline buffer holds 32 entries without touching the heap.
//! In practice this is never exhausted; if it were, `SmallVec` would spill to
//! the heap transparently, but that path is effectively unreachable in real
//! games.

use crate::core::WindowKey;
use smallvec::SmallVec;

/// Compact set of "hot" win-window keys for both players.
///
/// A window is "hot" when it contains at least 4 of one player's stones
/// and 0 of the opponent's stones.  These are the highest-priority threats
/// on the board.
///
/// # Invariants
/// * `by_player[p]` never contains duplicate `WindowKey`s.
/// * Every key in `by_player[p]` genuinely satisfies the hot condition
///   (this is enforced by `EvalState` callers).
#[derive(Clone, Debug)]
pub struct HotWindows {
    by_player: [SmallVec<[WindowKey; 32]>; 2],
}

impl HotWindows {
    /// Create an empty `HotWindows` with no entries for either player.
    #[inline]
    pub fn new() -> Self {
        Self {
            by_player: [SmallVec::new(), SmallVec::new()],
        }
    }

    /// Insert `k` into the hot set for `player`.
    ///
    /// # Arguments
    /// * `player` — `0` or `1`.
    /// * `k`      — the window key to mark as hot.
    ///
    /// # Complexity
    /// Amortised `O(1)` push onto the inline `SmallVec`.  Duplicate insertion
    /// is a logic bug (caught by `debug_assert!` in debug builds).
    #[inline]
    pub fn insert(&mut self, player: u8, k: WindowKey) {
        let vec = &mut self.by_player[player as usize];
        debug_assert!(!vec.contains(&k), "duplicate hot-window insertion");
        vec.push(k);
    }

    /// Remove `k` from the hot set for `player`, if present.
    ///
    /// # Arguments
    /// * `player` — `0` or `1`.
    /// * `k`      — the window key to remove.
    ///
    /// # Complexity
    /// `O(n)` to find the key, then `O(1)` `swap_remove`.
    #[inline]
    pub fn remove(&mut self, player: u8, k: WindowKey) {
        let vec = &mut self.by_player[player as usize];
        if let Some(idx) = vec.iter().position(|&x| x == k) {
            vec.swap_remove(idx);
        }
    }

    /// Iterate over the hot windows for `player`.
    ///
    /// Yields owned `WindowKey` values (they are `Copy`, so this is free).
    #[inline]
    pub fn iter(&self, player: u8) -> impl Iterator<Item = WindowKey> + '_ {
        self.by_player[player as usize].iter().copied()
    }

    /// Number of hot windows currently tracked for `player`.
    #[inline]
    pub fn len(&self, player: u8) -> usize {
        self.by_player[player as usize].len()
    }

    /// Returns `true` if `player` has no hot windows.
    #[inline]
    pub fn is_empty(&self, player: u8) -> bool {
        self.by_player[player as usize].is_empty()
    }

    /// Clear both players' hot sets, returning to an empty state.
    #[inline]
    #[cfg(test)]
    pub fn clear(&mut self) {
        self.by_player[0].clear();
        self.by_player[1].clear();
    }
}

impl Default for HotWindows {
    fn default() -> Self {
        Self::new()
    }
}
