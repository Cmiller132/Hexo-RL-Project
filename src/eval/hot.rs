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
//! the number of hot windows (typically < 20 in normal play).
//!
//! # Zero-allocation design
//!
//! `HotWindows` stores keys in a [`SmallVec<[WindowKey; 32]>`](smallvec::SmallVec)
//! per player.  The inline buffer holds 32 entries without touching the heap.
//! In practice this is never exhausted; if it were, `SmallVec` would spill to
//! the heap transparently, but that path is effectively unreachable in real
//! games.

use smallvec::SmallVec;
use crate::core::WindowKey;

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

    /// Insert `k` into the hot set for `player` if it is not already present.
    ///
    /// # Arguments
    /// * `player` — `0` or `1`.
    /// * `k`      — the window key to mark as hot.
    ///
    /// # Complexity
    /// `O(n)` in the number of hot windows for `player` because of the
    /// `contains` guard.  Since `n` is tiny (≈ 0–10) this is cheaper than
    /// using a hash table.
    #[inline]
    pub fn insert(&mut self, player: u8, k: WindowKey) {
        let vec = &mut self.by_player[player as usize];
        if !vec.contains(&k) {
            vec.push(k);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn k(q: i32, r: i32, dir: u8) -> WindowKey {
        WindowKey::new(q, r, dir)
    }

    #[test]
    fn new_is_empty() {
        let hw = HotWindows::new();
        assert!(hw.is_empty(0));
        assert!(hw.is_empty(1));
    }

    #[test]
    fn insert_and_len() {
        let mut hw = HotWindows::new();
        hw.insert(0, k(0, 0, 0));
        assert_eq!(hw.len(0), 1);
        assert_eq!(hw.len(1), 0);
    }

    #[test]
    fn duplicate_insert_is_idempotent() {
        let mut hw = HotWindows::new();
        let key = k(1, 2, 0);
        hw.insert(0, key);
        hw.insert(0, key);
        assert_eq!(hw.len(0), 1);
    }

    #[test]
    fn remove_existing() {
        let mut hw = HotWindows::new();
        hw.insert(0, k(0, 0, 0));
        hw.insert(0, k(1, 0, 0));
        hw.remove(0, k(0, 0, 0));
        assert_eq!(hw.len(0), 1);
    }

    #[test]
    fn clear_resets_everything() {
        let mut hw = HotWindows::new();
        hw.insert(0, k(0, 0, 0));
        hw.insert(1, k(1, 1, 1));
        hw.clear();
        assert!(hw.is_empty(0));
        assert!(hw.is_empty(1));
    }
}
