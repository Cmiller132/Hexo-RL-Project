use smallvec::SmallVec;
use crate::core::WindowKey;

/// Compact set of "hot" win-window keys for both players.
///
/// A window is "hot" when it contains at least 4 of one player's stones
/// and 0 of the opponent's stones.
#[derive(Clone, Debug)]
pub struct HotWindows {
    by_player: [SmallVec<[WindowKey; 32]>; 2],
}

impl HotWindows {
    #[inline]
    pub fn new() -> Self {
        Self {
            by_player: [SmallVec::new(), SmallVec::new()],
        }
    }

    #[inline]
    pub fn insert(&mut self, player: u8, k: WindowKey) {
        let vec = &mut self.by_player[player as usize];
        if !vec.contains(&k) {
            vec.push(k);
        }
    }

    #[inline]
    pub fn remove(&mut self, player: u8, k: WindowKey) {
        let vec = &mut self.by_player[player as usize];
        if let Some(idx) = vec.iter().position(|&x| x == k) {
            vec.swap_remove(idx);
        }
    }

    #[inline]
    pub fn iter(&self, player: u8) -> impl Iterator<Item = WindowKey> + '_ {
        self.by_player[player as usize].iter().copied()
    }

    #[inline]
    pub fn len(&self, player: u8) -> usize {
        self.by_player[player as usize].len()
    }

    #[inline]
    pub fn is_empty(&self, player: u8) -> bool {
        self.by_player[player as usize].is_empty()
    }

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
