use crate::core::WindowKey;
use crate::eval::hot::*;

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
