use crate::core::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_same_cell() {
        assert_eq!(hex_distance(Hex::ORIGIN, Hex::ORIGIN), 0);
    }

    #[test]
    fn distance_adjacent_along_each_direction() {
        for &(dq, dr) in &HEX_DIRECTIONS {
            assert_eq!(hex_distance(Hex::ORIGIN, Hex::new(dq, dr)), 1);
        }
    }

    #[test]
    fn distance_is_symmetric() {
        let a = Hex::new(3, -1);
        let b = Hex::new(-2, 4);
        assert_eq!(hex_distance(a, b), hex_distance(b, a));
    }

    #[test]
    fn distance_known_values() {
        assert_eq!(hex_distance(Hex::ORIGIN, Hex::new(3, 0)), 3);
        assert_eq!(hex_distance(Hex::ORIGIN, Hex::new(0, 5)), 5);
        assert_eq!(hex_distance(Hex::ORIGIN, Hex::new(2, -2)), 2);
        assert_eq!(hex_distance(Hex::new(1, 1), Hex::new(-1, -1)), 4);
    }

    #[test]
    fn distance_negative_coordinates() {
        assert_eq!(hex_distance(Hex::new(-3, -2), Hex::new(-3, -2)), 0);
        assert_eq!(hex_distance(Hex::new(-1, 0), Hex::new(1, 0)), 2);
    }

    #[test]
    fn hex_display() {
        assert_eq!(format!("{}", Hex::new(3, -1)), "(3, -1)");
        assert_eq!(format!("{}", Hex::ORIGIN), "(0, 0)");
    }

    #[test]
    fn hex_ordering() {
        let mut hexes = vec![Hex::new(1, 0), Hex::new(0, 1), Hex::new(0, 0)];
        hexes.sort();
        assert_eq!(hexes, vec![Hex::new(0, 0), Hex::new(0, 1), Hex::new(1, 0)]);
    }

    #[test]
    fn hex_equality_and_hashing() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Hex::new(0, 0));
        set.insert(Hex::new(1, 0));
        set.insert(Hex::new(0, 0)); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn directions_count() {
        assert_eq!(HEX_DIRECTIONS.len(), 3);
    }

    // ------------------------------------------------------------------
    // WindowKey round-trip and behaviour tests
    // ------------------------------------------------------------------

    #[test]
    fn windowkey_roundtrip_positive() {
        let key = WindowKey::new(42, 100, 2);
        assert_eq!(key.q(), 42);
        assert_eq!(key.r(), 100);
        assert_eq!(key.dir(), 2);
    }

    #[test]
    fn windowkey_roundtrip_negative() {
        let key = WindowKey::new(-100, -42, 1);
        assert_eq!(key.q(), -100);
        assert_eq!(key.r(), -42);
        assert_eq!(key.dir(), 1);
    }

    #[test]
    fn windowkey_roundtrip_mixed() {
        let key = WindowKey::new(i32::MIN, i32::MAX, 0);
        assert_eq!(key.q(), i32::MIN);
        assert_eq!(key.r(), i32::MAX);
        assert_eq!(key.dir(), 0);
    }

    #[test]
    fn windowkey_roundtrip_zero() {
        let key = WindowKey::new(0, 0, 0);
        assert_eq!(key.q(), 0);
        assert_eq!(key.r(), 0);
        assert_eq!(key.dir(), 0);
    }

    #[test]
    fn windowkey_full_i32_coords_do_not_alias() {
        let min = WindowKey::new(i32::MIN, -1, 0);
        let max = WindowKey::new(i32::MAX, -1, 0);
        assert_eq!(min.q(), i32::MIN);
        assert_eq!(max.q(), i32::MAX);
        assert_ne!(min, max);
    }

    #[test]
    fn windowkey_try_new_rejects_invalid_dir() {
        assert!(WindowKey::try_new(0, 0, 3).is_none());
        assert!(WindowKey::try_new(0, 0, u8::MAX).is_none());
        assert_eq!(WindowKey::try_new(0, 0, 2), Some(WindowKey::new(0, 0, 2)));
    }

    #[test]
    #[should_panic(expected = "dir must be a valid HEX_DIRECTIONS index")]
    fn windowkey_new_rejects_invalid_dir_in_all_builds() {
        let _ = WindowKey::new(0, 0, 3);
    }

    #[test]
    fn windowkey_equality_and_hashing() {
        use std::collections::HashSet;
        let a = WindowKey::new(1, 2, 0);
        let b = WindowKey::new(1, 2, 0);
        let c = WindowKey::new(1, 2, 1);
        assert_eq!(a, b);
        assert_ne!(a, c);

        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b); // duplicate
        set.insert(c);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn windowkey_copy_trait() {
        let a = WindowKey::new(5, -5, 2);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn windowkey_cell_at_along_axis_a() {
        // Direction 0 is (1, 0).
        let key = WindowKey::new(0, 0, 0);
        assert_eq!(key.cell_at(0), Hex::new(0, 0));
        assert_eq!(key.cell_at(1), Hex::new(1, 0));
        assert_eq!(key.cell_at(5), Hex::new(5, 0));
        assert_eq!(key.cell_at(-1), Hex::new(-1, 0));
    }

    #[test]
    fn windowkey_cell_at_along_axis_b() {
        // Direction 1 is (0, 1).
        let key = WindowKey::new(3, -2, 1);
        assert_eq!(key.cell_at(0), Hex::new(3, -2));
        assert_eq!(key.cell_at(2), Hex::new(3, 0));
        assert_eq!(key.cell_at(-2), Hex::new(3, -4));
    }

    #[test]
    fn windowkey_cell_at_along_axis_c() {
        // Direction 2 is (1, -1).
        let key = WindowKey::new(1, 1, 2);
        assert_eq!(key.cell_at(0), Hex::new(1, 1));
        assert_eq!(key.cell_at(1), Hex::new(2, 0));
        assert_eq!(key.cell_at(-1), Hex::new(0, 2));
    }

    #[test]
    fn windowkey_size_is_explicit_value() {
        assert_eq!(std::mem::size_of::<WindowKey>(), 12);
    }

    // ------------------------------------------------------------------
    // Turn tests
    // ------------------------------------------------------------------

    #[test]
    fn turn_single() {
        let t = Turn::single(Hex::new(1, 2));
        assert_eq!(t.first(), Hex::new(1, 2));
        assert_eq!(t.second(), None);
        assert_eq!(t.placements(), 1);
    }

    #[test]
    fn turn_pair_ordered() {
        let t = Turn::pair(Hex::new(1, 2), Hex::new(3, 4));
        assert_eq!(t.first(), Hex::new(1, 2));
        assert_eq!(t.second(), Some(Hex::new(3, 4)));
        assert_eq!(t.placements(), 2);
    }

    #[test]
    fn turn_pair_canonical_ordering() {
        // When passed in reverse order, pair should canonicalize (smaller first).
        let t = Turn::pair(Hex::new(3, 4), Hex::new(1, 2));
        assert_eq!(t.first(), Hex::new(1, 2));
        assert_eq!(t.second(), Some(Hex::new(3, 4)));
        assert_eq!(t.placements(), 2);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Turn::pair requires two distinct cells")]
    fn turn_pair_rejects_self_pair_in_debug() {
        // A self-pair is never a legal move and indicates a bug in move
        // generation.  In debug builds we panic so the mistake is caught
        // before it can poison a transposition table or HashSet.
        Turn::pair(Hex::new(2, 2), Hex::new(2, 2));
    }

    #[test]
    fn turn_copy_trait() {
        let a = Turn::single(Hex::new(0, 0));
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn turn_equality_and_hashing() {
        use std::collections::HashSet;
        let a = Turn::pair(Hex::new(1, 0), Hex::new(0, 1));
        let b = Turn::pair(Hex::new(0, 1), Hex::new(1, 0));
        assert_eq!(a, b);

        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b); // duplicate after canonicalization
        assert_eq!(set.len(), 1);
    }
}
