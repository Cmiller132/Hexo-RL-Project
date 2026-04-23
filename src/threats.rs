//! Threat analysis for Infinity Hexagonal Tic-Tac-Toe.
//!
//! This module adds threat-detection methods to [`HexGameState`] via an
//! `impl` block.  These methods are used by the alpha-beta search and MCTS
//! engines to filter moves, detect instant wins, and prune losing lines.

use rustc_hash::FxHashSet;
use crate::board::HexGameState;
use crate::core::{Hex, HEX_DIRECTIONS};
use crate::patterns::WIN_LENGTH;

impl HexGameState {
    /// Collect empty cells from a player's hot windows.
    ///
    /// Each returned inner set corresponds to one hot window and contains the
    /// empty cells that would need to be filled or blocked in that window.
    pub fn collect_threat_window_empties(&self, player: u8) -> Vec<FxHashSet<Hex>> {
        let p = player as usize;
        if self.window_fours[p] == 0 && self.window_fives[p] == 0 {
            return Vec::new();
        }

        let mut result = Vec::new();
        for &(wq, wr, dir) in &self.hot_windows[p] {
            let (dq, dr) = HEX_DIRECTIONS[dir as usize];
            let mut empties = FxHashSet::default();
            for k in 0..WIN_LENGTH {
                let h = Hex::new(wq + dq * k, wr + dr * k);
                if !self.board.contains_key(&h) {
                    empties.insert(h);
                }
            }
            if !empties.is_empty() && empties.len() <= 2 {
                result.push(empties);
            }
        }
        result
    }

    /// Collect the union of empty cells from a player's hot windows.
    pub fn collect_threat_cells(&self, player: u8) -> FxHashSet<Hex> {
        self.collect_threat_window_empties(player)
            .into_iter()
            .flatten()
            .collect()
    }

    /// Collect empty cells that would complete a win for `player` on this turn.
    ///
    /// Only returns cells when the number of empties in the hot window is
    /// less than or equal to the player's remaining placements.
    pub fn collect_winning_threat_cells(&self, player: u8) -> FxHashSet<Hex> {
        let p = player as usize;
        if self.window_fours[p] == 0 && self.window_fives[p] == 0 {
            return FxHashSet::default();
        }

        let available_placements = self.placements_remaining;
        let mut cells = FxHashSet::default();
        for empties in self.collect_threat_window_empties(player) {
            match empties.len() {
                1 if available_placements >= 1 => {
                    cells.extend(empties);
                }
                2 if available_placements >= 2 => {
                    cells.extend(empties);
                }
                _ => {}
            }
        }
        cells
    }

    /// Collect empty cells that block opponent wins.
    ///
    /// With 1 placement remaining, returns only cells that hit every threat
    /// window.  With 2 placements, returns cells that participate in at least
    /// one covering pair.
    pub fn collect_blocking_threat_cells(&self, player: u8) -> FxHashSet<Hex> {
        let must_hit = self.collect_threat_window_empties(player);
        if must_hit.is_empty() {
            return FxHashSet::default();
        }

        let available_placements = self.placements_remaining;

        // Deduplicated union of all block-candidate cells.
        let mut all_block_cells: Vec<Hex> =
            must_hit.iter().flat_map(|s| s.iter().copied()).collect();
        all_block_cells.sort();
        all_block_cells.dedup();

        if available_placements <= 1 {
            // Must find a single cell that hits every threat window.
            return all_block_cells
                .into_iter()
                .filter(|cell| must_hit.iter().all(|set| set.contains(cell)))
                .collect();
        }

        // With 2 placements: only return cells that participate in at least
        // one covering pair (a pair that hits every threat window).
        let mut valid = FxHashSet::default();
        for i in 0..all_block_cells.len() {
            for j in i..all_block_cells.len() {
                let c1 = all_block_cells[i];
                let c2 = all_block_cells[j];
                let covers_all = must_hit
                    .iter()
                    .all(|set| set.contains(&c1) || set.contains(&c2));
                if covers_all {
                    valid.insert(c1);
                    valid.insert(c2);
                }
            }
        }
        valid
    }

    /// Returns whether `player` has hot windows that cannot all be covered by
    /// the given number of available placements.
    pub fn is_player_win_unblockable(&self, player: u8, available_placements: u8) -> bool {
        let must_hit = self.collect_threat_window_empties(player);
        if must_hit.is_empty() {
            return false;
        }

        let mut all_block_cells: Vec<Hex> = must_hit.iter().flat_map(|s| s.iter().copied()).collect();
        all_block_cells.sort();
        all_block_cells.dedup();

        match available_placements {
            0 => true,
            1 => !all_block_cells
                .iter()
                .copied()
                .any(|cell| must_hit.iter().all(|set| set.contains(&cell))),
            _ => {
                for i in 0..all_block_cells.len() {
                    for j in i..all_block_cells.len() {
                        let c1 = all_block_cells[i];
                        let c2 = all_block_cells[j];
                        let covers_all = must_hit
                            .iter()
                            .all(|set| set.contains(&c1) || set.contains(&c2));
                        if covers_all {
                            return false;
                        }
                    }
                }
                true
            }
        }
    }

    /// Returns whether the opponent's hot windows cannot all be blocked with
    /// the current player's remaining placements.
    pub fn is_opponent_win_unblockable(&self, available_placements: u8) -> bool {
        self.is_player_win_unblockable(1 - self.current_player, available_placements)
    }

    /// Returns a threat-filtered subset of `legal`, or `None` when no
    /// hard constraint applies.
    pub fn compute_threat_constrained_moves(&self, legal: &[Hex], constrain: bool) -> Option<Vec<Hex>> {
        if !constrain || legal.is_empty() {
            return None;
        }

        let current = self.current_player;
        let current_idx = current as usize;
        let opp = 1 - current;
        let opp_idx = opp as usize;

        if self.window_fives[current_idx] > 0 || self.window_fours[current_idx] > 0 {
            let winning_cells = self.collect_winning_threat_cells(current);
            if !winning_cells.is_empty() {
                let filtered: Vec<Hex> = legal
                    .iter()
                    .copied()
                    .filter(|h| winning_cells.contains(h))
                    .collect();
                if !filtered.is_empty() {
                    return Some(filtered);
                }
            }
        }

        if self.window_fives[opp_idx] > 0 || self.window_fours[opp_idx] > 0 {
            if self.is_opponent_win_unblockable(self.placements_remaining) {
                return None;
            }

            let blocking_cells = self.collect_blocking_threat_cells(opp);
            if !blocking_cells.is_empty() {
                let filtered: Vec<Hex> = legal
                    .iter()
                    .copied()
                    .filter(|h| blocking_cells.contains(h))
                    .collect();
                if !filtered.is_empty() {
                    return Some(filtered);
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use crate::board::HexGameState;
    use crate::core::Hex;

    #[test]
    fn collect_winning_threat_cells_five_window() {
        let mut game = HexGameState::new();
        // P0 has a 5-stone run along (1,0): (0,0)..(4,0).
        // This creates multiple overlapping hot windows (5-windows and 4-windows).
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)],
            0,
            2,
        )
        .unwrap();

        let cells = game.collect_winning_threat_cells(0);
        // A bare 5-run produces 4 hot windows whose empty cells are:
        // (-2,0), (-1,0), (5,0), (6,0)
        assert_eq!(cells.len(), 4);
        assert!(cells.contains(&Hex::new(-2, 0)));
        assert!(cells.contains(&Hex::new(-1, 0)));
        assert!(cells.contains(&Hex::new(5, 0)));
        assert!(cells.contains(&Hex::new(6, 0)));
    }

    #[test]
    fn collect_winning_threat_cells_four_window() {
        let mut game = HexGameState::new();
        // P0 has a 4-stone run along (1,0): (0,0)..(3,0).
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
            0,
            2,
        )
        .unwrap();

        let cells = game.collect_winning_threat_cells(0);
        // A bare 4-run produces 3 hot 4-windows whose empty cells are:
        // (-2,0), (-1,0), (4,0), (5,0)
        assert_eq!(cells.len(), 4);
        assert!(cells.contains(&Hex::new(-2, 0)));
        assert!(cells.contains(&Hex::new(-1, 0)));
        assert!(cells.contains(&Hex::new(4, 0)));
        assert!(cells.contains(&Hex::new(5, 0)));
    }

    #[test]
    fn collect_blocking_threat_cells_single_placement() {
        let mut game = HexGameState::new();
        // P1 has a 4-stone run (0,0)..(3,0) and P0 has already blocked one end at (-2,0).
        // After the block, the remaining hot windows are:
        //   (-1,0)..(4,0) → empties {(-1,0), (4,0)}
        //   (0,0)..(5,0)  → empties {(4,0), (5,0)}
        // Both remaining windows contain (4,0), so a single placement there blocks all.
        game.set_position(
            &[
                (-2, 0, 0), // P0 blocker
                (0, 0, 1),
                (1, 0, 1),
                (2, 0, 1),
                (3, 0, 1),
            ],
            0,
            1,
        )
        .unwrap();

        let cells = game.collect_blocking_threat_cells(1);
        assert_eq!(cells.len(), 1);
        assert!(cells.contains(&Hex::new(4, 0)));
    }

    #[test]
    fn collect_blocking_threat_cells_two_placements() {
        let mut game = HexGameState::new();
        // P1 has a bare 4-stone run (0,0)..(3,0).
        // Hot windows and their empty cells:
        //   (-2,0)..(3,0) → {(-2,0), (-1,0)}
        //   (-1,0)..(4,0) → {(-1,0), (4,0)}
        //   (0,0)..(5,0)  → {(4,0), (5,0)}
        // Every boundary cell participates in at least one covering pair,
        // so all four are returned.
        game.set_position(
            &[(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)],
            0,
            2,
        )
        .unwrap();

        let cells = game.collect_blocking_threat_cells(1);
        assert_eq!(cells.len(), 4);
        assert!(cells.contains(&Hex::new(-2, 0)));
        assert!(cells.contains(&Hex::new(-1, 0)));
        assert!(cells.contains(&Hex::new(4, 0)));
        assert!(cells.contains(&Hex::new(5, 0)));
    }

    #[test]
    fn is_player_win_unblockable_single_cell_blocks_all() {
        let mut game = HexGameState::new();
        // P0 has a 4-run (0,0)..(3,0) with P1 already blocking one end at (-2,0).
        // After the block, the remaining hot windows are:
        //   (-1,0)..(4,0) → empties {(-1,0), (4,0)}
        //   (0,0)..(5,0)  → empties {(4,0), (5,0)}
        // Both contain (4,0), so a single placement there blocks all.
        game.set_position(
            &[
                (-2, 0, 1), // P1 blocker
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (3, 0, 0),
            ],
            0,
            1,
        )
        .unwrap();

        assert!(!game.is_player_win_unblockable(0, 1));
    }

    #[test]
    fn is_player_win_unblockable_disjoint_threats() {
        let mut game = HexGameState::new();
        // P0 has two disjoint 5-stone runs.
        // A single 5-run is unblockable with 1 placement (multiple disjoint empty cells).
        // Two disjoint 5-runs are also unblockable with 2 placements
        // because each run needs its own pair of blockers.
        game.set_position(
            &[
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (3, 0, 0),
                (4, 0, 0),
                (10, 0, 0),
                (11, 0, 0),
                (12, 0, 0),
                (13, 0, 0),
                (14, 0, 0),
            ],
            0,
            2,
        )
        .unwrap();

        // Even with 2 placements, two disjoint 5-runs are unblockable.
        assert!(game.is_player_win_unblockable(0, 2));
    }

    #[test]
    fn is_opponent_win_unblockable_true() {
        let mut game = HexGameState::new();
        // P1 has two disjoint 5-windows, P0 has only 1 placement.
        game.set_position(
            &[
                (0, 0, 1),
                (1, 0, 1),
                (2, 0, 1),
                (3, 0, 1),
                (4, 0, 1),
                (10, 0, 1),
                (11, 0, 1),
                (12, 0, 1),
                (13, 0, 1),
                (14, 0, 1),
            ],
            0,
            1,
        )
        .unwrap();

        assert!(game.is_opponent_win_unblockable(1));
    }

    #[test]
    fn threat_constrained_moves_own_win() {
        let mut game = HexGameState::new();
        // P0 has a 5-window and is to move with 2 placements.
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)],
            0,
            2,
        )
        .unwrap();

        let legal = vec![Hex::new(5, 0), Hex::new(100, 0)];
        let constrained = game.compute_threat_constrained_moves(&legal, true);
        assert!(constrained.is_some());
        let moves = constrained.unwrap();
        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0], Hex::new(5, 0));
    }

    #[test]
    fn threat_constrained_moves_opponent_block() {
        let mut game = HexGameState::new();
        // P1 has a 5-window. P0 must block.
        game.set_position(
            &[(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1), (4, 0, 1)],
            0,
            2,
        )
        .unwrap();

        let legal = vec![Hex::new(5, 0), Hex::new(100, 0)];
        let constrained = game.compute_threat_constrained_moves(&legal, true);
        assert!(constrained.is_some());
        let moves = constrained.unwrap();
        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0], Hex::new(5, 0));
    }

    #[test]
    fn threat_constrained_moves_unblockable_returns_none() {
        let mut game = HexGameState::new();
        // P1 has two disjoint 5-windows. P0 cannot block both with 1 placement.
        game.set_position(
            &[
                (0, 0, 1),
                (1, 0, 1),
                (2, 0, 1),
                (3, 0, 1),
                (4, 0, 1),
                (10, 0, 1),
                (11, 0, 1),
                (12, 0, 1),
                (13, 0, 1),
                (14, 0, 1),
            ],
            0,
            1,
        )
        .unwrap();

        let legal = vec![Hex::new(5, 0), Hex::new(15, 0)];
        let constrained = game.compute_threat_constrained_moves(&legal, true);
        // When unblockable, all moves are losing so no hard constraint is returned.
        assert!(constrained.is_none());
    }
}
