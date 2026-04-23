//! Threat analysis for Infinity Hexagonal Tic-Tac-Toe.
//!
//! This module adds threat-detection methods to [`HexGameState`] via an
//! `impl` block.  These methods are used by the alpha-beta search and MCTS
//! engines to filter moves, detect instant wins, and prune losing lines.

use rustc_hash::{FxHashMap, FxHashSet};
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
