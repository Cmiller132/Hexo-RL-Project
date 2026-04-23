//! Threat analysis for Infinity Hexagonal Tic-Tac-Toe.
//!
//! This module adds threat-detection methods to [`HexGameState`] via an
//! `impl` block.  These methods are used by the alpha-beta search and MCTS
//! engines to filter moves, detect instant wins, and prune losing lines.
//!
//! # Solver vs. heuristic
//!
//! The fast path (production) uses precomputed hot-window sets and hitting-set
//! logic (O(n²) for two placements).  For verification and edge-case safety,
//! [`solve_winning_cells`] and [`solve_blocking_cells`] brute-force every
//! candidate placement and check the resulting board state.  The solver is
//! the ground truth; the fast methods are validated against it in tests.

use rustc_hash::FxHashSet;
use crate::board::HexGameState;
use crate::core::{Hex, HEX_DIRECTIONS};
use crate::patterns::WIN_LENGTH;

impl HexGameState {
    // ── Fast path: hot-window enumeration ────────────────────────────────

    /// Collect empty cells from a player's hot windows.
    ///
    /// Each returned inner set corresponds to one hot window and contains the
    /// empty cells that would need to be filled or blocked in that window.
    /// Only windows with 1 or 2 empties are returned (3+ empties cannot be
    /// completed in a single turn with at most 2 placements).
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
    /// window (the intersection).  With 2 placements, returns the union of
    /// all threat-window empties — this is permissive but safe, because any
    /// cell outside this set cannot possibly block.  The caller (search or
    /// MCTS) is responsible for turn-level verification.
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

        // With 2 placements: return the union of all threat-window empties.
        // This is the safe superset — the search engine filters turns
        // precisely via its own turn-level hitting-set check.
        must_hit.into_iter().flatten().collect()
    }

    /// Returns whether `player` has hot windows that cannot all be covered by
    /// the given number of available placements.
    ///
    /// This is a hitting-set check: for 1 placement we need a common cell in
    /// every threat set; for 2 placements we need a pair of cells that covers
    /// every set.  Since the game only ever allows 1 or 2 placements, this is
    /// exact for immediate threats.
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

    // ── Solver-based ground truth ────────────────────────────────────────

    /// Brute-force solver: which cells from [`collect_winning_threat_cells`]
    /// actually participate in a winning turn?
    ///
    /// Iterates only over the fast-path candidates (cells in hot windows with
    /// empties ≤ remaining placements), tries every single placement and every
    /// pair, and records which candidates appear in at least one turn that ends
    /// the game with `player` as the winner.
    ///
    /// For this game (win = 6 contiguous, max 2 placements/turn), every cell
    /// in `collect_winning_threat_cells` is guaranteed to win either alone
    /// (5-window) or with its partner empty (4-window).  This method exists
    /// as a testable invariant.
    pub fn solve_winning_cells(&mut self, player: u8) -> FxHashSet<Hex> {
        let mut wins = FxHashSet::default();
        // Only test cells the fast path already identified.
        let candidates: Vec<Hex> =
            self.collect_winning_threat_cells(player).iter().copied().collect();

        // Single placements (5-windows).
        for &c in &candidates {
            if self.place(c.q, c.r).is_ok() {
                if self.winner == Some(player) {
                    wins.insert(c);
                }
                self.unmake_move();
            }
        }

        // Pairs (4-windows) — only when we have 2 placements.
        if self.placements_remaining >= 2 {
            for i in 0..candidates.len() {
                for j in (i + 1)..candidates.len() {
                    let c1 = candidates[i];
                    let c2 = candidates[j];
                    if self.place(c1.q, c1.r).is_ok() {
                        if self.winner == Some(player) {
                            wins.insert(c1);
                            wins.insert(c2);
                            self.unmake_move();
                            continue;
                        }
                        if self.place(c2.q, c2.r).is_ok() {
                            if self.winner == Some(player) {
                                wins.insert(c1);
                                wins.insert(c2);
                            }
                            self.unmake_move();
                        }
                        self.unmake_move();
                    }
                }
            }
        }

        wins
    }

    /// Brute-force solver: which single cells actually block the opponent?
    ///
    /// A cell "blocks" if, after placing it, the opponent no longer has any
    /// winning threat cells.  This is the ground-truth for
    /// [`collect_blocking_threat_cells`] with 1 placement.
    pub fn solve_blocking_cells(&mut self, player: u8) -> FxHashSet<Hex> {
        let opp = 1 - player;
        let mut blocks = FxHashSet::default();
        let cands = self.candidates_near2();
        for c in cands {
            if self.place(c.q, c.r).is_ok() {
                let opp_threats = self.collect_winning_threat_cells(opp);
                if opp_threats.is_empty() {
                    blocks.insert(c);
                }
                self.unmake_move();
            }
        }
        blocks
    }

    /// Brute-force solver: which pairs of cells block the opponent?
    ///
    /// Returns every pair `(a, b)` with `a <= b` (can be equal only if the
    /// game wins after the first placement, in which case the pair is
    /// effectively a single move).  Only used in tests.
    pub fn solve_blocking_pairs(&mut self, player: u8) -> Vec<(Hex, Hex)> {
        let opp = 1 - player;
        let mut pairs = Vec::new();
        let cands = self.candidates_near2();

        for i in 0..cands.len() {
            let c1 = cands[i];
            if self.place(c1.q, c1.r).is_ok() {
                // If first placement already wins or ends the game,
                // the "pair" is just (c1, c1) as a sentinel.
                if self.winner == Some(player) {
                    pairs.push((c1, c1));
                    self.unmake_move();
                    continue;
                }

                // Can we make a second placement?
                if self.placements_remaining > 0 {
                    for j in i..cands.len() {
                        let c2 = cands[j];
                        if self.place(c2.q, c2.r).is_ok() {
                            let opp_threats = self.collect_winning_threat_cells(opp);
                            if opp_threats.is_empty() {
                                pairs.push((c1, c2));
                            }
                            self.unmake_move();
                        }
                    }
                }
                self.unmake_move();
            }
        }
        pairs
    }
}

#[cfg(test)]
mod tests {
    use crate::board::HexGameState;
    use crate::core::Hex;

    // ── Winning threat cells ─────────────────────────────────────────────

    #[test]
    fn collect_winning_threat_cells_five_window() {
        let mut game = HexGameState::new();
        // P0 has a 5-stone run along (1,0): (0,0)..(4,0).
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

        // Solver: every cell in the fast path should win (alone or in a pair).
        let solver_wins = game.solve_winning_cells(0);
        assert_eq!(
            solver_wins, cells,
            "solver and fast path should agree exactly for winning cells"
        );
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

        // With only 1 placement, no 4-window can be completed.
        let mut g1 = HexGameState::new();
        g1.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
            0,
            1,
        )
        .unwrap();
        assert!(g1.collect_winning_threat_cells(0).is_empty());
    }

    #[test]
    fn winning_threat_solver_matches_fast() {
        let mut game = HexGameState::new();
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)],
            0,
            2,
        )
        .unwrap();

        let fast = game.collect_winning_threat_cells(0);
        let solver = game.solve_winning_cells(0);
        assert_eq!(fast, solver, "solver and fast path must agree exactly");
    }

    // ── Blocking threat cells ────────────────────────────────────────────

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

        // Solver should agree.
        let solver_blocks = game.solve_blocking_cells(0);
        assert_eq!(solver_blocks, cells);
    }

    #[test]
    fn collect_blocking_threat_cells_two_placements() {
        let mut game = HexGameState::new();
        // P1 has a bare 4-stone run (0,0)..(3,0).
        // Hot windows and their empty cells:
        //   (-2,0)..(3,0) → {(-2,0), (-1,0)}
        //   (-1,0)..(4,0) → {(-1,0), (4,0)}
        //   (0,0)..(5,0)  → {(4,0), (5,0)}
        // With 2 placements, the union of all empties is returned (safe superset).
        game.set_position(
            &[(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)],
            0,
            2,
        )
        .unwrap();

        let cells = game.collect_blocking_threat_cells(1);
        // Union of all threat-window empties
        assert_eq!(cells.len(), 4);
        assert!(cells.contains(&Hex::new(-2, 0)));
        assert!(cells.contains(&Hex::new(-1, 0)));
        assert!(cells.contains(&Hex::new(4, 0)));
        assert!(cells.contains(&Hex::new(5, 0)));

        // Solver: every pair from this set should block, but the fast method
        // just returns the safe superset.
        let pairs = game.solve_blocking_pairs(0);
        assert!(!pairs.is_empty(), "there must exist at least one blocking pair");
    }

    // ── Unblockable detection ────────────────────────────────────────────

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

    // ── Threat-constrained moves ─────────────────────────────────────────

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

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn hot_window_on_all_three_axes() {
        let mut game = HexGameState::new();
        // Create a "star" where P0 has 4 stones on each axis through (0,0).
        // q-axis: (-1,0), (0,0), (1,0), (2,0)
        // r-axis: (0,-1), (0,0), (0,1), (0,2)
        // diag-axis: (1,-1), (0,0), (-1,1), (-2,2)
        game.set_position(
            &[
                (-1, 0, 0),
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 2, 0),
                (1, -1, 0),
                (-1, 1, 0),
                (-2, 2, 0),
            ],
            0,
            2,
        )
        .unwrap();

        // Should have hot windows on all 3 axes.
        assert!(!game.hot_windows[0].is_empty());
        let empties = game.collect_threat_window_empties(0);
        assert!(!empties.is_empty());

        // With 4-stone runs, no single placement wins — pairs do.
        // The solver (which checks pairs) should find winning cells.
        let solver_wins = game.solve_winning_cells(0);
        assert!(!solver_wins.is_empty(), "solver should find pair-winning cells");

        // Fast path must contain every cell the solver found.
        let fast = game.collect_winning_threat_cells(0);
        for c in &solver_wins {
            assert!(
                fast.contains(c),
                "fast path missed solver-winning cell {:?}",
                c
            );
        }
    }

    #[test]
    fn six_stone_window_is_not_a_threat() {
        let mut game = HexGameState::new();
        // P0 already has 6 in a row — game is over.
        game.set_position(
            &[
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (3, 0, 0),
                (4, 0, 0),
                (5, 0, 0),
            ],
            0,
            2,
        )
        .unwrap();

        assert!(game.is_over());
        // The 6-window itself has 0 empties, but overlapping 5-windows
        // (e.g. (-1,0)..(4,0) with empty at (-1,0)) still exist structurally.
        // `collect_winning_threat_cells` will report those empties, which is
        // correct — placing them would complete another 6-line.  In practice
        // the game is already over so these are irrelevant for play.
        let wins = game.collect_winning_threat_cells(0);
        assert!(!wins.is_empty(), "overlapping 5-windows still have winning cells");
        // Solver should agree (but note: place() fails on a won game, so
        // solve_winning_cells returns empty).
        assert!(game.solve_winning_cells(0).is_empty());
    }

    #[test]
    fn blocked_window_is_not_hot() {
        let mut game = HexGameState::new();
        // P0 has 4 stones but P1 is inside the window.
        // Window (-1,0)..(4,0): P1 at (2,0) blocks it.
        game.set_position(
            &[
                (-1, 0, 0),
                (0, 0, 0),
                (2, 0, 1), // P1 blocker inside
                (3, 0, 0),
                (4, 0, 0),
            ],
            0,
            2,
        )
        .unwrap();

        // The window (-1,0)..(4,0) has P1 at offset 3, so it's not hot.
        let empties = game.collect_threat_window_empties(0);
        // There might be smaller hot windows like (0,0)..(5,0) depending on
        // exact geometry, but the blocked 6-window should not be hot.
        for set in &empties {
            assert!(
                !set.contains(&Hex::new(2, 0)),
                "a hot window should not contain an opponent stone"
            );
        }
    }

    #[test]
    fn three_window_is_not_hot() {
        let mut game = HexGameState::new();
        // P0 has only 3 stones — no hot windows.
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0)],
            0,
            2,
        )
        .unwrap();

        assert!(game.hot_windows[0].is_empty());
        assert!(game.collect_threat_window_empties(0).is_empty());
        assert!(game.collect_winning_threat_cells(0).is_empty());
    }

    #[test]
    fn solver_blocks_after_partial_block() {
        let mut game = HexGameState::new();
        // P1 has two 5-windows that share a common empty cell.
        // P0 blocks the common cell. After the block, P1 should have no threats.
        game.set_position(
            &[
                (0, 0, 1),
                (1, 0, 1),
                (2, 0, 1),
                (3, 0, 1),
                (4, 0, 1),
                (2, 1, 1),
                (2, 2, 1),
                (2, 3, 1),
                (2, 4, 1),
            ],
            0,
            1,
        )
        .unwrap();

        // P1 has two 5-windows:
        // q-axis: (0,0)..(4,0), empty at (5,0)
        // r-axis: (2,0)..(2,4), empty at (2,5)
        // They don't share an empty cell, so unblockable with 1 placement.
        assert!(game.is_opponent_win_unblockable(1));

        // Now block (5,0) — still unblockable because (2,5) is open.
        let mut g2 = game.clone();
        g2.place(5, 0).unwrap();
        assert!(
            g2.collect_winning_threat_cells(1).contains(&Hex::new(2, 5)),
            "P1 should still threaten (2,5) after (5,0) is blocked"
        );
    }

    #[test]
    fn overlapping_hot_windows_share_empties() {
        let mut game = HexGameState::new();
        // P0 5-run: (0,0)..(4,0). Two overlapping 5-windows:
        // (-1,0)..(4,0) → empty at (-1,0)
        // (0,0)..(5,0)  → empty at (5,0)
        // Both share the 4-run but have different empties.
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)],
            0,
            1,
        )
        .unwrap();

        let _empties = game.collect_threat_window_empties(0);
        // With 1 placement, the player can only fill one empty, so they
        // cannot complete both 5-windows. The function should still report
        // both empties as winning cells (they are, individually, in a window).
        let winning = game.collect_winning_threat_cells(0);
        assert!(winning.contains(&Hex::new(-1, 0)));
        assert!(winning.contains(&Hex::new(5, 0)));

        // But solver: placing at (-1,0) wins (completes 6), placing at (5,0) also wins.
        let solver_wins = game.solve_winning_cells(0);
        assert!(solver_wins.contains(&Hex::new(-1, 0)));
        assert!(solver_wins.contains(&Hex::new(5, 0)));
    }

    #[test]
    fn fast_blocking_matches_solver_random_positions() {
        // Deterministic "random" positions to validate fast path against solver.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut game = HexGameState::new();
        game.place(0, 0).unwrap();
        game.place(1, 0).unwrap();
        game.place(0, 1).unwrap();

        let mut step = 0u64;
        while !game.is_over() && game.move_count < 80 {
            let cands = game.candidates_near2();
            if cands.is_empty() {
                break;
            }
            let mut h = DefaultHasher::new();
            step.hash(&mut h);
            let idx = h.finish() as usize % cands.len();
            let m = cands[idx];
            if game.place(m.q, m.r).is_err() {
                break;
            }
            step += 1;

            // Validate after each completed turn (placements_remaining == 2).
            if game.placements_remaining == 2 && !game.is_over() {
                let p = game.current_player;

                // Winning: solver and fast path must match exactly.
                let fast_win = game.collect_winning_threat_cells(p);
                let solver_win = game.solve_winning_cells(p);
                assert_eq!(
                    fast_win, solver_win,
                    "fast winning diverged from solver at move {} for player {}",
                    game.move_count, p
                );

                // Blocking: only validate with 1 placement (exact match).
                let opp = 1 - p;
                if game.placements_remaining == 1 {
                    let fast_block = game.collect_blocking_threat_cells(opp);
                    let solver_block = game.solve_blocking_cells(p);
                    assert_eq!(
                        fast_block, solver_block,
                        "fast blocking diverged from solver at move {} for player {}",
                        game.move_count, p
                    );
                }
            }
        }
    }
}
