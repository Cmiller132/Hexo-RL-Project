//! Threat analysis for Hexo.
//!
//! This module provides free functions for classifying the tactical situation,
//! checking whether a turn satisfies threat constraints, and enumerating live
//! cells.  It replaces the old `impl HexGameState` threat methods.
//!
//! # Threat model
//!
//! In Hexo, a "threat window" is a length-6 line
//! (one of the 3 axial directions) that contains 4 or 5 stones of the same
//! player and only empty cells otherwise.  A 5-window can be completed in one
//! placement; a 4-window can be completed in two placements.  Because the game
//! ends immediately when a player forms 6-in-a-row, blocking all opponent
//! threat windows is usually mandatory.
//!
//! The core workflow is:
//! 1. `tactical_status(game)` — classify the position (Quiet / WinningTurns /
//!    MustBlock / Unblockable).
//! 2. `turn_satisfies_tactical(&status, turn)` — test a candidate turn against
//!    the pre-computed status.
//! 3. `generate_threat_turns(game, out, opp_buf, my_buf)` — produce candidate
//!    turns for quiescence search from live cells (crate-internal).
//! 4. `live_cells(game, player, out)` — enumerate empty cells that appear in
//!    at least one of the player's hot windows (useful for quiescence search
//!    and neural-network feature planes).

use crate::board::HexGameState;
use crate::core::{Hex, Turn, WindowKey, HEX_DIRECTIONS, WIN_LENGTH};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

// -------------------------------------------------------------------------
// Test-only compact compatibility status
// -------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
#[cfg(test)]
pub(crate) enum ThreatStatus {
    Quiet,
    WinningTurn(Turn),
    MustBlock(BlockConstraint),
    Unblockable,
}

/// Complete tactical classification for filtering and training masks.
///
/// This representation keeps every immediate winning continuation. Search
/// and encoders use it as the source of truth for pruning and legal masks.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
pub enum TacticalStatus {
    /// No immediate wins or mandatory blocks.
    Quiet,
    /// Every turn that wins immediately for the current player.
    WinningTurns(SmallVec<[Turn; 32]>),
    /// The current player must block opponent winning threats.
    MustBlock(BlockConstraint),
    /// Opponent threats cannot be covered with this turn's placements.
    Unblockable,
}

// -------------------------------------------------------------------------
// BlockConstraint
// -------------------------------------------------------------------------

/// Exact blocking constraint when the opponent has immediate threats.
///
/// When `MustBlock` is returned, the current player must place stones on
/// cells that intersect every opponent threat window.  This struct encodes
/// the valid blocking sets efficiently.
///
/// # Semantics by remaining placements
///
/// * **1 placement remaining** — only `cells` matters.  The single stone must
///   land in `cells` (the intersection of all threat windows).
/// * **2 placements remaining** — the pair of stones must together cover every
///   threat window.  Valid pairs are enumerated in `pairs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockConstraint {
    /// Single cells that block **every** threat window.
    ///
    /// These are the cells in the intersection of all opponent hot windows.
    /// If a turn contains any of these cells, it is automatically valid
    /// regardless of the other placement(s).
    cells: SmallVec<[Hex; 16]>,

    /// Valid pairs of **distinct** cells that together block every threat
    /// window (used when 2 placements remain).
    ///
    /// Pairs are stored in canonical order (`c1 < c2` by the internal
    /// ordering of [`Hex`]).  There are **no** self-pairs `(c, c)`.
    pairs: SmallVec<[(Hex, Hex); 32]>,
}

impl BlockConstraint {
    /// Single cells that block every threat window.
    pub fn cells(&self) -> &[Hex] {
        &self.cells
    }

    /// Valid pairs that together block every threat window.
    pub fn pairs(&self) -> &[(Hex, Hex)] {
        &self.pairs
    }
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct ThreatWindow {
    empties: SmallVec<[Hex; 2]>,
}

/// Scan every six-cell window touching one of `player`'s stones.
///
/// This intentionally does not use `EvalState::hot_windows`: the incremental
/// eval grid is bounded around the origin, while tactical legality must remain
/// correct on the sparse infinite board.
fn full_board_threat_windows(game: &HexGameState, player: u8) -> SmallVec<[ThreatWindow; 32]> {
    let mut seen = FxHashSet::default();
    let mut windows = SmallVec::<[ThreatWindow; 32]>::new();

    for (&stone, &owner) in game.stones() {
        if owner != player {
            continue;
        }

        for (dir, &(dq, dr)) in HEX_DIRECTIONS.iter().enumerate() {
            for off in 0..WIN_LENGTH {
                let sq = stone.q - dq * off;
                let sr = stone.r - dr * off;
                let key = WindowKey::new(sq, sr, dir as u8);
                if !seen.insert(key) {
                    continue;
                }

                let mut own = 0u8;
                let mut blocked = false;
                let mut empties = SmallVec::<[Hex; 2]>::new();

                for k in 0..WIN_LENGTH {
                    let h = Hex::new(sq + dq * k, sr + dr * k);
                    match game.stones().get(&h) {
                        Some(&p) if p == player => own += 1,
                        Some(_) => {
                            blocked = true;
                            break;
                        }
                        None => {
                            if empties.len() < 3 {
                                empties.push(h);
                            }
                        }
                    }
                }

                if !blocked && (own == 4 || own == 5) && !empties.is_empty() {
                    windows.push(ThreatWindow { empties });
                }
            }
        }
    }

    windows
}

fn winning_turns_from_windows(windows: &[ThreatWindow], remaining: u8) -> SmallVec<[Turn; 32]> {
    let mut turns = SmallVec::<[Turn; 32]>::new();
    for window in windows {
        match window.empties.as_slice() {
            [cell] if remaining >= 1 => turns.push(Turn::single(*cell)),
            [a, b] if remaining >= 2 => turns.push(Turn::pair(*a, *b)),
            _ => {}
        }
    }
    turns.sort_by(|a, b| {
        a.placements()
            .cmp(&b.placements())
            .then(a.first().cmp(&b.first()))
            .then(a.second().cmp(&b.second()))
    });
    turns.dedup();
    turns
}

fn block_constraint_from_windows(
    windows: &[ThreatWindow],
    placements: u8,
) -> Option<BlockConstraint> {
    if windows.is_empty() {
        return None;
    }

    let mut all_cells: SmallVec<[Hex; 32]> = windows
        .iter()
        .flat_map(|w| w.empties.iter().copied())
        .collect();
    all_cells.sort();
    all_cells.dedup();

    let mut cells = SmallVec::<[Hex; 16]>::new();
    for &cell in &all_cells {
        if windows.iter().all(|w| w.empties.contains(&cell)) {
            cells.push(cell);
        }
    }

    if placements <= 1 {
        if cells.is_empty() {
            return None;
        }
        return Some(BlockConstraint {
            cells,
            pairs: SmallVec::new(),
        });
    }

    let mut pairs = SmallVec::<[(Hex, Hex); 32]>::new();
    for i in 0..all_cells.len() {
        for j in (i + 1)..all_cells.len() {
            let c1 = all_cells[i];
            let c2 = all_cells[j];
            if windows
                .iter()
                .all(|w| w.empties.contains(&c1) || w.empties.contains(&c2))
            {
                pairs.push((c1, c2));
            }
        }
    }

    if cells.is_empty() && pairs.is_empty() {
        None
    } else {
        Some(BlockConstraint { cells, pairs })
    }
}

// -------------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------------

/// Classify the current tactical situation.
///
/// This is the entry point for threat-aware pruning.  It scans sparse windows
/// touching actual stones rather than the bounded incremental eval grid, then
/// enumerates hitting sets over the resulting threat-window empties.
///
/// # Algorithm
/// 1. **Game over?** → `Quiet` (no further constraints).
/// 2. **Winning turns?** — scan the current player's full-board threat
///    windows.  A 5-window with 1 empty is an instant win; a 4-window with 2
///    empties is a win if we have ≥2 placements remaining.
/// 3. **Opponent threats?** — collect all opponent full-board threat windows
///    and their empty cells.  If none, the position is `Quiet`.
/// 4. **Build exact `BlockConstraint`**:
///    - `cells` = intersection of all threat-window empties.
///    - `pairs` = all distinct pairs `(c1, c2)` that together intersect every
///      threat window.
/// 5. **Unblockable?** — if `cells` is empty and no valid pair exists, the
///    threats cannot be stopped.
pub fn tactical_status(game: &HexGameState) -> TacticalStatus {
    if game.winner().is_some() {
        return TacticalStatus::Quiet;
    }

    let current = game.current_player();
    let remaining = game.placements_remaining();

    let own_windows = full_board_threat_windows(game, current);
    let winning = winning_turns_from_windows(&own_windows, remaining);
    if !winning.is_empty() {
        return TacticalStatus::WinningTurns(winning);
    }

    let opp = 1 - current;
    let opp_windows = full_board_threat_windows(game, opp);
    if opp_windows.is_empty() {
        return TacticalStatus::Quiet;
    }

    match block_constraint_from_windows(&opp_windows, remaining) {
        Some(block) => TacticalStatus::MustBlock(block),
        None => TacticalStatus::Unblockable,
    }
}

#[cfg(test)]
pub(crate) fn threat_status(game: &HexGameState) -> ThreatStatus {
    match tactical_status(game) {
        TacticalStatus::Quiet => ThreatStatus::Quiet,
        TacticalStatus::WinningTurns(turns) => ThreatStatus::WinningTurn(turns[0]),
        TacticalStatus::MustBlock(block) => ThreatStatus::MustBlock(block),
        TacticalStatus::Unblockable => ThreatStatus::Unblockable,
    }
}

#[cfg(test)]
pub(crate) fn turn_satisfies_status(status: &ThreatStatus, turn: Turn) -> bool {
    match status {
        // Quiet positions impose no restrictions.
        ThreatStatus::Quiet => true,

        // WinningTurn: the only legal move is the exact winning turn.
        ThreatStatus::WinningTurn(w) => turn == *w,

        ThreatStatus::MustBlock(bc) => block_constraint_satisfied(bc, turn),

        // Unblockable means the threat filter does not constrain moves.
        ThreatStatus::Unblockable => true,
    }
}

fn block_constraint_satisfied(bc: &BlockConstraint, turn: Turn) -> bool {
    if turn.placements() == 1 {
        // With only 1 placement, the single stone must block every threat
        // window by itself.
        bc.cells().contains(&turn.first())
    } else {
        let second = turn.second().unwrap();

        // If either cell alone blocks every threat window, any pair containing
        // it is valid.
        if bc.cells().contains(&turn.first()) || bc.cells().contains(&second) {
            return true;
        }

        // Otherwise the pair must exactly match a valid blocking pair.
        bc.pairs().iter().any(|&(a, b_pair)| {
            (a == turn.first() && b_pair == second) || (a == second && b_pair == turn.first())
        })
    }
}

/// Check whether a turn is legal under complete tactical constraints.
pub fn turn_satisfies_tactical(status: &TacticalStatus, turn: Turn) -> bool {
    match status {
        TacticalStatus::Quiet => true,
        TacticalStatus::WinningTurns(turns) => turns.contains(&turn),
        TacticalStatus::MustBlock(bc) => block_constraint_satisfied(bc, turn),
        TacticalStatus::Unblockable => true,
    }
}

/// Collect every cell that participates in a forced tactical turn.
///
/// Returns `false` when the status imposes no mask-level constraint
/// (`Quiet`/`Unblockable`).
pub fn tactical_mask_cells(status: &TacticalStatus, out: &mut Vec<Hex>) -> bool {
    out.clear();
    match status {
        TacticalStatus::Quiet | TacticalStatus::Unblockable => return false,
        TacticalStatus::WinningTurns(turns) => {
            for &turn in turns {
                out.push(turn.first());
                if let Some(second) = turn.second() {
                    out.push(second);
                }
            }
        }
        TacticalStatus::MustBlock(block) => {
            out.extend(block.cells().iter().copied());
            for &(a, b) in block.pairs() {
                out.push(a);
                out.push(b);
            }
        }
    }
    out.sort();
    out.dedup();
    true
}

// -------------------------------------------------------------------------
// Threat turn generation (for quiescence)
// -------------------------------------------------------------------------

/// Generate turns from hot-window cells only (for quiescence search).
///
/// Quiescence does not need the full candidate list — it only extends
/// along tactically relevant cells (threats and blocks).  This keeps
/// the quiescence tree narrow while resolving immediate tactical
/// sequences.
///
/// The caller should supply reusable `out`, `opp_buf`, and `my_buf` buffers
/// (e.g. from `SearchState` scratch fields) to avoid per-call heap allocation.
pub(crate) fn generate_threat_turns(
    game: &HexGameState,
    out: &mut Vec<Turn>,
    opp_buf: &mut Vec<Hex>,
    my_buf: &mut Vec<Hex>,
) {
    out.clear();
    match tactical_status(game) {
        TacticalStatus::WinningTurns(turns) => {
            out.extend(turns);
            return;
        }
        TacticalStatus::MustBlock(block) => {
            if game.placements_remaining() == 1 {
                out.extend(block.cells().iter().copied().map(Turn::single));
                return;
            }

            for &(a, b) in block.pairs() {
                out.push(Turn::pair(a, b));
            }

            let mut mask_cells = Vec::new();
            let status = TacticalStatus::MustBlock(block);
            tactical_mask_cells(&status, &mut mask_cells);
            if let TacticalStatus::MustBlock(block) = &status {
                for &cell in block.cells() {
                    for &other in &mask_cells {
                        if cell != other {
                            out.push(Turn::pair(cell, other));
                        }
                    }
                }
            }
            out.sort_by(|a, b| a.first().cmp(&b.first()).then(a.second().cmp(&b.second())));
            out.dedup();
            return;
        }
        TacticalStatus::Unblockable => return,
        TacticalStatus::Quiet => {}
    }

    let player = game.current_player();
    let opp = 1 - player;

    live_cells(game, opp, opp_buf);
    live_cells(game, player, my_buf);

    // Single-placement quiescence
    if game.placements_remaining() == 1 {
        let cells = if !opp_buf.is_empty() {
            &opp_buf[..]
        } else {
            &my_buf[..]
        };
        for &h in cells.iter().take(6) {
            out.push(Turn::single(h));
        }
        return;
    }

    // Two-placement quiescence: collect live cells from both sides.
    let primary: &[Hex] = if !opp_buf.is_empty() {
        &opp_buf[..]
    } else {
        &my_buf[..]
    };
    if primary.is_empty() {
        return;
    }

    // Combine all threat cells from both sides using a stack buffer.
    let mut all_threats = SmallVec::<[Hex; 32]>::new();
    all_threats.extend(opp_buf.iter().copied());
    all_threats.extend(my_buf.iter().copied());
    all_threats.sort();
    all_threats.dedup();

    // Generate pairs within the top threat cells (most important).
    let n = all_threats.len().min(8);
    for i in 0..n {
        for j in (i + 1)..n {
            out.push(Turn::pair(all_threats[i], all_threats[j]));
        }
    }

    out.sort_by(|a, b| a.first().cmp(&b.first()).then(a.second().cmp(&b.second())));
    out.dedup();
    out.truncate(16);
}

// -------------------------------------------------------------------------
// Live cells
// -------------------------------------------------------------------------

/// Cells that are tactically "live" for the given player.
///
/// A cell is **live** if it appears as an empty cell in at least one of the
/// player's hot windows (4- or 5-windows).  These are the only cells that
/// can immediately create or block a win.  Live cells are used for:
///
/// * **Quiescence search** — restrict move generation to cells that matter.
/// * **Neural-network encoders** — channels 9/10 mark live cells for each
///   player, giving the network tactical focus.
///
/// # Parameters
/// * `game` — the current board state.
/// * `player` — the player whose live cells we want (0 or 1).
/// * `out` — reusable buffer; cleared on entry and filled with live cells.
pub fn live_cells(game: &HexGameState, player: u8, out: &mut Vec<Hex>) {
    out.clear();

    let mut seen = FxHashSet::default();
    for window in full_board_threat_windows(game, player) {
        for h in window.empties {
            if seen.insert(h) {
                out.push(h);
            }
        }
    }
}
