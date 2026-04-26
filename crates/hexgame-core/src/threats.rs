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
//! 1. `threat_status(game)` — classify the position (Quiet / WinningTurn /
//!    MustBlock / Unblockable).
//! 2. `turn_satisfies_status(&status, turn)` — test a candidate turn against
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
// ThreatStatus
// -------------------------------------------------------------------------

/// Classification of the current tactical situation.
///
/// This enum tells the search layer what kind of moves are legal at the
/// current node.  It is cheap to compute (O(hot_windows)) and should be
/// evaluated **once per node**, then reused for every candidate turn.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
pub enum ThreatStatus {
    /// No immediate threats for either side.
    ///
    /// The search is free to consider any legal turn.  This is the common
    /// case in the opening and middlegame.
    Quiet,

    /// The current player can force a win this turn.
    ///
    /// The returned `Turn` is the **only** winning continuation.  All other
    /// turns can be pruned because the game ends as soon as the 6th stone
    /// is placed.  A 5-window produces a single-placement win; a 4-window
    /// with at least 2 placements remaining produces a two-placement win.
    WinningTurn(Turn),

    /// The current player must block one or more opponent threat windows.
    ///
    /// The enclosed [`BlockConstraint`] describes exactly which cells (and
    /// which pairs of cells) cover every opponent threat window.  Any turn
    /// that does not satisfy the constraint loses immediately.
    MustBlock(BlockConstraint),

    /// Opponent threats cannot be blocked with the remaining placements.
    ///
    /// This means the opponent has at least two disjoint threat windows
    /// (or a single window with more empty cells than we have placements).
    /// The position is effectively lost; the search returns a large negative
    /// score.  The threat filter does **not** constrain moves in this state
    /// (the branch is hopeless regardless of what we play).
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

/// Collect opponent hot windows together with their empty cells.
///
/// Only windows with at least one empty cell are returned (fully-occupied
/// 6-windows are filtered out — they represent an already-won game).
///
/// The return type is a stack-allocated `SmallVec` to avoid heap allocation
/// in the common case where the opponent has fewer than 16 hot windows.
///
/// # Precondition
/// Callers must handle an empty result (no opponent threats) gracefully;
/// this function returns empty when the opponent has no fours or fives.
/// Scan a single window and push its empty cells into `out`.
///
/// `out` is **not** cleared on entry; callers should call `.clear()` first
/// if they want a fresh buffer.
#[inline]
fn window_empties(game: &HexGameState, key: WindowKey, out: &mut SmallVec<[Hex; 2]>) {
    let (dq, dr) = HEX_DIRECTIONS[key.dir() as usize];
    for k in 0..WIN_LENGTH {
        let h = Hex::new(key.q() + dq * k, key.r() + dr * k);
        if !game.stones().contains_key(&h) {
            out.push(h);
        }
    }
}

fn opponent_threat_windows(game: &HexGameState) -> (SmallVec<[Hex; 32]>, SmallVec<[u8; 16]>) {
    let opp = 1 - game.current_player();
    debug_assert!(
        game.eval().has_threats(opp),
        "opponent_threat_windows called with no threats"
    );
    let mut flat = SmallVec::<[Hex; 32]>::new();
    let mut lengths = SmallVec::<[u8; 16]>::new();
    let mut empties = SmallVec::<[Hex; 2]>::new();
    for key in game.eval().hot_windows(opp) {
        empties.clear();
        window_empties(game, key, &mut empties);
        if !empties.is_empty() {
            lengths.push(empties.len() as u8);
            flat.extend_from_slice(&empties);
        }
    }
    (flat, lengths)
}

// -------------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------------

/// Classify the current tactical situation.
///
/// This is the entry point for threat-aware pruning.  It runs in
/// O(opponent_hot_windows × WIN_LENGTH + all_cells²) time, which is
/// negligible compared to search overhead.
///
/// # Algorithm
/// 1. **Game over?** → `Quiet` (no further constraints).
/// 2. **Fast exit** — if neither side has fours or fives, the position is
///    tactically quiet.
/// 3. **Winning turn?** — scan the current player's hot windows.  A 5-window
///    with 1 empty is an instant win; a 4-window with 2 empties is a win if
///    we have ≥2 placements remaining.
/// 4. **Opponent threats?** — collect all opponent hot windows and their
///    empty cells.  If none, the position is `Quiet`.
/// 5. **Build exact `BlockConstraint`**:
///    - `cells` = intersection of all threat-window empties.
///    - `pairs` = all distinct pairs `(c1, c2)` that together intersect every
///      threat window.
/// 6. **Unblockable?** — if `cells` is empty and no valid pair exists, the
///    threats cannot be stopped.
pub fn threat_status(game: &HexGameState) -> ThreatStatus {
    if game.winner().is_some() {
        return ThreatStatus::Quiet;
    }

    let current = game.current_player();
    if !game.eval().has_any_threats() {
        return ThreatStatus::Quiet;
    }

    // Can the current player win immediately?
    //
    // We do two passes: first look for 5-windows (single empty) so we always
    // prefer a one-stone win over a two-stone win. Then look for 4-windows.
    let remaining = game.placements_remaining();
    let mut pair_win: Option<Turn> = None;

    let mut empties = SmallVec::<[Hex; 2]>::new();
    for key in game.eval().hot_windows(current) {
        empties.clear();
        window_empties(game, key, &mut empties);

        match empties.len() {
            1 => {
                // A single empty in a 5-window wins immediately, even with
                // 2 placements remaining — the game ends as soon as the 6th
                // stone is placed.
                return ThreatStatus::WinningTurn(Turn::single(empties[0]));
            }
            2 if remaining >= 2 && pair_win.is_none() => {
                // Remember the first 4-window, but keep scanning in case a
                // 5-window appears later in the iteration order.
                pair_win = Some(Turn::pair(empties[0], empties[1]));
            }
            _ => {}
        }
    }

    if let Some(turn) = pair_win {
        return ThreatStatus::WinningTurn(turn);
    }

    let opp = 1 - current;
    if !game.eval().has_threats(opp) {
        return ThreatStatus::Quiet;
    }

    let (flat_empties, window_lengths) = opponent_threat_windows(game);
    if flat_empties.is_empty() {
        return ThreatStatus::Quiet;
    }

    let mut must_hit = SmallVec::<[&[Hex]; 16]>::new();
    let mut offset = 0usize;
    for &len in &window_lengths {
        let s = &flat_empties[offset..offset + len as usize];
        must_hit.push(s);
        offset += len as usize;
    }

    // Build exact BlockConstraint.
    let placements = game.placements_remaining();

    // Collect all unique empty cells across threat windows.
    let mut all_cells: SmallVec<[Hex; 32]> =
        must_hit.iter().flat_map(|w| w.iter().copied()).collect();
    all_cells.sort();
    all_cells.dedup();

    // Intersection: cells that appear in EVERY threat window.
    // These single-handedly block all threats, so any turn containing one
    // of them is valid regardless of the other placement(s).
    let mut cells = SmallVec::<[Hex; 16]>::new();
    for &cell in &all_cells {
        if must_hit.iter().all(|w| w.contains(&cell)) {
            cells.push(cell);
        }
    }

    // With only 1 placement left, we cannot play a pair.  If there is no
    // single cell that blocks everything, the position is unblockable.
    if placements <= 1 {
        if cells.is_empty() {
            return ThreatStatus::Unblockable;
        }
        let pairs = SmallVec::<[(Hex, Hex); 32]>::new();
        return ThreatStatus::MustBlock(BlockConstraint { cells, pairs });
    }

    // placements >= 2: enumerate every distinct pair of candidate cells.
    // A pair is valid if, for every threat window, at least one of the two
    // cells lies inside that window.
    let mut pairs = SmallVec::<[(Hex, Hex); 32]>::new();
    for i in 0..all_cells.len() {
        for j in (i + 1)..all_cells.len() {
            let c1 = all_cells[i];
            let c2 = all_cells[j];
            debug_assert_ne!(c1, c2, "self-pair detected in BlockConstraint enumeration");
            if must_hit.iter().all(|w| w.contains(&c1) || w.contains(&c2)) {
                pairs.push((c1, c2));
            }
        }
    }

    // If neither single-cell blocks nor any valid pair exists, we lose.
    if pairs.is_empty() && cells.is_empty() {
        return ThreatStatus::Unblockable;
    }

    ThreatStatus::MustBlock(BlockConstraint { cells, pairs })
}

/// Check whether a single turn is legal under threat constraints,
/// given a pre-computed threat status.
///
/// This function is O(pairs) in the worst case and should be called via
/// `retain` on a turn vector after computing `threat_status` once per node.
pub fn turn_satisfies_status(status: &ThreatStatus, turn: Turn) -> bool {
    match status {
        // Quiet positions impose no restrictions.
        ThreatStatus::Quiet => true,

        // WinningTurn: the only legal move is the exact winning turn.
        ThreatStatus::WinningTurn(w) => turn == *w,

        ThreatStatus::MustBlock(bc) => {
            if turn.placements() == 1 {
                // With only 1 placement, the single stone must block every
                // threat window by itself.
                bc.cells().contains(&turn.first())
            } else {
                let second = turn.second().unwrap();

                // If either cell alone blocks every threat window, any pair
                // containing it is valid.
                if bc.cells().contains(&turn.first()) || bc.cells().contains(&second) {
                    return true;
                }

                // Otherwise the pair must exactly match a valid blocking pair.
                bc.pairs().iter().any(|&(a, b_pair)| {
                    (a == turn.first() && b_pair == second)
                        || (a == second && b_pair == turn.first())
                })
            }
        }

        // Unblockable means the threat filter does not constrain moves.
        ThreatStatus::Unblockable => true,
    }
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

    // Fast exit: no fours or fives means no live cells.
    if !game.eval().has_threats(player) {
        return;
    }

    let mut empties = SmallVec::<[Hex; 2]>::new();
    let mut seen = FxHashSet::default();
    for key in game.eval().hot_windows(player) {
        empties.clear();
        window_empties(game, key, &mut empties);
        for &h in &empties {
            if seen.insert(h) {
                out.push(h);
            }
        }
    }
}
