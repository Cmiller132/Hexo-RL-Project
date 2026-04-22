//! Turn-based alpha-beta search with iterative deepening and transposition table.
//!
//! Key design: the search unit is a **Turn** (pair of moves), not individual
//! placements. This doubles effective depth vs placement-based search.
//!
//! Features inspired by SealBot's architecture:
//! - Turn-based search (2-move atomic unit)
//! - Instant-win detection + unblockable-win pruning at every node
//! - Threat-filtered move generation (prune moves that don't block threats)
//! - Deep quiescence search (depth 16) on threat moves only
//! - PVS + LMR + killer/history heuristics (advantage over SealBot)
//! - Aspiration windows in iterative deepening
//! - Pair-sum constraint for focused branching
//! - Mate-distance TT scoring
//! - Root candidates returned for temperature-based sampling in self-play

use rustc_hash::FxHashMap;
use std::cmp::Reverse;
use std::time::{Duration, Instant, SystemTime};

use crate::core::{hex_distance, Hex, HEX_DIRECTIONS};
use crate::eval::{evaluate, WIN_SCORE};
use crate::game::HexGameState;
use crate::game::WIN_LENGTH;

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------

/// Maximum ply depth tracked for killer moves.
const MAX_PLY: usize = 64;

/// Initial aspiration window.
const ASPIRATION_WINDOW: i32 = 500;

/// Maximum quiescence search depth (in turns).
const QUIESCE_DEPTH: i32 = 6;

/// Maximum candidate cells per node (non-root).
const CANDIDATE_CAP: usize = 12;

/// Maximum candidate cells at root.
const ROOT_CANDIDATE_CAP: usize = 14;

/// Pair-sum cap: only generate pairs (i,j) where i+j <= this value.
const PAIR_SUM_CAP: usize = 12;

/// Weight for eval delta in move ordering.
const DELTA_WEIGHT: i32 = 15;

/// TT entries cap before clearing.
const TT_MAX_SIZE: usize = 2_000_000;

// -------------------------------------------------------------------------
// Turn type
// -------------------------------------------------------------------------

/// A turn consists of 1 or 2 placements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Turn {
    pub m1: Hex,
    pub m2: Option<Hex>,
}

impl Turn {
    #[inline]
    pub fn one(m: Hex) -> Self {
        Turn { m1: m, m2: None }
    }

    #[inline]
    pub fn two(a: Hex, b: Hex) -> Self {
        // Canonical ordering for TT consistency
        if a <= b {
            Turn { m1: a, m2: Some(b) }
        } else {
            Turn { m1: b, m2: Some(a) }
        }
    }
}

// -------------------------------------------------------------------------
// Transposition table
// -------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub enum TTFlag {
    Exact,
    LowerBound,
    UpperBound,
}

#[derive(Clone, Copy)]
pub struct TTEntry {
    pub depth: i32,
    pub score: i32,
    pub flag: TTFlag,
    pub best_turn: Option<Turn>,
}

// -------------------------------------------------------------------------
// Search state
// -------------------------------------------------------------------------

pub struct SearchState {
    pub tt: FxHashMap<u64, TTEntry>,
    pub nodes: u64,
    pub deadline: Option<Instant>,
    pub aborted: bool,
    /// Killer turns per ply.
    killers: [Option<Turn>; MAX_PLY],
    /// History heuristic: per-cell score (depth² on cutoff).
    history: FxHashMap<Hex, i32>,
    /// Random seed for tiebreaking and non-deterministic play.
    noise_seed: u64,
    /// Noise level for non-deterministic play (0.0 = deterministic).
    /// Affects candidate ordering to produce varied games for training.
    /// Threat blocking and instant-win detection remain deterministic.
    noise_level: f32,
}

impl SearchState {
    pub fn new(noise_level: f32) -> Self {
        Self {
            tt: FxHashMap::default(),
            nodes: 0,
            deadline: None,
            aborted: false,
            killers: [None; MAX_PLY],
            history: FxHashMap::default(),
            noise_seed: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42),
            noise_level,
        }
    }

    #[inline]
    fn timed_out(&self) -> bool {
        if let Some(dl) = self.deadline {
            Instant::now() >= dl
        } else {
            false
        }
    }

    fn update_killers(&mut self, ply: usize, t: Turn) {
        let idx = ply.min(MAX_PLY - 1);
        self.killers[idx] = Some(t);
    }

    fn update_history(&mut self, t: Turn, depth: i32) {
        let bonus = depth * depth;
        *self.history.entry(t.m1).or_insert(0) += bonus;
        if let Some(m2) = t.m2 {
            *self.history.entry(m2).or_insert(0) += bonus;
        }
    }

    fn maybe_clear_tt(&mut self) {
        if self.tt.len() > TT_MAX_SIZE {
            self.tt.clear();
        }
    }
}

// -------------------------------------------------------------------------
// TT hash
// -------------------------------------------------------------------------

/// TT hash: board + side-to-move + placements remaining.
#[inline]
fn tt_hash(game: &HexGameState) -> u64 {
    let side = if game.current_player == 0 {
        0x9e37_79b9_7f4a_7c15u64
    } else {
        0xc2b2_ae3d_27d4_eb4fu64
    };
    let phase = match game.placements_remaining {
        0 => 0x1656_67b1_9e37_79f9u64,
        1 => 0x27d4_eb2f_1656_67c5u64,
        _ => 0x94d0_49bb_1331_11ebu64,
    };
    game.zobrist_hash ^ side ^ phase
}

// -------------------------------------------------------------------------
// Make/unmake turns
// -------------------------------------------------------------------------

/// Execute a full turn (1 or 2 placements). Returns (game_over, placements_made).
fn make_turn(game: &mut HexGameState, t: Turn) -> (bool, u8) {
    game.place(t.m1.q, t.m1.r).unwrap_or(true);
    if game.is_over() {
        return (true, 1);
    }
    if let Some(m2) = t.m2 {
        game.place(m2.q, m2.r).unwrap_or(true);
        if game.is_over() {
            return (true, 2);
        }
        return (false, 2);
    }
    (false, 1)
}

/// Undo a full turn. `placed` = number of placements that were actually made.
fn unmake_turn(game: &mut HexGameState, placed: u8) {
    for _ in 0..placed {
        game.unmake_move();
    }
}

// -------------------------------------------------------------------------
// Candidate scoring and turn generation
// -------------------------------------------------------------------------

/// Score a single candidate cell for sorting.
/// Includes hot-window bonuses for threat blocking/completing.
/// When `noise_level > 0`, adds randomization for training data variety.
/// Threat-related bonuses remain unaffected by noise (deterministic blocking).
#[inline]
fn score_candidate(
    game: &HexGameState,
    cell: Hex,
    history: &FxHashMap<Hex, i32>,
    sign: i32,
    noise_level: f32,
    noise_seed: u64,
) -> i64 {
    let delta = game.move_eval_delta(cell, game.current_player) as i64;
    let hist = history.get(&cell).copied().unwrap_or(0).min(500_000) as i64;

    // Hot-window bonus: cells in opponent's hot windows (blocking threats)
    // These bonuses are NOT affected by noise — blocking must remain deterministic.
    let opp = (1 - game.current_player) as usize;
    let mut tactical = 0i64;
    for &(wq, wr, dir) in &game.hot_windows[opp] {
        let (dq, dr) = HEX_DIRECTIONS[dir as usize];
        for k in 0..WIN_LENGTH {
            if cell.q == wq + dq * k && cell.r == wr + dr * k {
                tactical += 50_000;
                break;
            }
        }
    }
    // Cells in our own hot windows (completing threats)
    let p = game.current_player as usize;
    for &(wq, wr, dir) in &game.hot_windows[p] {
        let (dq, dr) = HEX_DIRECTIONS[dir as usize];
        for k in 0..WIN_LENGTH {
            if cell.q == wq + dq * k && cell.r == wr + dr * k {
                tactical += 40_000;
                break;
            }
        }
    }

    let base = delta * sign as i64 * DELTA_WEIGHT as i64 + hist;

    // Inject noise into the non-tactical portion of the score.
    // This shuffles candidate ordering for varied training games while
    // preserving deterministic threat handling.
    let noisy_base = if noise_level > 0.0 {
        // Deterministic-per-cell pseudo-random noise from cell coords + seed
        let cell_hash = (cell.q as u64).wrapping_mul(2654435761)
            ^ (cell.r as u64).wrapping_mul(40503)
            ^ noise_seed;
        // Map to [-1.0, 1.0] range
        let rand_frac = ((cell_hash % 10000) as f64 / 5000.0) - 1.0;
        // Scale noise relative to the base score magnitude
        let noise_mag =
            (base.abs() as f64 * noise_level as f64 * 0.5 + 500.0 * noise_level as f64) * rand_frac;
        base + noise_mag as i64
    } else {
        base
    };

    noisy_base + tactical
}

/// Generate candidate cells sorted by score, capped.
fn generate_sorted_candidates(
    game: &HexGameState,
    history: &FxHashMap<Hex, i32>,
    cap: usize,
    noise_level: f32,
    noise_seed: u64,
) -> Vec<Hex> {
    let mut cands: Vec<Hex> = game.candidates_near2();
    if cands.is_empty() {
        return cands;
    }

    let sign = if game.current_player == 0 {
        1i32
    } else {
        -1i32
    };
    cands.sort_by_cached_key(|&m| {
        Reverse(score_candidate(
            game,
            m,
            history,
            sign,
            noise_level,
            noise_seed,
        ))
    });
    cands.truncate(cap);
    cands
}

/// Generate turn pairs from sorted candidates with pair-sum constraint.
fn generate_turn_pairs(cands: &[Hex], max_pair_sum: usize) -> Vec<Turn> {
    let n = cands.len();
    let mut turns = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            if i + j <= max_pair_sum {
                turns.push(Turn::two(cands[i], cands[j]));
            }
        }
    }
    turns
}

// -------------------------------------------------------------------------
// Instant-win detection
// -------------------------------------------------------------------------

/// Check if the current player can win this turn (with remaining placements).
/// Uses incrementally-maintained hot_windows for O(hot_set) performance.
fn find_instant_win(game: &HexGameState, player: u8) -> Option<Turn> {
    let p = player as usize;
    if game.window_fours[p] == 0 && game.window_fives[p] == 0 {
        return None;
    }
    let remaining = game.placements_remaining as usize;

    for &(wq, wr, dir) in &game.hot_windows[p] {
        let (dq, dr) = HEX_DIRECTIONS[dir as usize];
        let mut empties = Vec::new();
        for k in 0..WIN_LENGTH {
            let h = Hex::new(wq + dq * k, wr + dr * k);
            if !game.board.contains_key(&h) {
                empties.push(h);
            }
        }
        if empties.is_empty() || empties.len() > remaining {
            continue;
        }
        match empties.len() {
            1 => {
                if remaining == 1 {
                    return Some(Turn::one(empties[0]));
                }
                // 2 placements: fill the win cell + any legal cell
                let win_cell = empties[0];
                for c in game.candidates_near2() {
                    if c != win_cell {
                        return Some(Turn::two(win_cell, c));
                    }
                }
            }
            2 if remaining >= 2 => {
                return Some(Turn::two(empties[0], empties[1]));
            }
            _ => {}
        }
    }
    None
}

// -------------------------------------------------------------------------
// Threat analysis
// -------------------------------------------------------------------------

/// Filter turns to only keep those that block all opponent threats.
fn filter_turns_by_threats(game: &HexGameState, turns: &mut Vec<Turn>) {
    let opp = 1 - game.current_player;
    let must_hit = game.collect_threat_window_empties(opp);

    if must_hit.is_empty() || turns.is_empty() {
        return;
    }

    let filtered: Vec<Turn> = turns
        .iter()
        .copied()
        .filter(|t| {
            must_hit
                .iter()
                .all(|set| set.contains(&t.m1) || t.m2.map_or(false, |m2| set.contains(&m2)))
        })
        .collect();

    if !filtered.is_empty() {
        *turns = filtered;
    }
}

// -------------------------------------------------------------------------
// Turn generation
// -------------------------------------------------------------------------

/// Generate root turns with colony candidate.
fn generate_root_turns(
    game: &HexGameState,
    history: &FxHashMap<Hex, i32>,
    noise_level: f32,
    noise_seed: u64,
) -> Vec<Turn> {
    if game.board.is_empty() {
        return vec![Turn::one(Hex::ORIGIN)];
    }

    // Opening: player 0 only places 1 stone
    if game.placements_remaining == 1 && game.move_count == 0 {
        return vec![Turn::one(Hex::ORIGIN)];
    }

    // Check for instant wins (always deterministic)
    if let Some(win_turn) = find_instant_win(game, game.current_player) {
        return vec![win_turn];
    }

    let mut cands =
        generate_sorted_candidates(game, history, ROOT_CANDIDATE_CAP, noise_level, noise_seed);

    // Add colony candidate: far from centroid
    if cands.len() >= 2 && !game.board.is_empty() {
        let (sq, sr, n) = game
            .board
            .keys()
            .fold((0i64, 0i64, 0u32), |(sq, sr, n), h| {
                (sq + h.q as i64, sr + h.r as i64, n + 1)
            });
        let cq = (sq as f64 / n as f64).round() as i32;
        let cr = (sr as f64 / n as f64).round() as i32;

        let max_r = game
            .board
            .keys()
            .map(|h| hex_distance(*h, Hex::new(cq, cr)))
            .max()
            .unwrap_or(0);

        let colony_dist = max_r + 3;
        let dirs: [(i32, i32); 6] = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)];
        let dir_idx = (noise_seed as usize ^ game.zobrist_hash as usize) % 6;
        let (dq, dr) = dirs[dir_idx];
        let colony = Hex::new(cq + dq * colony_dist, cr + dr * colony_dist);

        if !game.board.contains_key(&colony)
            && game
                .board
                .keys()
                .any(|&h| hex_distance(h, colony) <= crate::game::PLACEMENT_RADIUS)
        {
            if !cands.contains(&colony) {
                cands.push(colony);
            }
        }
    }

    // Handle single-placement turns
    if game.placements_remaining == 1 {
        return cands.into_iter().map(Turn::one).collect();
    }

    // Generate pairs with pair-sum constraint at root too
    let mut turns = generate_turn_pairs(&cands, PAIR_SUM_CAP);

    filter_turns_by_threats(game, &mut turns);
    turns
}

/// Generate inner (non-root) turns.
/// Inner nodes always use deterministic ordering (noise_level=0) to preserve
/// search quality. Noise is only injected at the root to vary game openings.
fn generate_inner_turns(
    game: &HexGameState,
    history: &FxHashMap<Hex, i32>,
    noise_seed: u64,
) -> Vec<Turn> {
    if game.board.is_empty() {
        return vec![Turn::one(Hex::ORIGIN)];
    }

    // Single-placement turn
    if game.placements_remaining == 1 {
        let cands = generate_sorted_candidates(game, history, CANDIDATE_CAP, 0.0, noise_seed);
        return cands.into_iter().map(Turn::one).collect();
    }

    // Note: instant-win check is done by the caller (alphabeta) before
    // calling this function, so we skip it here to avoid redundant work.

    let cands = generate_sorted_candidates(game, history, CANDIDATE_CAP, 0.0, noise_seed);
    let mut turns = generate_turn_pairs(&cands, PAIR_SUM_CAP);

    filter_turns_by_threats(game, &mut turns);
    turns
}

// -------------------------------------------------------------------------
// Turn ordering
// -------------------------------------------------------------------------

/// Promote TT best and killer to front. Candidates are already sorted by
/// eval delta from generate_sorted_candidates, so pairs inherit good ordering.
fn promote_best_turns(turns: &mut Vec<Turn>, tt_best: Option<Turn>, killer: Option<Turn>) {
    let mut start = 0;
    if let Some(tt_t) = tt_best {
        if let Some(pos) = turns.iter().position(|t| *t == tt_t) {
            turns.swap(0, pos);
            start = 1;
        }
    }
    if let Some(k) = killer {
        if let Some(pos) = turns[start..].iter().position(|t| *t == k) {
            turns.swap(start, pos + start);
        }
    }
}

// -------------------------------------------------------------------------
// Threat turn generation (for quiescence)
// -------------------------------------------------------------------------

/// Generate turns from hot window cells only (for quiescence search).
fn generate_threat_turns(game: &HexGameState) -> Vec<Turn> {
    let player = game.current_player;
    let opp = 1 - player;

    // Single-placement quiescence
    if game.placements_remaining == 1 {
        let opp_threats = game.collect_threat_cells(opp);
        let my_threats = game.collect_threat_cells(player);
        let cells = if !opp_threats.is_empty() {
            opp_threats
        } else {
            my_threats
        };
        if cells.is_empty() {
            return Vec::new();
        }
        return cells.into_iter().take(6).map(Turn::one).collect();
    }

    let opp_threats = game.collect_threat_cells(opp);
    let my_threats = game.collect_threat_cells(player);
    let primary = if !opp_threats.is_empty() {
        &opp_threats
    } else {
        &my_threats
    };
    if primary.is_empty() {
        return Vec::new();
    }

    // Combine all threat cells from both sides
    let mut all_threats: Vec<Hex> = opp_threats
        .iter()
        .chain(my_threats.iter())
        .copied()
        .collect();
    all_threats.sort();
    all_threats.dedup();

    let mut turns = Vec::new();

    // Pairs within threat cells (most important)
    let n = all_threats.len().min(8);
    for i in 0..n {
        for j in (i + 1)..n {
            turns.push(Turn::two(all_threats[i], all_threats[j]));
        }
    }

    turns.sort_by(|a, b| a.m1.cmp(&b.m1).then(a.m2.cmp(&b.m2)));
    turns.dedup();
    turns.truncate(16);
    turns
}

// -------------------------------------------------------------------------
// Mate-distance TT adjustments
// -------------------------------------------------------------------------

#[inline]
fn adjust_mate_store(score: i32, ply: usize) -> i32 {
    if score > WIN_SCORE - 200 {
        score + ply as i32
    } else if score < -(WIN_SCORE - 200) {
        score - ply as i32
    } else {
        score
    }
}

#[inline]
fn adjust_mate_load(score: i32, ply: usize) -> i32 {
    if score > WIN_SCORE - 200 {
        score - ply as i32
    } else if score < -(WIN_SCORE - 200) {
        score + ply as i32
    } else {
        score
    }
}

// -------------------------------------------------------------------------
// Quiescence search (turn-based)
// -------------------------------------------------------------------------

fn quiesce(
    game: &mut HexGameState,
    ss: &mut SearchState,
    mut alpha: i32,
    beta: i32,
    qdepth: i32,
    ply: usize,
) -> i32 {
    if ss.aborted {
        return 0;
    }
    ss.nodes += 1;
    if ss.nodes % 1024 == 0 && ss.timed_out() {
        ss.aborted = true;
        return 0;
    }

    let player = game.current_player;

    if game.is_over() {
        return if game.winner == Some(player) {
            WIN_SCORE - ply as i32
        } else {
            -(WIN_SCORE - ply as i32)
        };
    }

    // Stand-pat
    let stand_pat = evaluate(game, player);
    if stand_pat >= beta {
        return stand_pat;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    if qdepth <= 0 {
        return alpha;
    }

    // Only extend when threats exist
    let p = player as usize;
    let o = 1 - p;
    if game.window_fives[p] == 0
        && game.window_fives[o] == 0
        && game.window_fours[p] == 0
        && game.window_fours[o] == 0
    {
        return alpha;
    }

    // Check instant win
    if let Some(win_turn) = find_instant_win(game, player) {
        let (over, placed) = make_turn(game, win_turn);
        let score = if over && game.winner == Some(player) {
            WIN_SCORE - ply as i32
        } else if over {
            let s = -quiesce(game, ss, -beta, -alpha, qdepth - 1, ply + 1);
            s
        } else {
            -quiesce(game, ss, -beta, -alpha, qdepth - 1, ply + 1)
        };
        unmake_turn(game, placed);
        return if over && game.winner == Some(player) {
            WIN_SCORE - ply as i32
        } else {
            score
        };
    }

    // Check unblockable opponent win
    if game.is_opponent_win_unblockable(game.placements_remaining) {
        return -(WIN_SCORE - ply as i32 - 1);
    }

    let turns = generate_threat_turns(game);
    if turns.is_empty() {
        return alpha;
    }

    for t in &turns {
        let (over, placed) = make_turn(game, *t);
        let score = if over {
            if game.winner == Some(player) {
                WIN_SCORE - ply as i32
            } else {
                -(WIN_SCORE - ply as i32)
            }
        } else {
            -quiesce(game, ss, -beta, -alpha, qdepth - 1, ply + 1)
        };
        unmake_turn(game, placed);
        if ss.aborted {
            return 0;
        }

        if score >= beta {
            return score;
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

// -------------------------------------------------------------------------
// Main alpha-beta search (turn-based)
// -------------------------------------------------------------------------

fn alphabeta(
    game: &mut HexGameState,
    ss: &mut SearchState,
    depth: i32,
    ply: usize,
    mut alpha: i32,
    beta: i32,
) -> i32 {
    if ss.aborted {
        return 0;
    }
    if ss.nodes % 1024 == 0 && ss.timed_out() {
        ss.aborted = true;
        return 0;
    }
    ss.nodes += 1;

    let player = game.current_player;

    // Terminal check
    if game.is_over() {
        return if game.winner == Some(player) {
            WIN_SCORE - ply as i32
        } else {
            -(WIN_SCORE - ply as i32)
        };
    }

    // Leaf: quiescence
    if depth <= 0 {
        return quiesce(game, ss, alpha, beta, QUIESCE_DEPTH, ply);
    }

    // Instant win check
    if let Some(win_turn) = find_instant_win(game, player) {
        let (over, placed) = make_turn(game, win_turn);
        let score = if over && game.winner == Some(player) {
            WIN_SCORE - ply as i32
        } else {
            -alphabeta(game, ss, depth - 1, ply + 1, -beta, -alpha)
        };
        unmake_turn(game, placed);
        return score;
    }

    // Unblockable opponent win check
    if game.is_opponent_win_unblockable(game.placements_remaining) {
        return -(WIN_SCORE - ply as i32 - 1);
    }

    // Reverse futility pruning
    if depth <= 2 && ply > 0 {
        let margin = 2000 * depth;
        let static_eval = evaluate(game, player);
        if static_eval - margin >= beta {
            return static_eval;
        }
    }

    // TT probe
    let hash = tt_hash(game);
    let tt_entry = ss.tt.get(&hash).copied();
    let mut tt_best_turn = None;
    if let Some(entry) = tt_entry {
        if entry.depth >= depth {
            let adj_score = adjust_mate_load(entry.score, ply);
            match entry.flag {
                TTFlag::Exact => return adj_score,
                TTFlag::LowerBound => {
                    if adj_score >= beta {
                        return adj_score;
                    }
                    if adj_score > alpha {
                        alpha = adj_score;
                    }
                }
                TTFlag::UpperBound => {
                    if adj_score <= alpha {
                        return adj_score;
                    }
                }
            }
        }
        tt_best_turn = entry.best_turn;
    }

    // Generate turns (inner nodes always deterministic)
    let mut turns = generate_inner_turns(game, &ss.history, ss.noise_seed);
    if turns.is_empty() {
        return evaluate(game, player);
    }

    let ply_idx = ply.min(MAX_PLY - 1);
    promote_best_turns(&mut turns, tt_best_turn, ss.killers[ply_idx]);

    let mut best_score = i32::MIN + 1;
    let mut best_turn = turns[0];
    let orig_alpha = alpha;

    for (move_idx, t) in turns.iter().enumerate() {
        let (over, placed) = make_turn(game, *t);

        let score = if over {
            if game.winner == Some(player) {
                WIN_SCORE - ply as i32
            } else if game.winner == Some(1 - player) {
                -(WIN_SCORE - ply as i32)
            } else {
                0
            }
        } else {
            // Late move pruning: at low depth, skip moves late in order
            if depth <= 2 && move_idx >= 20 && best_score > -(WIN_SCORE - 100) {
                unmake_turn(game, placed);
                continue;
            }

            // PVS + LMR (aggressive: start reduction at move 2)
            let do_lmr = depth >= 2 && move_idx >= 2 && best_score > -(WIN_SCORE - 100);
            let lmr_reduction = if move_idx >= 10 { 2 } else { 1 };

            if move_idx == 0 {
                -alphabeta(game, ss, depth - 1, ply + 1, -beta, -alpha)
            } else if do_lmr {
                let reduced = -alphabeta(
                    game,
                    ss,
                    depth - 1 - lmr_reduction,
                    ply + 1,
                    -(alpha + 1),
                    -alpha,
                );
                if reduced > alpha && !ss.aborted {
                    let nw = -alphabeta(game, ss, depth - 1, ply + 1, -(alpha + 1), -alpha);
                    if nw > alpha && nw < beta && !ss.aborted {
                        -alphabeta(game, ss, depth - 1, ply + 1, -beta, -alpha)
                    } else {
                        nw
                    }
                } else {
                    reduced
                }
            } else {
                let nw = -alphabeta(game, ss, depth - 1, ply + 1, -(alpha + 1), -alpha);
                if nw > alpha && nw < beta && !ss.aborted {
                    -alphabeta(game, ss, depth - 1, ply + 1, -beta, -alpha)
                } else {
                    nw
                }
            }
        };

        unmake_turn(game, placed);
        if ss.aborted {
            return 0;
        }

        if score > best_score {
            best_score = score;
            best_turn = *t;
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            if best_score.abs() < WIN_SCORE - 100 {
                ss.update_killers(ply_idx, *t);
                ss.update_history(*t, depth);
            }
            break;
        }
    }

    // TT store with mate-distance adjustment
    let flag = if best_score <= orig_alpha {
        TTFlag::UpperBound
    } else if best_score >= beta {
        TTFlag::LowerBound
    } else {
        TTFlag::Exact
    };
    let store_score = adjust_mate_store(best_score, ply);
    ss.tt.insert(
        hash,
        TTEntry {
            depth,
            score: store_score,
            flag,
            best_turn: Some(best_turn),
        },
    );

    best_score
}

// -------------------------------------------------------------------------
// Root search with full move info
// -------------------------------------------------------------------------

fn search_root(
    game: &mut HexGameState,
    ss: &mut SearchState,
    depth: i32,
    root_turns: &[Turn],
    init_alpha: i32,
    init_beta: i32,
) -> (Turn, i32, Vec<(Turn, i32)>) {
    let player = game.current_player;
    let mut alpha = init_alpha;
    let beta = init_beta;
    let mut best_turn = root_turns[0];
    let mut best_score = i32::MIN + 1;
    let mut scores: Vec<(Turn, i32)> = Vec::with_capacity(root_turns.len());

    for (move_idx, t) in root_turns.iter().enumerate() {
        let (over, placed) = make_turn(game, *t);

        let score = if over {
            if game.winner == Some(player) {
                WIN_SCORE
            } else {
                -WIN_SCORE
            }
        } else {
            // PVS at root
            if move_idx == 0 {
                -alphabeta(game, ss, depth - 1, 1, -beta, -alpha)
            } else {
                let nw = -alphabeta(game, ss, depth - 1, 1, -(alpha + 1), -alpha);
                if nw > alpha && nw < beta && !ss.aborted {
                    -alphabeta(game, ss, depth - 1, 1, -beta, -alpha)
                } else {
                    nw
                }
            }
        };

        unmake_turn(game, placed);
        if ss.aborted {
            break;
        }

        scores.push((*t, score));

        if score > best_score {
            best_score = score;
            best_turn = *t;
        }
        if score > alpha {
            alpha = score;
        }
    }

    (best_turn, best_score, scores)
}

// -------------------------------------------------------------------------
// Iterative deepening (turn-based)
// -------------------------------------------------------------------------

/// Result of a turn-based search.
pub struct SearchResult {
    pub best_turn: Turn,
    pub best_move: Hex,
    pub score: i32,
    pub depth_reached: i32,
    pub nodes: u64,
    pub root_candidates: Vec<(Hex, i32)>,
}

/// Run iterative-deepening turn-based alpha-beta search.
///
/// `noise_level`: 0.0 = deterministic (strongest play), >0.0 = inject noise
/// into root candidate ordering for varied training data. Threat blocking
/// and instant-win detection remain deterministic regardless of noise level.
pub fn iterative_deepening(
    game: &HexGameState,
    time_limit: Duration,
    max_depth: i32,
    _near_radius: i32, // kept for API compat, always uses radius-2 internally
    collect_candidates: bool,
    noise_level: f32,
) -> SearchResult {
    let mut ss = SearchState::new(noise_level);
    ss.deadline = Some(Instant::now() + time_limit);
    ss.maybe_clear_tt();

    // Handle opening move
    if game.board.is_empty() {
        return SearchResult {
            best_turn: Turn::one(Hex::ORIGIN),
            best_move: Hex::ORIGIN,
            score: 0,
            depth_reached: 0,
            nodes: 0,
            root_candidates: vec![(Hex::ORIGIN, 0)],
        };
    }

    // Generate root turns (using immutable game)
    let mut root_turns = generate_root_turns(game, &ss.history, ss.noise_level, ss.noise_seed);
    if root_turns.is_empty() {
        let c = game.candidates_near2();
        if c.len() >= 2 && game.placements_remaining >= 2 {
            root_turns.push(Turn::two(c[0], c[1]));
        } else if !c.is_empty() {
            root_turns.push(Turn::one(c[0]));
        } else {
            return SearchResult {
                best_turn: Turn::one(Hex::ORIGIN),
                best_move: Hex::ORIGIN,
                score: 0,
                depth_reached: 0,
                nodes: 0,
                root_candidates: vec![],
            };
        }
    }

    // Clone game ONCE for search (search_root uses make/unmake instead of cloning per move)
    let mut search_game = game.clone();

    let mut best_turn = root_turns[0];
    let mut best_score = 0i32;
    let mut depth_reached = 0;

    for depth in 1..=max_depth {
        ss.aborted = false;

        // Root search at this depth with aspiration windows
        let (bt, score, scores) = if depth >= 3 && best_score.abs() < WIN_SCORE - 100 {
            let lo = best_score - ASPIRATION_WINDOW;
            let hi = best_score + ASPIRATION_WINDOW;

            let (bt, s, scores) =
                search_root(&mut search_game, &mut ss, depth, &root_turns, lo, hi);
            if ss.aborted {
                break;
            }

            if s <= lo || s >= hi {
                // Re-search with full window
                ss.aborted = false;
                search_root(
                    &mut search_game,
                    &mut ss,
                    depth,
                    &root_turns,
                    i32::MIN + 1,
                    i32::MAX - 1,
                )
            } else {
                (bt, s, scores)
            }
        } else {
            search_root(
                &mut search_game,
                &mut ss,
                depth,
                &root_turns,
                i32::MIN + 1,
                i32::MAX - 1,
            )
        };

        if ss.aborted {
            break;
        }

        best_turn = bt;
        best_score = score;
        depth_reached = depth;

        // Re-sort root turns for next iteration
        if !scores.is_empty() {
            let score_map: FxHashMap<Turn, i32> = scores.into_iter().collect();
            root_turns.sort_by(|a, b| {
                let sa = score_map.get(a).copied().unwrap_or(i32::MIN);
                let sb = score_map.get(b).copied().unwrap_or(i32::MIN);
                sb.cmp(&sa)
            });
        }

        // Early exit on forced win/loss
        if score.abs() >= WIN_SCORE - 100 {
            break;
        }
    }

    // Build root_candidates for NN training compatibility (always deterministic)
    let root_candidates = if collect_candidates {
        let cands = generate_sorted_candidates(game, &ss.history, 12, 0.0, ss.noise_seed);
        cands
            .iter()
            .map(|&m| {
                let sign = if game.current_player == 0 { 1 } else { -1 };
                let score = game.move_eval_delta(m, game.current_player) * sign;
                (m, score)
            })
            .collect()
    } else {
        vec![]
    };

    SearchResult {
        best_turn,
        best_move: best_turn.m1,
        score: best_score,
        depth_reached,
        nodes: ss.nodes,
        root_candidates,
    }
}
