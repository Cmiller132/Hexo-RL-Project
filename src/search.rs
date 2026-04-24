//! Turn-based alpha-beta search with iterative deepening and transposition table.
//!
//! Key design: the search unit is a **Turn** (pair of moves), not individual
//! placements. This doubles effective depth vs placement-based search.
//!
//! # Pruning strategy
//!
//! The engine uses a multi-layered pruning stack:
//!
//! 1. **Instant-win detection** — at every node, check if the current player
//!    can win immediately (`find_instant_win`). If yes, return the winning
//!    score without expanding children.
//! 2. **Unblockable-loss pruning** — if `threat_status` reports `Unblockable`,
//!    the opponent has disjoint threats we cannot stop; return a large
//!    negative score immediately.
//! 3. **Threat-filtered move generation** — `generate_root_turns` and
//!    `generate_inner_turns` compute `threat_status` **once per node**, then
//!    retain only turns that satisfy the blocking constraint.  This often
//!    reduces the branching factor from hundreds to a handful.
//! 4. **Reverse futility pruning** — at shallow depth (`depth <= 2`), if the
//!    static evaluation is so far above beta that even a generous margin
//!    cannot save it, return the static score early.
//! 5. **Late-move pruning** — at low depth, skip moves late in the ordering
//!    when the position is not in danger of mate.
//! 6. **PVS + LMR** — Principal Variation Search with Late Move Reduction.
//!    The first move is searched with a full window; subsequent moves use a
//!    null window.  Moves indexed ≥2 are reduced by 1 or 2 plies; if they
//!    beat alpha, a re-search at full depth is performed.
//!
//! # Move ordering
//!
//! Candidates are scored by:
//! - Incremental evaluation delta (`hypothetical_score_delta`).
//! - History heuristic (depth² bonus on cutoffs).
//! - Tactical bonuses for cells in hot windows (+50k for blocking opponent,
//!   +40k for completing our own threats).
//! - Optional noise for training data variety (applied only to the non-tactical
//!   portion of the score so blocking remains deterministic).
//!
//! At each ply the TT best move and killer move are promoted to the front.
//!
//! # Transposition table
//!
//! A simple FxHashMap stores exact, lower-bound, and upper-bound entries
//! keyed by Zobrist hash XOR side-to-move XOR phase.  Mate scores are
//! adjusted by ply distance so that "mate in 3" and "mate in 5" are
//! distinguished correctly.

use rustc_hash::FxHashMap;
use std::cmp::Reverse;
use std::time::{Duration, Instant, SystemTime};

use crate::board::GameError;
use crate::board::HexGameState;
use crate::core::Turn;
use crate::core::WIN_LENGTH;
use crate::core::{hex_distance, Hex, HEX_DIRECTIONS};
use crate::encoder::WIN_SCORE;
use crate::threats::{generate_threat_turns, threat_status, turn_satisfies_status, ThreatStatus};
use smallvec::SmallVec;

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------

/// Maximum ply depth tracked for killer moves.
const MAX_PLY: usize = 64;

/// Initial aspiration window around the previous iteration's score.
const ASPIRATION_WINDOW: i32 = 500;

/// Maximum quiescence search depth (in turns).
///
/// Quiescence only searches threat moves, so a depth of 6 turns is
/// deep enough to resolve most tactical sequences without exploding
/// the node count.
const QUIESCE_DEPTH: i32 = 6;

/// Maximum candidate cells per node (non-root).
const CANDIDATE_CAP: usize = 12;

/// Maximum candidate cells at root.
const ROOT_CANDIDATE_CAP: usize = 14;

/// Pair-sum cap: only generate pairs (i,j) where i+j <= this value.
///
/// Because candidates are sorted by quality, indices 0..N represent the
/// best cells.  The pair-sum constraint limits pair generation to cells
/// that are both high-quality, dramatically reducing branching factor.
const PAIR_SUM_CAP: usize = 12;

/// Weight for eval delta in move ordering.
const DELTA_WEIGHT: i32 = 15;

/// Tempo bonus for the side to move.
const TEMPO_BONUS: i32 = 15;

/// TT entries cap before clearing to avoid unbounded memory growth.
const TT_MAX_SIZE: usize = 2_000_000;

// -------------------------------------------------------------------------
// Evaluation helper (local to search — not part of public eval API)
// -------------------------------------------------------------------------

/// Minimal O(1) evaluation from `player`'s perspective using incremental state.
///
/// Returns `WIN_SCORE` if `player` has won, `-WIN_SCORE` if they have lost,
/// otherwise the signed incremental score plus a small tempo bonus.
#[inline]
fn evaluate(game: &HexGameState, player: u8) -> i32 {
    if let Some(w) = game.winner() {
        return if w == player { WIN_SCORE } else { -WIN_SCORE };
    }
    let mut score = if player == 0 {
        game.eval().score()
    } else {
        -game.eval().score()
    };
    if game.current_player() == player {
        score += TEMPO_BONUS;
    }
    score
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
    ///
    /// A "killer" is a non-capturing move that caused a beta cutoff at this
    /// ply in a sibling subtree.  It is a cheap but effective proxy for
    /// move quality and is promoted to the front of the move list.
    killers: [Option<Turn>; MAX_PLY],
    /// History heuristic: per-cell score (depth² on cutoff).
    ///
    /// Cells that frequently cause cutoffs accumulate history score, which
    /// boosts their candidate ranking in later nodes.
    history: FxHashMap<Hex, i32>,
    /// Random seed for tiebreaking and non-deterministic play.
    noise_seed: u64,
    /// Noise level for non-deterministic play (0.0 = deterministic).
    ///
    /// Affects candidate ordering to produce varied games for training.
    /// Threat blocking and instant-win detection remain deterministic.
    noise_level: f32,
    /// Reusable buffer for inner-turn generation (avoids per-node allocations).
    scratch_inner: Vec<Turn>,
    /// Reusable buffers for quiescence search (avoids per-node allocations).
    scratch_turns: Vec<Turn>,
    scratch_opp: Vec<Hex>,
    scratch_my: Vec<Hex>,
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
            scratch_inner: Vec::new(),
            scratch_turns: Vec::new(),
            scratch_opp: Vec::new(),
            scratch_my: Vec::new(),
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
        *self.history.entry(t.first()).or_insert(0) += bonus;
        if let Some(m2) = t.second() {
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
///
/// Uses fixed magic constants so that flipping side or phase always
/// changes the hash, preventing collisions across different game states.
#[inline]
fn tt_hash(game: &HexGameState) -> u64 {
    let side = if game.current_player() == 0 {
        0x9e37_79b9_7f4a_7c15u64
    } else {
        0xc2b2_ae3d_27d4_eb4fu64
    };
    let phase = match game.placements_remaining() {
        0 => 0x1656_67b1_9e37_79f9u64,
        1 => 0x27d4_eb2f_1656_67c5u64,
        _ => 0x94d0_49bb_1331_11ebu64,
    };
    game.zobrist() ^ side ^ phase
}

// -------------------------------------------------------------------------
// Make/unmake turns
// -------------------------------------------------------------------------

/// Execute a full turn (1 or 2 placements). Returns `(game_over, placements_made)`.
///
/// Propagates `GameError` so that an illegal move from the move generator
/// surfaces immediately instead of silently corrupting the undo stack.
fn make_turn(game: &mut HexGameState, t: Turn) -> Result<(bool, u8), GameError> {
    let m1 = t.first();
    game.place(m1.q, m1.r)?;
    if game.is_over() {
        return Ok((true, 1));
    }
    if let Some(m2) = t.second() {
        game.place(m2.q, m2.r)?;
        if game.is_over() {
            return Ok((true, 2));
        }
        return Ok((false, 2));
    }
    Ok((false, 1))
}

/// Undo a full turn. `placed` = number of placements that were actually made.
fn unmake_turn(game: &mut HexGameState, placed: u8) {
    for _ in 0..placed {
        game.unplace();
    }
}

// -------------------------------------------------------------------------
// Candidate scoring and turn generation
// -------------------------------------------------------------------------

/// Score a single candidate cell for sorting.
///
/// The score has three components:
/// 1. **Evaluation delta** — how much `hypothetical_score_delta` changes the position.
/// 2. **History bonus** — cells that caused cutoffs in sibling subtrees.
/// 3. **Tactical bonus** — cells in hot windows get huge bonuses so they
///    bubble to the top (+50k for blocking opponent, +40k for completing
///    our own threat).
///
/// When `noise_level > 0`, pseudo-random noise is injected into the
/// non-tactical portion.  This shuffles candidate ordering for varied
/// training games while preserving deterministic threat handling.
#[inline]
fn score_candidate(
    game: &HexGameState,
    cell: Hex,
    history: &FxHashMap<Hex, i32>,
    sign: i32,
    noise_level: f32,
    noise_seed: u64,
) -> i64 {
    let delta = game
        .eval()
        .hypothetical_score_delta(cell, game.current_player()) as i64;
    let hist = history.get(&cell).copied().unwrap_or(0).min(500_000) as i64;

    // ── Tactical bonuses (deterministic, not affected by noise) ──
    let opp = (1 - game.current_player()) as usize;
    let mut tactical = 0i64;

    // Blocking opponent threats: cell lies in an opponent hot window.
    for key in game.eval().hot_windows(opp as u8) {
        let (dq, dr) = HEX_DIRECTIONS[key.dir() as usize];
        for k in 0..WIN_LENGTH {
            if cell.q == key.q() + dq * k && cell.r == key.r() + dr * k {
                tactical += 50_000;
                break;
            }
        }
    }

    // Completing our own threats: cell lies in our hot window.
    let curr = game.current_player() as usize;
    for key in game.eval().hot_windows(curr as u8) {
        let (dq, dr) = HEX_DIRECTIONS[key.dir() as usize];
        for k in 0..WIN_LENGTH {
            if cell.q == key.q() + dq * k && cell.r == key.r() + dr * k {
                tactical += 40_000;
                break;
            }
        }
    }

    let base = delta * sign as i64 * DELTA_WEIGHT as i64 + hist;

    // ── Noise injection (training variety) ──
    let noisy_base = if noise_level > 0.0 {
        // Deterministic-per-cell pseudo-random noise from cell coords + seed.
        let cell_hash = (cell.q as u64).wrapping_mul(2654435761)
            ^ (cell.r as u64).wrapping_mul(40503)
            ^ noise_seed;
        // Map to [-1.0, 1.0] range.
        let rand_frac = ((cell_hash % 10000) as f64 / 5000.0) - 1.0;
        // Scale noise relative to the base score magnitude.
        let noise_mag =
            (base.abs() as f64 * noise_level as f64 * 0.5 + 500.0 * noise_level as f64) * rand_frac;
        base + noise_mag as i64
    } else {
        base
    };

    noisy_base + tactical
}

/// Generate candidate cells sorted by score, then cap to `cap`.
///
/// Uses `candidates_near2()` as the seed set, scores each cell, sorts
/// descending, and truncates.  The cap keeps branching factor manageable.
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

    let sign = if game.current_player() == 0 {
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
///
/// `cands` is assumed to be sorted from best to worst.  Only pairs whose
/// indices satisfy `i + j <= max_pair_sum` are generated.  Because the
/// best cells cluster at low indices, this heuristic keeps the pair set
/// small and high-quality without evaluating every combination.
fn generate_turn_pairs(cands: &[Hex], max_pair_sum: usize) -> Vec<Turn> {
    let n = cands.len();
    let mut turns = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            if i + j <= max_pair_sum {
                turns.push(Turn::pair(cands[i], cands[j]));
            }
        }
    }
    turns
}

// -------------------------------------------------------------------------
// Instant-win detection
// -------------------------------------------------------------------------

/// Check if the current player can win this turn (with remaining placements).
///
/// Uses incrementally-maintained hot_windows for O(hot_set) performance.
/// A 5-window (1 empty) is an instant win regardless of whether 1 or 2
/// placements remain — the game ends as soon as the 6th stone is placed.
/// A 4-window (2 empties) is a win only if we have at least 2 placements.
fn find_instant_win(game: &HexGameState, player: u8) -> Option<Turn> {
    if !game.eval().has_threats(player) {
        return None;
    }
    let remaining = game.placements_remaining() as usize;

    for key in game.eval().hot_windows(player) {
        let (dq, dr) = HEX_DIRECTIONS[key.dir() as usize];
        // Stack-allocated buffer: a length-6 window can have at most 2 empties
        // when fours or fives are present.
        let mut empties: SmallVec<[Hex; 2]> = SmallVec::new();

        for k in 0..WIN_LENGTH {
            let h = Hex::new(key.q() + dq * k, key.r() + dr * k);
            if !game.stones().contains_key(&h) {
                empties.push(h);
            }
        }

        // A full window means the game is already over; skip it.
        // A window needing more empties than placements is unreachable.
        if empties.is_empty() || empties.len() > remaining {
            continue;
        }

        match empties.len() {
            1 => {
                // A single empty in a 5-window wins immediately.
                // Even with 2 placements remaining, Turn::single wins because
                // the game ends after the 6th stone is placed.
                return Some(Turn::single(empties[0]));
            }
            2 if remaining >= 2 => {
                return Some(Turn::pair(empties[0], empties[1]));
            }
            _ => {}
        }
    }
    None
}

// -------------------------------------------------------------------------
// Turn generation
// -------------------------------------------------------------------------

/// Generate root turns with colony candidate and optional noise.
///
/// Opening moves, instant wins, candidate scoring, colony injection,
/// pair generation, and threat filtering are applied in sequence.
fn generate_root_turns(
    game: &HexGameState,
    history: &FxHashMap<Hex, i32>,
    noise_level: f32,
    noise_seed: u64,
) -> Vec<Turn> {
    // Opening: empty board.
    if game.stones().is_empty() {
        return vec![Turn::single(Hex::ORIGIN)];
    }

    // Opening: player 0 only places 1 stone on move 0.
    if game.placements_remaining() == 1 && game.move_count() == 0 {
        return vec![Turn::single(Hex::ORIGIN)];
    }

    // Check for instant wins (always deterministic, highest priority).
    if let Some(win_turn) = find_instant_win(game, game.current_player()) {
        return vec![win_turn];
    }

    let mut cands =
        generate_sorted_candidates(game, history, ROOT_CANDIDATE_CAP, noise_level, noise_seed);

    // ── Colony candidate ──
    // Add a cell far from the stone centroid to encourage multi-front play.
    // This prevents the engine from always clustering stones in one region.
    if cands.len() >= 2 && !game.stones().is_empty() {
        let (sq, sr, n) = game
            .stones()
            .keys()
            .fold((0i64, 0i64, 0u32), |(sq, sr, n), h| {
                (sq + h.q as i64, sr + h.r as i64, n + 1)
            });
        let cq = (sq as f64 / n as f64).round() as i32;
        let cr = (sr as f64 / n as f64).round() as i32;

        let max_r = game
            .stones()
            .keys()
            .map(|h| hex_distance(*h, Hex::new(cq, cr)))
            .max()
            .unwrap_or(0);

        let colony_dist = max_r + 3;
        let dirs: [(i32, i32); 6] = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)];
        let dir_idx = (noise_seed as usize ^ game.zobrist() as usize) % 6;
        let (dq, dr) = dirs[dir_idx];
        let colony = Hex::new(cq + dq * colony_dist, cr + dr * colony_dist);

        // Only add the colony if it is empty and within placement radius
        // of at least one existing stone (otherwise it would be illegal).
        if !game.stones().contains_key(&colony)
            && game
                .stones()
                .keys()
                .any(|&h| hex_distance(h, colony) <= crate::core::PLACEMENT_RADIUS)
            && !cands.contains(&colony)
        {
            cands.push(colony);
        }
    }

    // Handle single-placement turns.
    if game.placements_remaining() == 1 {
        return cands.into_iter().map(Turn::single).collect();
    }

    // Generate pairs with pair-sum constraint at root too.
    let mut turns = generate_turn_pairs(&cands, PAIR_SUM_CAP);

    // ── Threat filter ──
    // Compute threat_status ONCE and retain only legal turns.
    let ts = threat_status(game);
    turns.retain(|t| turn_satisfies_status(&ts, *t));
    turns
}

/// Generate inner (non-root) turns.
///
/// Inner nodes always use deterministic ordering (`noise_level = 0`) to
/// preserve search quality.  Noise is only injected at the root to vary
/// game openings.
///
/// The instant-win check is skipped here because the caller (`alphabeta`)
/// already performs it before move generation, avoiding redundant work.
fn generate_inner_turns(
    game: &HexGameState,
    history: &FxHashMap<Hex, i32>,
    noise_seed: u64,
    ts: &ThreatStatus,
    out: &mut Vec<Turn>,
) {
    out.clear();
    if game.stones().is_empty() {
        out.push(Turn::single(Hex::ORIGIN));
        return;
    }

    // Single-placement turn: no pairs to generate.
    if game.placements_remaining() == 1 {
        let cands = generate_sorted_candidates(game, history, CANDIDATE_CAP, 0.0, noise_seed);
        out.extend(cands.into_iter().map(Turn::single));
        return;
    }

    let cands = generate_sorted_candidates(game, history, CANDIDATE_CAP, 0.0, noise_seed);
    let mut turns = generate_turn_pairs(&cands, PAIR_SUM_CAP);

    // Retain only turns that satisfy the pre-computed threat status.
    turns.retain(|t| turn_satisfies_status(ts, *t));
    out.extend(turns);
}

// -------------------------------------------------------------------------
// Turn ordering
// -------------------------------------------------------------------------

/// Promote TT best and killer moves to the front of the turn list.
///
/// Candidates are already sorted by `score_candidate`, so pairs inherit
/// good ordering.  This function simply moves the two most trusted moves
/// to indices 0 and 1 without changing the relative order of the rest.
fn promote_best_turns(turns: &mut [Turn], tt_best: Option<Turn>, killer: Option<Turn>) {
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
// Mate-distance TT adjustments
// -------------------------------------------------------------------------

/// Adjust a mate score before storing it in the TT.
///
/// "Mate in N" scores are stored as `WIN_SCORE - ply` so that deeper
/// nodes store larger (less negative) values.  When loaded back, the
/// ply offset is subtracted to recover the original distance.
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

/// Adjust a mate score after loading it from the TT.
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

/// Quiescence search: extend the evaluation along tactical lines only.
///
/// Quiescence resolves immediate threats (wins, blocks, and counter-threats)
/// until the position is "quiet" (no fours or fives for either side).  It
/// uses `generate_threat_turns` instead of the full move generator, keeping
/// the node count tiny even at depth 6.
///
/// # Parameters
/// * `alpha`, `beta` — standard alpha-beta bounds.
/// * `qdepth` — remaining quiescence depth (in turns).  Stops at 0.
/// * `ply` — distance from root, used for mate-distance scoring.
/// * `ts_opt` — pre-computed `ThreatStatus` from caller (avoids recomputation).
///   Scratch buffers (`ss.scratch_turns/opp/my`) are reused via `SearchState`.
fn quiesce(
    game: &mut HexGameState,
    ss: &mut SearchState,
    mut alpha: i32,
    beta: i32,
    qdepth: i32,
    ply: usize,
    ts_opt: Option<&ThreatStatus>,
) -> Result<i32, GameError> {
    if ss.aborted {
        return Ok(0);
    }
    ss.nodes += 1;
    if ss.nodes.is_multiple_of(1024) && ss.timed_out() {
        ss.aborted = true;
        return Ok(0);
    }

    let player = game.current_player();

    // Terminal position: return mate score from player's perspective.
    if game.is_over() {
        return Ok(if game.winner() == Some(player) {
            WIN_SCORE - ply as i32
        } else {
            -(WIN_SCORE - ply as i32)
        });
    }

    // Stand-pat: the static evaluation is a lower bound on the score.
    let stand_pat = evaluate(game, player);
    if stand_pat >= beta {
        return Ok(stand_pat);
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    // Max quiescence depth reached.
    if qdepth <= 0 {
        return Ok(alpha);
    }

    // Only extend when threats exist on either side.
    if !game.eval().has_any_threats() {
        return Ok(alpha);
    }

    // Instant win check.
    if let Some(win_turn) = find_instant_win(game, player) {
        let (over, placed) = make_turn(game, win_turn)?;

        let score = if over && game.winner() == Some(player) {
            WIN_SCORE - ply as i32
        } else {
            // If the game ended unexpectedly, the recursive quiesce call
            // will immediately see game.is_over() and return the correct
            // terminal score.
            -quiesce(game, ss, -beta, -alpha, qdepth - 1, ply + 1, None)?
        };

        unmake_turn(game, placed);

        return Ok(score);
    }

    // Unblockable opponent win: we cannot stop all threats.
    let ts = ts_opt.cloned().unwrap_or_else(|| threat_status(game));
    if matches!(ts, ThreatStatus::Unblockable) {
        return Ok(-(WIN_SCORE - ply as i32 - 1));
    }

    ss.scratch_turns.clear();
    generate_threat_turns(
        game,
        &mut ss.scratch_turns,
        &mut ss.scratch_opp,
        &mut ss.scratch_my,
    );
    if ss.scratch_turns.is_empty() {
        return Ok(alpha);
    }

    let turns: SmallVec<[Turn; 16]> = ss.scratch_turns.iter().copied().collect();

    for &t in &turns {
        let (over, placed) = make_turn(game, t)?;
        let score = if over {
            if game.winner() == Some(player) {
                WIN_SCORE - ply as i32
            } else {
                -(WIN_SCORE - ply as i32)
            }
        } else {
            -quiesce(game, ss, -beta, -alpha, qdepth - 1, ply + 1, None)?
        };
        unmake_turn(game, placed);

        if ss.aborted {
            return Ok(0);
        }

        if score >= beta {
            return Ok(score);
        }
        if score > alpha {
            alpha = score;
        }
    }

    Ok(alpha)
}

// -------------------------------------------------------------------------
// Main alpha-beta search (turn-based)
// -------------------------------------------------------------------------

/// Recursive alpha-beta search with PVS and LMR.
///
/// # Pruning inside this function
/// - Instant-win return before move generation.
/// - Unblockable-loss return before move generation.
/// - Reverse futility pruning at shallow depth (`depth <= 2`).
/// - Late-move pruning: skip moves indexed ≥20 at low depth.
/// - PVS: full window for first move, null window for rest.
/// - LMR: reduce depth by 1 or 2 for moves indexed ≥2; re-search if they
///   beat alpha.
fn alphabeta(
    game: &mut HexGameState,
    ss: &mut SearchState,
    depth: i32,
    ply: usize,
    mut alpha: i32,
    beta: i32,
) -> Result<i32, GameError> {
    if ss.aborted {
        return Ok(0);
    }
    if ss.nodes.is_multiple_of(1024) && ss.timed_out() {
        ss.aborted = true;
        return Ok(0);
    }
    ss.nodes += 1;

    let player = game.current_player();

    // Terminal check.
    if game.is_over() {
        return Ok(if game.winner() == Some(player) {
            WIN_SCORE - ply as i32
        } else {
            -(WIN_SCORE - ply as i32)
        });
    }

    // Compute threat status once for quiescence, unblockable check, and move generation.
    let ts = threat_status(game);

    // Leaf: drop into quiescence search.
    if depth <= 0 {
        return quiesce(game, ss, alpha, beta, QUIESCE_DEPTH, ply, Some(&ts));
    }

    // Instant win check: if we can win right now, return immediately.
    if let Some(win_turn) = find_instant_win(game, player) {
        let (over, placed) = make_turn(game, win_turn)?;
        let score = if over && game.winner() == Some(player) {
            WIN_SCORE - ply as i32
        } else {
            -alphabeta(game, ss, depth - 1, ply + 1, -beta, -alpha)?
        };
        unmake_turn(game, placed);
        return Ok(score);
    }

    // Unblockable opponent win check.
    if matches!(ts, ThreatStatus::Unblockable) {
        return Ok(-(WIN_SCORE - ply as i32 - 1));
    }

    // Reverse futility pruning.
    // If the static eval is so good that even a large margin cannot bring
    // the opponent back to beta, return early.
    if depth <= 2 && ply > 0 {
        let margin = 2000 * depth;
        let static_eval = evaluate(game, player);
        if static_eval - margin >= beta {
            return Ok(static_eval);
        }
    }

    // ── TT probe ──
    let hash = tt_hash(game);
    let tt_entry = ss.tt.get(&hash).copied();
    let mut tt_best_turn = None;
    if let Some(entry) = tt_entry {
        if entry.depth >= depth {
            let adj_score = adjust_mate_load(entry.score, ply);
            match entry.flag {
                TTFlag::Exact => return Ok(adj_score),
                TTFlag::LowerBound => {
                    if adj_score >= beta {
                        return Ok(adj_score);
                    }
                    if adj_score > alpha {
                        alpha = adj_score;
                    }
                }
                TTFlag::UpperBound => {
                    if adj_score <= alpha {
                        return Ok(adj_score);
                    }
                }
            }
        }
        tt_best_turn = entry.best_turn;
    }

    // Generate turns (inner nodes always deterministic).
    generate_inner_turns(game, &ss.history, ss.noise_seed, &ts, &mut ss.scratch_inner);
    let mut turns = std::mem::take(&mut ss.scratch_inner);
    if turns.is_empty() {
        return Ok(evaluate(game, player));
    }

    let ply_idx = ply.min(MAX_PLY - 1);
    promote_best_turns(&mut turns, tt_best_turn, ss.killers[ply_idx]);

    let mut best_score = i32::MIN + 1;
    let mut best_turn = turns[0];
    let orig_alpha = alpha;

    for (move_idx, t) in turns.iter().enumerate() {
        let (over, placed) = make_turn(game, *t)?;

        let score = if over {
            if game.winner() == Some(player) {
                WIN_SCORE - ply as i32
            } else if game.winner() == Some(1 - player) {
                -(WIN_SCORE - ply as i32)
            } else {
                0
            }
        } else {
            // Late move pruning: at low depth, skip moves late in order
            // when the position is not in a forced-mate situation.
            if depth <= 2 && move_idx >= 20 && best_score > -(WIN_SCORE - 100) {
                unmake_turn(game, placed);
                continue;
            }

            // ── PVS + LMR ──
            // Aggressive reduction: start reducing at move index 2.
            let do_lmr = depth >= 2 && move_idx >= 2 && best_score > -(WIN_SCORE - 100);
            let lmr_reduction = if move_idx >= 10 { 2 } else { 1 };

            if move_idx == 0 {
                // Full-window search for the first (presumably best) move.
                -alphabeta(game, ss, depth - 1, ply + 1, -beta, -alpha)?
            } else if do_lmr {
                // Reduced-depth null-window search.
                let reduced = -alphabeta(
                    game,
                    ss,
                    depth - 1 - lmr_reduction,
                    ply + 1,
                    -(alpha + 1),
                    -alpha,
                )?;
                if reduced > alpha && !ss.aborted {
                    // Re-search with null window at full depth.
                    let nw = -alphabeta(game, ss, depth - 1, ply + 1, -(alpha + 1), -alpha)?;
                    if nw > alpha && nw < beta && !ss.aborted {
                        // Full-window re-search.
                        -alphabeta(game, ss, depth - 1, ply + 1, -beta, -alpha)?
                    } else {
                        nw
                    }
                } else {
                    reduced
                }
            } else {
                // Null-window search without reduction.
                let nw = -alphabeta(game, ss, depth - 1, ply + 1, -(alpha + 1), -alpha)?;
                if nw > alpha && nw < beta && !ss.aborted {
                    -alphabeta(game, ss, depth - 1, ply + 1, -beta, -alpha)?
                } else {
                    nw
                }
            }
        };

        unmake_turn(game, placed);
        if ss.aborted {
            return Ok(0);
        }

        if score > best_score {
            best_score = score;
            best_turn = *t;
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            // Update killer and history on non-mate cutoffs.
            if best_score.abs() < WIN_SCORE - 100 {
                ss.update_killers(ply_idx, *t);
                ss.update_history(*t, depth);
            }
            break;
        }
    }

    // ── TT store with mate-distance adjustment ──
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

    ss.scratch_inner = turns;
    ss.scratch_inner.clear();

    Ok(best_score)
}

// -------------------------------------------------------------------------
// Root search with full move info
// -------------------------------------------------------------------------

type RootSearchResult = (Turn, i32, Vec<(Turn, i32)>);

/// Search all root turns and return the best one, its score, and per-turn scores.
///
/// Uses PVS at the root: the first move gets a full window, subsequent moves
/// get a null window with re-search on failure.
fn search_root(
    game: &mut HexGameState,
    ss: &mut SearchState,
    depth: i32,
    root_turns: &[Turn],
    init_alpha: i32,
    init_beta: i32,
) -> Result<RootSearchResult, GameError> {
    let player = game.current_player();
    let mut alpha = init_alpha;
    let beta = init_beta;
    let mut best_turn = root_turns[0];
    let mut best_score = i32::MIN + 1;
    let mut scores: Vec<(Turn, i32)> = Vec::with_capacity(root_turns.len());

    for (move_idx, t) in root_turns.iter().enumerate() {
        let (over, placed) = make_turn(game, *t)?;

        let score = if over {
            if game.winner() == Some(player) {
                WIN_SCORE
            } else {
                -WIN_SCORE
            }
        } else {
            // PVS at root.
            if move_idx == 0 {
                -alphabeta(game, ss, depth - 1, 1, -beta, -alpha)?
            } else {
                let nw = -alphabeta(game, ss, depth - 1, 1, -(alpha + 1), -alpha)?;
                if nw > alpha && nw < beta && !ss.aborted {
                    -alphabeta(game, ss, depth - 1, 1, -beta, -alpha)?
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

    Ok((best_turn, best_score, scores))
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
    /// Root candidates for temperature-based sampling in self-play.
    /// Only populated when `collect_candidates` is true.
    pub root_candidates: Vec<(Hex, i32)>,
}

/// Run iterative-deepening turn-based alpha-beta search.
///
/// `time_limit` caps the total search time.  The engine loops over depths
/// 1..max_depth, reusing the TT between iterations.  After depth 3,
/// aspiration windows (`best_score ± 500`) are used; if the score falls
/// outside the window, a re-search with a full window is performed.
///
/// `noise_level`: 0.0 = deterministic (strongest play), >0.0 = inject noise
/// into root candidate ordering for varied training data. Threat blocking
/// and instant-win detection remain deterministic regardless of noise level.
///
/// `collect_candidates`: if true, populate `root_candidates` with the top
/// 12 cells and their eval deltas for NN training compatibility.
pub fn iterative_deepening(
    game: &HexGameState,
    time_limit: Duration,
    max_depth: i32,
    _near_radius: i32, // kept for API compat, always uses radius-2 internally
    collect_candidates: bool,
    noise_level: f32,
) -> Result<SearchResult, GameError> {
    let mut ss = SearchState::new(noise_level);
    ss.deadline = Some(Instant::now() + time_limit);
    ss.maybe_clear_tt();

    // Handle opening move.
    if game.stones().is_empty() {
        return Ok(SearchResult {
            best_turn: Turn::single(Hex::ORIGIN),
            best_move: Hex::ORIGIN,
            score: 0,
            depth_reached: 0,
            nodes: 0,
            root_candidates: vec![(Hex::ORIGIN, 0)],
        });
    }

    // Generate root turns (using immutable game).
    let mut root_turns = generate_root_turns(game, &ss.history, ss.noise_level, ss.noise_seed);
    if root_turns.is_empty() {
        let c = game.candidates_near2();
        if c.len() >= 2 && game.placements_remaining() >= 2 {
            root_turns.push(Turn::pair(c[0], c[1]));
        } else if !c.is_empty() {
            root_turns.push(Turn::single(c[0]));
        } else {
            return Ok(SearchResult {
                best_turn: Turn::single(Hex::ORIGIN),
                best_move: Hex::ORIGIN,
                score: 0,
                depth_reached: 0,
                nodes: 0,
                root_candidates: vec![],
            });
        }
    }

    // Clone game ONCE for search (search_root uses make/unmake instead of cloning per move).
    let mut search_game = game.clone();

    let mut best_turn = root_turns[0];
    let mut best_score = 0i32;
    let mut depth_reached = 0;

    for depth in 1..=max_depth {
        ss.aborted = false;

        // Root search at this depth with aspiration windows.
        let (bt, score, scores) = if depth >= 3 && best_score.abs() < WIN_SCORE - 100 {
            let lo = best_score - ASPIRATION_WINDOW;
            let hi = best_score + ASPIRATION_WINDOW;

            let (bt, s, scores) =
                search_root(&mut search_game, &mut ss, depth, &root_turns, lo, hi)?;
            if ss.aborted {
                break;
            }

            if s <= lo || s >= hi {
                // Re-search with full window.
                ss.aborted = false;
                search_root(
                    &mut search_game,
                    &mut ss,
                    depth,
                    &root_turns,
                    i32::MIN + 1,
                    i32::MAX - 1,
                )?
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
            )?
        };

        if ss.aborted {
            break;
        }

        best_turn = bt;
        best_score = score;
        depth_reached = depth;

        // Re-sort root turns for next iteration (best-first for PVS).
        // Linear scan is cheaper than a HashMap for the small root move set.
        if !scores.is_empty() {
            root_turns.sort_by_key(|t| {
                Reverse(
                    scores
                        .iter()
                        .find_map(|(turn, s)| if turn == t { Some(*s) } else { None })
                        .unwrap_or(i32::MIN),
                )
            });
        }

        // Early exit on forced win/loss.
        if score.abs() >= WIN_SCORE - 100 {
            break;
        }
    }

    // Build root_candidates for NN training compatibility (always deterministic).
    let root_candidates = if collect_candidates {
        let cands = generate_sorted_candidates(game, &ss.history, 12, 0.0, ss.noise_seed);
        cands
            .iter()
            .map(|&m| {
                let sign = if game.current_player() == 0 { 1 } else { -1 };
                let score = game
                    .eval()
                    .hypothetical_score_delta(m, game.current_player())
                    * sign;
                (m, score)
            })
            .collect()
    } else {
        vec![]
    };

    Ok(SearchResult {
        best_turn,
        best_move: best_turn.first(),
        score: best_score,
        depth_reached,
        nodes: ss.nodes,
        root_candidates,
    })
}
