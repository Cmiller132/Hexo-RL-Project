//! Unified neural-network board encoder for Infinity Hex.
//!
//! Infinity Hex is a variant of Hex played on an infinite hexagonal grid where
//! the win condition is six stones in a straight line. Each turn (except the
//! opening move) consists of **two** stone placements. This module converts a
//! [`HexGameState`] into a fixed-size 13-channel 33×33 float32 tensor that
//! feeds the neural-network policy and value heads.
//!
//! Both the MCTS search tree and the Python training pipeline call into here,
//! eliminating the previous duplication between `pybridge.rs` and `mcts.rs`.

use std::sync::OnceLock;

use crate::board::HexGameState;
use crate::core::{hex_distance, Hex, HEX_DIRECTIONS, WIN_LENGTH};
use crate::threats::{live_cells, threat_status, ThreatStatus};

// ── Pre-computed channel ─────────────────────────────────────────────────

static CENTROID_DIST_CHANNEL: OnceLock<[f32; BOARD_AREA]> = OnceLock::new();

fn centroid_dist_channel() -> &'static [f32; BOARD_AREA] {
    CENTROID_DIST_CHANNEL.get_or_init(|| {
        let center = Hex::new(HALF_BOARD, HALF_BOARD);
        let mut buf = [0.0f32; BOARD_AREA];
        for gi in 0..BOARD_SIZE {
            for gj in 0..BOARD_SIZE {
                let h = Hex::new(gi, gj);
                buf[(gi * BOARD_SIZE + gj) as usize] =
                    hex_distance(h, center) as f32 / HALF_BOARD as f32;
            }
        }
        buf
    })
}

// ── Constants ───────────────────────────────────────────────────────────

/// Width and height of the square tensor used for NN input.
///
/// The infinite board is cropped to a 33×33 window centered on the board's
/// centroid (banker's-rounded mean of all occupied cells). Stones or legal
/// moves that fall outside this window are clipped.
pub const BOARD_SIZE: i32 = 33;

/// Half of [`BOARD_SIZE`], i.e. the centre coordinate of the tensor.
pub const HALF_BOARD: i32 = 16; // BOARD_SIZE / 2

/// Number of feature channels in the encoded tensor.
pub const NUM_CHANNELS: usize = 13;

/// Total number of spatial elements in one channel (`BOARD_SIZE * BOARD_SIZE`).
pub const BOARD_AREA: usize = (BOARD_SIZE * BOARD_SIZE) as usize; // 1089

/// Total flat tensor length (`NUM_CHANNELS * BOARD_AREA`).
pub const TENSOR_SIZE: usize = NUM_CHANNELS * BOARD_AREA; // 14157

// ── Types ───────────────────────────────────────────────────────────────

/// Result of encoding a board for neural-network input.
pub struct EncodedBoard {
    /// Flat f32 tensor of shape (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE).
    /// Layout: [ch0_plane, ch1_plane, ..., ch12_plane] where each plane is row-major.
    pub tensor: Vec<f32>,
    /// Spatial offset: the board coordinate that maps to tensor index (0, 0).
    ///
    /// To convert board `(q, r)` to tensor `(gi, gj)`:
    /// `gi = q - offset_q`, `gj = r - offset_r`.
    pub offset_q: i32,
    pub offset_r: i32,
    /// Legal moves used for channel 3 (same set that the NN policy head should predict).
    legal_moves: Vec<Hex>,
}

impl EncodedBoard {
    /// Legal moves at the encoded position.
    pub fn legal_moves(&self) -> &[Hex] {
        &self.legal_moves
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Python-compatible "banker's rounding" (round half to even).
///
/// The centroid of all occupied cells is computed, then rounded with this
/// function to match Python's built-in `round()` behaviour exactly. This
/// guarantees that the Rust encoder and any Python data-preprocessing scripts
/// produce bitwise-identical offsets.
pub(crate) fn bankers_round(v: f64) -> i32 {
    let frac = v - v.floor();
    if (frac - 0.5).abs() < 1e-9 {
        let lo = v.floor() as i32;
        let hi = v.ceil() as i32;
        if lo % 2 == 0 {
            lo
        } else {
            hi
        }
    } else {
        v.round() as i32
    }
}

// ── Encoding ────────────────────────────────────────────────────────────

/// Encode the full board state as an NN input tensor.
///
/// # Steps
/// 1. Compute board centroid (mean of all occupied cells), banker's-rounded.
/// 2. Compute spatial offset so the centroid lands at tensor centre `(16, 16)`.
/// 3. Gather legal moves (with optional threat constraint).
/// 4. Fill 13 channels (see [`encode_board_into`] for a detailed breakdown).
///
/// Returns [`EncodedBoard`] with tensor, offsets, and legal moves.
pub fn encode_board(
    game: &HexGameState,
    near_radius: i32,
    constrain_threats: bool,
) -> EncodedBoard {
    let mut tensor = vec![0.0f32; TENSOR_SIZE];
    let mut hot_buf = Vec::new();
    let mut legal_moves = Vec::new();
    let (offset_q, offset_r) = encode_board_into(
        game,
        near_radius,
        constrain_threats,
        &mut tensor,
        &mut hot_buf,
        &mut legal_moves,
    );
    EncodedBoard {
        tensor,
        offset_q,
        offset_r,
        legal_moves,
    }
}

/// Encode into a pre-allocated buffer. Returns `(offset_q, offset_r)`.
///
/// The `out` buffer must be at least [`TENSOR_SIZE`] elements. This variant is used
/// by MCTS to avoid repeated tensor allocations during tree search.
/// The `legal_out` vector is provided by the caller and will be overwritten with
/// the set of legal (or threat-constrained) moves.
///
/// # Channel layout (13 channels)
///
/// | Ch | Name | Description |
/// |---|---|---|
/// | 0 | **Own stones** | `1.0` on every cell occupied by the current player. |
/// | 1 | **Opponent stones** | `1.0` on every cell occupied by the opponent. |
/// | 2 | **Empty mask** | `1.0 - ch0 - ch1`. Marks every unoccupied cell. |
/// | 3 | **Legal moves** | `1.0` on each legal move inside the 33×33 window. When `constrain_threats` is `true`, only threat-constrained moves are marked (e.g. must-block cells or winning-turn cells). |
/// | 4 | **Turn phase** | All `1.0` when the current turn is on its **second** placement (`placements_remaining == 1 && move_count > 0`), otherwise all `0.0`. Tells the net whether one or two stones remain to be placed this turn. |
/// | 5 | **First stone of turn** | `1.0` on the cell of the most recent move in history (the first placement of the current turn). Only active during phase 2 (see ch 4). |
/// | 6 | **Player colour** | All `1.0` if the current player is player 0, all `0.0` if player 1. |
/// | 7 | **Own recency** | `1 / (1 + plies_ago)` for each own stone, decaying from most-recent to oldest. |
/// | 8 | **Opponent recency** | Same as ch 7 but for opponent stones. |
/// | 9 | **Opponent hot cells** | Empty cells that lie inside the opponent's "hot windows" (windows with 4+ opponent stones and 0 own stones). These are the cells the opponent could use to extend a threat. Computed via [`live_cells`] to share logic with the tactical engine. |
/// | 10 | **Own hot cells** | Empty cells inside the current player's hot windows. Same semantics as ch 9 but from the current player's perspective. |
/// | 11 | **Distance from centre** | Normalised hex distance from the board centroid: `hex_dist(cell, centre) / HALF_BOARD`. Values are in `[0, 1]`. |
/// | 12 | **Opponent's last turn** | Marks the cells placed by the opponent during their most recently completed turn (1 or 2 cells). |
///
/// Channels 9 and 10 are populated by calling [`live_cells`] with a single
/// reusable buffer. The caller should own and reuse a `Vec<Hex>` to amortize
/// allocations across encode calls. The `legal_out` vector is owned by the
/// caller; its previous contents are cleared and replaced on each call.
pub fn encode_board_into(
    game: &HexGameState,
    near_radius: i32,
    constrain_threats: bool,
    out: &mut [f32],
    hot_buf: &mut Vec<Hex>,
    legal_out: &mut Vec<Hex>,
) -> (i32, i32) {
    debug_assert!(
        out.len() >= TENSOR_SIZE,
        "encode_board_into: buffer too small ({} < {})",
        out.len(),
        TENSOR_SIZE
    );

    let board = game.stones();

    // ── Compute centroid and offsets ──
    // Python's round() uses banker's rounding (round half to even).
    let (offset_q, offset_r) = if board.is_empty() {
        (-HALF_BOARD, -HALF_BOARD)
    } else {
        let n = board.len() as f64;
        let (mut sq, mut sr) = (0i64, 0i64);
        for &h in board.keys() {
            sq += h.q as i64;
            sr += h.r as i64;
        }
        let cq = bankers_round(sq as f64 / n);
        let cr = bankers_round(sr as f64 / n);
        (cq - HALF_BOARD, cr - HALF_BOARD)
    };

    let current = game.current_player();
    let mc = game.move_count();
    let pr = game.placements_remaining();
    let is_phase_2 = pr == 1 && mc > 0;

    out[..TENSOR_SIZE].fill(0.0);

    // Helper: index into flat tensor [ch, gi, gj]
    #[inline(always)]
    fn idx(ch: usize, gi: i32, gj: i32) -> usize {
        ch * BOARD_AREA + (gi as usize) * (BOARD_SIZE as usize) + gj as usize
    }

    // ── Channels 0-1: player stones ──
    for (&h, &player) in board.iter() {
        let gi = h.q - offset_q;
        let gj = h.r - offset_r;
        if (0..BOARD_SIZE).contains(&gi) && (0..BOARD_SIZE).contains(&gj) {
            if player == current {
                out[idx(0, gi, gj)] = 1.0;
            } else {
                out[idx(1, gi, gj)] = 1.0;
            }
        }
    }

    // ── Channels 7-8: stone recency ──
    // `1/(1 + plies_ago)` for each stone in the move history, split by
    // whether the stone belongs to the current player (ch7) or opponent (ch8).
    for (ply_idx, rec) in game.move_history().iter().enumerate() {
        let gi = rec.cell.q - offset_q;
        let gj = rec.cell.r - offset_r;
        if (0..BOARD_SIZE).contains(&gi) && (0..BOARD_SIZE).contains(&gj) {
            let plies_ago = mc - ply_idx as u32;
            let recency = 1.0 / (1.0 + plies_ago as f32);
            let ch = if rec.player == current { 7 } else { 8 };
            out[idx(ch, gi, gj)] = recency;
        }
    }

    // ── Channel 2: empty cells mask ──
    // After ch0 and ch1 are filled, ch2 is simply 1.0 minus their sum.
    let ch2_start = 2 * BOARD_AREA;
    for i in 0..BOARD_AREA {
        out[ch2_start + i] = 1.0 - out[i] - out[BOARD_AREA + i];
    }

    // ── Channel 3: legal moves mask ──
    legal_out.clear();
    legal_out.extend(game.legal_moves_near(near_radius));
    if constrain_threats {
        let maybe_constrained = match threat_status(game) {
            ThreatStatus::Quiet | ThreatStatus::Unblockable => None,
            ThreatStatus::WinningTurn(t) => {
                let mut allowed = vec![t.first()];
                if let Some(s) = t.second() {
                    allowed.push(s);
                }
                Some(
                    legal_out
                        .iter()
                        .copied()
                        .filter(|h| allowed.contains(h))
                        .collect::<Vec<_>>(),
                )
            }
            ThreatStatus::MustBlock(b) => Some(
                legal_out
                    .iter()
                    .copied()
                    .filter(|h| b.cells().contains(h))
                    .collect::<Vec<_>>(),
            ),
        };
        if let Some(constrained) = maybe_constrained {
            if !constrained.is_empty() {
                *legal_out = constrained;
            }
        }
    }
    for h in legal_out.iter() {
        let gi = h.q - offset_q;
        let gj = h.r - offset_r;
        if (0..BOARD_SIZE).contains(&gi) && (0..BOARD_SIZE).contains(&gj) {
            out[idx(3, gi, gj)] = 1.0;
        }
    }

    // ── Channel 4: turn phase ──
    // All 1.0 when the current turn is on its second placement (phase 2).
    if is_phase_2 {
        let start = 4 * BOARD_AREA;
        out[start..start + BOARD_AREA].fill(1.0);
    }

    // ── Channel 5: first stone of current turn (phase 2 only) ──
    // Marks the cell of the most recent move in history, which is the first
    // placement of the current turn when `is_phase_2` is true.
    if is_phase_2 {
        if let Some(last) = game.move_history().last() {
            let gi = last.cell.q - offset_q;
            let gj = last.cell.r - offset_r;
            if (0..BOARD_SIZE).contains(&gi) && (0..BOARD_SIZE).contains(&gj) {
                out[idx(5, gi, gj)] = 1.0;
            }
        }
    }

    // ── Channel 6: current player colour ──
    // All 1.0 if the current player is player 0, all 0.0 if player 1.
    if current == 0 {
        let start = 6 * BOARD_AREA;
        out[start..start + BOARD_AREA].fill(1.0);
    }

    // ── Channel 11: distance from centroid (pre-computed) ──
    {
        let ch11_start = 11 * BOARD_AREA as usize;
        out[ch11_start..ch11_start + BOARD_AREA as usize]
            .copy_from_slice(centroid_dist_channel());
    }

    // ── Channel 12: opponent's most recent completed turn ──
    // Marks the cells placed by the opponent during their last full turn.
    // For player 0's opening turn this is a single cell; otherwise two cells.
    for h in game.opponent_last_turn_cells() {
        let gi = h.q - offset_q;
        let gj = h.r - offset_r;
        if (0..BOARD_SIZE).contains(&gi) && (0..BOARD_SIZE).contains(&gj) {
            out[idx(12, gi, gj)] = 1.0;
        }
    }

    // ── Channels 9-10: hot cells ──
    //
    // Ch 9: empty cells that lie inside the opponent's "hot windows"
    //     (windows with 4+ opponent stones and 0 own stones).
    // Ch 10: empty cells that lie inside the current player's hot windows.
    //
    // Both channels are filled by calling [`live_cells`] with a caller-owned
    // reusable buffer (`hot_buf`) to avoid per-call allocation.
    // Channel 10 (own live cells)
    hot_buf.clear();
    live_cells(game, current, hot_buf);
    for h in hot_buf.iter() {
        let gi = h.q - offset_q;
        let gj = h.r - offset_r;
        if (0..BOARD_SIZE).contains(&gi) && (0..BOARD_SIZE).contains(&gj) {
            out[idx(10, gi, gj)] = 1.0;
        }
    }

    // Channel 9 (opponent live cells)
    let opp = 1 - current;
    hot_buf.clear();
    live_cells(game, opp, hot_buf);
    for h in hot_buf.iter() {
        let gi = h.q - offset_q;
        let gj = h.r - offset_r;
        if (0..BOARD_SIZE).contains(&gi) && (0..BOARD_SIZE).contains(&gj) {
            out[idx(9, gi, gj)] = 1.0;
        }
    }

    (offset_q, offset_r)
}

// -------------------------------------------------------------------------
// Classical feature extraction (moved from eval/mod.rs to resolve layer
// hierarchy violation — encoder may depend on board, eval must not).
// -------------------------------------------------------------------------

/// Total number of scalar features emitted by [`extract_features`].
///
/// The vector is structured as:
/// ```text
/// [P0_live5, P0_dead5, P0_live4, P0_dead4, P0_live3, P0_live2,
///  P1_live5, P1_dead5, P1_live4, P1_dead4, P1_live3, P1_live2,
///  tempo]
/// ```
/// where `tempo` is `+1.0` when P0 is to move and `-1.0` when P1 is to move.
pub const FEATURE_COUNT: usize = 13;

/// Static evaluation bonus for an immediate win.
pub const WIN_SCORE: i32 = 1_000_000;

const FEATURES_PER_PLAYER: usize = 6;
const LIVE5: usize = 0;
const DEAD5: usize = 1;
const LIVE4: usize = 2;
const DEAD4: usize = 3;
const LIVE3: usize = 4;
const LIVE2: usize = 5;

// -------------------------------------------------------------------------
// Run counting
// -------------------------------------------------------------------------

/// Count contiguous stones of `player` starting from `start` along `(dq, dr)`.
///
/// # Arguments
/// * `game`   — the board state.
/// * `start`  — the first cell of the run (already known to hold `player`).
/// * `dq`     — q-step per cell along the line.
/// * `dr`     — r-step per cell along the line.
/// * `player` — the player whose stones we are counting.
///
/// # Returns
/// A tuple `(count, open_end)` where:
/// * `count`    — number of *additional* consecutive `player` stones after
///   `start` (not counting `start` itself).
/// * `open_end` — `true` if the run ended because the next cell is empty;
///   `false` if it ended because the next cell holds an opponent stone.
#[inline]
fn count_run(game: &HexGameState, start: Hex, dq: i32, dr: i32, player: u8) -> (i32, bool) {
    let mut count = 0;
    let mut q = start.q + dq;
    let mut r = start.r + dr;
    loop {
        let h = Hex::new(q, r);
        match game.stones().get(&h) {
            Some(&p) if p == player => count += 1,
            Some(_) => return (count, false), // blocked by opponent
            None => return (count, true),     // open end
        }
        q += dq;
        r += dr;
    }
}

// -------------------------------------------------------------------------
// Feature extraction
// -------------------------------------------------------------------------

/// Extract a 13-dimensional classical feature vector from the board.
///
/// The features count "runs" of contiguous stones for each player along all
/// three principal hex directions.  A run is only counted if the cell
/// immediately before its start is **not** the same player's stone; this
/// prevents double-counting the same line segment.
///
/// # Feature semantics
///
/// | Index | Name    | Meaning                                           |
/// |-------|---------|---------------------------------------------------|
/// | 0     | P0 live5| P0 has ≥5 in a row with at least one open end     |
/// | 1     | P0 dead5| P0 has exactly 5 in a row with zero open ends     |
/// | 2     | P0 live4| P0 has exactly 4 in a row with **two** open ends  |
/// | 3     | P0 dead4| P0 has exactly 4 in a row with **one** open end   |
/// | 4     | P0 live3| P0 has exactly 3 in a row with two open ends      |
/// | 5     | P0 live2| P0 has exactly 2 in a row with two open ends      |
/// | 6–11  | P1 …    | Same six features mirrored for player 1           |
/// | 12    | tempo   | `+1.0` if P0 to move, `-1.0` otherwise            |
///
/// "Open end" means the adjacent cell past the run is empty.  A run of 6
/// or more stones is treated as a `live5` and multiplied by 10 to emphasise
/// its decisiveness.
pub fn extract_features(game: &HexGameState) -> [f32; FEATURE_COUNT] {
    let mut feats = [0.0f32; FEATURE_COUNT];
    // counts[p][i] accumulates the raw integer counts for player p, feature i.
    let mut counts = [[0i32; FEATURES_PER_PLAYER]; 2];

    // Iterate over every stone on the board.
    for (&cell, &player) in game.stones() {
        let p = player as usize;
        for &(dq, dr) in &HEX_DIRECTIONS {
            // Only start a run if the previous cell in this direction is
            // NOT the same player's stone.  This guarantees each maximal
            // contiguous run is counted exactly once.
            let prev = Hex::new(cell.q - dq, cell.r - dr);
            if game.stones().get(&prev) == Some(&player) {
                continue;
            }

            // Count how far the run extends forward from `cell`.
            let (fwd, fwd_open) = count_run(game, cell, dq, dr, player);
            let run_len = 1 + fwd; // include `cell` itself

            // Determine whether the cell just before `cell` is open.
            let bwd_open = game.stones().get(&prev).is_none();

            let open_ends = (bwd_open as i32) + (fwd_open as i32);

            // Classify the run into one of the threat categories.
            if run_len >= WIN_LENGTH {
                counts[p][LIVE5] += 10;
            } else if run_len == 5 {
                if open_ends >= 1 {
                    counts[p][LIVE5] += 1;
                } else {
                    counts[p][DEAD5] += 1;
                }
            } else if run_len == 4 {
                if open_ends == 2 {
                    counts[p][LIVE4] += 1;
                } else if open_ends == 1 {
                    counts[p][DEAD4] += 1;
                }
            } else if run_len == 3 {
                if open_ends == 2 {
                    counts[p][LIVE3] += 1;
                }
            } else if run_len == 2 && open_ends == 2 {
                counts[p][LIVE2] += 1;
            }
        }
    }

    // Copy the integer counts into the float feature vector.
    for p in 0..2 {
        for i in 0..FEATURES_PER_PLAYER {
            feats[p * FEATURES_PER_PLAYER + i] = counts[p][i] as f32;
        }
    }

    // Final feature: tempo (whose turn it is).
    feats[FEATURE_COUNT - 1] = if game.current_player() == 0 {
        1.0
    } else {
        -1.0
    };
    feats
}
