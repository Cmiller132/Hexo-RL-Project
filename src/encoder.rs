//! Unified neural network board encoder for a hexagonal tic-tac-toe engine.
//!
//! This module provides the single canonical implementation for encoding a
//! [`HexGameState`] into a 13-channel 33×33 float32 tensor. Both the MCTS
//! search and the Python training pipeline call into here, eliminating the
//! previous duplication between `pybridge.rs` and `mcts.rs`.

use crate::board::HexGameState;
use crate::core::{hex_distance, Hex, HEX_DIRECTIONS};

// ── Constants ───────────────────────────────────────────────────────────

/// Width and height of the square tensor used for NN input.
pub const BOARD_SIZE: i32 = 33;

/// Half of [`BOARD_SIZE`], i.e. the center coordinate of the tensor.
pub const HALF_BOARD: i32 = 16; // BOARD_SIZE / 2

/// Number of feature channels in the encoded tensor.
pub const NUM_CHANNELS: usize = 13;

/// Total number of spatial elements in one channel (`BOARD_SIZE * BOARD_SIZE`).
pub const BOARD_AREA: usize = (BOARD_SIZE * BOARD_SIZE) as usize; // 1089

/// Total flat tensor length (`NUM_CHANNELS * BOARD_AREA`).
pub const TENSOR_SIZE: usize = NUM_CHANNELS * BOARD_AREA; // 14157

// ── Types ───────────────────────────────────────────────────────────────

/// Result of encoding a board for neural network input.
pub struct EncodedBoard {
    /// Flat f32 tensor of shape (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE).
    /// Layout: [ch0_plane, ch1_plane, ..., ch12_plane] where each plane is row-major.
    pub tensor: Vec<f32>,
    /// Spatial offset: the board coordinate that maps to tensor index (0, 0).
    /// To convert board (q, r) to tensor (gi, gj): gi = q - offset_q, gj = r - offset_r.
    pub offset_q: i32,
    pub offset_r: i32,
    /// Legal moves used for channel 3 (same as what the NN policy head should predict).
    pub legal_moves: Vec<Hex>,
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Python-compatible "banker's rounding" (round half to even).
///
/// The centroid of all occupied cells is computed, then rounded with this function
/// to match Python's `round()` behavior exactly.
pub fn bankers_round(v: f64) -> i32 {
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
/// 2. Compute spatial offset so centroid lands at tensor center `(16, 16)`.
/// 3. Gather legal moves (with optional threat constraint).
/// 4. Fill 13 channels (see [`encode_board_into`] for channel details).
///
/// Returns [`EncodedBoard`] with tensor, offsets, and legal moves.
pub fn encode_board(game: &HexGameState, near_radius: i32, constrain_threats: bool) -> EncodedBoard {
    let mut tensor = vec![0.0f32; TENSOR_SIZE];
    let (offset_q, offset_r, legal_moves) =
        encode_board_into(game, near_radius, constrain_threats, &mut tensor);
    EncodedBoard {
        tensor,
        offset_q,
        offset_r,
        legal_moves,
    }
}

/// Encode into a pre-allocated buffer. Returns `(offset_q, offset_r, legal_moves)`.
///
/// The buffer must be at least [`TENSOR_SIZE`] elements. This variant is used
/// by MCTS to avoid repeated allocations during search.
///
/// # Channel layout
///
/// | Ch | Content |
/// |---|---|
/// | 0 | Current player's stones |
/// | 1 | Opponent's stones |
/// | 2 | Empty cell mask (= 1.0 - ch0 - ch1) |
/// | 3 | Legal moves mask (constrained if `constrain_threats=true`) |
/// | 4 | Turn phase: all 1.0 if `placements_remaining == 1 && move_count > 0` |
/// | 5 | First stone of current turn (phase 2 only): marks last move in history |
/// | 6 | Current player color: all 1.0 if player 0 |
/// | 7 | Own stone recency: `1/(1 + plies_ago)` |
/// | 8 | Opponent stone recency |
/// | 9 | Opponent "hot cells" (empty cells in opponent's 4+ windows) |
/// | 10 | Own "hot cells" |
/// | 11 | Distance from centroid: `hex_dist(cell, center) / HALF_BOARD` |
/// | 12 | Opponent's most recent completed turn cells |
pub fn encode_board_into(
    game: &HexGameState,
    near_radius: i32,
    constrain_threats: bool,
    out: &mut [f32],
) -> (i32, i32, Vec<Hex>) {
    debug_assert!(
        out.len() >= TENSOR_SIZE,
        "encode_board_into: buffer too small ({} < {})",
        out.len(),
        TENSOR_SIZE
    );

    let board = &game.board;

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

    let current = game.current_player;
    let mc = game.move_count;
    let pr = game.placements_remaining;
    let is_phase_2 = pr == 1 && mc > 0;

    // Zero the active region of the buffer.
    out[..TENSOR_SIZE].fill(0.0);

    // Helper: index into flat tensor [ch, gi, gj]
    #[inline(always)]
    fn idx(ch: usize, gi: i32, gj: i32) -> usize {
        ch * BOARD_AREA + (gi as usize) * (BOARD_SIZE as usize) + gj as usize
    }

    // ── Channels 0-1: player stones ──
    // For every occupied cell, write 1.0 into channel 0 (current player) or
    // channel 1 (opponent). Cells outside the 33×33 view are clipped.
    for (&h, &player) in board.iter() {
        let gi = h.q - offset_q;
        let gj = h.r - offset_r;
        if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
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
    for (ply_idx, rec) in game.move_history.iter().enumerate() {
        let gi = rec.cell.q - offset_q;
        let gj = rec.cell.r - offset_r;
        if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
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
    // Gather legal moves via `legal_moves_near`, optionally constrained by
    // threat analysis. Each legal move gets a 1.0 in channel 3 if it falls
    // inside the tensor window. The (possibly constrained) move list is
    // returned so callers can map policy logits back to moves.
    let mut legal = game.legal_moves_near(near_radius);
    if let Some(constrained) = game.compute_threat_constrained_moves(&legal, constrain_threats) {
        legal = constrained;
    }
    for h in &legal {
        let gi = h.q - offset_q;
        let gj = h.r - offset_r;
        if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
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
        if let Some(last) = game.move_history.last() {
            let gi = last.cell.q - offset_q;
            let gj = last.cell.r - offset_r;
            if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
                out[idx(5, gi, gj)] = 1.0;
            }
        }
    }

    // ── Channel 6: current player color ──
    // All 1.0 if the current player is player 0, all 0.0 if player 1.
    if current == 0 {
        let start = 6 * BOARD_AREA;
        out[start..start + BOARD_AREA].fill(1.0);
    }

    // ── Channel 11: distance from centroid ──
    // For every tensor cell, compute the hex distance to the board centroid
    // (which maps to tensor coordinate (HALF_BOARD, HALF_BOARD)), then
    // normalize by HALF_BOARD so values are in [0, 1].
    {
        let center = Hex::new(offset_q + HALF_BOARD, offset_r + HALF_BOARD);
        for gi in 0..BOARD_SIZE {
            for gj in 0..BOARD_SIZE {
                let h = Hex::new(gi + offset_q, gj + offset_r);
                let dist = hex_distance(h, center) as f32 / HALF_BOARD as f32;
                out[idx(11, gi, gj)] = dist;
            }
        }
    }

    // ── Channel 12: opponent's most recent completed turn ──
    // Marks the cells placed by the opponent during their last full turn.
    // For player 0's opening turn this is a single cell; otherwise two cells.
    for h in game.opponent_last_turn_cells() {
        let gi = h.q - offset_q;
        let gj = h.r - offset_r;
        if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
            out[idx(12, gi, gj)] = 1.0;
        }
    }

    // ── Channels 9-10: hot cells ──
    // Ch 9: empty cells that lie inside the opponent's "hot windows"
    //     (windows with 4+ opponent stones and 0 own stones).
    // Ch 10: empty cells that lie inside the current player's hot windows.
    {
        use crate::game::WIN_LENGTH;
        let opp = (1 - current) as usize;
        let own = current as usize;

        for (ch, player_idx) in [(9usize, opp), (10usize, own)] {
            for &(wq, wr, dir) in &game.hot_windows[player_idx] {
                let (dq, dr) = HEX_DIRECTIONS[dir as usize];
                for k in 0..WIN_LENGTH {
                    let cq = wq + dq * k;
                    let cr = wr + dr * k;
                    let h = Hex::new(cq, cr);
                    if !board.contains_key(&h) {
                        let gi = cq - offset_q;
                        let gj = cr - offset_r;
                        if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
                            out[idx(ch, gi, gj)] = 1.0;
                        }
                    }
                }
            }
        }
    }

    (offset_q, offset_r, legal)
}
