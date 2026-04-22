//! Arena-allocated MCTS engine with PUCT, virtual loss, and batch leaf selection.
//!
//! The tree lives entirely in Rust. Python calls:
//!   1. `select_leaves(batch_size)` → board tensors for GPU inference
//!   2. `expand_and_backprop(policies, values)` → updates tree
//!   3. `get_results(temperature)` → (moves, probs, root_value)

use crate::core::Hex;
use crate::game::HexGameState;
use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonically increasing counter seeded into the Gumbel RNG so that every
/// call to `init_gumbel` produces fresh noise even for repeated positions.
static GUMBEL_CALL_COUNTER: AtomicU64 = AtomicU64::new(1);

// ── Constants ────────────────────────────────────────────────────────
const BOARD_SIZE: i32 = 33;
const HALF_BOARD: i32 = BOARD_SIZE / 2;
const NUM_CHANNELS: usize = 13;
const BOARD_AREA: usize = (BOARD_SIZE * BOARD_SIZE) as usize;
const TENSOR_SIZE: usize = NUM_CHANNELS * BOARD_AREA;

const FPU_REDUCTION: f32 = 0.2;
const VIRTUAL_LOSS_VISITS: u32 = 1;
// Pre-allocate capacity for this many nodes per simulation.  Each
// simulation triggers one leaf expansion that pushes N children (N =
// filtered legal moves) into the arena, so real peak for a 1000-sim
// search is ~1000 * branching ≈ 50k nodes at near_radius=8.  The old
// value 128 reserved ~4 MB upfront (2× over-allocation); hint=2 would
// require 5–6 Vec doublings per search (worse heap fragmentation).
// A hint of 64 matches realistic peak usage (~64k nodes ≈ 2 MB) with
// effectively zero reallocations in the common case.
const CHILD_CAPACITY_HINT: usize = 64;

/// Sentinel value for "no parent".
const NO_PARENT: u32 = u32::MAX;

// ── Selector enum ────────────────────────────────────────────────────

/// Tree selection policy for child selection during MCTS traversal.
#[derive(Clone, Copy, PartialEq)]
enum Selector {
    /// Standard PUCT (AlphaZero-style) with dynamic c_puct.
    Puct,
    /// UCT-V-P: variance-aware prior-based UCT (Weichart 2026, canonical RPO form).
    /// Uses O(√log N) exploration scaling — conservative, self-regulating.
    UctVP,
    /// PUCT-V: variance-aware PUCT (heuristic form).
    /// Uses O(√N) exploration scaling — more aggressive.
    PuctV,
}

impl Selector {
    fn from_str(s: &str) -> Self {
        match s {
            "uct_v_p" => Selector::UctVP,
            "puct_v" => Selector::PuctV,
            _ => Selector::Puct,
        }
    }
}

// ── MCTSNode ─────────────────────────────────────────────────────────

#[derive(Clone)]
struct MCTSNode {
    parent: u32,
    action: (i16, i16), // (q, r) relative to origin
    prior: f32,
    visit_count: u32,
    total_value: f32,
    /// Welford M2 accumulator for online variance estimation.
    /// Variance = m2 / visit_count (population variance).
    m2: f32,
    children_start: u32, // index into arena
    children_count: u16,
    player: u8, // 0 or 1, 255 = unset
    is_expanded: bool,
}

impl MCTSNode {
    fn new(parent: u32, action: (i16, i16), prior: f32) -> Self {
        Self {
            parent,
            action,
            prior,
            visit_count: 0,
            total_value: 0.0,
            m2: 0.0,
            children_start: 0,
            children_count: 0,
            player: 255,
            is_expanded: false,
        }
    }

    #[inline(always)]
    fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f32
        }
    }
}

// ── Pending leaf info ────────────────────────────────────────────────

struct PendingLeaf {
    node_idx: u32,
    search_path: Vec<u32>,
    is_terminal: bool,
    terminal_value: f32,
    depth: u32,
    offset_q: i32,
    offset_r: i32,
    legal_moves: Vec<Hex>,
    current_player: u8,
}

// ── Board encoding (reused from pybridge) ────────────────────────────

/// Python-compatible "banker's rounding" (round half to even).
fn bankers_round(v: f64) -> i32 {
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

/// Encode board into a flat f32 tensor and return legal moves + offsets.
fn encode_board(
    game: &HexGameState,
    near_radius: i32,
    constrain_threats: bool,
    tensor: &mut [f32; TENSOR_SIZE],
) -> (i32, i32, Vec<Hex>) {
    let board = &game.board;

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

    // Zero the tensor
    tensor.fill(0.0);

    #[inline(always)]
    fn idx(ch: usize, gi: i32, gj: i32) -> usize {
        ch * BOARD_AREA + (gi as usize) * (BOARD_SIZE as usize) + gj as usize
    }

    // Ch 0-1: player stones (from board HashMap)
    for (&h, &player) in board.iter() {
        let gi = h.q - offset_q;
        let gj = h.r - offset_r;
        if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
            if player == current {
                tensor[idx(0, gi, gj)] = 1.0;
            } else {
                tensor[idx(1, gi, gj)] = 1.0;
            }
        }
    }

    // Ch 7-8: stone recency 1/(1+plies_ago), split by player
    for (ply_idx, rec) in game.move_history.iter().enumerate() {
        let gi = rec.cell.q - offset_q;
        let gj = rec.cell.r - offset_r;
        if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
            let plies_ago = mc - ply_idx as u32;
            let recency = 1.0 / (1.0 + plies_ago as f32);
            let ch = if rec.player == current { 7 } else { 8 };
            tensor[idx(ch, gi, gj)] = recency;
        }
    }

    // Ch 2: empty cells mask
    let ch2_start = 2 * BOARD_AREA;
    for i in 0..BOARD_AREA {
        tensor[ch2_start + i] = 1.0 - tensor[i] - tensor[BOARD_AREA + i];
    }

    let is_phase_2 = pr == 1 && mc > 0;

    // Ch 3: legal moves
    let mut legal = game.legal_moves_near(near_radius);
    if let Some(constrained) = game.compute_threat_constrained_moves(&legal, constrain_threats) {
        legal = constrained;
    }
    for h in &legal {
        let gi = h.q - offset_q;
        let gj = h.r - offset_r;
        if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
            tensor[idx(3, gi, gj)] = 1.0;
        }
    }

    // Ch 4: turn phase
    if is_phase_2 {
        let start = 4 * BOARD_AREA;
        tensor[start..start + BOARD_AREA].fill(1.0);
    }

    // Ch 5: first stone of current turn (phase 2 only)
    if is_phase_2 {
        if let Some(last) = game.move_history.last() {
            let gi = last.cell.q - offset_q;
            let gj = last.cell.r - offset_r;
            if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
                tensor[idx(5, gi, gj)] = 1.0;
            }
        }
    }

    // Ch 6: current player color
    if current == 0 {
        let start = 6 * BOARD_AREA;
        tensor[start..start + BOARD_AREA].fill(1.0);
    }

    // Ch 11: distance from centroid — hex_dist(cell, centroid) / 16.0
    {
        use crate::core::hex_distance;
        let center = Hex::new(HALF_BOARD, HALF_BOARD);
        for gi in 0..BOARD_SIZE {
            for gj in 0..BOARD_SIZE {
                let cell = Hex::new(gi, gj);
                let dist = hex_distance(cell, center) as f32 / 16.0;
                tensor[idx(11, gi, gj)] = dist;
            }
        }
    }

    // Ch 12: opponent's most recent completed turn
    for h in game.opponent_last_turn_cells() {
        let gi = h.q - offset_q;
        let gj = h.r - offset_r;
        if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
            tensor[idx(12, gi, gj)] = 1.0;
        }
    }

    // Ch 9: opponent's hot cells (empty cells in opp's 4+ windows)
    // Ch 10: own hot cells (empty cells in own's 4+ windows)
    {
        use crate::core::HEX_DIRECTIONS;
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
                            tensor[idx(ch, gi, gj)] = 1.0;
                        }
                    }
                }
            }
        }
    }

    (offset_q, offset_r, legal)
}

/// Encode board into a mutable f32 slice (must be >= TENSOR_SIZE).
/// Same as encode_board but avoids the fixed-size array intermediate —
/// callers can pass a slice of a larger pre-allocated buffer.
fn encode_board_slice(
    game: &HexGameState,
    near_radius: i32,
    constrain_threats: bool,
    tensor: &mut [f32],
) -> (i32, i32, Vec<Hex>) {
    debug_assert!(tensor.len() >= TENSOR_SIZE);
    let board = &game.board;

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

    // Zero the tensor
    tensor[..TENSOR_SIZE].fill(0.0);

    #[inline(always)]
    fn idx(ch: usize, gi: i32, gj: i32) -> usize {
        ch * BOARD_AREA + (gi as usize) * (BOARD_SIZE as usize) + gj as usize
    }

    // Ch 0-1: player stones (from board HashMap)
    for (&h, &player) in board.iter() {
        let gi = h.q - offset_q;
        let gj = h.r - offset_r;
        if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
            if player == current {
                tensor[idx(0, gi, gj)] = 1.0;
            } else {
                tensor[idx(1, gi, gj)] = 1.0;
            }
        }
    }

    // Ch 7-8: stone recency 1/(1+plies_ago), split by player
    for (ply_idx, rec) in game.move_history.iter().enumerate() {
        let gi = rec.cell.q - offset_q;
        let gj = rec.cell.r - offset_r;
        if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
            let plies_ago = mc - ply_idx as u32;
            let recency = 1.0 / (1.0 + plies_ago as f32);
            let ch = if rec.player == current { 7 } else { 8 };
            tensor[idx(ch, gi, gj)] = recency;
        }
    }

    // Ch 2: empty cells mask
    let ch2_start = 2 * BOARD_AREA;
    for i in 0..BOARD_AREA {
        tensor[ch2_start + i] = 1.0 - tensor[i] - tensor[BOARD_AREA + i];
    }

    let is_phase_2 = pr == 1 && mc > 0;

    // Ch 3: legal moves
    let mut legal = game.legal_moves_near(near_radius);
    if let Some(constrained) = game.compute_threat_constrained_moves(&legal, constrain_threats) {
        legal = constrained;
    }
    for h in &legal {
        let gi = h.q - offset_q;
        let gj = h.r - offset_r;
        if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
            tensor[idx(3, gi, gj)] = 1.0;
        }
    }

    // Ch 4: turn phase
    if is_phase_2 {
        let start = 4 * BOARD_AREA;
        tensor[start..start + BOARD_AREA].fill(1.0);
    }

    // Ch 5: first stone of current turn (phase 2 only)
    if is_phase_2 {
        if let Some(last) = game.move_history.last() {
            let gi = last.cell.q - offset_q;
            let gj = last.cell.r - offset_r;
            if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
                tensor[idx(5, gi, gj)] = 1.0;
            }
        }
    }

    // Ch 6: current player color
    if current == 0 {
        let start = 6 * BOARD_AREA;
        tensor[start..start + BOARD_AREA].fill(1.0);
    }

    // Ch 11: distance from centroid — hex_dist(cell, centroid) / 16.0
    {
        use crate::core::hex_distance;
        let center = Hex::new(HALF_BOARD, HALF_BOARD);
        for gi in 0..BOARD_SIZE {
            for gj in 0..BOARD_SIZE {
                let cell = Hex::new(gi, gj);
                let dist = hex_distance(cell, center) as f32 / 16.0;
                tensor[idx(11, gi, gj)] = dist;
            }
        }
    }

    // Ch 12: opponent's most recent completed turn
    for h in game.opponent_last_turn_cells() {
        let gi = h.q - offset_q;
        let gj = h.r - offset_r;
        if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
            tensor[idx(12, gi, gj)] = 1.0;
        }
    }

    // Ch 9: opponent's hot cells (empty cells in opp's 4+ windows)
    // Ch 10: own hot cells (empty cells in own's 4+ windows)
    {
        use crate::core::HEX_DIRECTIONS;
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
                            tensor[idx(ch, gi, gj)] = 1.0;
                        }
                    }
                }
            }
        }
    }

    (offset_q, offset_r, legal)
}

// ── Policy gathering ──────────────────────────────────────────────────

/// Gather policy logits for legal moves and normalize them with softmax.
/// Returns (moves, priors).
fn gather_policy(
    moves: &[Hex],
    policy_logits: &[f32], // flat (BOARD_AREA,) = 1089
    offset_q: i32,
    offset_r: i32,
) -> (Vec<Hex>, Vec<f32>) {
    let n = moves.len();
    let mut raw = vec![-10.0f64; n];

    for (i, h) in moves.iter().enumerate() {
        let gi = h.q - offset_q;
        let gj = h.r - offset_r;
        if gi >= 0 && gi < BOARD_SIZE && gj >= 0 && gj < BOARD_SIZE {
            let flat = (gi as usize) * (BOARD_SIZE as usize) + gj as usize;
            raw[i] = policy_logits[flat] as f64;
        }
    }

    let gathered_moves = moves.to_vec();
    let gathered_raw = raw;

    // Softmax in f64 for numerical stability, then convert to f32 priors
    let max_val = gathered_raw
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = gathered_raw.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let priors: Vec<f32> = if sum > 0.0 {
        exps.iter().map(|&e| (e / sum) as f32).collect()
    } else {
        vec![1.0 / gathered_moves.len() as f32; gathered_moves.len()]
    };

    (gathered_moves, priors)
}

// ── MCTSEngine ───────────────────────────────────────────────────────

pub struct MCTSEngine {
    arena: Vec<MCTSNode>,
    game: HexGameState,
    root_idx: u32,
    near_radius: i32,
    constrain_threats: bool,
    c_puct: f32,
    /// Dynamic c_puct scaling constant (AlphaZero/KataGo formula).
    /// effective_c_puct = c_puct + ln((N_parent + c_puct_init) / c_puct_init)
    /// Set to 0.0 to disable (use static c_puct).
    pub c_puct_init: f32,
    /// Tree selection policy (PUCT, UCT-V-P, or PUCT-V).
    selector: Selector,
    /// Variance term coefficient for UCT-V-P / PUCT-V selectors.
    c1: f32,
    /// Bias term coefficient for UCT-V-P / PUCT-V selectors.
    c2: f32,
    sims_done: u32,
    num_simulations: u32,
    // State for pipelined batch processing
    pending: Vec<PendingLeaf>,
    /// Previous batch's pending leaves (for pipeline overlap with GPU).
    prev_pending: Vec<PendingLeaf>,
    tensor_buf: [f32; TENSOR_SIZE],
    /// Pre-allocated output buffer for select_leaves (avoids per-call Vec alloc).
    batch_buf: Vec<f32>,

    // ── Gumbel Sequential Halving state ─────────────────────────────
    /// Whether Gumbel+SH mode is active (replaces Dirichlet noise).
    pub use_gumbel: bool,
    /// sigma_a = gumbel_a + log(prior_a) for each root child (indexed by
    /// offset from children_start).
    gumbel_sigmas: Vec<f32>,
    /// Arena indices of current SH candidates (subset of root children).
    gumbel_candidates: Vec<u32>,
    /// Number of SH rounds = ceil(log2(initial_m)).
    gumbel_num_rounds: u32,
    /// Current round (0-indexed).
    gumbel_current_round: u32,
    /// Simulations allocated per round = num_simulations / num_rounds.
    gumbel_sims_per_round: u32,
    /// Simulations completed in the current SH round.
    gumbel_sims_in_round: u32,
    /// Simulations selected but not yet expanded in the pipelined path.
    gumbel_sims_in_flight: u32,
}

impl MCTSEngine {
    pub fn new(
        game: HexGameState,
        num_simulations: u32,
        c_puct: f32,
        near_radius: i32,
        constrain_threats: bool,
    ) -> Self {
        Self::with_arena_sim_hint(
            game,
            num_simulations,
            num_simulations,
            c_puct,
            near_radius,
            constrain_threats,
        )
    }

    /// Construct with an explicit arena sim-count hint.
    ///
    /// Under subtree reuse, the same arena is shared across both placements of
    /// a turn, so the p1 sim count underestimates the true node budget. Pass
    /// `p1_sims + p2_sims` here to avoid mid-search Vec reallocation.
    pub fn with_arena_sim_hint(
        game: HexGameState,
        num_simulations: u32,
        arena_sim_hint: u32,
        c_puct: f32,
        near_radius: i32,
        constrain_threats: bool,
    ) -> Self {
        let mut arena = Vec::with_capacity(arena_sim_hint as usize * CHILD_CAPACITY_HINT + 64);
        // Create root node
        arena.push(MCTSNode::new(NO_PARENT, (0, 0), 1.0));
        // Batch buffer grows naturally from empty.  The old formula
        // min(sims, 256) * TENSOR_SIZE = 256 * 13 * 33 * 33 * 4 bytes
        // reserved ~13.8 MB per engine upfront, but actual peak usage is
        // leaf_batch_size * TENSOR_SIZE ≈ 2.7 MB at leaf_batch_size=48.
        // The 11 MB of waste per engine, multiplied by 30k engine
        // create/destroy cycles per epoch, fragments the Windows heap
        // until contiguous allocations fail (epoch 19 OOM at 32 MB).
        let batch_capacity = 0usize;
        Self {
            arena,
            game,
            root_idx: 0,
            near_radius,
            constrain_threats,
            c_puct,
            c_puct_init: 19652.0,
            selector: Selector::Puct,
            c1: 1.4,
            c2: 3.0,
            sims_done: 0,
            num_simulations,
            pending: Vec::new(),
            prev_pending: Vec::new(),
            tensor_buf: [0.0f32; TENSOR_SIZE],
            batch_buf: Vec::with_capacity(batch_capacity),
            use_gumbel: false,
            gumbel_sigmas: Vec::new(),
            gumbel_candidates: Vec::new(),
            gumbel_num_rounds: 0,
            gumbel_current_round: 0,
            gumbel_sims_per_round: 0,
            gumbel_sims_in_round: 0,
            gumbel_sims_in_flight: 0,
        }
    }

    /// Initialize root: encode board and return tensor bytes for GPU eval.
    /// Returns (tensor_bytes, offset_q, offset_r, legal_moves) or None if game is over.
    pub fn init_root(&mut self) -> Option<(Vec<f32>, i32, i32, Vec<Hex>)> {
        if self.game.is_over() {
            return None;
        }
        let (oq, or_, legal) = encode_board(
            &self.game,
            self.near_radius,
            self.constrain_threats,
            &mut self.tensor_buf,
        );
        Some((self.tensor_buf.to_vec(), oq, or_, legal))
    }

    /// Expand the root node with policy output from GPU.
    pub fn expand_root(
        &mut self,
        policy_logits: &[f32],
        _value: f32,
        offset_q: i32,
        offset_r: i32,
        legal_moves: &[Hex],
    ) {
        let player = self.game.current_player;
        self.expand_node(
            self.root_idx,
            legal_moves,
            player,
            policy_logits,
            offset_q,
            offset_r,
        );
        self.arena[self.root_idx as usize].player = player;
    }

    /// Add Dirichlet noise to root priors for exploration.
    pub fn add_dirichlet_noise(&mut self, noise: &[f32], noise_fraction: f32) {
        let root = &self.arena[self.root_idx as usize];
        let start = root.children_start as usize;
        let count = root.children_count as usize;
        let n = count.min(noise.len());
        for i in 0..n {
            let child = &mut self.arena[start + i];
            child.prior = (1.0 - noise_fraction) * child.prior + noise_fraction * noise[i];
        }
    }

    /// Are we done simulating?
    pub fn done(&self) -> bool {
        self.sims_done >= self.num_simulations
    }

    /// Select up to `batch_size` leaves, encode their boards.
    /// Returns a reference to the internal batch buffer and the count of non-terminal leaves.
    /// The buffer contains (non_terminal_count × TENSOR_SIZE) f32 values.
    pub fn select_leaves(&mut self, batch_size: u32) -> (&[f32], u32) {
        let mut actual_batch = batch_size.min(self.num_simulations - self.sims_done);
        if self.use_gumbel
            && self.gumbel_current_round + 1 < self.gumbel_num_rounds
            && self.gumbel_sims_per_round > 0
        {
            let committed = self.gumbel_sims_in_round + self.gumbel_sims_in_flight;
            let remaining_in_round = self.gumbel_sims_per_round.saturating_sub(committed);
            actual_batch = actual_batch.min(remaining_in_round);
        }
        self.pending.clear();
        // Pre-allocate batch buffer for worst case (all non-terminal)
        let max_floats = actual_batch as usize * TENSOR_SIZE;
        self.batch_buf.clear();
        self.batch_buf.reserve(max_floats);
        let mut non_terminal_count = 0u32;

        for _ in 0..actual_batch {
            let mut node_idx = self.root_idx;
            let mut depth = 0u32;
            let mut search_path = vec![self.root_idx];

            // Selection: traverse tree via PUCT
            while self.arena[node_idx as usize].is_expanded
                && self.arena[node_idx as usize].children_count > 0
            {
                node_idx = self.select_child_puct(node_idx);
                let child = &self.arena[node_idx as usize];
                self.game
                    .place(child.action.0 as i32, child.action.1 as i32)
                    .expect("MCTS: illegal place during tree traversal");
                depth += 1;
                search_path.push(node_idx);
            }

            if self.arena[node_idx as usize].player == 255 {
                self.arena[node_idx as usize].player = self.game.current_player;
            }

            // Apply virtual loss before moving search_path into PendingLeaf
            // (visit count only — avoids sign-flip interaction with
            // player-perspective Q negation)
            for &ni in &search_path {
                let n = &mut self.arena[ni as usize];
                n.visit_count += VIRTUAL_LOSS_VISITS;
            }

            if self.game.is_over() {
                let node_player = self.arena[node_idx as usize].player;
                let value = if self.game.winner == Some(node_player) {
                    1.0
                } else {
                    -1.0
                };
                self.pending.push(PendingLeaf {
                    node_idx,
                    search_path,
                    is_terminal: true,
                    terminal_value: value,
                    depth,
                    offset_q: 0,
                    offset_r: 0,
                    legal_moves: Vec::new(),
                    current_player: 0,
                });
            } else {
                // Encode directly into batch_buf (avoids intermediate copy)
                let start = non_terminal_count as usize * TENSOR_SIZE;
                self.batch_buf.resize(start + TENSOR_SIZE, 0.0);
                let tensor_slice = &mut self.batch_buf[start..start + TENSOR_SIZE];
                // Only constrain threats at root (init_root), not internal
                // nodes — the O(n²) unblockable check is too expensive to run
                // on every leaf expansion during tree search.
                let (oq, or_, legal) = encode_board_slice(
                    &self.game,
                    self.near_radius,
                    false,
                    tensor_slice,
                );
                let cp = self.game.current_player;
                non_terminal_count += 1;
                self.pending.push(PendingLeaf {
                    node_idx,
                    search_path,
                    is_terminal: false,
                    terminal_value: 0.0,
                    depth,
                    offset_q: oq,
                    offset_r: or_,
                    legal_moves: legal,
                    current_player: cp,
                });
            }

            // Undo moves to restore root state
            for _ in 0..depth {
                self.game.unmake_move();
            }
        }

        self.sims_done += actual_batch;

        (
            &self.batch_buf[..non_terminal_count as usize * TENSOR_SIZE],
            non_terminal_count,
        )
    }

    /// Expand leaves and backpropagate using GPU-provided policies and values.
    /// `policies` is flat (N * BOARD_AREA) f32, `values` is flat (N,) f32.
    pub fn expand_and_backprop(&mut self, policies: &[f32], values: &[f32]) {
        let mut eval_idx = 0usize;

        // Take pending out to avoid borrow issues
        let pending = std::mem::take(&mut self.pending);
        // Count before processing — this is the number of simulations whose
        // Q-values will be fresh by the time we reach the halving check below.
        // Deliberately counted here (not in select_leaves) so that the halving
        // boundary is based on *completed* sims, not *selected* sims.  In the
        // pipelined path (select N+1 → expand N), counting in select_leaves
        // would trigger halving with batch N+1 already in flight but its
        // Q-values not yet updated, causing stale elimination decisions.
        let sims_completed = pending.len() as u32;

        for leaf in &pending {
            // Remove virtual loss
            for &ni in &leaf.search_path {
                let n = &mut self.arena[ni as usize];
                n.visit_count -= VIRTUAL_LOSS_VISITS;
            }

            let value;
            if leaf.is_terminal {
                value = leaf.terminal_value;
            } else {
                // Extract this leaf's policy slice
                let policy_start = eval_idx * BOARD_AREA;
                let policy_slice = &policies[policy_start..policy_start + BOARD_AREA];
                let v = values[eval_idx];

                self.expand_node(
                    leaf.node_idx,
                    &leaf.legal_moves,
                    leaf.current_player,
                    policy_slice,
                    leaf.offset_q,
                    leaf.offset_r,
                );
                value = v;
                eval_idx += 1;
            }

            // Backpropagate with Welford online variance tracking.
            // Welford update uses the player-perspective value (post-flip),
            // matching the sign convention of Q-values seen by the selector.
            let leaf_player = self.arena[leaf.node_idx as usize].player;
            for &ni in leaf.search_path.iter().rev() {
                let n = &mut self.arena[ni as usize];
                // Player-perspective value (same sign as what gets added to total_value)
                let pv_value = if n.player == leaf_player { value } else { -value };
                let old_mean = if n.visit_count > 0 {
                    n.total_value / n.visit_count as f32
                } else {
                    0.0
                };
                n.visit_count += 1;
                n.total_value += pv_value;
                let new_mean = n.total_value / n.visit_count as f32;
                // Welford M2 update: accumulates sum of squared deviations
                // from the running mean. Only meaningful from visit 2 onward,
                // but the math is correct for visit 1 too (delta * delta2 = 0
                // when old_mean was 0 and new_mean == pv_value).
                if n.visit_count >= 2 {
                    n.m2 += (pv_value - old_mean) * (pv_value - new_mean);
                }
            }
        }

        // Restore pending (now empty)
        self.pending = pending;
        self.pending.clear();

        // Gumbel SH: accumulate completed sims then halve when round budget is
        // exhausted.  Both the increment and the halving check live here so
        // that we only fire on actually-committed Q-values.
        if self.use_gumbel {
            self.gumbel_sims_in_round += sims_completed;
            if self.gumbel_candidates.len() > 2
                && self.gumbel_sims_in_round >= self.gumbel_sims_per_round
                && self.gumbel_current_round + 1 < self.gumbel_num_rounds
            {
                self.gumbel_halve();
            }
        }
    }

    /// Pipeline variant of select_leaves: saves current pending → prev_pending
    /// before selecting new leaves. This allows GPU inference for the current
    /// batch to overlap with Rust select for the next batch.
    pub fn select_leaves_pipeline(&mut self, batch_size: u32) -> (&[f32], u32) {
        if self.use_gumbel {
            self.gumbel_sims_in_flight = self.pending.len() as u32;
            if self.gumbel_current_round + 1 < self.gumbel_num_rounds
                && self.gumbel_sims_per_round > 0
                && self.gumbel_sims_in_round + self.gumbel_sims_in_flight >= self.gumbel_sims_per_round
            {
                std::mem::swap(&mut self.pending, &mut self.prev_pending);
                self.pending.clear();
                self.batch_buf.clear();
                return (&self.batch_buf[..0], 0);
            }
        }
        // Move current pending to prev_pending (previous batch awaiting GPU results)
        std::mem::swap(&mut self.pending, &mut self.prev_pending);
        self.select_leaves(batch_size)
    }

    /// Expand and backpropagate the PREVIOUS batch (stored in prev_pending).
    /// Used in the pipeline pattern: select N+1 → expand N → GPU N+1 → ...
    pub fn expand_prev_and_backprop(&mut self, policies: &[f32], values: &[f32]) {
        if self.use_gumbel {
            self.gumbel_sims_in_flight = 0;
        }
        // Swap prev_pending into pending for expand_and_backprop
        std::mem::swap(&mut self.pending, &mut self.prev_pending);
        self.expand_and_backprop(policies, values);
        // Swap back: pending is restored to current batch, prev_pending is cleared
        std::mem::swap(&mut self.pending, &mut self.prev_pending);
    }

    /// Get search results: (moves_q, moves_r, visit_counts, root_value).
    pub fn get_results(&self) -> (Vec<i32>, Vec<i32>, Vec<u32>, f32) {
        let root = &self.arena[self.root_idx as usize];
        let root_value = root.q_value();
        let start = root.children_start as usize;
        let count = root.children_count as usize;

        let mut moves_q = Vec::with_capacity(count);
        let mut moves_r = Vec::with_capacity(count);
        let mut visits = Vec::with_capacity(count);

        for i in start..start + count {
            let child = &self.arena[i];
            moves_q.push(child.action.0 as i32);
            moves_r.push(child.action.1 as i32);
            visits.push(child.visit_count);
        }

        (moves_q, moves_r, visits, root_value)
    }

    /// Set the c_puct exploration constant.
    pub fn set_c_puct(&mut self, c_puct: f32) {
        self.c_puct = c_puct;
    }

    /// Set the tree selection policy and variance-aware coefficients.
    pub fn set_selector(&mut self, selector_str: &str, c1: f32, c2: f32) {
        self.selector = Selector::from_str(selector_str);
        self.c1 = c1;
        self.c2 = c2;
    }

    /// Re-root the tree at the child matching action `(q, r)`.
    ///
    /// Used for subtree reuse across placements: after placement 1 is selected,
    /// re-root at that child so placement 2's MCTS starts from the surviving
    /// subtree with all visit/Q/variance statistics intact.
    ///
    /// **Precondition:** Pipeline must be fully flushed before calling — both
    /// `pending` and `prev_pending` must be empty (no unprocessed virtual loss).
    ///
    /// The arena is not compacted — dead sibling subtrees remain in memory but
    /// are never referenced. With typical tree sizes this wastes ~400KB, which
    /// is acceptable since the arena is recreated per turn.
    pub fn re_root(&mut self, q: i16, r: i16, new_num_simulations: u32) {
        assert!(
            self.pending.is_empty() && self.prev_pending.is_empty(),
            "re_root: pipeline must be flushed (pending/prev_pending not empty)"
        );

        let root = &self.arena[self.root_idx as usize];
        let start = root.children_start as usize;
        let count = root.children_count as usize;

        // Find the child matching the action
        let mut child_idx = None;
        for i in start..start + count {
            let child = &self.arena[i];
            if child.action.0 == q && child.action.1 == r {
                child_idx = Some(i as u32);
                break;
            }
        }

        let child_idx = child_idx.unwrap_or_else(|| {
            panic!("re_root: no child found for action ({}, {})", q, r);
        });

        // Apply the placement to the internal game state so that subsequent
        // select_leaves traversals start from the correct board position.
        self.game
            .place(q as i32, r as i32)
            .expect("re_root: illegal placement");

        // Update root pointer
        self.root_idx = child_idx;
        self.arena[child_idx as usize].parent = NO_PARENT;

        // If the new root was already expanded during P1's tree traversal, its
        // children were computed with constrain_threats=false (encode_board_slice
        // always passes false for internal nodes to avoid the O(n²) unblockable
        // check). When threat constraints are enabled and P2's position has active
        // threats, those unconstrained children are wrong — clear them so that
        // Python calls init_root again and gets the correct constrained legal set.
        //
        // The descendant subtree must also be purged: extract_tree_node_states
        // iterates the full arena and includes any node with is_expanded=true
        // whose parent chain reaches self.root_idx. Without purging, the
        // grandchildren reached via threat-illegal paths would leak into RGSC
        // candidates as spurious off-policy positions.
        if self.constrain_threats {
            let me = self.game.current_player as usize;
            let opp = 1 - me;
            let has_threats = self.game.window_fives[me] > 0
                || self.game.window_fours[me] > 0
                || self.game.window_fives[opp] > 0
                || self.game.window_fours[opp] > 0;
            if has_threats && self.arena[child_idx as usize].is_expanded {
                let (old_start, old_count) = {
                    let node = &self.arena[child_idx as usize];
                    (node.children_start as usize, node.children_count as usize)
                };
                // Mark new root as needing re-expansion.
                {
                    let node = &mut self.arena[child_idx as usize];
                    node.is_expanded = false;
                    node.children_count = 0;
                }
                // BFS-invalidate the orphaned subtree so extract_tree_node_states
                // does not surface descendants reached via unconstrained paths.
                let mut stack: Vec<u32> = (old_start..old_start + old_count)
                    .map(|i| i as u32)
                    .collect();
                while let Some(idx) = stack.pop() {
                    let (start, count, was_expanded) = {
                        let node = &self.arena[idx as usize];
                        (
                            node.children_start as usize,
                            node.children_count as usize,
                            node.is_expanded,
                        )
                    };
                    let node = &mut self.arena[idx as usize];
                    node.is_expanded = false;
                    node.children_count = 0;
                    node.parent = NO_PARENT;
                    if was_expanded {
                        for i in start..start + count {
                            stack.push(i as u32);
                        }
                    }
                }
            }
        }

        // Reset simulation state for the new search
        self.sims_done = 0;
        self.num_simulations = new_num_simulations;

        // Clear Gumbel Sequential Halving state
        self.use_gumbel = false;
        self.gumbel_sigmas.clear();
        self.gumbel_candidates.clear();
        self.gumbel_num_rounds = 0;
        self.gumbel_current_round = 0;
        self.gumbel_sims_per_round = 0;
        self.gumbel_sims_in_round = 0;
        self.gumbel_sims_in_flight = 0;
    }

    /// Get the number of children at root.
    pub fn root_child_count(&self) -> u16 {
        self.arena[self.root_idx as usize].children_count
    }

    /// Get the prior probabilities of root children (for shaped Dirichlet noise).
    pub fn root_child_priors(&self) -> Vec<f32> {
        let root = &self.arena[self.root_idx as usize];
        let start = root.children_start as usize;
        let count = root.children_count as usize;
        (start..start + count)
            .map(|i| self.arena[i].prior)
            .collect()
    }

    /// Get Q-values of root children from the root player's perspective.
    pub fn root_child_q_values(&self) -> Vec<f32> {
        let root = &self.arena[self.root_idx as usize];
        let root_player = root.player;
        let root_q = root.q_value();
        let start = root.children_start as usize;
        let count = root.children_count as usize;

        (start..start + count)
            .map(|i| {
                let child = &self.arena[i];
                if child.visit_count > 0 {
                    let raw_q = child.q_value();
                    if root_player != 255 && child.player != 255 && child.player != root_player {
                        -raw_q
                    } else {
                        raw_q
                    }
                } else {
                    root_q
                }
            })
            .collect()
    }

    /// Extract encoded board states and move histories for expanded tree nodes.
    pub fn extract_tree_node_states(
        &mut self,
        min_visits: u32,
    ) -> (Vec<f32>, Vec<Vec<(u8, i16, i16)>>, usize) {
        let mut packed = Vec::new();
        let mut histories = Vec::new();

        // Cap candidates to avoid OOM on large trees (48MB+ with 800+ nodes).
        const MAX_CANDIDATES: usize = 128;

        let mut candidates: Vec<u32> = self
            .arena
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| {
                let idx = idx as u32;
                if idx == self.root_idx || !node.is_expanded || node.visit_count < min_visits {
                    return None;
                }
                Some(idx)
            })
            .collect();

        // Keep only the highest-visited candidates if over the cap.
        if candidates.len() > MAX_CANDIDATES {
            candidates.sort_unstable_by(|a, b| {
                self.arena[*b as usize]
                    .visit_count
                    .cmp(&self.arena[*a as usize].visit_count)
            });
            candidates.truncate(MAX_CANDIDATES);
        }

        if let Err(_) = packed.try_reserve(candidates.len() * TENSOR_SIZE) {
            // Allocation would exceed available memory — return empty.
            return (Vec::new(), Vec::new(), 0);
        }
        histories.reserve(candidates.len());

        for node_idx in candidates {
            let mut actions: Vec<(i16, i16)> = Vec::new();
            let mut cur = node_idx;
            let mut valid = true;
            while cur != self.root_idx {
                let node = &self.arena[cur as usize];
                if node.parent == NO_PARENT {
                    valid = false;
                    break; // safety: orphan node
                }
                actions.push(node.action);
                cur = node.parent;
            }
            actions.reverse();

            let mut placed = 0usize;
            for &(q, r) in &actions {
                if self.game.is_over() {
                    valid = false;
                    break;
                }
                if self.game.place(q as i32, r as i32).is_err() {
                    valid = false;
                    break;
                }
                placed += 1;
            }

            if valid && !self.game.is_over() {
                let start = packed.len();
                packed.resize(start + TENSOR_SIZE, 0.0);
                let tensor_slice = &mut packed[start..start + TENSOR_SIZE];
                encode_board_slice(&self.game, self.near_radius, false, tensor_slice);

                let history: Vec<(u8, i16, i16)> = self
                    .game
                    .move_history
                    .iter()
                    .map(|rec| (rec.player, rec.cell.q as i16, rec.cell.r as i16))
                    .collect();
                histories.push(history);
            }

            for _ in 0..placed {
                self.game.unmake_move();
            }
        }

        let count = histories.len();
        (packed, histories, count)
    }

    // ── Gumbel Sequential Halving ────────────────────────────────────

    /// Initialize Gumbel+Sequential Halving mode. Call after `expand_root`.
    ///
    /// `num_considered` (m): number of top actions to consider (typically 16).
    /// Gumbel(0,1) noise is sampled for each root child, combined with log-prior
    /// to form sigma scores. The top-m by sigma are the initial SH candidates.
    /// Replaces Dirichlet noise — do NOT also call `add_dirichlet_noise`.
    pub fn init_gumbel(&mut self, num_considered: u32) {
        let root = &self.arena[self.root_idx as usize];
        let start = root.children_start as usize;
        let count = root.children_count as usize;

        if count == 0 {
            return;
        }

        // Sample Gumbel(0,1) noise: -log(-log(U)) where U ~ Uniform(0,1)
        // Seed = global call counter XOR board hash.  The counter guarantees
        // uniqueness across calls so repeated positions (e.g. the opening)
        // produce fresh noise every game, restoring self-play diversity.
        let call_id = GUMBEL_CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut rng_state: u64 = {
            let mut h: u64 = 0x517cc1b727220a95;
            // call_id is the primary uniqueness source
            h ^= call_id.wrapping_mul(0x6c62272e07bb0142);
            h = h.wrapping_mul(0x94d049bb133111eb).rotate_left(31);
            for (hex, &player) in self.game.board.iter() {
                h ^= (hex.q as u64).wrapping_mul(0x9e3779b97f4a7c15);
                h ^= (hex.r as u64).wrapping_mul(0x6c62272e07bb0142);
                h ^= (player as u64).wrapping_mul(0xbf58476d1ce4e5b9);
                h = h.wrapping_mul(0x94d049bb133111eb).rotate_left(31);
            }
            // Mix in move count for additional positional diversity
            h ^= (self.game.move_history.len() as u64).wrapping_mul(0x517cc1b727220a95);
            if h == 0 { h = 1; }
            h
        };

        // Compute sigma_a = gumbel_a + log(prior_a) for each root child
        self.gumbel_sigmas.clear();
        self.gumbel_sigmas.reserve(count);
        for i in start..start + count {
            // xorshift64
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let u = (rng_state as f64) / (u64::MAX as f64);
            // Clamp to avoid log(0)
            let u_clamped = u.max(1e-20).min(1.0 - 1e-10);
            let gumbel = -((-u_clamped.ln()).ln()) as f32;

            let prior = self.arena[i].prior.max(1e-8);
            let sigma = gumbel + prior.ln();
            self.gumbel_sigmas.push(sigma);
        }

        // Select top-m candidates by sigma score
        let m = (num_considered as usize).min(count);
        let mut indices: Vec<usize> = (0..count).collect();
        indices.sort_by(|&a, &b| {
            self.gumbel_sigmas[b]
                .partial_cmp(&self.gumbel_sigmas[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(m);

        // Store as arena indices
        self.gumbel_candidates = indices
            .iter()
            .map(|&i| (start + i) as u32)
            .collect();

        // Compute SH round structure
        let m32 = m as u32;
        self.gumbel_num_rounds = if m32 <= 1 { 1 } else { (m32 as f32).log2().ceil() as u32 };
        self.gumbel_current_round = 0;
        self.gumbel_sims_per_round = self.num_simulations / self.gumbel_num_rounds.max(1);
        self.gumbel_sims_in_round = 0;
        self.gumbel_sims_in_flight = 0;
        self.use_gumbel = true;
    }

    /// Encode board states for each current Gumbel candidate.
    ///
    /// For each candidate P1 move, applies the move, encodes the resulting
    /// board, and undoes the move.  Returns packed f32 tensors of shape
    /// (num_candidates, 13, 33, 33) suitable for batched GPU evaluation.
    pub fn encode_gumbel_candidate_states(&mut self) -> Vec<f32> {
        let n = self.gumbel_candidates.len();
        let mut buf = vec![0.0f32; n * TENSOR_SIZE];
        for (i, &idx) in self.gumbel_candidates.clone().iter().enumerate() {
            let (q, r) = self.arena[idx as usize].action;
            self.game
                .place(q as i32, r as i32)
                .expect("synergy boost: illegal P1 placement");
            let slice = &mut buf[i * TENSOR_SIZE..(i + 1) * TENSOR_SIZE];
            encode_board_slice(&self.game, self.near_radius, false, slice);
            self.game.unmake_move();
        }
        buf
    }

    /// Boost Gumbel sigma scores with per-candidate synergy values and
    /// re-select the top-m candidates.
    ///
    /// `boosts[i]` is the max P2 prior reachable from candidate `i`.
    /// The boost is applied in log-space:
    ///     sigma_a += beta * ln(max_p2_prior_a)
    ///
    /// After boosting, re-ranks all root children (original sigma + boost
    /// for candidates that were evaluated, original sigma for the rest)
    /// and selects a new top-m set.
    pub fn boost_gumbel_sigmas(&mut self, boosts: &[f32], beta: f32) {
        if boosts.len() != self.gumbel_candidates.len() || boosts.is_empty() {
            return;
        }

        let root = &self.arena[self.root_idx as usize];
        let start = root.children_start as usize;
        let count = root.children_count as usize;
        if count == 0 {
            return;
        }

        // Apply boost to the sigma of each evaluated candidate
        for (i, &idx) in self.gumbel_candidates.iter().enumerate() {
            let child_offset = idx as usize - start;
            if child_offset < self.gumbel_sigmas.len() {
                let b = boosts[i].max(1e-8);
                self.gumbel_sigmas[child_offset] += beta * b.ln();
            }
        }

        // Re-select top-m from all children with updated sigmas
        let m = self.gumbel_candidates.len(); // keep same pool size
        let mut indices: Vec<usize> = (0..count).collect();
        indices.sort_by(|&a, &b| {
            self.gumbel_sigmas[b]
                .partial_cmp(&self.gumbel_sigmas[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(m);

        self.gumbel_candidates = indices
            .iter()
            .map(|&i| (start + i) as u32)
            .collect();
    }

    /// Perform Sequential Halving: keep top half of candidates by Q-value.
    /// Called automatically when a round's simulation budget is exhausted.
    fn gumbel_halve(&mut self) {
        if self.gumbel_candidates.len() <= 2 {
            return; // Don't halve below 2 candidates
        }

        let root = &self.arena[self.root_idx as usize];
        let root_player = root.player;

        // Score each candidate by its Q-value (from the root player's perspective)
        let mut scored: Vec<(u32, f32)> = self
            .gumbel_candidates
            .iter()
            .map(|&idx| {
                let child = &self.arena[idx as usize];
                let q = if child.visit_count > 0 {
                    let raw_q = child.q_value();
                    // Negate if child represents opponent's perspective
                    if root_player != 255
                        && child.player != 255
                        && child.player != root_player
                    {
                        -raw_q
                    } else {
                        raw_q
                    }
                } else {
                    f32::NEG_INFINITY
                };
                (idx, q)
            })
            .collect();

        // Sort by Q descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep top half
        let keep = (scored.len() + 1) / 2; // Round up
        scored.truncate(keep);
        self.gumbel_candidates = scored.iter().map(|&(idx, _)| idx).collect();

        self.gumbel_current_round += 1;
        self.gumbel_sims_in_round = 0;
    }

    // ── Internal helpers ─────────────────────────────────────────────

    fn select_child_puct(&self, node_idx: u32) -> u32 {
        let node = &self.arena[node_idx as usize];
        let parent_player = node.player;
        let parent_n = node.visit_count;
        let sqrt_parent = if parent_n > 0 {
            (parent_n as f32).sqrt()
        } else {
            1.0
        };
        let parent_q = if parent_n > 0 {
            node.q_value()
        } else {
            0.0
        };
        let fpu_value = parent_q - FPU_REDUCTION;

        let start = node.children_start as usize;
        let count = node.children_count as usize;

        // In Gumbel mode at root: equal-budget allocation among SH candidates.
        // Pick the candidate with the fewest visits so each surviving action
        // receives an equal number of simulations per round.  This is the
        // correct Sequential Halving behaviour — PUCT must NOT govern root
        // selection here or high-prior actions steal the budget.
        if self.use_gumbel && node_idx == self.root_idx && !self.gumbel_candidates.is_empty() {
            let mut best_idx = self.gumbel_candidates[0];
            let mut min_visits = u32::MAX;
            for &idx in &self.gumbel_candidates {
                let vc = self.arena[idx as usize].visit_count;
                if vc < min_visits {
                    min_visits = vc;
                    best_idx = idx;
                }
            }
            return best_idx;
        }

        match self.selector {
            Selector::Puct => {
                // Standard PUCT with dynamic c_puct (AlphaZero/KataGo formula)
                let effective_c_puct = if self.c_puct_init > 0.0 {
                    let n = parent_n as f32;
                    self.c_puct + ((n + self.c_puct_init) / self.c_puct_init).ln()
                } else {
                    self.c_puct
                };

                let mut best_idx = start as u32;
                let mut best_score = f32::NEG_INFINITY;

                for i in start..start + count {
                    let child = &self.arena[i];
                    let vc = child.visit_count;
                    let q = if vc > 0 {
                        let raw_q = child.total_value / vc as f32;
                        if parent_player != 255 && child.player != 255 && child.player != parent_player {
                            -raw_q
                        } else {
                            raw_q
                        }
                    } else {
                        fpu_value
                    };

                    let score = q + effective_c_puct * child.prior * sqrt_parent / (1.0 + vc as f32);
                    if score > best_score {
                        best_score = score;
                        best_idx = i as u32;
                    }
                }
                best_idx
            }
            Selector::UctVP => {
                // UCT-V-P (Weichart 2026, canonical RPO form):
                // S_a = q_a + c1 * σ̂_a * √(π(a) * log(N) / (1+n_a))
                //             + c2 * π(a) * log(N) / (1+n_a)
                let log_parent = if parent_n > 1 {
                    (parent_n as f32).ln()
                } else {
                    1.0f32  // avoid log(0/1) edge case
                };

                let mut best_idx = start as u32;
                let mut best_score = f32::NEG_INFINITY;

                for i in start..start + count {
                    let child = &self.arena[i];
                    let vc = child.visit_count;
                    let q = if vc > 0 {
                        let raw_q = child.total_value / vc as f32;
                        if parent_player != 255 && child.player != 255 && child.player != parent_player {
                            -raw_q
                        } else {
                            raw_q
                        }
                    } else {
                        fpu_value
                    };

                    let denom = 1.0 + vc as f32;
                    let bias = self.c2 * child.prior * log_parent / denom;

                    let variance_term = if vc >= 2 {
                        let variance = child.m2 / vc as f32;
                        let sigma = variance.max(0.0).sqrt();
                        self.c1 * sigma * (child.prior * log_parent / denom).max(0.0).sqrt()
                    } else {
                        0.0
                    };

                    let score = q + variance_term + bias;
                    if score > best_score {
                        best_score = score;
                        best_idx = i as u32;
                    }
                }
                best_idx
            }
            Selector::PuctV => {
                // PUCT-V (heuristic form):
                // S_a = q_a + c1 * π(a) * σ̂_a * √N / (1+n_a)
                //             + c2 * π(a) * log(N) / (1+n_a)
                let log_parent = if parent_n > 1 {
                    (parent_n as f32).ln()
                } else {
                    1.0f32
                };

                let mut best_idx = start as u32;
                let mut best_score = f32::NEG_INFINITY;

                for i in start..start + count {
                    let child = &self.arena[i];
                    let vc = child.visit_count;
                    let q = if vc > 0 {
                        let raw_q = child.total_value / vc as f32;
                        if parent_player != 255 && child.player != 255 && child.player != parent_player {
                            -raw_q
                        } else {
                            raw_q
                        }
                    } else {
                        fpu_value
                    };

                    let denom = 1.0 + vc as f32;
                    let bias = self.c2 * child.prior * log_parent / denom;

                    let variance_term = if vc >= 2 {
                        let variance = child.m2 / vc as f32;
                        let sigma = variance.max(0.0).sqrt();
                        self.c1 * child.prior * sigma * sqrt_parent / denom
                    } else {
                        0.0
                    };

                    let score = q + variance_term + bias;
                    if score > best_score {
                        best_score = score;
                        best_idx = i as u32;
                    }
                }
                best_idx
            }
        }
    }

    fn expand_node(
        &mut self,
        node_idx: u32,
        legal_moves: &[Hex],
        player: u8,
        policy_logits: &[f32],
        offset_q: i32,
        offset_r: i32,
    ) {
        if legal_moves.is_empty() {
            return;
        }

        let (filtered_moves, priors) =
            gather_policy(legal_moves, policy_logits, offset_q, offset_r);

        // Allocate children contiguously in arena
        let children_start = self.arena.len() as u32;
        let children_count = filtered_moves.len() as u16;

        for (i, h) in filtered_moves.iter().enumerate() {
            let child = MCTSNode::new(node_idx, (h.q as i16, h.r as i16), priors[i]);
            self.arena.push(child);
        }

        let node = &mut self.arena[node_idx as usize];
        node.children_start = children_start;
        node.children_count = children_count;
        node.player = player;
        node.is_expanded = true;
    }
}

