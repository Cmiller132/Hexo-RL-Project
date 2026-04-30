//! Arena-allocated MCTS engine with PUCT, virtual loss, and batch leaf selection.
//!
//! The tree lives entirely in Rust. Callers exchange root and batch generation
//! tokens with the engine so stale neural-network responses fail explicitly
//! instead of mutating the wrong search tree.
//!
//! # MCTS algorithm overview
//!
//! 1. **Selection** — starting from the root, repeatedly choose the child
//!    with the highest PUCT score until an unexpanded node or terminal
//!    position is reached.
//! 2. **Expansion** — for a non-terminal leaf, create children for all
//!    legal moves and attach neural-network policy priors.
//! 3. **Evaluation** — terminal leaves use the game outcome; non-terminal
//!    leaves use the neural-network value head.
//! 4. **Backpropagation** — walk back up the search path, incrementing
//!    visit counts and accumulating values.  Values are stored from each
//!    node's own player perspective, so the sign flips at every ply.
//!
//! # UCB / PUCT formula
//!
//! The selection score for a child is:
//!
//! ```text
//! score = Q + c_puct * prior * sqrt(parent_visits) / (1 + child_visits)
//! ```
//!
//! Where:
//! - `Q` is the average value of the child from the **parent player's**
//!   perspective.  For unvisited children, `Q = parent_Q - FPU_REDUCTION`
//!   (First-Play Urgency, encouraging exploration of high-prior moves).
//! - `c_puct` is the exploration constant.  If `c_puct_init > 0`, it grows
//!   logarithmically with parent visits (AlphaZero / KataGo formula):
//!   ```text
//!   effective_c = c_puct + ln((parent_visits + c_puct_init) / c_puct_init)
//!   ```
//! - `prior` is the neural-network policy probability for the move.
//! - The denominator `(1 + child_visits)` shrinks the exploration bonus
//!   as the child is visited more often.
//!
//! # Virtual loss
//!
//! During batch selection, every node on the search path receives a
//! temporary `VIRTUAL_LOSS_VISITS`.  This discourages multiple batch slots
//! (or parallel threads) from exploring the same branch simultaneously,
//! improving diversity in the batch.

use crate::board::HexGameState;
use crate::core::Hex;
use crate::encoder::{self, BOARD_AREA, BOARD_SIZE, TENSOR_SIZE};
use smallvec::SmallVec;

/// Errors that can occur during MCTS operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MCTSError {
    /// re_root was called with an action that is not a child of the current root.
    ChildNotFound { q: i32, r: i32 },
    /// A root expansion response was submitted for an older root/init state.
    StaleRootToken { expected: u64, received: u64 },
    /// A leaf batch response was submitted for an older selected batch.
    StaleBatchToken { expected: u64, received: u64 },
    /// A root operation was called in an invalid state.
    InvalidRootState(&'static str),
    /// A batch operation was called in an invalid state.
    InvalidBatchState(&'static str),
    /// A dense policy slice had the wrong flat length.
    WrongPolicyLength { expected: usize, actual: usize },
    /// A value slice had the wrong flat length.
    WrongValueLength { expected: usize, actual: usize },
    /// Sparse policy metadata did not match the selected non-terminal leaves.
    WrongSparseMetadataLength {
        expected: usize,
        actions: usize,
        logits: usize,
        sources: Option<usize>,
    },
    /// A dense policy logit was NaN or infinite.
    NonFinitePolicy { index: usize },
    /// A value was NaN or infinite.
    NonFiniteValue { index: usize },
    /// Sparse policy metadata contained a NaN or infinite logit.
    NonFiniteSparsePolicy { leaf: usize, index: usize },
    /// Dirichlet noise was shorter than the current root child count.
    WrongNoiseLength {
        expected_at_least: usize,
        actual: usize,
    },
    /// Dirichlet noise had invalid values or zero mass.
    InvalidNoise(&'static str),
    /// Prior metadata was malformed or inconsistent with the current root.
    InvalidPrior(String),
    /// The tree contained an action that the game state rejected during traversal.
    IllegalTreeAction { q: i32, r: i32, node_idx: u32 },
}

// ── Constants ──────────────────────────────────────────────────────────

/// Sentinel value for "no parent".
const NO_PARENT: u32 = u32::MAX;

/// First-Play Urgency reduction for unvisited children.
///
/// Unvisited children are assigned Q = parent_Q - FPU_REDUCTION.
/// This makes the engine prefer exploring high-prior moves that have not
/// yet been tried, rather than treating unvisited children as Q = 0.
const FPU_REDUCTION: f32 = 0.2;

/// Number of virtual visits added to each node on the search path during
/// leaf selection.  This discourages multiple threads (or sequential batch
/// slots) from exploring the same branch simultaneously.
const VIRTUAL_LOSS_VISITS: u32 = 1;

/// Pre-allocate arena capacity for this many nodes per simulation hint.
///
/// Each leaf expansion pushes N children (N = filtered legal moves) into
/// the arena, so realistic peak for a 1000-sim search is ~1000 × branching
/// factor ≈ 50k nodes.  A hint of 64 avoids mid-search Vec reallocation.
const CHILD_CAPACITY_HINT: usize = 64;

/// Simple XOR-shift RNG step, returns a uniform float in [0, 1).
fn next_uniform(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x as f32 / u64::MAX as f32
}

// ── MCTSNode ───────────────────────────────────────────────────────────

/// A single node in the MCTS tree, stored contiguously in the engine's arena.
///
/// Each node tracks its parent, the move that led to it, policy prior,
/// visit count, total accumulated value, and child range within the arena.
/// The arena layout is:
///
/// ```text
/// arena[0]              = root node
/// arena[1..N]           = root's children (contiguous)
/// arena[N+1..]          = grandchildren, great-grandchildren, etc.
/// ```
///
/// Children of a given node are always stored contiguously, so a node's
/// children span `children_start .. children_start + children_count`.
#[derive(Clone, Copy)]
pub(crate) struct MCTSNode {
    /// Arena index of the parent node. `NO_PARENT` for the root.
    parent: u32,
    /// The move `(q, r)` that led to this node, relative to board origin.
    action: (i32, i32),
    /// Neural network policy prior for this node (probability of selecting
    /// the move that leads here, as assigned by the parent expansion).
    prior: f32,
    /// Total visit count (includes virtual-loss visits while the node is
    /// in-flight during batch selection).
    pub(crate) visit_count: u32,
    /// Sum of all backpropagated values from this node's perspective.
    ///
    /// Because values are stored from each node's own player perspective,
    /// `total_value / visit_count` gives the node's Q-value directly.
    total_value: f32,
    /// Arena index where this node's children begin (if expanded).
    children_start: u32,
    /// Number of children allocated in the arena.
    children_count: u16,
    /// Player whose turn it is at this node (0 or 1). 255 = unset.
    player: u8,
    /// Whether children have been allocated for this node.
    is_expanded: bool,
}

impl MCTSNode {
    /// Create a new unexpanded node.
    fn new(parent: u32, action: (i32, i32), prior: f32) -> Self {
        Self {
            parent,
            action,
            prior,
            visit_count: 0,
            total_value: 0.0,
            children_start: 0,
            children_count: 0,
            player: 255,
            is_expanded: false,
        }
    }

    /// Average Q-value from the perspective of the player at this node.
    #[inline(always)]
    fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f32
        }
    }
}

// ── Pending leaf info ──────────────────────────────────────────────────

/// Temporary record of a leaf selected during `select_leaves`, stored
/// between selection and expansion so that `expand_and_backprop` knows
/// which nodes to expand and which search paths to update.
struct PendingLeaf {
    /// Arena index of the leaf node.
    node_idx: u32,
    /// Sequence of arena indices from root to this leaf (inclusive).
    ///
    /// Used to apply / remove virtual loss and to backpropagate values.
    search_path: SmallVec<[u32; 32]>,
    /// Whether this leaf represents a terminal game position.
    is_terminal: bool,
    /// Game outcome from the leaf node's player perspective (+1 win, -1 loss).
    terminal_value: f32,
    /// Tensor spatial offset: the board coordinate that maps to tensor index (0, 0).
    offset_q: i32,
    /// Tensor spatial offset for the r-axis.
    offset_r: i32,
    /// Legal moves at this leaf position (needed for node expansion).
    legal_moves: Vec<Hex>,
    /// Packed move history at this leaf as `(player, q, r)` little-endian i32 triples.
    move_history: Vec<u8>,
}

pub struct RootInit {
    pub tensor: Vec<f32>,
    pub offset_q: i32,
    pub offset_r: i32,
    pub legal_moves: Vec<Hex>,
    pub root_generation: u64,
}

pub struct LeafBatch<'a> {
    pub tensors: &'a [f32],
    pub non_terminal_count: u32,
    pub root_generation: u64,
    pub batch_generation: u64,
}

fn pack_move_history(game: &HexGameState) -> Vec<u8> {
    let hist = game.move_history();
    let mut buf = Vec::with_capacity(hist.len() * 12);
    for rec in hist {
        buf.extend_from_slice(&(rec.player() as i32).to_le_bytes());
        buf.extend_from_slice(&rec.cell().q.to_le_bytes());
        buf.extend_from_slice(&rec.cell().r.to_le_bytes());
    }
    buf
}

// ── Policy gathering ───────────────────────────────────────────────────

const PRIOR_SOURCE_SPARSE: u8 = 1;
const PRIOR_SOURCE_DENSE: u8 = 2;
const PRIOR_SOURCE_DEFAULT: u8 = 3;
const PRIOR_SOURCE_PAIR: u8 = 4;

#[derive(Debug, Clone, Copy, Default)]
pub struct PriorSourceTelemetry {
    pub root_total_count: u32,
    pub root_sparse_count: u32,
    pub root_dense_count: u32,
    pub root_default_count: u32,
    pub leaf_total_count: u32,
    pub leaf_sparse_count: u32,
    pub leaf_dense_count: u32,
    pub leaf_default_count: u32,
    pub root_pair_count: u32,
    pub leaf_pair_count: u32,
    pub root_sparse_candidate_count: u32,
    pub leaf_sparse_candidate_count: u32,
    pub root_pair_candidate_count: u32,
    pub leaf_expansion_count: u32,
}

fn dense_source(q: i32, r: i32, offset_q: i32, offset_r: i32) -> u8 {
    let gi = q - offset_q;
    let gj = r - offset_r;
    if (0..BOARD_SIZE).contains(&gi) && (0..BOARD_SIZE).contains(&gj) {
        PRIOR_SOURCE_DENSE
    } else {
        PRIOR_SOURCE_DEFAULT
    }
}

fn summarize_sources(sources: &[u8]) -> (u32, u32, u32, u32, u32) {
    let mut sparse = 0u32;
    let mut dense = 0u32;
    let mut default = 0u32;
    let mut pair = 0u32;
    for &source in sources {
        match source {
            PRIOR_SOURCE_SPARSE => sparse += 1,
            PRIOR_SOURCE_DENSE => dense += 1,
            PRIOR_SOURCE_DEFAULT => default += 1,
            PRIOR_SOURCE_PAIR => pair += 1,
            _ => {}
        }
    }
    (sources.len() as u32, sparse, dense, default, pair)
}

fn count_prior_source(sources: &[u8], wanted: u8) -> u32 {
    sources.iter().filter(|&&source| source == wanted).count() as u32
}

fn validate_policy_slice(policy_logits: &[f32], expected: usize) -> Result<(), MCTSError> {
    if policy_logits.len() != expected {
        return Err(MCTSError::WrongPolicyLength {
            expected,
            actual: policy_logits.len(),
        });
    }
    if let Some(index) = policy_logits.iter().position(|value| !value.is_finite()) {
        return Err(MCTSError::NonFinitePolicy { index });
    }
    Ok(())
}

fn validate_values(values: &[f32], expected: usize) -> Result<(), MCTSError> {
    if values.len() != expected {
        return Err(MCTSError::WrongValueLength {
            expected,
            actual: values.len(),
        });
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(MCTSError::NonFiniteValue { index });
    }
    Ok(())
}

fn validate_root_value(value: f32) -> Result<(), MCTSError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(MCTSError::NonFiniteValue { index: 0 })
    }
}

/// Gather policy logits for legal moves and normalize them with softmax.
///
/// For each legal move, looks up the corresponding logit in the flat
/// `policy_logits` array (shape BOARD_AREA) using the board-to-tensor
/// offset, then applies numerically-stable softmax.
///
/// Moves that fall outside the tensor window receive a default logit of -10.0.
///
/// The caller must supply reusable `raw` and `priors` buffers to avoid
/// per-expansion heap allocation.
fn gather_policy(
    moves: &[Hex],
    policy_logits: &[f32],
    offset_q: i32,
    offset_r: i32,
    raw: &mut Vec<f32>,
    priors: &mut Vec<f32>,
) {
    let n = moves.len();
    raw.clear();
    raw.resize(n, -10.0f32);

    for (i, h) in moves.iter().enumerate() {
        let gi = h.q - offset_q;
        let gj = h.r - offset_r;
        if (0..BOARD_SIZE).contains(&gi) && (0..BOARD_SIZE).contains(&gj) {
            let flat = (gi as usize) * (BOARD_SIZE as usize) + gj as usize;
            raw[i] = policy_logits[flat];
        }
    }

    let max_val = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in raw.iter_mut() {
        *v = (*v - max_val).exp();
    }
    let sum: f32 = raw.iter().sum();

    priors.clear();
    if sum > 0.0 {
        priors.extend(raw.iter().map(|&e| e / sum));
    } else {
        // Uniform fallback if all logits are identical or invalid.
        priors.extend(std::iter::repeat_n(1.0 / moves.len() as f32, moves.len()));
    }
}

fn gather_dense_sources(moves: &[Hex], offset_q: i32, offset_r: i32, sources: &mut Vec<u8>) {
    sources.clear();
    sources.extend(
        moves
            .iter()
            .map(|h| dense_source(h.q, h.r, offset_q, offset_r)),
    );
}

fn softmax_in_place(raw: &mut [f32], priors: &mut Vec<f32>) {
    let max_val = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in raw.iter_mut() {
        *v = (*v - max_val).exp();
    }
    let sum: f32 = raw.iter().sum();
    priors.clear();
    if sum > 0.0 && sum.is_finite() {
        priors.extend(raw.iter().map(|&e| e / sum));
    } else if !raw.is_empty() {
        priors.extend(std::iter::repeat_n(1.0 / raw.len() as f32, raw.len()));
    }
}

fn sparse_logit_for(
    q: i32,
    r: i32,
    sparse_actions: &[(i32, i32)],
    sparse_logits: &[f32],
) -> Option<f32> {
    sparse_actions
        .iter()
        .position(|&(sq, sr)| sq == q && sr == r)
        .and_then(|idx| sparse_logits.get(idx).copied())
}

#[allow(clippy::too_many_arguments)]
fn gather_policy_with_sparse(
    moves: &[Hex],
    policy_logits: &[f32],
    offset_q: i32,
    offset_r: i32,
    sparse_actions: &[(i32, i32)],
    sparse_logits: &[f32],
    stage: u8,
    sparse_mix: f32,
    sparse_source: u8,
    raw: &mut Vec<f32>,
    priors: &mut Vec<f32>,
    sources: &mut Vec<u8>,
) {
    if stage == 0 || sparse_actions.is_empty() || sparse_logits.is_empty() {
        gather_policy(moves, policy_logits, offset_q, offset_r, raw, priors);
        gather_dense_sources(moves, offset_q, offset_r, sources);
        return;
    }

    raw.clear();
    raw.resize(moves.len(), -10.0f32);
    sources.clear();
    sources.resize(moves.len(), PRIOR_SOURCE_DEFAULT);
    if stage == 2 {
        for (i, h) in moves.iter().enumerate() {
            if let Some(logit) = sparse_logit_for(h.q, h.r, sparse_actions, sparse_logits) {
                raw[i] = logit;
                sources[i] = sparse_source;
            } else {
                let gi = h.q - offset_q;
                let gj = h.r - offset_r;
                if (0..BOARD_SIZE).contains(&gi) && (0..BOARD_SIZE).contains(&gj) {
                    let flat = (gi as usize) * (BOARD_SIZE as usize) + gj as usize;
                    raw[i] = policy_logits[flat];
                    sources[i] = PRIOR_SOURCE_DENSE;
                }
            }
        }
        softmax_in_place(raw, priors);
        return;
    }

    let mut dense_priors = Vec::new();
    gather_policy(
        moves,
        policy_logits,
        offset_q,
        offset_r,
        raw,
        &mut dense_priors,
    );

    raw.clear();
    raw.resize(moves.len(), f32::NEG_INFINITY);
    for (i, h) in moves.iter().enumerate() {
        if let Some(logit) = sparse_logit_for(h.q, h.r, sparse_actions, sparse_logits) {
            raw[i] = logit;
        }
    }
    let mut sparse_priors = Vec::new();
    if raw.iter().any(|v| v.is_finite()) {
        let max_val = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in raw.iter_mut() {
            if v.is_finite() {
                *v = (*v - max_val).exp();
            } else {
                *v = 0.0;
            }
        }
        let sum: f32 = raw.iter().sum();
        if sum > 0.0 {
            sparse_priors.extend(raw.iter().map(|&e| e / sum));
        }
    }
    if sparse_priors.len() != moves.len() {
        sparse_priors.resize(moves.len(), 0.0);
    }
    let sparse_total: f32 = sparse_priors.iter().sum();
    if sparse_total <= 0.0 {
        priors.clear();
        priors.extend(dense_priors);
        gather_dense_sources(moves, offset_q, offset_r, sources);
        return;
    }

    let mix = sparse_mix.clamp(0.0, 1.0);
    priors.clear();
    priors.extend(
        dense_priors
            .iter()
            .zip(sparse_priors.iter())
            .map(|(&d, &s)| (1.0 - mix) * d + mix * s),
    );
    let total: f32 = priors.iter().sum();
    if total > 0.0 {
        for p in priors.iter_mut() {
            *p /= total;
        }
    }

    sources.clear();
    sources.extend(moves.iter().zip(sparse_priors.iter()).map(|(h, &s)| {
        if s > 0.0 {
            sparse_source
        } else {
            dense_source(h.q, h.r, offset_q, offset_r)
        }
    }));
}

// ── MCTSEngine ─────────────────────────────────────────────────────────

pub struct MCTSEngine {
    /// Contiguous storage for all MCTS nodes.
    ///
    /// Nodes are never deleted; `re_root` simply moves the root pointer.
    pub(crate) arena: Vec<MCTSNode>,
    /// Internal game state used during tree traversal.
    ///
    /// Moves are applied during `select_leaves` and undone before the
    /// function returns, so the game state is always valid for the root.
    game: HexGameState,
    /// Arena index of the current root node.
    pub(crate) root_idx: u32,
    /// Controls how far from existing stones legal moves are generated.
    near_radius: i32,
    /// Enables threat-based move filtering at the root.
    ///
    /// When true, `init_root` calls the encoder with threat constraints.
    /// Internal nodes never use threat constraints (too expensive).
    constrain_threats: bool,
    /// Base exploration constant for PUCT.
    c_puct: f32,
    /// Dynamic c_puct scaling constant (AlphaZero/KataGo formula).
    ///
    /// ```text
    /// effective_c_puct = c_puct + ln((N_parent + c_puct_init) / c_puct_init)
    /// ```
    ///
    /// Set to 0.0 to disable (use static c_puct).
    c_puct_init: f32,
    /// Number of simulations already completed.
    sims_done: u32,
    /// Total number of simulations to perform.
    num_simulations: u32,
    /// Pre-allocated output buffer for `select_leaves`.  Holds packed tensors
    /// for all non-terminal leaves in the current batch.
    batch_buf: Vec<f32>,
    /// Pending leaves from the most recent `select_leaves` call, consumed by
    /// the next `expand_and_backprop` call.
    pending: Vec<PendingLeaf>,
    /// Reusable scratch buffer for `gather_policy` (raw logits before softmax).
    scratch_raw: Vec<f32>,
    /// Reusable scratch buffer for `gather_policy` (normalized priors).
    scratch_priors: Vec<f32>,
    /// Reusable scratch buffer for prior-source labels aligned with priors.
    scratch_sources: Vec<u8>,
    /// Prior-source labels for the current root children.
    root_prior_sources: Vec<u8>,
    root_generation: u64,
    batch_generation: u64,
    last_root_init_generation: Option<u64>,
    pending_batch_generation: Option<u64>,
    root_sparse_candidate_count: u32,
    root_pair_candidate_count: u32,
    leaf_sparse_candidate_count: u32,
    leaf_expansion_count: u32,
    leaf_source_total: u32,
    leaf_source_sparse: u32,
    leaf_source_dense: u32,
    leaf_source_default: u32,
    leaf_source_pair: u32,
    /// Reusable buffer for `encode_board_into` live-cells channels.
    hot_buf: Vec<Hex>,
    legal_buf: Vec<Hex>,
    seed: u64,
}

type TreeNodeStates = (Vec<f32>, Vec<Vec<(u8, i32, i32)>>, usize);

impl MCTSEngine {
    // ── Construction ───────────────────────────────────────────────────

    /// Create a new MCTS engine for the given game state.
    ///
    /// `num_simulations` is the total number of MCTS rollouts to perform.
    /// `c_puct` is the base exploration constant.
    /// `near_radius` controls how far from existing stones legal moves are
    /// generated (clamped to `PLACEMENT_RADIUS`).
    /// `constrain_threats` enables threat-based move filtering at the root.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        game: HexGameState,
        num_simulations: u32,
        c_puct: f32,
        near_radius: i32,
        constrain_threats: bool,
        seed: u64,
    ) -> Self {
        Self::with_arena_sim_hint(
            game,
            num_simulations,
            num_simulations,
            c_puct,
            near_radius,
            constrain_threats,
            19652.0,
            seed,
        )
    }

    /// Construct with an explicit arena simulation-count hint.
    ///
    /// Under subtree reuse, the same arena is shared across both placements of
    /// a turn, so the p1 sim count underestimates the true node budget.  Pass
    /// `p1_sims + p2_sims` here to avoid mid-search Vec reallocation.
    #[allow(clippy::too_many_arguments)]
    pub fn with_arena_sim_hint(
        game: HexGameState,
        num_simulations: u32,
        arena_sim_hint: u32,
        c_puct: f32,
        near_radius: i32,
        constrain_threats: bool,
        c_puct_init: f32,
        seed: u64,
    ) -> Self {
        let mut arena = Vec::with_capacity(arena_sim_hint as usize * CHILD_CAPACITY_HINT + 64);
        // Root node has no parent and a dummy action.
        arena.push(MCTSNode::new(NO_PARENT, (0, 0), 1.0));

        Self {
            arena,
            game,
            root_idx: 0,
            near_radius,
            constrain_threats,
            c_puct,
            c_puct_init,
            sims_done: 0,
            num_simulations,
            batch_buf: Vec::new(),
            pending: Vec::new(),
            scratch_raw: Vec::new(),
            scratch_priors: Vec::new(),
            scratch_sources: Vec::new(),
            root_prior_sources: Vec::new(),
            root_generation: 0,
            batch_generation: 0,
            last_root_init_generation: None,
            pending_batch_generation: None,
            root_sparse_candidate_count: 0,
            root_pair_candidate_count: 0,
            leaf_sparse_candidate_count: 0,
            leaf_expansion_count: 0,
            leaf_source_total: 0,
            leaf_source_sparse: 0,
            leaf_source_dense: 0,
            leaf_source_default: 0,
            leaf_source_pair: 0,
            hot_buf: Vec::new(),
            legal_buf: Vec::new(),
            seed,
        }
    }

    fn reset_prior_telemetry(&mut self) {
        self.root_prior_sources.clear();
        self.root_sparse_candidate_count = 0;
        self.root_pair_candidate_count = 0;
        self.leaf_sparse_candidate_count = 0;
        self.leaf_expansion_count = 0;
        self.leaf_source_total = 0;
        self.leaf_source_sparse = 0;
        self.leaf_source_dense = 0;
        self.leaf_source_default = 0;
        self.leaf_source_pair = 0;
    }

    fn record_leaf_prior_sources(&mut self, sparse_candidate_count: usize) {
        let (total, sparse, dense, default, pair) = summarize_sources(&self.scratch_sources);
        self.leaf_source_total += total;
        self.leaf_source_sparse += sparse;
        self.leaf_source_dense += dense;
        self.leaf_source_default += default;
        self.leaf_source_pair += pair;
        self.leaf_sparse_candidate_count += sparse_candidate_count as u32;
        self.leaf_expansion_count += 1;
    }

    fn validate_root_generation(&self, received: u64) -> Result<(), MCTSError> {
        let Some(expected) = self.last_root_init_generation else {
            return Err(MCTSError::InvalidRootState(
                "root must be initialized before expansion",
            ));
        };
        if received != expected || received != self.root_generation {
            return Err(MCTSError::StaleRootToken { expected, received });
        }
        Ok(())
    }

    fn validate_batch_generation(&self, received: u64) -> Result<(), MCTSError> {
        let Some(expected) = self.pending_batch_generation else {
            return Err(MCTSError::InvalidBatchState(
                "no selected leaf batch is pending backpropagation",
            ));
        };
        if received != expected {
            return Err(MCTSError::StaleBatchToken { expected, received });
        }
        Ok(())
    }

    // ── Root initialization ────────────────────────────────────────────

    /// Encode the root board state and return tensor data for GPU evaluation.
    ///
    /// Returns `Ok(None)` if the game is already over.
    pub fn init_root(&mut self) -> Result<Option<RootInit>, MCTSError> {
        if self.game.is_over() {
            return Ok(None);
        }

        // Allocate a local tensor buffer and encode directly into it.
        let mut tensor = vec![0.0f32; TENSOR_SIZE];
        self.legal_buf.clear();
        let (oq, or_) = encoder::encode_board_into(
            &self.game,
            self.near_radius,
            self.constrain_threats,
            &mut tensor,
            &mut self.hot_buf,
            &mut self.legal_buf,
        );

        self.root_generation = self.root_generation.wrapping_add(1);
        self.last_root_init_generation = Some(self.root_generation);

        Ok(Some(RootInit {
            tensor,
            offset_q: oq,
            offset_r: or_,
            legal_moves: self.legal_buf.clone(),
            root_generation: self.root_generation,
        }))
    }

    /// Expand the root node with policy output from the GPU.
    ///
    /// `policy_logits` is a flat slice of length `BOARD_AREA` (1089).
    /// `value` is ignored for root expansion (the root's value comes from
    /// subsequent backpropagation).
    /// `offset_q` and `offset_r` map board coordinates to tensor indices.
    /// `legal` is the list of legal moves at the root.
    pub fn expand_root(
        &mut self,
        root_generation: u64,
        policy_logits: &[f32],
        value: f32,
        offset_q: i32,
        offset_r: i32,
        legal: &[Hex],
    ) -> Result<(), MCTSError> {
        self.validate_root_generation(root_generation)?;
        validate_policy_slice(policy_logits, BOARD_AREA)?;
        validate_root_value(value)?;
        self.expand_root_impl(policy_logits, offset_q, offset_r, legal);
        Ok(())
    }

    fn expand_root_impl(
        &mut self,
        policy_logits: &[f32],
        offset_q: i32,
        offset_r: i32,
        legal: &[Hex],
    ) {
        self.reset_prior_telemetry();
        let player = self.game.current_player();
        self.expand_node(
            self.root_idx,
            legal,
            player,
            policy_logits,
            offset_q,
            offset_r,
        );
        self.root_prior_sources.clear();
        self.root_prior_sources
            .extend_from_slice(&self.scratch_sources);
        self.arena[self.root_idx as usize].player = player;
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn expand_root_with_sparse_priors(
        &mut self,
        root_generation: u64,
        policy_logits: &[f32],
        value: f32,
        offset_q: i32,
        offset_r: i32,
        legal: &[Hex],
        sparse_actions: &[(i32, i32)],
        sparse_logits: &[f32],
        stage: u8,
        sparse_mix: f32,
    ) -> Result<(), MCTSError> {
        self.validate_root_generation(root_generation)?;
        validate_policy_slice(policy_logits, BOARD_AREA)?;
        validate_root_value(value)?;
        if let Some(index) = sparse_logits.iter().position(|logit| !logit.is_finite()) {
            return Err(MCTSError::NonFiniteSparsePolicy { leaf: 0, index });
        }
        self.expand_root_with_sparse_priors_impl(
            policy_logits,
            offset_q,
            offset_r,
            legal,
            sparse_actions,
            sparse_logits,
            stage,
            sparse_mix,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_root_with_sparse_priors_impl(
        &mut self,
        policy_logits: &[f32],
        offset_q: i32,
        offset_r: i32,
        legal: &[Hex],
        sparse_actions: &[(i32, i32)],
        sparse_logits: &[f32],
        stage: u8,
        sparse_mix: f32,
    ) {
        self.reset_prior_telemetry();
        if legal.is_empty() {
            return;
        }
        let player = self.game.current_player();
        self.root_sparse_candidate_count = sparse_actions.len().min(sparse_logits.len()) as u32;
        gather_policy_with_sparse(
            legal,
            policy_logits,
            offset_q,
            offset_r,
            sparse_actions,
            sparse_logits,
            stage,
            sparse_mix,
            PRIOR_SOURCE_SPARSE,
            &mut self.scratch_raw,
            &mut self.scratch_priors,
            &mut self.scratch_sources,
        );
        self.expand_node_with_priors(self.root_idx, legal, player);
        self.root_prior_sources.clear();
        self.root_prior_sources
            .extend_from_slice(&self.scratch_sources);
        self.arena[self.root_idx as usize].player = player;
    }

    fn expand_root_with_global_priors_impl(
        &mut self,
        legal: &[Hex],
        global_actions: &[(i32, i32)],
        global_logits: &[f32],
        _value: f32,
    ) -> Result<(), MCTSError> {
        self.reset_prior_telemetry();
        if legal.len() != global_actions.len() {
            return Err(MCTSError::InvalidPrior(format!(
                "global prior legal row count mismatch: legal={} global_actions={}",
                legal.len(),
                global_actions.len()
            )));
        }
        if global_logits.len() < global_actions.len() {
            return Err(MCTSError::InvalidPrior(format!(
                "global prior logits length {} is smaller than legal rows {}",
                global_logits.len(),
                global_actions.len()
            )));
        }
        for (idx, (legal_hex, &(q, r))) in legal.iter().zip(global_actions.iter()).enumerate() {
            if legal_hex.q != q || legal_hex.r != r {
                return Err(MCTSError::InvalidPrior(format!(
                    "global prior legal row mismatch at {idx}: Rust legal=({}, {}) graph legal=({}, {})",
                    legal_hex.q, legal_hex.r, q, r
                )));
            }
        }
        self.scratch_raw.clear();
        self.scratch_raw
            .extend(global_logits.iter().take(global_actions.len()).copied());
        if self.scratch_raw.iter().any(|value| !value.is_finite()) {
            return Err(MCTSError::InvalidPrior(
                "global prior logits contain non-finite values".to_string(),
            ));
        }
        softmax_in_place(&mut self.scratch_raw, &mut self.scratch_priors);
        if self.scratch_priors.len() != legal.len() {
            return Err(MCTSError::InvalidPrior(
                "global prior logits could not be normalized".to_string(),
            ));
        }
        self.scratch_sources.clear();
        self.scratch_sources
            .extend(std::iter::repeat_n(PRIOR_SOURCE_SPARSE, legal.len()));
        let player = self.game.current_player();
        self.expand_node_with_priors(self.root_idx, legal, player);
        self.root_prior_sources.clear();
        self.root_prior_sources
            .extend_from_slice(&self.scratch_sources);
        self.root_sparse_candidate_count = legal.len() as u32;
        self.arena[self.root_idx as usize].player = player;
        Ok(())
    }

    pub fn expand_root_with_global_priors(
        &mut self,
        root_generation: u64,
        legal: &[Hex],
        global_actions: &[(i32, i32)],
        global_logits: &[f32],
        value: f32,
    ) -> Result<(), MCTSError> {
        self.validate_root_generation(root_generation)?;
        validate_root_value(value)?;
        self.expand_root_with_global_priors_impl(legal, global_actions, global_logits, value)
    }

    /// Blend joint pair-action logits into root first-placement priors.
    ///
    /// `pair_actions` stores unordered pairs `(q1, r1, q2, r2)` keyed by
    /// global coordinates.  Each coordinate must be a current root child and
    /// duplicate cells are illegal.  Pair logits are normalized over pair rows,
    /// then incident pair mass is summed back onto first-placement children.
    pub fn apply_root_pair_priors(
        &mut self,
        pair_actions: &[(i32, i32, i32, i32)],
        pair_logits: &[f32],
        pair_mix: f32,
    ) -> Result<(), MCTSError> {
        if self.game.placements_remaining() < 2 {
            self.root_pair_candidate_count = 0;
            return Ok(());
        }
        let root = &self.arena[self.root_idx as usize];
        let start = root.children_start as usize;
        let count = root.children_count as usize;
        if count == 0 {
            return Err(MCTSError::InvalidPrior(
                "root must be expanded before applying pair priors".to_string(),
            ));
        }
        if pair_logits.len() < pair_actions.len() {
            return Err(MCTSError::InvalidPrior(format!(
                "pair_logits length {} is smaller than pair_actions rows {}",
                pair_logits.len(),
                pair_actions.len()
            )));
        }
        if pair_actions.is_empty() {
            self.root_pair_candidate_count = 0;
            return Ok(());
        }

        let root_actions: Vec<(i32, i32)> = (0..count)
            .map(|idx| {
                let child = &self.arena[start + idx];
                (child.action.0, child.action.1)
            })
            .collect();

        let mut valid_pairs: Vec<(usize, usize, f32)> = Vec::with_capacity(pair_actions.len());
        let mut seen_pairs = std::collections::HashSet::with_capacity(pair_actions.len());
        for (row, &(q1, r1, q2, r2)) in pair_actions.iter().enumerate() {
            if q1 == q2 && r1 == r2 {
                return Err(MCTSError::InvalidPrior(format!(
                    "duplicate coordinates are illegal for pair policy: ({q1}, {r1})"
                )));
            }
            let Some(first_idx) = root_actions.iter().position(|&(q, r)| q == q1 && r == r1) else {
                return Err(MCTSError::InvalidPrior(format!(
                    "pair policy target contains illegal first action: ({q1}, {r1})"
                )));
            };
            let Some(second_idx) = root_actions.iter().position(|&(q, r)| q == q2 && r == r2)
            else {
                return Err(MCTSError::InvalidPrior(format!(
                    "pair policy target contains illegal second action: ({q2}, {r2})"
                )));
            };
            let logit = pair_logits[row];
            if !logit.is_finite() {
                return Err(MCTSError::InvalidPrior(format!(
                    "pair policy logit at row {row} is not finite"
                )));
            }
            let key = if first_idx <= second_idx {
                (first_idx, second_idx)
            } else {
                (second_idx, first_idx)
            };
            if !seen_pairs.insert(key) {
                return Err(MCTSError::InvalidPrior(format!(
                    "duplicate unordered pair policy row at {row}: ({q1}, {r1}) <-> ({q2}, {r2})"
                )));
            }
            valid_pairs.push((first_idx, second_idx, logit));
        }

        let max_logit = valid_pairs
            .iter()
            .map(|(_, _, logit)| *logit)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut pair_mass = vec![0.0f32; count];
        let mut denom = 0.0f32;
        for &(_first_idx, _second_idx, logit) in &valid_pairs {
            denom += (logit - max_logit).exp();
        }
        if denom <= 0.0 || !denom.is_finite() {
            return Err(MCTSError::InvalidPrior(
                "pair policy logits could not be normalized".to_string(),
            ));
        }
        for &(first_idx, second_idx, logit) in &valid_pairs {
            let p = (logit - max_logit).exp() / denom;
            pair_mass[first_idx] += p;
            pair_mass[second_idx] += p;
        }
        let mass_total: f32 = pair_mass.iter().sum();
        if mass_total <= 0.0 {
            return Err(MCTSError::InvalidPrior(
                "pair policy produced zero incident action mass".to_string(),
            ));
        }
        for mass in &mut pair_mass {
            *mass /= mass_total;
        }

        let mix = pair_mix.clamp(0.0, 1.0);
        let mut total_prior = 0.0f32;
        for (idx, pair_prior) in pair_mass.iter().enumerate() {
            let child = &mut self.arena[start + idx];
            child.prior = (1.0 - mix) * child.prior + mix * *pair_prior;
            total_prior += child.prior;
        }
        if total_prior > 0.0 {
            for idx in 0..count {
                self.arena[start + idx].prior /= total_prior;
            }
        }
        if self.root_prior_sources.len() != count {
            self.root_prior_sources.resize(count, PRIOR_SOURCE_DEFAULT);
        }
        for (idx, pair_prior) in pair_mass.iter().enumerate() {
            if *pair_prior > 0.0 && mix > 0.0 {
                self.root_prior_sources[idx] = PRIOR_SOURCE_PAIR;
            }
        }
        self.root_pair_candidate_count = valid_pairs.len() as u32;
        Ok(())
    }

    /// Blend first-placement pair-policy logits into root child priors.
    ///
    /// The logits are keyed by the current root legal row table. This is the
    /// direct all-legal `policy_pair_first` head; joint pair logits can still be
    /// applied afterwards with `apply_root_pair_priors`.
    pub fn apply_root_pair_first_priors(
        &mut self,
        action_logits: &[f32],
        pair_mix: f32,
    ) -> Result<(), MCTSError> {
        if self.game.placements_remaining() < 2 {
            return Ok(());
        }
        let root = &self.arena[self.root_idx as usize];
        let start = root.children_start as usize;
        let count = root.children_count as usize;
        if count == 0 {
            return Err(MCTSError::InvalidPrior(
                "root must be expanded before applying pair-first priors".to_string(),
            ));
        }
        if action_logits.len() < count {
            return Err(MCTSError::InvalidPrior(format!(
                "pair-first logits length {} is smaller than root legal rows {}",
                action_logits.len(),
                count
            )));
        }
        let logits = &action_logits[..count];
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(MCTSError::InvalidPrior(
                "pair-first logits contain non-finite values".to_string(),
            ));
        }
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut mass = vec![0.0f32; count];
        let mut denom = 0.0f32;
        for (idx, logit) in logits.iter().enumerate() {
            let p = (*logit - max_logit).exp();
            mass[idx] = p;
            denom += p;
        }
        if denom <= 0.0 || !denom.is_finite() {
            return Err(MCTSError::InvalidPrior(
                "pair-first logits could not be normalized".to_string(),
            ));
        }
        for value in &mut mass {
            *value /= denom;
        }
        let mix = pair_mix.clamp(0.0, 1.0);
        let mut total_prior = 0.0f32;
        for (idx, pair_prior) in mass.iter().enumerate() {
            let child = &mut self.arena[start + idx];
            child.prior = (1.0 - mix) * child.prior + mix * *pair_prior;
            total_prior += child.prior;
        }
        if total_prior > 0.0 {
            for idx in 0..count {
                self.arena[start + idx].prior /= total_prior;
            }
        }
        if self.root_prior_sources.len() != count {
            self.root_prior_sources.resize(count, PRIOR_SOURCE_DEFAULT);
        }
        if mix > 0.0 {
            for source in &mut self.root_prior_sources {
                *source = PRIOR_SOURCE_PAIR;
            }
        }
        self.root_pair_candidate_count = count as u32;
        Ok(())
    }

    /// Blend conditional pair-action logits into second-placement root priors.
    ///
    /// This is used after re-rooting from the first placement of a normal turn.
    /// `pair_actions` stores ordered `(first_q, first_r, second_q, second_r)`
    /// rows.  The first coordinate must be the immediately preceding placement
    /// by the current player; the second coordinate must be a current root child.
    pub fn apply_root_pair_second_priors(
        &mut self,
        pair_actions: &[(i32, i32, i32, i32)],
        pair_logits: &[f32],
        pair_mix: f32,
    ) -> Result<(), MCTSError> {
        if self.game.placements_remaining() != 1 {
            self.root_pair_candidate_count = 0;
            return Ok(());
        }
        let Some(last_move) = self.game.move_history().last() else {
            return Err(MCTSError::InvalidPrior(
                "second-placement pair prior requires a previous placement".to_string(),
            ));
        };
        if last_move.player() != self.game.current_player() {
            return Err(MCTSError::InvalidPrior(
                "previous placement does not belong to the current player".to_string(),
            ));
        }
        let first_cell = last_move.cell();
        let root = &self.arena[self.root_idx as usize];
        let start = root.children_start as usize;
        let count = root.children_count as usize;
        if count == 0 {
            return Err(MCTSError::InvalidPrior(
                "root must be expanded before applying second-placement pair priors".to_string(),
            ));
        }
        if pair_logits.len() < pair_actions.len() {
            return Err(MCTSError::InvalidPrior(format!(
                "pair_logits length {} is smaller than pair_actions rows {}",
                pair_logits.len(),
                pair_actions.len()
            )));
        }
        if pair_actions.is_empty() {
            self.root_pair_candidate_count = 0;
            return Ok(());
        }

        let root_actions: Vec<(i32, i32)> = (0..count)
            .map(|idx| {
                let child = &self.arena[start + idx];
                (child.action.0, child.action.1)
            })
            .collect();

        let mut valid_pairs: Vec<(usize, f32)> = Vec::with_capacity(pair_actions.len());
        for (row, &(q1, r1, q2, r2)) in pair_actions.iter().enumerate() {
            if q1 == q2 && r1 == r2 {
                return Err(MCTSError::InvalidPrior(format!(
                    "duplicate coordinates are illegal for pair policy: ({q1}, {r1})"
                )));
            }
            if q1 != first_cell.q || r1 != first_cell.r {
                return Err(MCTSError::InvalidPrior(format!(
                    "pair policy first action ({q1}, {r1}) does not match current turn first placement ({}, {})",
                    first_cell.q, first_cell.r
                )));
            }
            let Some(second_idx) = root_actions.iter().position(|&(q, r)| q == q2 && r == r2)
            else {
                return Err(MCTSError::InvalidPrior(format!(
                    "pair policy target contains illegal second action: ({q2}, {r2})"
                )));
            };
            let logit = pair_logits[row];
            if !logit.is_finite() {
                return Err(MCTSError::InvalidPrior(format!(
                    "pair policy logit at row {row} is not finite"
                )));
            }
            valid_pairs.push((second_idx, logit));
        }

        let max_logit = valid_pairs
            .iter()
            .map(|(_, logit)| *logit)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for &(_, logit) in &valid_pairs {
            denom += (logit - max_logit).exp();
        }
        if denom <= 0.0 || !denom.is_finite() {
            return Err(MCTSError::InvalidPrior(
                "pair policy logits could not be normalized".to_string(),
            ));
        }
        let mut pair_mass = vec![0.0f32; count];
        for &(second_idx, logit) in &valid_pairs {
            pair_mass[second_idx] += (logit - max_logit).exp() / denom;
        }

        let mix = pair_mix.clamp(0.0, 1.0);
        let mut total_prior = 0.0f32;
        for (idx, pair_prior) in pair_mass.iter().enumerate() {
            let child = &mut self.arena[start + idx];
            child.prior = (1.0 - mix) * child.prior + mix * *pair_prior;
            total_prior += child.prior;
        }
        if total_prior > 0.0 {
            for idx in 0..count {
                self.arena[start + idx].prior /= total_prior;
            }
        }
        if self.root_prior_sources.len() != count {
            self.root_prior_sources.resize(count, PRIOR_SOURCE_DEFAULT);
        }
        for (idx, pair_prior) in pair_mass.iter().enumerate() {
            if *pair_prior > 0.0 && mix > 0.0 {
                self.root_prior_sources[idx] = PRIOR_SOURCE_PAIR;
            }
        }
        self.root_pair_candidate_count = valid_pairs.len() as u32;
        Ok(())
    }

    /// Add Dirichlet noise to root priors for exploration.
    ///
    /// `noise` must have at least as many elements as root children.  Noise
    /// values are normalized over the current root children before blending.
    /// `noise_fraction` controls the blend between original prior and noise
    /// (0.0 = no noise, 1.0 = pure noise).
    pub fn add_dirichlet_noise(
        &mut self,
        noise: &[f32],
        noise_fraction: f32,
    ) -> Result<(), MCTSError> {
        let count = self.arena[self.root_idx as usize].children_count as usize;
        if noise.len() < count {
            return Err(MCTSError::WrongNoiseLength {
                expected_at_least: count,
                actual: noise.len(),
            });
        }
        let root = &self.arena[self.root_idx as usize];
        let start = root.children_start as usize;
        let count = root.children_count as usize;
        let n = count.min(noise.len());

        let noise_slice = &noise[..n];
        if !noise_slice
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        {
            return Err(MCTSError::InvalidNoise(
                "noise values must be finite and non-negative",
            ));
        }
        let noise_sum: f32 = noise_slice.iter().sum();
        if noise_sum <= 0.0 || !noise_sum.is_finite() {
            return Err(MCTSError::InvalidNoise(
                "noise values must have positive finite mass",
            ));
        }

        let mix = noise_fraction.clamp(0.0, 1.0);
        for (i, &noise_val) in noise_slice.iter().enumerate() {
            let child = &mut self.arena[start + i];
            child.prior = (1.0 - mix) * child.prior + mix * (noise_val / noise_sum);
        }
        let total_prior: f32 = (start..start + count)
            .map(|idx| self.arena[idx].prior)
            .sum();
        if total_prior > 0.0 && total_prior.is_finite() {
            for idx in start..start + count {
                self.arena[idx].prior /= total_prior;
            }
        }
        Ok(())
    }

    // ── Core MCTS loop ─────────────────────────────────────────────────

    /// Are we done simulating?
    pub fn done(&self) -> bool {
        self.sims_done >= self.num_simulations
    }

    /// Select up to `batch_size` leaves, encode their boards into tensors.
    ///
    /// # Steps for each leaf
    /// 1. Traverse from root using `select_child_puct` until an unexpanded
    ///    node or terminal position is reached.
    /// 2. Apply virtual loss to every node on the search path.
    /// 3. If terminal, record the terminal value.  If non-terminal, encode
    ///    the board into `batch_buf` using `encoder::encode_board_into`.
    /// 4. Undo all moves to restore the root game state.
    ///
    pub fn select_leaves(&mut self, batch_size: u32) -> Result<LeafBatch<'_>, MCTSError> {
        // Clamp batch size to the remaining simulation budget.
        let actual_batch = batch_size.min(self.num_simulations.saturating_sub(self.sims_done));

        // Clear previous pending state from any prior batch, rolling back
        // virtual loss if a caller selected a new batch before backpropagating
        // the old one.
        self.clear_pending_leaves();

        // Reserve batch buffer space for the worst case (all non-terminal).
        let max_floats = actual_batch as usize * TENSOR_SIZE;
        self.batch_buf.clear();
        self.batch_buf.reserve(max_floats);

        let mut non_terminal_count = 0u32;

        for _ in 0..actual_batch {
            let mut node_idx = self.root_idx;
            let mut search_path = SmallVec::<[u32; 32]>::new();
            search_path.push(self.root_idx);

            // Traverse via PUCT until an unexpanded node or terminal.
            while self.arena[node_idx as usize].is_expanded
                && self.arena[node_idx as usize].children_count > 0
            {
                node_idx = self.select_child_puct(node_idx);
                let child = &self.arena[node_idx as usize];
                if let Err(_err) = self.game.place(child.action.0, child.action.1) {
                    for _ in 0..search_path.len().saturating_sub(1) {
                        self.game
                            .unplace()
                            .expect("selected path rollback should match applied moves");
                    }
                    return Err(MCTSError::IllegalTreeAction {
                        q: child.action.0,
                        r: child.action.1,
                        node_idx,
                    });
                }
                search_path.push(node_idx);
            }

            // Set player if this is the first time we've reached this node.
            if self.arena[node_idx as usize].player == 255 {
                self.arena[node_idx as usize].player = self.game.current_player();
            }

            // Virtual loss discourages other batch slots from exploring the
            // same branch while this leaf is awaiting GPU evaluation.
            for &ni in &search_path {
                let n = &mut self.arena[ni as usize];
                n.visit_count += VIRTUAL_LOSS_VISITS;
                n.total_value -= VIRTUAL_LOSS_VISITS as f32;
            }

            // We replayed `search_path.len() - 1` moves (root is not a move).
            let depth = search_path.len() as u32 - 1;

            if self.game.is_over() {
                // Terminal value is from the leaf player's perspective.
                let node_player = self.arena[node_idx as usize].player;
                let value = if self.game.winner() == Some(node_player) {
                    1.0
                } else {
                    -1.0
                };
                self.pending.push(PendingLeaf {
                    node_idx,
                    search_path,
                    is_terminal: true,
                    terminal_value: value,
                    offset_q: 0,
                    offset_r: 0,
                    legal_moves: Vec::new(),
                    move_history: pack_move_history(&self.game),
                });
            } else {
                let start = non_terminal_count as usize * TENSOR_SIZE;
                self.batch_buf.resize(start + TENSOR_SIZE, 0.0);
                let tensor_slice = &mut self.batch_buf[start..start + TENSOR_SIZE];

                // Only constrain threats at root (init_root), not internal
                // nodes — the O(n²) unblockable check is too expensive to run
                // on every leaf expansion during tree search.
                self.legal_buf.clear();
                let (oq, or_) = encoder::encode_board_into(
                    &self.game,
                    self.near_radius,
                    false,
                    tensor_slice,
                    &mut self.hot_buf,
                    &mut self.legal_buf,
                );

                non_terminal_count += 1;
                self.pending.push(PendingLeaf {
                    node_idx,
                    search_path,
                    is_terminal: false,
                    terminal_value: 0.0,
                    offset_q: oq,
                    offset_r: or_,
                    legal_moves: self.legal_buf.clone(),
                    move_history: pack_move_history(&self.game),
                });
            }

            // Restore root state for the next simulation.
            for _ in 0..depth {
                self.game
                    .unplace()
                    .expect("selection depth rollback should match applied moves");
            }
        }

        self.batch_generation = self.batch_generation.wrapping_add(1);
        self.pending_batch_generation = if self.pending.is_empty() {
            None
        } else {
            Some(self.batch_generation)
        };
        Ok(LeafBatch {
            tensors: &self.batch_buf[..non_terminal_count as usize * TENSOR_SIZE],
            non_terminal_count,
            root_generation: self.root_generation,
            batch_generation: self.batch_generation,
        })
    }

    /// Expand leaves and backpropagate using GPU-provided policies and values.
    ///
    /// `policies` is flat `(N × BOARD_AREA)` f32, `values` is flat `(N,)` f32,
    /// where `N` is the number of non-terminal leaves from the last
    /// `select_leaves` call.
    ///
    /// # Steps for each leaf
    /// 1. Remove virtual loss from the search path.
    /// 2. If terminal: use the pre-recorded terminal value.
    ///    If non-terminal: expand the node with its policy slice and use the
    ///    NN value.
    /// 3. Backpropagate the value up the search path, flipping sign at each
    ///    level so that every node stores values from its own player's
    ///    perspective.
    pub fn expand_and_backprop(
        &mut self,
        batch_generation: u64,
        policies: &[f32],
        values: &[f32],
    ) -> Result<(), MCTSError> {
        self.validate_batch_generation(batch_generation)?;
        let non_terminal_count = self.pending.iter().filter(|l| !l.is_terminal).count();
        validate_policy_slice(policies, non_terminal_count * BOARD_AREA)?;
        validate_values(values, non_terminal_count)?;

        let mut eval_idx = 0usize;

        // Move pending leaves out temporarily to avoid borrow issues with
        // &mut self, then return the allocated storage for reuse next batch.
        let mut leaves = Vec::new();
        std::mem::swap(&mut leaves, &mut self.pending);
        self.sims_done += leaves.len() as u32;

        for leaf in &leaves {
            // Remove virtual loss before recording real visits.
            for &ni in &leaf.search_path {
                let n = &mut self.arena[ni as usize];
                n.visit_count -= VIRTUAL_LOSS_VISITS;
                n.total_value += VIRTUAL_LOSS_VISITS as f32;
            }

            let value;
            if leaf.is_terminal {
                value = leaf.terminal_value;
            } else {
                // Extract this leaf's policy slice from the flat GPU output.
                let policy_start = eval_idx * BOARD_AREA;
                let policy_slice = &policies[policy_start..policy_start + BOARD_AREA];
                let v = values[eval_idx];

                // Expand the leaf node with its policy and legal moves.
                self.expand_node(
                    leaf.node_idx,
                    &leaf.legal_moves,
                    self.arena[leaf.node_idx as usize].player,
                    policy_slice,
                    leaf.offset_q,
                    leaf.offset_r,
                );
                self.record_leaf_prior_sources(0);
                value = v;
                eval_idx += 1;
            }

            // Flip only when crossing to a node owned by the other player.
            // Hexo turns may contain two same-player placement edges.
            let mut parity_value = value;
            let mut value_player = self.arena[leaf.node_idx as usize].player;
            for &ni in leaf.search_path.iter().rev() {
                let n = &mut self.arena[ni as usize];
                if n.player != 255 && value_player != 255 && n.player != value_player {
                    parity_value = -parity_value;
                    value_player = n.player;
                }
                n.visit_count += 1;
                n.total_value += parity_value;
            }
        }

        leaves.clear();
        std::mem::swap(&mut leaves, &mut self.pending);
        self.pending_batch_generation = None;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn expand_and_backprop_with_sparse(
        &mut self,
        batch_generation: u64,
        policies: &[f32],
        values: &[f32],
        sparse_actions: &[Vec<(i32, i32)>],
        sparse_logits: &[Vec<f32>],
        stage: u8,
        sparse_mix: f32,
    ) -> Result<(), MCTSError> {
        self.expand_and_backprop_with_sparse_impl(
            batch_generation,
            policies,
            values,
            sparse_actions,
            sparse_logits,
            None,
            stage,
            sparse_mix,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn expand_and_backprop_with_sparse_sources(
        &mut self,
        batch_generation: u64,
        policies: &[f32],
        values: &[f32],
        sparse_actions: &[Vec<(i32, i32)>],
        sparse_logits: &[Vec<f32>],
        sparse_sources: &[Vec<u8>],
        stage: u8,
        sparse_mix: f32,
    ) -> Result<(), MCTSError> {
        self.expand_and_backprop_with_sparse_impl(
            batch_generation,
            policies,
            values,
            sparse_actions,
            sparse_logits,
            Some(sparse_sources),
            stage,
            sparse_mix,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_and_backprop_with_sparse_impl(
        &mut self,
        batch_generation: u64,
        policies: &[f32],
        values: &[f32],
        sparse_actions: &[Vec<(i32, i32)>],
        sparse_logits: &[Vec<f32>],
        sparse_sources: Option<&[Vec<u8>]>,
        stage: u8,
        sparse_mix: f32,
    ) -> Result<(), MCTSError> {
        self.validate_batch_generation(batch_generation)?;
        let non_terminal_count = self.pending.iter().filter(|l| !l.is_terminal).count();
        validate_policy_slice(policies, non_terminal_count * BOARD_AREA)?;
        validate_values(values, non_terminal_count)?;
        let source_len = sparse_sources.map(|sources| sources.len());
        if sparse_actions.len() != non_terminal_count
            || sparse_logits.len() != non_terminal_count
            || source_len.is_some_and(|len| len != non_terminal_count)
        {
            return Err(MCTSError::WrongSparseMetadataLength {
                expected: non_terminal_count,
                actions: sparse_actions.len(),
                logits: sparse_logits.len(),
                sources: source_len,
            });
        }
        for (leaf_idx, logits) in sparse_logits.iter().enumerate() {
            if let Some(index) = logits.iter().position(|logit| !logit.is_finite()) {
                return Err(MCTSError::NonFiniteSparsePolicy {
                    leaf: leaf_idx,
                    index,
                });
            }
        }

        let mut eval_idx = 0usize;
        let mut leaves = Vec::new();
        std::mem::swap(&mut leaves, &mut self.pending);
        self.sims_done += leaves.len() as u32;

        for leaf in &leaves {
            for &ni in &leaf.search_path {
                let n = &mut self.arena[ni as usize];
                n.visit_count -= VIRTUAL_LOSS_VISITS;
                n.total_value += VIRTUAL_LOSS_VISITS as f32;
            }

            let value;
            if leaf.is_terminal {
                value = leaf.terminal_value;
            } else {
                let policy_start = eval_idx * BOARD_AREA;
                let policy_slice = &policies[policy_start..policy_start + BOARD_AREA];
                let v = values[eval_idx];
                let sparse_source = sparse_sources
                    .and_then(|sources| sources.get(eval_idx))
                    .and_then(|row| row.first().copied())
                    .unwrap_or(PRIOR_SOURCE_SPARSE);
                self.expand_node_with_sparse(
                    leaf.node_idx,
                    &leaf.legal_moves,
                    self.arena[leaf.node_idx as usize].player,
                    policy_slice,
                    leaf.offset_q,
                    leaf.offset_r,
                    &sparse_actions[eval_idx],
                    &sparse_logits[eval_idx],
                    stage,
                    sparse_mix,
                    sparse_source,
                );
                let sparse_count = if stage >= 2 {
                    sparse_actions[eval_idx]
                        .len()
                        .min(sparse_logits[eval_idx].len())
                } else {
                    0
                };
                self.record_leaf_prior_sources(sparse_count);
                value = v;
                eval_idx += 1;
            }

            let mut parity_value = value;
            let mut value_player = self.arena[leaf.node_idx as usize].player;
            for &ni in leaf.search_path.iter().rev() {
                let n = &mut self.arena[ni as usize];
                if n.player != 255 && value_player != 255 && n.player != value_player {
                    parity_value = -parity_value;
                    value_player = n.player;
                }
                n.visit_count += 1;
                n.total_value += parity_value;
            }
        }

        leaves.clear();
        std::mem::swap(&mut leaves, &mut self.pending);
        self.pending_batch_generation = None;
        Ok(())
    }

    pub fn pending_leaf_metadata(&self) -> Vec<(i32, i32, Vec<Hex>, Vec<u8>)> {
        self.pending
            .iter()
            .filter(|leaf| !leaf.is_terminal)
            .map(|leaf| {
                (
                    leaf.offset_q,
                    leaf.offset_r,
                    leaf.legal_moves.clone(),
                    leaf.move_history.clone(),
                )
            })
            .collect()
    }

    pub fn clear_pending_leaves(&mut self) {
        if self.pending.is_empty() {
            self.pending_batch_generation = None;
            return;
        }
        let mut leaves = Vec::new();
        std::mem::swap(&mut leaves, &mut self.pending);
        for leaf in &leaves {
            for &ni in &leaf.search_path {
                let n = &mut self.arena[ni as usize];
                n.visit_count = n.visit_count.saturating_sub(VIRTUAL_LOSS_VISITS);
                n.total_value += VIRTUAL_LOSS_VISITS as f32;
            }
        }
        leaves.clear();
        std::mem::swap(&mut leaves, &mut self.pending);
        self.pending_batch_generation = None;
    }

    // ── Tree management ────────────────────────────────────────────────

    /// Re-root the tree at the child matching action `(q, r)`.
    ///
    /// Used for subtree reuse across placements: after placement 1 is selected,
    /// re-root at that child so placement 2's MCTS starts from the surviving
    /// subtree with all visit/Q statistics intact.
    ///
    /// **Precondition:** No pending leaves must exist (call after a fully
    /// completed expand_and_backprop cycle).
    ///
    /// The arena is not compacted — dead sibling subtrees remain in memory but
    /// are never referenced.  With typical tree sizes this wastes ~400KB, which
    /// is acceptable since the arena is recreated per turn.
    pub fn re_root(&mut self, q: i32, r: i32, new_num_simulations: u32) -> Result<(), MCTSError> {
        self.clear_pending_leaves();

        let root = &self.arena[self.root_idx as usize];
        let start = root.children_start as usize;
        let count = root.children_count as usize;

        // Find the child matching the action.
        let mut child_idx = None;
        for i in start..start + count {
            let child = &self.arena[i];
            if child.action.0 == q && child.action.1 == r {
                child_idx = Some(i as u32);
                break;
            }
        }

        let child_idx = match child_idx {
            Some(idx) => idx,
            None => return Err(MCTSError::ChildNotFound { q, r }),
        };

        // Apply the placement to the internal game state so that subsequent
        // select_leaves traversals start from the correct board position.
        self.game.place(q, r).expect("re_root: illegal placement");

        // Update root pointer.
        self.root_idx = child_idx;
        self.arena[child_idx as usize].parent = NO_PARENT;

        // If the new root was already expanded during P1's tree traversal, its
        // children were computed with constrain_threats=false (internal nodes
        // never run the O(n²) unblockable check).  When threat constraints are
        // enabled and the new position has active threats, those unconstrained
        // children are wrong — clear them so Python calls init_root again and
        // gets the correct constrained legal set.
        //
        // Descendant parent-clearing is unnecessary: once the ancestor's
        // children_count is zero, descendants are unreachable from the new root.
        // extract_tree_node_states only follows expanded nodes, so orphaned
        // subtrees are naturally excluded.
        if self.constrain_threats
            && self.arena[child_idx as usize].is_expanded
            && self.game.eval().has_any_threats()
        {
            let node = &mut self.arena[child_idx as usize];
            node.is_expanded = false;
            node.children_count = 0;
        }

        // Reset simulation state for the new search.
        self.sims_done = 0;
        self.num_simulations = new_num_simulations;
        self.root_generation = self.root_generation.wrapping_add(1);
        self.last_root_init_generation = None;

        Ok(())
    }

    // ── Results ────────────────────────────────────────────────────────

    /// Get search results: `(moves_q, moves_r, visit_counts, root_value)`.
    ///
    /// Returns parallel vectors for all root children.  `root_value` is the
    /// average Q-value of the root node.
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
            moves_q.push(child.action.0);
            moves_r.push(child.action.1);
            visits.push(child.visit_count);
        }

        (moves_q, moves_r, visits, root_value)
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

    pub fn root_child_prior_sources(&self) -> Vec<u8> {
        self.root_prior_sources.clone()
    }

    pub fn prior_source_telemetry(&self) -> PriorSourceTelemetry {
        let (root_total, root_sparse, root_dense, root_default, _root_pair) =
            summarize_sources(&self.root_prior_sources);
        PriorSourceTelemetry {
            root_total_count: root_total,
            root_sparse_count: root_sparse,
            root_dense_count: root_dense,
            root_default_count: root_default,
            leaf_total_count: self.leaf_source_total,
            leaf_sparse_count: self.leaf_source_sparse,
            leaf_dense_count: self.leaf_source_dense,
            leaf_default_count: self.leaf_source_default,
            root_pair_count: count_prior_source(&self.root_prior_sources, PRIOR_SOURCE_PAIR),
            leaf_pair_count: self.leaf_source_pair,
            root_sparse_candidate_count: self.root_sparse_candidate_count,
            leaf_sparse_candidate_count: self.leaf_sparse_candidate_count,
            root_pair_candidate_count: self.root_pair_candidate_count,
            leaf_expansion_count: self.leaf_expansion_count,
        }
    }

    /// Get Q-values of root children from the root player's perspective.
    ///
    /// For unvisited children, returns the root's Q-value (before FPU
    /// reduction, since this is for external reporting, not selection).
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
                    // Flip sign if the child is from the opponent's perspective.
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

    /// Return search-observed joint first/second placement targets at the root.
    ///
    /// Rows are `(first_q, first_r, second_q, second_r, visit_count)` and are
    /// extracted from expanded root-child subtrees. This gives training a real
    /// MCTS joint-pair signal instead of reconstructing a pair from the single
    /// first move that happened to be sampled into the played trajectory.
    pub fn root_pair_visit_targets(&self) -> Vec<(i32, i32, i32, i32, u32)> {
        if self.game.placements_remaining() < 2 {
            return Vec::new();
        }
        let root = &self.arena[self.root_idx as usize];
        let start = root.children_start as usize;
        let count = root.children_count as usize;
        let mut rows = Vec::new();
        for first_idx in start..start + count {
            let first = &self.arena[first_idx];
            if !first.is_expanded || first.children_count == 0 {
                continue;
            }
            let second_start = first.children_start as usize;
            let second_count = first.children_count as usize;
            for second_idx in second_start..second_start + second_count {
                let second = &self.arena[second_idx];
                if second.visit_count == 0 {
                    continue;
                }
                rows.push((
                    first.action.0,
                    first.action.1,
                    second.action.0,
                    second.action.1,
                    second.visit_count,
                ));
            }
        }
        rows
    }

    // ── Training data extraction ───────────────────────────────────────

    /// Extract encoded board states and move histories for expanded tree nodes
    /// that have at least `min_visits` visits.
    ///
    /// # Returns
    /// `(packed_tensors, histories, count)` where:
    /// - `packed_tensors` is a flat `Vec<f32>` of shape `(count, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)`
    /// - `histories` is a parallel vector of move histories as `(player, q, r)` tuples
    /// - `count` is the number of valid candidates extracted
    pub fn extract_tree_node_states(
        &mut self,
        min_visits: u32,
    ) -> Result<TreeNodeStates, &'static str> {
        let mut packed = Vec::new();
        let mut histories = Vec::new();

        // Cap candidates to avoid OOM on large trees (48MB+ with 800+ nodes).
        const MAX_CANDIDATES: usize = 128;

        let mut candidates: Vec<u32> = Vec::new();
        let mut reachable_stack = vec![self.root_idx];
        while let Some(node_idx) = reachable_stack.pop() {
            let node = &self.arena[node_idx as usize];
            if node_idx != self.root_idx && node.is_expanded && node.visit_count >= min_visits {
                candidates.push(node_idx);
            }
            if node.is_expanded && node.children_count > 0 {
                let start = node.children_start as usize;
                let count = node.children_count as usize;
                for child_idx in (start..start + count).rev() {
                    reachable_stack.push(child_idx as u32);
                }
            }
        }

        if candidates.len() > MAX_CANDIDATES {
            candidates.sort_unstable_by(|a, b| {
                self.arena[*b as usize]
                    .visit_count
                    .cmp(&self.arena[*a as usize].visit_count)
            });
            candidates.truncate(MAX_CANDIDATES);
        }

        if packed.try_reserve(candidates.len() * TENSOR_SIZE).is_err() {
            return Ok((Vec::new(), Vec::new(), 0));
        }
        histories.reserve(candidates.len());

        let candidate_set: std::collections::HashSet<u32> = candidates.iter().copied().collect();

        #[derive(Clone, Copy)]
        enum Frame {
            Visit(u32),
            VisitChild {
                parent_idx: u32,
                child_offset: usize,
            },
            Unplace,
        }

        /// Depth of `node_idx` from the current root, measured in placed edges.
        fn depth_from_root(node_idx: u32, root_idx: u32, arena: &[MCTSNode]) -> u32 {
            let mut depth = 0;
            let mut idx = node_idx;
            while idx != root_idx && idx != NO_PARENT {
                depth += 1;
                idx = arena[idx as usize].parent;
            }
            depth
        }

        let mut scratch_history: Vec<(u8, i32, i32)> = Vec::new();
        let mut stack: Vec<Frame> = Vec::new();
        stack.push(Frame::Visit(self.root_idx));

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Visit(node_idx) => {
                    if candidate_set.contains(&node_idx) && node_idx != self.root_idx {
                        if self.game.is_over() {
                            // Restore game state before returning the error.
                            let d = depth_from_root(node_idx, self.root_idx, &self.arena);
                            for _ in 0..d {
                                self.game
                                    .unplace()
                                    .expect("extraction rollback should match traversal depth");
                            }
                            return Err("sampled node is terminal during extraction");
                        }
                        let start = packed.len();
                        packed.resize(start + TENSOR_SIZE, 0.0);
                        self.legal_buf.clear();
                        encoder::encode_board_into(
                            &self.game,
                            self.near_radius,
                            false,
                            &mut packed[start..start + TENSOR_SIZE],
                            &mut self.hot_buf,
                            &mut self.legal_buf,
                        );
                        scratch_history.clear();
                        scratch_history.extend(
                            self.game
                                .move_history()
                                .iter()
                                .map(|rec| (rec.player, rec.cell.q, rec.cell.r)),
                        );
                        histories.push(scratch_history.clone());
                    }

                    let node = &self.arena[node_idx as usize];
                    if node.is_expanded && node.children_count > 0 {
                        let count = node.children_count as usize;
                        // Schedule children for traversal.
                        for offset in (0..count).rev() {
                            stack.push(Frame::VisitChild {
                                parent_idx: node_idx,
                                child_offset: offset,
                            });
                        }
                    }
                }
                Frame::VisitChild {
                    parent_idx,
                    child_offset,
                } => {
                    let node = &self.arena[parent_idx as usize];
                    let start = node.children_start as usize;
                    let child = &self.arena[start + child_offset];
                    if self.game.place(child.action.0, child.action.1).is_err() {
                        let d = depth_from_root(parent_idx, self.root_idx, &self.arena);
                        for _ in 0..d {
                            self.game
                                .unplace()
                                .expect("extraction error rollback should match traversal depth");
                        }
                        return Err("illegal move during tree node extraction");
                    }
                    // After the child's entire subtree is processed, unplace it.
                    stack.push(Frame::Unplace);
                    stack.push(Frame::Visit(start as u32 + child_offset as u32));
                }
                Frame::Unplace => {
                    self.game
                        .unplace()
                        .expect("tree extraction stack should unplace an applied child");
                }
            }
        }

        let count = histories.len();
        Ok((packed, histories, count))
    }

    // ── Internal helpers ───────────────────────────────────────────────

    /// Select the best child using the PUCT formula.
    ///
    /// ```text
    /// score = Q + c_puct * prior * sqrt(parent_visits) / (1 + child_visits)
    /// ```
    ///
    /// Where `Q` is the average value from the **parent player's** perspective.
    /// For unvisited children, `Q = parent_Q - FPU_REDUCTION` (First-Play Urgency).
    ///
    /// Dynamic c_puct: if `c_puct_init > 0`, `c_puct` grows logarithmically:
    /// ```text
    /// effective_c = c_puct + ln((parent_visits + c_puct_init) / c_puct_init)
    /// ```
    fn select_child_puct(&self, node_idx: u32) -> u32 {
        let node = &self.arena[node_idx as usize];
        let parent_player = node.player;
        let parent_n = node.visit_count;
        let sqrt_parent = if parent_n > 0 {
            (parent_n as f32).sqrt()
        } else {
            1.0
        };
        let parent_q = if parent_n > 0 { node.q_value() } else { 0.0 };
        let fpu_value = parent_q - FPU_REDUCTION;

        let start = node.children_start as usize;
        let count = node.children_count as usize;

        // Compute dynamic c_puct if enabled.
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

            // Q-value from the parent player's perspective.
            // If the child was reached by the opponent, flip the sign.
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

            // PUCT exploration bonus.
            let score = q + effective_c_puct * child.prior * sqrt_parent / (1.0 + vc as f32);
            debug_assert!(
                score.is_finite(),
                "PUCT score non-finite for child {i}: q={q}, cpuct={effective_c_puct}, prior={}, sqrt_parent={sqrt_parent}, vc={vc}",
                child.prior
            );
            if score > best_score {
                best_score = score;
                best_idx = i as u32;
            }
        }

        best_idx
    }

    /// Expand a node by creating children for all legal moves.
    ///
    /// Gathers policy priors for the legal moves, runs softmax, then appends
    /// children contiguously to the arena.  Sets the node's `children_start`,
    /// `children_count`, `player`, and `is_expanded`.
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

        gather_policy(
            legal_moves,
            policy_logits,
            offset_q,
            offset_r,
            &mut self.scratch_raw,
            &mut self.scratch_priors,
        );
        gather_dense_sources(legal_moves, offset_q, offset_r, &mut self.scratch_sources);
        self.expand_node_with_priors(node_idx, legal_moves, player);
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_node_with_sparse(
        &mut self,
        node_idx: u32,
        legal_moves: &[Hex],
        player: u8,
        policy_logits: &[f32],
        offset_q: i32,
        offset_r: i32,
        sparse_actions: &[(i32, i32)],
        sparse_logits: &[f32],
        stage: u8,
        sparse_mix: f32,
        sparse_source: u8,
    ) {
        if legal_moves.is_empty() {
            return;
        }
        let effective_stage = if stage >= 2 { 2 } else { 0 };
        gather_policy_with_sparse(
            legal_moves,
            policy_logits,
            offset_q,
            offset_r,
            sparse_actions,
            sparse_logits,
            effective_stage,
            sparse_mix,
            sparse_source,
            &mut self.scratch_raw,
            &mut self.scratch_priors,
            &mut self.scratch_sources,
        );
        self.expand_node_with_priors(node_idx, legal_moves, player);
    }

    fn expand_node_with_priors(&mut self, node_idx: u32, legal_moves: &[Hex], player: u8) {
        // Allocate children contiguously in arena.
        let children_start = self.arena.len() as u32;
        let child_count = legal_moves.len();
        assert!(
            child_count <= u16::MAX as usize,
            "expand_node: legal_moves count {} exceeds u16::MAX",
            child_count
        );
        let children_count = child_count as u16;

        for (i, h) in legal_moves.iter().enumerate() {
            let child = MCTSNode::new(node_idx, (h.q, h.r), self.scratch_priors[i]);
            self.arena.push(child);
        }

        let node = &mut self.arena[node_idx as usize];
        node.children_start = children_start;
        node.children_count = children_count;
        node.player = player;
        node.is_expanded = true;
    }

    /// Sample a move from the root's children using temperature-based visit weighting.
    ///
    /// When `temperature == 0.0`, returns the child with the highest visit count.
    /// When `temperature > 0.0`, samples with probability proportional to
    /// `visit_count^(1/temperature)`.
    ///
    /// `rng_state` is an in-out XOR-shift state. The caller should seed it
    /// from the deterministic seed chain and pass it by mutable reference.
    pub fn sample_action(
        &mut self,
        temperature: f32,
        rng_state: &mut u64,
    ) -> Result<(i32, i32), MCTSError> {
        if *rng_state == 0 {
            *rng_state = self.seed.max(1);
        }
        let root = &self.arena[self.root_idx as usize];
        let start = root.children_start as usize;
        let count = root.children_count as usize;

        if count == 0 {
            return Err(MCTSError::InvalidRootState(
                "sample_action requires an expanded root with at least one child",
            ));
        }

        if temperature == 0.0 {
            let best = (start..start + count)
                .max_by_key(|&i| self.arena[i].visit_count)
                .unwrap();
            let action = (self.arena[best].action.0, self.arena[best].action.1);
            self.seed = *rng_state;
            return Ok(action);
        }

        let inv_t = 1.0 / temperature.max(1e-8);
        let weights: Vec<f32> = (start..start + count)
            .map(|i| {
                let vc = self.arena[i].visit_count as f32;
                if vc <= 0.0 {
                    1e-12f32.powf(inv_t)
                } else {
                    vc.powf(inv_t)
                }
            })
            .collect();
        let sum: f32 = weights.iter().sum();
        if sum <= 0.0 {
            let action = (self.arena[start].action.0, self.arena[start].action.1);
            self.seed = *rng_state;
            return Ok(action);
        }
        let r = next_uniform(rng_state) * sum;
        let mut acc = 0.0f32;
        for (offset, &w) in weights.iter().enumerate() {
            acc += w;
            if acc >= r {
                let n = &self.arena[start + offset];
                let action = (n.action.0, n.action.1);
                self.seed = *rng_state;
                return Ok(action);
            }
        }
        let last = start + count - 1;
        let action = (self.arena[last].action.0, self.arena[last].action.1);
        self.seed = *rng_state;
        Ok(action)
    }

    /// Returns true if the root Q-value is below the resign threshold.
    ///
    /// This should be called after the MCTS search completes (all sims done).
    /// The root value is from the current player's perspective, so a negative
    /// value means the engine thinks it's losing.
    pub fn should_resign(&self, threshold: f32) -> bool {
        let root = &self.arena[self.root_idx as usize];
        if root.visit_count == 0 {
            return false;
        }
        root.q_value() < threshold
    }
}
