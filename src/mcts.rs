//! Arena-allocated MCTS engine with PUCT, virtual loss, and batch leaf selection.
//!
//! The tree lives entirely in Rust. Python calls:
//!   1. `select_leaves(batch_size)` → board tensors for GPU inference
//!   2. `expand_and_backprop(policies, values)` → updates tree
//!   3. `get_results()` → (moves, visits, root_value)
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

use crate::core::Hex;
use crate::board::HexGameState;
use crate::encoder::{self, BOARD_SIZE, BOARD_AREA, TENSOR_SIZE};
use smallvec::SmallVec;

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
struct MCTSNode {
    /// Arena index of the parent node. `NO_PARENT` for the root.
    parent: u32,
    /// The move `(q, r)` that led to this node, relative to board origin.
    action: (i16, i16),
    /// Neural network policy prior for this node (probability of selecting
    /// the move that leads here, as assigned by the parent expansion).
    prior: f32,
    /// Total visit count (includes virtual-loss visits while the node is
    /// in-flight during batch selection).
    visit_count: u32,
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
    fn new(parent: u32, action: (i16, i16), prior: f32) -> Self {
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
}

// ── Policy gathering ───────────────────────────────────────────────────

/// Gather policy logits for legal moves and normalize them with softmax.
///
/// For each legal move, looks up the corresponding logit in the flat
/// `policy_logits` array (shape BOARD_AREA) using the board-to-tensor
/// offset, then applies numerically-stable softmax in f64 space.
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
    raw: &mut Vec<f64>,
    priors: &mut Vec<f32>,
) {
    let n = moves.len();
    raw.clear();
    raw.resize(n, -10.0f64);

    for (i, h) in moves.iter().enumerate() {
        let gi = h.q - offset_q;
        let gj = h.r - offset_r;
        if (0..BOARD_SIZE).contains(&gi) && (0..BOARD_SIZE).contains(&gj) {
            let flat = (gi as usize) * (BOARD_SIZE as usize) + gj as usize;
            raw[i] = policy_logits[flat] as f64;
        }
    }

    // Softmax in f64 for numerical stability, then convert to f32 priors.
    let max_val = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for v in raw.iter_mut() {
        *v = (*v - max_val).exp();
    }
    let sum: f64 = raw.iter().sum();

    priors.clear();
    if sum > 0.0 {
        priors.extend(raw.iter().map(|&e| (e / sum) as f32));
    } else {
        // Uniform fallback if all logits are identical or invalid.
        priors.extend(std::iter::repeat_n(1.0 / moves.len() as f32, moves.len()));
    }
}

// ── MCTSEngine ─────────────────────────────────────────────────────────

pub struct MCTSEngine {
    /// Contiguous storage for all MCTS nodes.
    ///
    /// Nodes are never deleted; `re_root` simply moves the root pointer.
    arena: Vec<MCTSNode>,
    /// Internal game state used during tree traversal.
    ///
    /// Moves are applied during `select_leaves` and undone before the
    /// function returns, so the game state is always valid for the root.
    game: HexGameState,
    /// Arena index of the current root node.
    root_idx: u32,
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
    pub c_puct_init: f32,
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
    scratch_raw: Vec<f64>,
    /// Reusable scratch buffer for `gather_policy` (normalized priors).
    scratch_priors: Vec<f32>,
}

type TreeNodeStates = (Vec<f32>, Vec<Vec<(u8, i16, i16)>>, usize);

impl MCTSEngine {
    // ── Construction ───────────────────────────────────────────────────

    /// Create a new MCTS engine for the given game state.
    ///
    /// `num_simulations` is the total number of MCTS rollouts to perform.
    /// `c_puct` is the base exploration constant.
    /// `near_radius` controls how far from existing stones legal moves are
    /// generated (clamped to `PLACEMENT_RADIUS`).
    /// `constrain_threats` enables threat-based move filtering at the root.
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

    /// Construct with an explicit arena simulation-count hint.
    ///
    /// Under subtree reuse, the same arena is shared across both placements of
    /// a turn, so the p1 sim count underestimates the true node budget.  Pass
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
        // Root node has no parent and a dummy action.
        arena.push(MCTSNode::new(NO_PARENT, (0, 0), 1.0));

        Self {
            arena,
            game,
            root_idx: 0,
            near_radius,
            constrain_threats,
            c_puct,
            c_puct_init: 19652.0,
            sims_done: 0,
            num_simulations,
            batch_buf: Vec::new(),
            pending: Vec::new(),
            scratch_raw: Vec::new(),
            scratch_priors: Vec::new(),
        }
    }

    // ── Root initialization ────────────────────────────────────────────

    /// Encode the root board state and return tensor data for GPU evaluation.
    ///
    /// # Returns
    /// `Some((tensor, offset_q, offset_r, legal_moves))` where `tensor` is a
    /// flat `Vec<f32>` of length `TENSOR_SIZE` (13×33×33).  Returns `None` if
    /// the game is already over.
    pub fn init_root(&mut self) -> Option<(Vec<f32>, i32, i32, Vec<Hex>)> {
        if self.game.is_over() {
            return None;
        }

        // Allocate a local tensor buffer and encode directly into it.
        let mut tensor = vec![0.0f32; TENSOR_SIZE];
        let (oq, or_, legal) =
            encoder::encode_board_into(&self.game, self.near_radius, self.constrain_threats, &mut tensor);

        Some((tensor, oq, or_, legal))
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
        policy_logits: &[f32],
        _value: f32,
        offset_q: i32,
        offset_r: i32,
        legal: &[Hex],
    ) {
        let player = self.game.current_player();
        self.expand_node(
            self.root_idx,
            legal,
            player,
            policy_logits,
            offset_q,
            offset_r,
        );
        self.arena[self.root_idx as usize].player = player;
    }

    /// Add Dirichlet noise to root priors for exploration.
    ///
    /// `noise` must have at least as many elements as root children.
    /// `noise_fraction` controls the blend between original prior and noise
    /// (0.0 = no noise, 1.0 = pure noise).
    pub fn add_dirichlet_noise(&mut self, noise: &[f32], noise_fraction: f32) {
        let root = &self.arena[self.root_idx as usize];
        let start = root.children_start as usize;
        let count = root.children_count as usize;
        let n = count.min(noise.len());
        for (i, &noise_val) in noise.iter().enumerate().take(n) {
            let child = &mut self.arena[start + i];
            child.prior = (1.0 - noise_fraction) * child.prior + noise_fraction * noise_val;
        }
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
    /// # Returns
    /// `(&[f32], u32)` — a reference to the internal batch buffer containing
    /// `non_terminal_count × TENSOR_SIZE` floats, and the count of
    /// non-terminal leaves.
    pub fn select_leaves(&mut self, batch_size: u32) -> (&[f32], u32) {
        // Clamp batch size to the remaining simulation budget.
        let actual_batch = batch_size.min(self.num_simulations - self.sims_done);

        // Clear previous pending state from any prior batch.
        self.pending.clear();

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
                self.game
                    .place(child.action.0 as i32, child.action.1 as i32)
                    .expect("MCTS: illegal place during tree traversal");
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
                });
            } else {
                let start = non_terminal_count as usize * TENSOR_SIZE;
                self.batch_buf.resize(start + TENSOR_SIZE, 0.0);
                let tensor_slice = &mut self.batch_buf[start..start + TENSOR_SIZE];

                // Only constrain threats at root (init_root), not internal
                // nodes — the O(n²) unblockable check is too expensive to run
                // on every leaf expansion during tree search.
                let (oq, or_, legal) =
                    encoder::encode_board_into(&self.game, self.near_radius, false, tensor_slice);

                non_terminal_count += 1;
                self.pending.push(PendingLeaf {
                    node_idx,
                    search_path,
                    is_terminal: false,
                    terminal_value: 0.0,
                    offset_q: oq,
                    offset_r: or_,
                    legal_moves: legal,
                });
            }

            // Restore root state for the next simulation.
            for _ in 0..depth {
                self.game.unplace();
            }
        }

        self.sims_done += actual_batch;

        (
            &self.batch_buf[..non_terminal_count as usize * TENSOR_SIZE],
            non_terminal_count,
        )
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
    pub fn expand_and_backprop(&mut self, policies: &[f32], values: &[f32]) {
        let mut eval_idx = 0usize;

        // Take pending out temporarily to avoid borrow issues with &mut self.
        let leaves = std::mem::take(&mut self.pending);

        for leaf in &leaves {
            // Remove virtual loss before recording real visits.
            for &ni in &leaf.search_path {
                let n = &mut self.arena[ni as usize];
                n.visit_count -= VIRTUAL_LOSS_VISITS;
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
                value = v;
                eval_idx += 1;
            }

            // Flip sign when the parent player differs from the leaf player
            // so every node stores values from its own perspective.
            let leaf_player = self.arena[leaf.node_idx as usize].player;
            for &ni in leaf.search_path.iter().rev() {
                let n = &mut self.arena[ni as usize];
                let pv_value = if n.player == leaf_player { value } else { -value };
                n.visit_count += 1;
                n.total_value += pv_value;
            }
        }

        // `leaves` is dropped; `self.pending` is now empty.
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
    pub fn re_root(&mut self, q: i16, r: i16, new_num_simulations: u32) {
        assert!(
            self.pending.is_empty(),
            "re_root: pending leaves must be flushed"
        );

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

        let child_idx = child_idx.unwrap_or_else(|| {
            panic!("re_root: no child found for action ({}, {})", q, r);
        });

        // Apply the placement to the internal game state so that subsequent
        // select_leaves traversals start from the correct board position.
        self.game
            .place(q as i32, r as i32)
            .expect("re_root: illegal placement");

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
            moves_q.push(child.action.0 as i32);
            moves_r.push(child.action.1 as i32);
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
            Unplace,
        }

        /// Depth of `node_idx` from the root (number of ancestors).
        fn depth_from_root(node_idx: u32, arena: &[MCTSNode]) -> u32 {
            let mut depth = 0;
            let mut idx = node_idx;
            while idx != NO_PARENT {
                depth += 1;
                idx = arena[idx as usize].parent;
            }
            depth
        }

        let mut stack: Vec<Frame> = Vec::new();
        stack.push(Frame::Visit(self.root_idx));

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Visit(node_idx) => {
                    if candidate_set.contains(&node_idx) && node_idx != self.root_idx {
                        if self.game.is_over() {
                            // Restore game state before returning the error.
                            let d = depth_from_root(node_idx, &self.arena);
                            for _ in 0..d {
                                self.game.unplace();
                            }
                            return Err("sampled node is terminal during extraction");
                        }
                        let start = packed.len();
                        packed.resize(start + TENSOR_SIZE, 0.0);
                        encoder::encode_board_into(
                            &self.game,
                            self.near_radius,
                            false,
                            &mut packed[start..start + TENSOR_SIZE],
                        );
                        let history: Vec<(u8, i16, i16)> = self
                            .game
                            .move_history()
                            .iter()
                            .map(|rec| (rec.player, rec.cell.q as i16, rec.cell.r as i16))
                            .collect();
                        histories.push(history);
                    }

                    let node = &self.arena[node_idx as usize];
                    if node.is_expanded && node.children_count > 0 {
                        if node_idx != self.root_idx {
                            stack.push(Frame::Unplace);
                        }
                        let start = node.children_start as usize;
                        let count = node.children_count as usize;
                        for i in (start..start + count).rev() {
                            let child = &self.arena[i];
                            if self.game.place(
                                child.action.0 as i32,
                                child.action.1 as i32,
                            ).is_err() {
                                let d = depth_from_root(node_idx, &self.arena);
                                for _ in 0..d {
                                    self.game.unplace();
                                }
                                return Err("illegal move during tree node extraction");
                            }
                            stack.push(Frame::Visit(i as u32));
                        }
                    } else if node_idx != self.root_idx {
                        stack.push(Frame::Unplace);
                    }
                }
                Frame::Unplace => {
                    self.game.unplace();
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
        let parent_q = if parent_n > 0 {
            node.q_value()
        } else {
            0.0
        };
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

        // Allocate children contiguously in arena.
        let children_start = self.arena.len() as u32;
        let children_count = legal_moves.len() as u16;

        for (i, h) in legal_moves.iter().enumerate() {
            let child = MCTSNode::new(node_idx, (h.q as i16, h.r as i16), self.scratch_priors[i]);
            self.arena.push(child);
        }

        let node = &mut self.arena[node_idx as usize];
        node.children_start = children_start;
        node.children_count = children_count;
        node.player = player;
        node.is_expanded = true;
    }
}
