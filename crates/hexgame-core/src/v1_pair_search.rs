//! Pair-native V1 search foundation.
//!
//! This module is intentionally separate from [`crate::mcts::MCTSEngine`].
//! It consumes V1 row-identity objects and returns V1 pair actions directly,
//! without projecting pair policy back to single-placement priors.

use crate::board::{GameError, HexGameState};
use crate::core::Hex;
use crate::v1::{
    canonical_pair_rows_ordered_v1, legal_row_table_v1, terminal_tactical_set_v1, LegalRowTableV1,
    PairRowErrorV1, PairRowV1, TerminalTacticalSetV1, TurnPhaseV1, PAIR_ROW_SCHEMA_HASH_V1,
    PAIR_ROW_SCHEMA_VERSION_V1,
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;

const TARGET_SUPPORT_TERMINAL_EXACT: u32 = 1;
const TARGET_SUPPORT_TERMINAL_EQUIVALENT: u32 = 1 << 1;
const TARGET_SUPPORT_HOT_COVER: u32 = 1 << 2;
const MIN_PRIOR_TEMPERATURE: f32 = 1.0e-4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposalCorrectionModeV1 {
    ExactImportance,
    ClippedPropensity,
    UncorrectedLogged,
    TrainingForbidden,
}

impl ProposalCorrectionModeV1 {
    pub const fn as_str(self) -> &'static str {
        match self {
            ProposalCorrectionModeV1::ExactImportance => "exact_importance",
            ProposalCorrectionModeV1::ClippedPropensity => "clipped_propensity",
            ProposalCorrectionModeV1::UncorrectedLogged => "uncorrected_logged",
            ProposalCorrectionModeV1::TrainingForbidden => "training_forbidden",
        }
    }

    pub const fn code(self) -> u8 {
        match self {
            ProposalCorrectionModeV1::ExactImportance => 0,
            ProposalCorrectionModeV1::ClippedPropensity => 1,
            ProposalCorrectionModeV1::UncorrectedLogged => 2,
            ProposalCorrectionModeV1::TrainingForbidden => 3,
        }
    }

    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(ProposalCorrectionModeV1::ExactImportance),
            1 => Some(ProposalCorrectionModeV1::ClippedPropensity),
            2 => Some(ProposalCorrectionModeV1::UncorrectedLogged),
            3 => Some(ProposalCorrectionModeV1::TrainingForbidden),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum V1PairSearchError {
    StaleRootToken { expected: u64, received: u64 },
    InvalidRootState(&'static str),
    InvalidCandidate(String),
    PairRows(PairRowErrorV1),
    ApplyIdentityMismatch(String),
    Game(GameError),
    DuplicateInteriorNode { node_key: u64 },
    InteriorNodeNotFound { node_key: u64 },
}

impl fmt::Display for V1PairSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            V1PairSearchError::StaleRootToken { expected, received } => write!(
                f,
                "V1 root token mismatch: got {received}, expected {expected}"
            ),
            V1PairSearchError::InvalidRootState(msg) => write!(f, "{msg}"),
            V1PairSearchError::InvalidCandidate(msg) => write!(f, "{msg}"),
            V1PairSearchError::PairRows(err) => write!(f, "{err}"),
            V1PairSearchError::ApplyIdentityMismatch(msg) => write!(f, "{msg}"),
            V1PairSearchError::Game(err) => write!(f, "{err}"),
            V1PairSearchError::DuplicateInteriorNode { node_key } => {
                write!(
                    f,
                    "V1 interior node {node_key} already has a cached reservoir"
                )
            }
            V1PairSearchError::InteriorNodeNotFound { node_key } => {
                write!(f, "V1 interior node {node_key} has no cached reservoir")
            }
        }
    }
}

impl std::error::Error for V1PairSearchError {}

impl From<PairRowErrorV1> for V1PairSearchError {
    fn from(value: PairRowErrorV1) -> Self {
        V1PairSearchError::PairRows(value)
    }
}

impl From<GameError> for V1PairSearchError {
    fn from(value: GameError) -> Self {
        V1PairSearchError::Game(value)
    }
}

#[derive(Debug, Clone)]
pub struct V1PairSearchConfig {
    pub num_simulations: u32,
    pub c_puct: f32,
    pub seed: u64,
    pub min_root_admitted: usize,
    pub max_root_admitted: Option<usize>,
    pub prior_temperature: f32,
    pub min_log_correction: f32,
    pub max_log_correction: f32,
    pub alpha_pw: f32,
    pub c_pw: f32,
}

impl Default for V1PairSearchConfig {
    fn default() -> Self {
        Self {
            num_simulations: 32,
            c_puct: 1.4,
            seed: 1,
            min_root_admitted: 1,
            max_root_admitted: None,
            prior_temperature: 1.0,
            min_log_correction: -4.0,
            max_log_correction: 4.0,
            alpha_pw: 0.5,
            c_pw: 2.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct V1RootInit {
    pub legal_table: LegalRowTableV1,
    pub tactical: TerminalTacticalSetV1,
    pub root_generation: u64,
    pub legal_pair_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct V1PairCandidate {
    pub candidate_id: u32,
    pub row: PairRowV1,
    pub model_logit: f32,
    pub proposal_correction_weight: f32,
    pub correction_mode: ProposalCorrectionModeV1,
    pub prior_logit: f32,
    pub prior: f32,
    pub gumbel: f32,
    pub visit_count: u32,
    pub total_value: f32,
    pub completed_q: f32,
    pub allocation: u32,
    pub admitted: bool,
    pub forced_exploration_flag: bool,
    pub terminal_exact_flag: bool,
    pub terminal_equivalence_flag: bool,
    pub target_support_flags: u32,
}

impl V1PairCandidate {
    pub fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f32
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum V1SelectedAction {
    Single {
        cell: Hex,
        reason: &'static str,
        root_generation: u64,
        legal_row_table_hash: u64,
    },
    Pair {
        row: PairRowV1,
        root_generation: u64,
        legal_row_table_hash: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V1AppliedAction {
    pub action_kind: &'static str,
    pub placements_applied: u8,
    pub first: Hex,
    pub second: Option<Hex>,
    pub root_generation: u64,
    pub legal_row_table_hash: u64,
    pub pair_key: Option<u64>,
    pub terminal_after_first: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct V1PairSearchTelemetry {
    pub root_generation: u64,
    pub phase: Option<TurnPhaseV1>,
    pub legal_row_table_hash: u64,
    pub legal_row_schema_version: u32,
    pub pair_row_schema_version: u32,
    pub pair_row_schema_hash: u64,
    pub legal_row_count: u32,
    pub legal_pair_count: u64,
    pub supplied_candidate_count: u32,
    pub admitted_pair_count: u32,
    pub selected_pair_key: Option<u64>,
    pub selected_single: Option<Hex>,
    pub search_performed: bool,
    pub hardcoded_action: bool,
    pub hardcoded_reason: Option<&'static str>,
    pub gumbel_rounds: u32,
    pub simulation_count: u32,
    pub root_gumbel_seed: u64,
    pub reservoir_build_count: u32,
    pub scoring_pass_count: u32,
    pub neural_calls_per_expanded_full_turn_node: u32,
    pub reservoir_refill_events: u32,
    pub interior_expanded_full_turn_nodes: u32,
    pub interior_reservoir_build_count: u32,
    pub interior_scoring_pass_count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct V1InteriorReservoirTelemetry {
    pub node_key: u64,
    pub candidate_count: u32,
    pub revealed_count: u32,
    pub reservoir_build_count: u32,
    pub scoring_pass_count: u32,
    pub widening_events: u32,
    pub reservoir_refill_events: u32,
}

#[derive(Debug, Clone)]
pub struct V1InteriorWideningResult {
    pub telemetry: V1InteriorReservoirTelemetry,
    pub revealed_rows: Vec<PairRowV1>,
    pub puct_scores: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct V1ExpansionRequest {
    pub node_key: u64,
    pub game: HexGameState,
    pub legal_table: LegalRowTableV1,
    pub tactical: TerminalTacticalSetV1,
    pub parent_visits: u32,
    pub node_visit_count: u32,
    pub root_generation: u64,
    pub legal_row_table_hash: u64,
    pub phase: TurnPhaseV1,
}

#[derive(Debug, Clone)]
struct V1RootIdentity {
    generation: u64,
    legal_table: LegalRowTableV1,
    tactical: TerminalTacticalSetV1,
}

#[derive(Debug, Clone)]
struct V1InteriorReservoir {
    telemetry: V1InteriorReservoirTelemetry,
    rows: Vec<PairRowV1>,
    prior_logits: Vec<f32>,
    priors: Vec<f32>,
    visits: Vec<u32>,
    total_values: Vec<f32>,
    child_keys: Vec<Option<u64>>,
}

#[derive(Debug, Clone)]
struct V1PendingExpansion {
    path: Vec<V1PathEdge>,
    node_path: Vec<u64>,
    root_player: u8,
    game: HexGameState,
    legal_table_hash: u64,
}

#[derive(Debug, Clone)]
struct V1SearchNode {
    game: HexGameState,
    legal_table: LegalRowTableV1,
    tactical: TerminalTacticalSetV1,
    parent_edge: Option<V1PathEdge>,
    incoming_pair: Option<PairRowV1>,
    current_player: u8,
    visit_count: u32,
    total_value: f32,
    terminal_value_root: Option<f32>,
    reservoir: Option<V1InteriorReservoir>,
}

#[derive(Debug, Clone, Copy)]
enum V1PathEdge {
    Root(usize),
    Interior { node_key: u64, edge_idx: usize },
}

#[derive(Debug, Clone)]
pub struct V1PairSearchEngine {
    game: HexGameState,
    config: V1PairSearchConfig,
    root_generation: u64,
    root_identity: Option<V1RootIdentity>,
    root_candidates: Vec<V1PairCandidate>,
    selected: Option<V1SelectedAction>,
    telemetry: V1PairSearchTelemetry,
    interior_reservoirs: FxHashMap<u64, V1InteriorReservoir>,
    pending_expansions: FxHashMap<u64, V1PendingExpansion>,
    search_nodes: FxHashMap<u64, V1SearchNode>,
}

impl V1PairSearchEngine {
    pub fn new(game: HexGameState, config: V1PairSearchConfig) -> Self {
        Self {
            game,
            config,
            root_generation: 0,
            root_identity: None,
            root_candidates: Vec::new(),
            selected: None,
            telemetry: V1PairSearchTelemetry {
                pair_row_schema_version: PAIR_ROW_SCHEMA_VERSION_V1,
                pair_row_schema_hash: PAIR_ROW_SCHEMA_HASH_V1,
                ..V1PairSearchTelemetry::default()
            },
            interior_reservoirs: FxHashMap::default(),
            pending_expansions: FxHashMap::default(),
            search_nodes: FxHashMap::default(),
        }
    }

    pub fn game(&self) -> &HexGameState {
        &self.game
    }

    pub fn init_root(&mut self) -> V1RootInit {
        let legal_table = legal_row_table_v1(&self.game);
        let tactical = terminal_tactical_set_v1(&self.game);
        let legal_pair_count = if legal_table.phase == TurnPhaseV1::NormalTwoPlacement {
            legal_table.rows.len() * legal_table.rows.len().saturating_sub(1) / 2
        } else {
            0
        };

        self.root_generation = self.root_generation.wrapping_add(1);
        self.root_candidates.clear();
        self.pending_expansions.clear();
        self.interior_reservoirs.clear();
        self.search_nodes.clear();
        self.selected = None;
        self.root_identity = Some(V1RootIdentity {
            generation: self.root_generation,
            legal_table: legal_table.clone(),
            tactical: tactical.clone(),
        });
        self.telemetry.root_generation = self.root_generation;
        self.telemetry.phase = Some(legal_table.phase);
        self.telemetry.legal_row_table_hash = legal_table.table_hash;
        self.telemetry.legal_row_schema_version = legal_table.schema_version;
        self.telemetry.pair_row_schema_version = PAIR_ROW_SCHEMA_VERSION_V1;
        self.telemetry.pair_row_schema_hash = PAIR_ROW_SCHEMA_HASH_V1;
        self.telemetry.legal_row_count = legal_table.rows.len() as u32;
        self.telemetry.legal_pair_count = legal_pair_count as u64;
        self.telemetry.supplied_candidate_count = 0;
        self.telemetry.admitted_pair_count = 0;
        self.telemetry.selected_pair_key = None;
        self.telemetry.selected_single = None;
        self.telemetry.search_performed = false;
        self.telemetry.hardcoded_action = false;
        self.telemetry.hardcoded_reason = None;
        self.telemetry.gumbel_rounds = 0;
        self.telemetry.simulation_count = 0;
        self.telemetry.root_gumbel_seed = self.config.seed;
        self.telemetry.reservoir_build_count = 0;
        self.telemetry.scoring_pass_count = 0;
        self.telemetry.neural_calls_per_expanded_full_turn_node = 0;
        self.telemetry.reservoir_refill_events = 0;
        self.telemetry.interior_expanded_full_turn_nodes = 0;
        self.telemetry.interior_reservoir_build_count = 0;
        self.telemetry.interior_scoring_pass_count = 0;

        V1RootInit {
            legal_table,
            tactical,
            root_generation: self.root_generation,
            legal_pair_count,
        }
    }

    pub fn admit_root_pairs(
        &mut self,
        root_generation: u64,
        pairs: &[(Hex, Hex)],
        model_logits: &[f32],
        correction_weights: &[f32],
        correction_modes: &[ProposalCorrectionModeV1],
    ) -> Result<(), V1PairSearchError> {
        let identity = self.validate_root_generation(root_generation)?.clone();
        if identity.legal_table.phase != TurnPhaseV1::NormalTwoPlacement {
            return Err(V1PairSearchError::InvalidRootState(
                "V1 root pair admission requires normal_two_placement phase",
            ));
        }
        validate_candidate_metadata(pairs, model_logits, correction_weights, correction_modes)?;

        let pair_table = canonical_pair_rows_ordered_v1(&identity.legal_table, pairs)?;
        let tactical_keys = tactical_pair_keys(&identity.tactical);
        let terminal_exact_keys = identity
            .tactical
            .hot_completion_pairs
            .iter()
            .map(|row| row.pair_key)
            .collect::<FxHashSet<_>>();
        let terminal_equiv_keys = identity
            .tactical
            .terminal_equivalent_pairs
            .iter()
            .map(|row| row.pair_key)
            .collect::<FxHashSet<_>>();
        let hot_cover_keys = identity
            .tactical
            .hot_cover_pairs
            .iter()
            .map(|row| row.pair_key)
            .collect::<FxHashSet<_>>();

        let mut candidates = Vec::with_capacity(pair_table.rows.len());
        for (idx, row) in pair_table.rows.into_iter().enumerate() {
            let mode = correction_modes[idx];
            let weight = correction_weights[idx];
            match mode {
                ProposalCorrectionModeV1::ExactImportance
                | ProposalCorrectionModeV1::ClippedPropensity => {
                    if weight <= 0.0 || !weight.is_finite() {
                        return Err(V1PairSearchError::InvalidCandidate(format!(
                            "proposal correction weight at {idx} must be finite and positive"
                        )));
                    }
                }
                ProposalCorrectionModeV1::UncorrectedLogged
                | ProposalCorrectionModeV1::TrainingForbidden => {}
            };
            let terminal_exact_flag = terminal_exact_keys.contains(&row.pair_key);
            let terminal_equivalence_flag = terminal_equiv_keys.contains(&row.pair_key);
            let mut target_support_flags = 0;
            if terminal_exact_flag {
                target_support_flags |= TARGET_SUPPORT_TERMINAL_EXACT;
            }
            if terminal_equivalence_flag {
                target_support_flags |= TARGET_SUPPORT_TERMINAL_EQUIVALENT;
            }
            if hot_cover_keys.contains(&row.pair_key) {
                target_support_flags |= TARGET_SUPPORT_HOT_COVER;
            }
            candidates.push(V1PairCandidate {
                candidate_id: idx as u32,
                row,
                model_logit: model_logits[idx],
                proposal_correction_weight: weight,
                correction_mode: mode,
                prior_logit: model_logits[idx],
                prior: 0.0,
                gumbel: gumbel_from_seed(self.config.seed, root_generation, row.pair_key),
                visit_count: 0,
                total_value: 0.0,
                completed_q: 0.0,
                allocation: 0,
                admitted: false,
                forced_exploration_flag: tactical_keys.contains(&row.pair_key),
                terminal_exact_flag,
                terminal_equivalence_flag,
                target_support_flags,
            });
        }

        assign_priors(
            &mut candidates,
            self.config.prior_temperature.max(MIN_PRIOR_TEMPERATURE),
        );
        self.root_candidates = candidates;
        self.selected = None;
        self.telemetry.supplied_candidate_count = self.root_candidates.len() as u32;
        self.telemetry.reservoir_build_count = 1;
        self.telemetry.scoring_pass_count = 1;
        self.telemetry.neural_calls_per_expanded_full_turn_node = 1;
        Ok(())
    }

    pub fn run_search_step(
        &mut self,
        max_expansions: usize,
    ) -> Result<Vec<V1ExpansionRequest>, V1PairSearchError> {
        let identity = self
            .root_identity
            .clone()
            .ok_or(V1PairSearchError::InvalidRootState(
                "V1 root must be initialized before search",
            ))?;
        if identity.legal_table.phase != TurnPhaseV1::NormalTwoPlacement {
            return Ok(Vec::new());
        }
        if self.root_candidates.is_empty() {
            return Err(V1PairSearchError::InvalidRootState(
                "normal V1 pair search requires admitted root candidate pairs",
            ));
        }
        if !self.root_candidates.iter().any(|candidate| candidate.admitted) {
            self.admit_gumbel_root_set()?;
        }
        let limit = max_expansions.max(1);
        let mut requests = Vec::new();
        while requests.len() < limit
            && self.telemetry.simulation_count + self.pending_expansions.len() as u32
                < self.config.num_simulations.max(1)
        {
            match self.start_recursive_simulation(&identity)? {
                Some(request) => requests.push(request),
                None => {
                    if self.pending_expansions.is_empty() {
                        break;
                    }
                    break;
                }
            }
        }
        Ok(requests)
    }

    fn start_recursive_simulation(
        &mut self,
        identity: &V1RootIdentity,
    ) -> Result<Option<V1ExpansionRequest>, V1PairSearchError> {
        let Some(root_idx) = self.select_root_puct_edge()? else {
            return Ok(None);
        };
        let mut path = vec![V1PathEdge::Root(root_idx)];
        let mut node_path = Vec::<u64>::new();
        let row = self.root_candidates[root_idx].row;
        let mut child = self.game.clone();
        let (first, second) = terminal_preferred_pair_order(&child, row);
        child.place(first.q, first.r)?;
        if child.is_over() {
            let value = outcome_value(&child, identity.legal_table.current_player);
            self.backup_path(&path, &node_path, value);
            return Ok(None);
        }
        child.place(second.q, second.r)?;
        if child.is_over() {
            let value = outcome_value(&child, identity.legal_table.current_player);
            self.backup_path(&path, &node_path, value);
            return Ok(None);
        }
        let mut node_key = child_node_key(0, row.pair_key);
        self.ensure_search_node(
            node_key,
            child,
            Some(V1PathEdge::Root(root_idx)),
            Some(row),
            identity.legal_table.current_player,
        )?;

        loop {
            let node = self
                .search_nodes
                .get(&node_key)
                .cloned()
                .ok_or(V1PairSearchError::InteriorNodeNotFound { node_key })?;
            if let Some(value) = node.terminal_value_root {
                self.backup_path(&path, &node_path, value);
                return Ok(None);
            }
            if node.legal_table.phase != TurnPhaseV1::NormalTwoPlacement {
                let value = self.evaluate_structural_leaf_root_value(
                    &node,
                    identity.legal_table.current_player,
                )?;
                self.backup_path(&path, &node_path, value);
                return Ok(None);
            }
            node_path.push(node_key);
            if node.reservoir.is_none() {
                if self.pending_expansions.contains_key(&node_key) {
                    return Ok(None);
                }
                self.pending_expansions.insert(
                    node_key,
                    V1PendingExpansion {
                        path,
                        node_path,
                        root_player: identity.legal_table.current_player,
                        game: node.game.clone(),
                        legal_table_hash: node.legal_table.table_hash,
                    },
                );
                let phase = node.legal_table.phase;
                let legal_row_table_hash = node.legal_table.table_hash;
                let parent_visits = node.visit_count.max(1);
                let node_visit_count = node.visit_count;
                return Ok(Some(V1ExpansionRequest {
                    node_key,
                    game: node.game,
                    phase,
                    legal_table: node.legal_table,
                    tactical: node.tactical,
                    parent_visits,
                    node_visit_count,
                    root_generation: identity.generation,
                    legal_row_table_hash,
                }));
            }

            let edge_idx = self.select_interior_puct_edge(node_key)?;
            let row = self
                .search_nodes
                .get(&node_key)
                .and_then(|node| node.reservoir.as_ref())
                .and_then(|reservoir| reservoir.rows.get(edge_idx))
                .copied()
                .ok_or(V1PairSearchError::InteriorNodeNotFound { node_key })?;
            path.push(V1PathEdge::Interior { node_key, edge_idx });

            let mut child = node.game.clone();
            let (first, second) = terminal_preferred_pair_order(&child, row);
            child.place(first.q, first.r)?;
            if child.is_over() {
                let value = outcome_value(&child, identity.legal_table.current_player);
                self.backup_path(&path, &node_path, value);
                return Ok(None);
            }
            child.place(second.q, second.r)?;
            if child.is_over() {
                let value = outcome_value(&child, identity.legal_table.current_player);
                self.backup_path(&path, &node_path, value);
                return Ok(None);
            }
            let child_key = child_node_key(node_key, row.pair_key);
            if let Some(parent) = self.search_nodes.get_mut(&node_key) {
                if let Some(reservoir) = parent.reservoir.as_mut() {
                    reservoir.child_keys[edge_idx] = Some(child_key);
                }
            }
            self.ensure_search_node(
                child_key,
                child,
                Some(V1PathEdge::Interior { node_key, edge_idx }),
                Some(row),
                identity.legal_table.current_player,
            )?;
            node_key = child_key;
        }
    }

    fn select_root_puct_edge(&self) -> Result<Option<usize>, V1PairSearchError> {
        let parent_visits = self
            .root_candidates
            .iter()
            .map(|candidate| candidate.visit_count)
            .sum::<u32>()
            .max(1);
        let parent_sqrt = (parent_visits as f32).sqrt();
        let mut best: Option<(usize, f32)> = None;
        for (idx, candidate) in self.root_candidates.iter().enumerate() {
            if !candidate.admitted
                || candidate.correction_mode == ProposalCorrectionModeV1::TrainingForbidden
            {
                continue;
            }
            let child_key = child_node_key(0, candidate.row.pair_key);
            if self.pending_expansions.contains_key(&child_key) {
                continue;
            }
            let q = candidate.q_value();
            let score = q
                + self.config.c_puct * candidate.prior * parent_sqrt
                    / (1.0 + candidate.visit_count as f32)
                + candidate.gumbel * 1.0e-6;
            if best
                .map(|(_, best_score)| score > best_score)
                .unwrap_or(true)
            {
                best = Some((idx, score));
            }
        }
        Ok(best.map(|(idx, _)| idx))
    }

    pub fn complete_expansion(
        &mut self,
        node_key: u64,
        node_value_current_player: f32,
        pairs: &[(Hex, Hex)],
        model_logits: &[f32],
        correction_weights: &[f32],
        correction_modes: &[ProposalCorrectionModeV1],
    ) -> Result<V1InteriorReservoirTelemetry, V1PairSearchError> {
        let pending = self
            .pending_expansions
            .get(&node_key)
            .cloned()
            .ok_or(V1PairSearchError::InteriorNodeNotFound { node_key })?;
        let node = self
            .search_nodes
            .get(&node_key)
            .ok_or(V1PairSearchError::InteriorNodeNotFound { node_key })?;
        if node.reservoir.is_some() {
            return Err(V1PairSearchError::DuplicateInteriorNode { node_key });
        }
        if node.legal_table.table_hash != pending.legal_table_hash {
            return Err(V1PairSearchError::ApplyIdentityMismatch(format!(
                "V1 expansion legal row table hash changed for node {node_key}: got {}, expected {}",
                node.legal_table.table_hash, pending.legal_table_hash
            )));
        }
        if !node_value_current_player.is_finite() {
            return Err(V1PairSearchError::InvalidCandidate(
                "V1 expansion value must be finite".to_string(),
            ));
        }
        let root_value = if pending.game.current_player() == pending.root_player {
            node_value_current_player
        } else {
            -node_value_current_player
        };
        let reservoir = self.build_interior_reservoir_for_game(
            node_key,
            &pending.game,
            pairs,
            model_logits,
            correction_weights,
            correction_modes,
        )?;
        let telemetry = reservoir.telemetry.clone();
        self.search_nodes
            .get_mut(&node_key)
            .ok_or(V1PairSearchError::InteriorNodeNotFound { node_key })?
            .reservoir = Some(reservoir);
        self.telemetry.interior_expanded_full_turn_nodes += 1;
        self.telemetry.interior_reservoir_build_count += 1;
        self.telemetry.interior_scoring_pass_count += 1;
        let _ = self.widen_tree_node_reservoir(node_key)?;
        self.pending_expansions.remove(&node_key);
        self.backup_path(&pending.path, &pending.node_path, root_value);
        Ok(self
            .search_nodes
            .get(&node_key)
            .and_then(|node| node.reservoir.as_ref())
            .map(|reservoir| reservoir.telemetry.clone())
            .unwrap_or(telemetry))
    }

    pub fn select_root_action(&mut self) -> Result<Option<V1SelectedAction>, V1PairSearchError> {
        let identity = self
            .root_identity
            .clone()
            .ok_or(V1PairSearchError::InvalidRootState(
                "V1 root must be initialized before search",
            ))?;
        match identity.legal_table.phase {
            TurnPhaseV1::OpeningSingle | TurnPhaseV1::OnePlacement | TurnPhaseV1::Terminal => {
                return self.run_root_search();
            }
            TurnPhaseV1::NormalTwoPlacement => {}
        }
        if !self.pending_expansions.is_empty() {
            return Err(V1PairSearchError::InvalidRootState(
                "V1 cannot select root action while interior expansions are pending",
            ));
        }
        if self.telemetry.simulation_count < self.config.num_simulations.max(1) {
            return Err(V1PairSearchError::InvalidRootState(
                "V1 root action selection requires the recursive simulation budget to complete",
            ));
        }
        if !self
            .root_candidates
            .iter()
            .any(|candidate| candidate.admitted && candidate.visit_count > 0)
        {
            return Err(V1PairSearchError::InvalidRootState(
                "V1 root action selection requires at least one backed-up pair evaluation",
            ));
        }
        self.complete_q_values();
        let selected_idx = self
            .root_candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| {
                candidate.admitted
                    && candidate.visit_count > 0
                    && candidate.correction_mode != ProposalCorrectionModeV1::TrainingForbidden
            })
            .max_by(|(_, left), (_, right)| compare_candidate_final(left, right))
            .map(|(idx, _)| idx)
            .ok_or(V1PairSearchError::InvalidRootState(
                "no selectable admitted V1 pair candidate remained after search",
            ))?;
        let row = self.root_candidates[selected_idx].row;
        let selected = V1SelectedAction::Pair {
            row,
            root_generation: identity.generation,
            legal_row_table_hash: identity.legal_table.table_hash,
        };
        self.telemetry.selected_pair_key = Some(row.pair_key);
        self.telemetry.selected_single = None;
        self.telemetry.search_performed = true;
        self.telemetry.hardcoded_action = false;
        self.telemetry.hardcoded_reason = None;
        self.selected = Some(selected.clone());
        Ok(Some(selected))
    }

    pub fn run_root_search(&mut self) -> Result<Option<V1SelectedAction>, V1PairSearchError> {
        let identity = self
            .root_identity
            .clone()
            .ok_or(V1PairSearchError::InvalidRootState(
                "V1 root must be initialized before search",
            ))?;

        match identity.legal_table.phase {
            TurnPhaseV1::OpeningSingle => {
                let cell = identity
                    .legal_table
                    .rows
                    .first()
                    .map(|row| row.cell)
                    .ok_or(V1PairSearchError::InvalidRootState(
                        "opening_single root has no legal row",
                    ))?;
                let selected = V1SelectedAction::Single {
                    cell,
                    reason: "opening_center",
                    root_generation: identity.generation,
                    legal_row_table_hash: identity.legal_table.table_hash,
                };
                self.record_single_selection(cell, "opening_center");
                self.selected = Some(selected.clone());
                return Ok(Some(selected));
            }
            TurnPhaseV1::OnePlacement => {
                let cell = self.terminal_single_cell(&identity).ok_or(
                    V1PairSearchError::InvalidRootState(
                        "one_placement V1 exception requires an immediate terminal single",
                    ),
                )?;
                let selected = V1SelectedAction::Single {
                    cell,
                    reason: "single_placement_terminal_win",
                    root_generation: identity.generation,
                    legal_row_table_hash: identity.legal_table.table_hash,
                };
                self.record_single_selection(cell, "single_placement_terminal_win");
                self.selected = Some(selected.clone());
                return Ok(Some(selected));
            }
            TurnPhaseV1::Terminal => {
                self.selected = None;
                return Ok(None);
            }
            TurnPhaseV1::NormalTwoPlacement => {}
        }

        if self.root_candidates.is_empty() {
            return Err(V1PairSearchError::InvalidRootState(
                "normal V1 pair search requires admitted root candidate pairs",
            ));
        }
        if !self.root_candidates.iter().any(|candidate| candidate.visit_count > 0) {
            return Err(V1PairSearchError::InvalidRootState(
                "normal V1 pair search requires run_search_step/complete_expansion before select_root_action",
            ));
        }
        return self.select_root_action();
    }

    fn backup_root_candidate(&mut self, idx: usize, value: f32) {
        let candidate = &mut self.root_candidates[idx];
        candidate.visit_count += 1;
        candidate.allocation += 1;
        candidate.total_value += value;
        self.telemetry.simulation_count = self
            .root_candidates
            .iter()
            .map(|candidate| candidate.allocation)
            .sum();
    }

    fn backup_path(&mut self, path: &[V1PathEdge], node_path: &[u64], root_value: f32) {
        let root_player = self
            .root_identity
            .as_ref()
            .map(|identity| identity.legal_table.current_player);
        for edge in path {
            match *edge {
                V1PathEdge::Root(idx) => {
                    self.backup_root_candidate(idx, root_value);
                }
                V1PathEdge::Interior { node_key, edge_idx } => {
                    let Some(node) = self.search_nodes.get(&node_key) else {
                        continue;
                    };
                    let local_value = if root_player
                        .map(|player| player == node.current_player)
                        .unwrap_or(true)
                    {
                        root_value
                    } else {
                        -root_value
                    };
                    if let Some(parent) = self.search_nodes.get_mut(&node_key) {
                        if let Some(reservoir) = parent.reservoir.as_mut() {
                            if edge_idx < reservoir.visits.len() {
                                reservoir.visits[edge_idx] += 1;
                                reservoir.total_values[edge_idx] += local_value;
                            }
                        }
                    }
                }
            }
        }
        for &node_key in node_path {
            if let Some(node) = self.search_nodes.get_mut(&node_key) {
                let local_value = if root_player
                    .map(|player| player == node.current_player)
                    .unwrap_or(true)
                {
                    root_value
                } else {
                    -root_value
                };
                node.visit_count += 1;
                node.total_value += local_value;
            }
        }
    }

    fn ensure_search_node(
        &mut self,
        node_key: u64,
        game: HexGameState,
        parent_edge: Option<V1PathEdge>,
        incoming_pair: Option<PairRowV1>,
        root_player: u8,
    ) -> Result<(), V1PairSearchError> {
        if self.search_nodes.contains_key(&node_key) {
            return Ok(());
        }
        let legal_table = legal_row_table_v1(&game);
        let tactical = terminal_tactical_set_v1(&game);
        let terminal_value_root = if game.is_over() {
            Some(outcome_value(&game, root_player))
        } else {
            None
        };
        self.search_nodes.insert(
            node_key,
            V1SearchNode {
                current_player: game.current_player(),
                game,
                legal_table,
                tactical,
                parent_edge,
                incoming_pair,
                visit_count: 0,
                total_value: 0.0,
                terminal_value_root,
                reservoir: None,
            },
        );
        Ok(())
    }

    fn select_interior_puct_edge(&mut self, node_key: u64) -> Result<usize, V1PairSearchError> {
        self.widen_tree_node_reservoir(node_key)?;
        let node = self
            .search_nodes
            .get(&node_key)
            .ok_or(V1PairSearchError::InteriorNodeNotFound { node_key })?;
        let reservoir = node
            .reservoir
            .as_ref()
            .ok_or(V1PairSearchError::InteriorNodeNotFound { node_key })?;
        let revealed = reservoir.telemetry.revealed_count as usize;
        if revealed == 0 {
            return Err(V1PairSearchError::InvalidRootState(
                "V1 expanded node has no revealed pair actions",
            ));
        }
        let parent_sqrt = (node.visit_count.max(1) as f32).sqrt();
        let revealed_priors = softmax(
            &reservoir.prior_logits[..revealed.min(reservoir.prior_logits.len())],
            self.config.prior_temperature.max(MIN_PRIOR_TEMPERATURE),
        );
        let mut best: Option<(usize, f32)> = None;
        for idx in 0..revealed.min(reservoir.rows.len()) {
            if let Some(child_key) = reservoir.child_keys[idx] {
                if self.pending_expansions.contains_key(&child_key) {
                    continue;
                }
            }
            let visits = reservoir.visits[idx];
            let q = if visits == 0 {
                0.0
            } else {
                reservoir.total_values[idx] / visits as f32
            };
            let prior = revealed_priors.get(idx).copied().unwrap_or(0.0);
            let score = q + self.config.c_puct * prior * parent_sqrt / (1.0 + visits as f32);
            if best
                .map(|(_, best_score)| score > best_score)
                .unwrap_or(true)
            {
                best = Some((idx, score));
            }
        }
        best.map(|(idx, _)| idx).ok_or(V1PairSearchError::InvalidRootState(
            "V1 expanded node has no selectable pair actions",
        ))
    }

    fn widen_tree_node_reservoir(
        &mut self,
        node_key: u64,
    ) -> Result<V1InteriorWideningResult, V1PairSearchError> {
        let node = self
            .search_nodes
            .get_mut(&node_key)
            .ok_or(V1PairSearchError::InteriorNodeNotFound { node_key })?;
        let reservoir = node
            .reservoir
            .as_mut()
            .ok_or(V1PairSearchError::InteriorNodeNotFound { node_key })?;
        let desired = progressive_widen_limit(
            node.visit_count.max(1),
            self.config.c_pw,
            self.config.alpha_pw,
            reservoir.rows.len(),
        );
        let start = reservoir.telemetry.revealed_count as usize;
        let end = desired.max(start).min(reservoir.rows.len());
        let revealed_rows = reservoir.rows[start..end].to_vec();
        reservoir.telemetry.revealed_count = end as u32;
        reservoir.telemetry.widening_events += 1;
        let parent_sqrt = (node.visit_count.max(1) as f32).sqrt();
        let revealed_priors = softmax(
            &reservoir.prior_logits[..end.min(reservoir.prior_logits.len())],
            self.config.prior_temperature.max(MIN_PRIOR_TEMPERATURE),
        );
        let puct_scores = reservoir
            .visits
            .iter()
            .zip(reservoir.total_values.iter())
            .zip(revealed_priors.iter())
            .take(end)
            .map(|((&visits, &total_value), &prior)| {
                let q = if visits == 0 {
                    0.0
                } else {
                    total_value / visits as f32
                };
                q + self.config.c_puct * prior * parent_sqrt / (1.0 + visits as f32)
            })
            .collect::<Vec<_>>();
        Ok(V1InteriorWideningResult {
            telemetry: reservoir.telemetry.clone(),
            revealed_rows,
            puct_scores,
        })
    }

    fn evaluate_structural_leaf_root_value(
        &self,
        node: &V1SearchNode,
        root_player: u8,
    ) -> Result<f32, V1PairSearchError> {
        if node.game.is_over() {
            return Ok(outcome_value(&node.game, root_player));
        }
        if node.legal_table.phase == TurnPhaseV1::OnePlacement {
            if let Some(row) = node.legal_table.rows.first() {
                let mut game = node.game.clone();
                game.place(row.cell.q, row.cell.r)?;
                return Ok(outcome_value(&game, root_player));
            }
        }
        Err(V1PairSearchError::InvalidRootState(
            "V1 recursive search reached a non-terminal non-full-turn leaf without a structural action",
        ))
    }

    pub fn selected_action(&self) -> Option<&V1SelectedAction> {
        self.selected.as_ref()
    }

    pub fn root_candidates(&self) -> &[V1PairCandidate] {
        &self.root_candidates
    }

    pub fn telemetry(&self) -> &V1PairSearchTelemetry {
        &self.telemetry
    }

    pub fn apply_selected_action(
        &mut self,
        root_generation: u64,
        legal_row_table_hash: u64,
        pair_key: Option<u64>,
    ) -> Result<V1AppliedAction, V1PairSearchError> {
        let selected = self
            .selected
            .clone()
            .ok_or(V1PairSearchError::InvalidRootState(
                "V1 apply requires a selected root action",
            ))?;
        let current_table = legal_row_table_v1(&self.game);
        if current_table.schema_version == 0 || current_table.schema_hash == 0 {
            return Err(V1PairSearchError::ApplyIdentityMismatch(
                "current V1 legal row schema identity is missing".to_string(),
            ));
        }
        if current_table.table_hash != legal_row_table_hash {
            return Err(V1PairSearchError::ApplyIdentityMismatch(format!(
                "V1 legal row table hash mismatch: got {legal_row_table_hash}, expected {}",
                current_table.table_hash
            )));
        }

        match selected {
            V1SelectedAction::Single {
                cell,
                root_generation: expected_generation,
                legal_row_table_hash: expected_hash,
                ..
            } => {
                validate_apply_generation(root_generation, expected_generation)?;
                if expected_hash != legal_row_table_hash {
                    return Err(V1PairSearchError::ApplyIdentityMismatch(format!(
                        "selected single legal hash {} does not match supplied {legal_row_table_hash}",
                        expected_hash
                    )));
                }
                if pair_key.is_some() {
                    return Err(V1PairSearchError::ApplyIdentityMismatch(
                        "single-action V1 exception must not supply a pair_key".to_string(),
                    ));
                }
                self.game.place(cell.q, cell.r)?;
                Ok(V1AppliedAction {
                    action_kind: "single",
                    placements_applied: 1,
                    first: cell,
                    second: None,
                    root_generation,
                    legal_row_table_hash,
                    pair_key: None,
                    terminal_after_first: self.game.is_over(),
                })
            }
            V1SelectedAction::Pair {
                row,
                root_generation: expected_generation,
                legal_row_table_hash: expected_hash,
            } => {
                validate_apply_generation(root_generation, expected_generation)?;
                if expected_hash != legal_row_table_hash {
                    return Err(V1PairSearchError::ApplyIdentityMismatch(format!(
                        "selected pair legal hash {} does not match supplied {legal_row_table_hash}",
                        expected_hash
                    )));
                }
                if pair_key != Some(row.pair_key) {
                    return Err(V1PairSearchError::ApplyIdentityMismatch(format!(
                        "V1 pair_key mismatch: got {:?}, expected {}",
                        pair_key, row.pair_key
                    )));
                }
                let backup = self.game.clone();
                let (first, second) = terminal_preferred_pair_order(&self.game, row);
                self.game.place(first.q, first.r)?;
                if self.game.is_over() {
                    return Ok(V1AppliedAction {
                        action_kind: "pair",
                        placements_applied: 1,
                        first,
                        second: Some(second),
                        root_generation,
                        legal_row_table_hash,
                        pair_key: Some(row.pair_key),
                        terminal_after_first: true,
                    });
                }
                if let Err(err) = self.game.place(second.q, second.r) {
                    self.game = backup;
                    return Err(V1PairSearchError::Game(err));
                }
                Ok(V1AppliedAction {
                    action_kind: "pair",
                    placements_applied: 2,
                    first,
                    second: Some(second),
                    root_generation,
                    legal_row_table_hash,
                    pair_key: Some(row.pair_key),
                    terminal_after_first: false,
                })
            }
        }
    }

    pub fn cache_interior_reservoir(
        &mut self,
        node_key: u64,
        pairs: &[(Hex, Hex)],
        model_logits: &[f32],
        correction_weights: &[f32],
        correction_modes: &[ProposalCorrectionModeV1],
    ) -> Result<V1InteriorReservoirTelemetry, V1PairSearchError> {
        let game = self.game.clone();
        self.cache_interior_reservoir_for_game(
            node_key,
            &game,
            pairs,
            model_logits,
            correction_weights,
            correction_modes,
        )
    }

    fn cache_interior_reservoir_for_game(
        &mut self,
        node_key: u64,
        game: &HexGameState,
        pairs: &[(Hex, Hex)],
        model_logits: &[f32],
        correction_weights: &[f32],
        correction_modes: &[ProposalCorrectionModeV1],
    ) -> Result<V1InteriorReservoirTelemetry, V1PairSearchError> {
        if self.interior_reservoirs.contains_key(&node_key) {
            return Err(V1PairSearchError::DuplicateInteriorNode { node_key });
        }
        let reservoir = self.build_interior_reservoir_for_game(
            node_key,
            game,
            pairs,
            model_logits,
            correction_weights,
            correction_modes,
        )?;
        let telemetry = reservoir.telemetry.clone();
        self.interior_reservoirs.insert(node_key, reservoir);
        self.telemetry.interior_expanded_full_turn_nodes += 1;
        self.telemetry.interior_reservoir_build_count += 1;
        self.telemetry.interior_scoring_pass_count += 1;
        Ok(telemetry)
    }

    fn build_interior_reservoir_for_game(
        &self,
        node_key: u64,
        game: &HexGameState,
        pairs: &[(Hex, Hex)],
        model_logits: &[f32],
        correction_weights: &[f32],
        correction_modes: &[ProposalCorrectionModeV1],
    ) -> Result<V1InteriorReservoir, V1PairSearchError> {
        validate_candidate_metadata(pairs, model_logits, correction_weights, correction_modes)?;
        let legal_table = legal_row_table_v1(game);
        if legal_table.phase != TurnPhaseV1::NormalTwoPlacement {
            return Err(V1PairSearchError::InvalidRootState(
                "V1 interior pair reservoir requires normal_two_placement phase",
            ));
        }
        let pair_table = canonical_pair_rows_ordered_v1(&legal_table, pairs)?;
        let mut prior_logits = Vec::with_capacity(pair_table.rows.len());
        for idx in 0..pair_table.rows.len() {
            let mode = correction_modes[idx];
            let weight = correction_weights[idx];
            match mode {
                ProposalCorrectionModeV1::ExactImportance
                | ProposalCorrectionModeV1::ClippedPropensity => {
                    if weight <= 0.0 || !weight.is_finite() {
                        return Err(V1PairSearchError::InvalidCandidate(format!(
                            "proposal correction weight at {idx} must be finite and positive"
                        )));
                    }
                }
                ProposalCorrectionModeV1::UncorrectedLogged
                | ProposalCorrectionModeV1::TrainingForbidden => {}
            };
            prior_logits.push(model_logits[idx]);
        }
        let priors = softmax(
            &prior_logits,
            self.config.prior_temperature.max(MIN_PRIOR_TEMPERATURE),
        );
        let mut order = (0..pair_table.rows.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            prior_logits[b]
                .partial_cmp(&prior_logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    pair_table.rows[a]
                        .pair_key
                        .cmp(&pair_table.rows[b].pair_key)
                })
        });
        let rows = order
            .iter()
            .map(|&idx| pair_table.rows[idx])
            .collect::<Vec<_>>();
        let prior_logits = order.iter().map(|&idx| prior_logits[idx]).collect::<Vec<_>>();
        let priors = order.iter().map(|&idx| priors[idx]).collect::<Vec<_>>();
        let telemetry = V1InteriorReservoirTelemetry {
            node_key,
            candidate_count: rows.len() as u32,
            revealed_count: 0,
            reservoir_build_count: 1,
            scoring_pass_count: 1,
            widening_events: 0,
            reservoir_refill_events: 0,
        };
        Ok(V1InteriorReservoir {
            telemetry,
            rows,
            prior_logits,
            priors,
            visits: vec![0; pair_table.rows.len()],
            total_values: vec![0.0; pair_table.rows.len()],
            child_keys: vec![None; pair_table.rows.len()],
        })
    }

    pub fn widen_interior_reservoir(
        &mut self,
        node_key: u64,
        parent_visits: u32,
    ) -> Result<V1InteriorWideningResult, V1PairSearchError> {
        let reservoir = self
            .interior_reservoirs
            .get_mut(&node_key)
            .ok_or(V1PairSearchError::InteriorNodeNotFound { node_key })?;
        let desired = progressive_widen_limit(
            parent_visits,
            self.config.c_pw,
            self.config.alpha_pw,
            reservoir.rows.len(),
        );
        let start = reservoir.telemetry.revealed_count as usize;
        let end = desired.max(start).min(reservoir.rows.len());
        let revealed_rows = reservoir.rows[start..end].to_vec();
        reservoir.telemetry.revealed_count = end as u32;
        reservoir.telemetry.widening_events += 1;
        let parent_sqrt = (parent_visits.max(1) as f32).sqrt();
        let revealed_priors = softmax(
            &reservoir.prior_logits[..end.min(reservoir.prior_logits.len())],
            self.config.prior_temperature.max(MIN_PRIOR_TEMPERATURE),
        );
        let puct_scores = reservoir
            .visits
            .iter()
            .zip(reservoir.total_values.iter())
            .zip(revealed_priors.iter())
            .take(end)
            .map(|((&visits, &total_value), &prior)| {
                let q = if visits == 0 {
                    0.0
                } else {
                    total_value / visits as f32
                };
                q + self.config.c_puct * prior * parent_sqrt / (1.0 + visits as f32)
            })
            .collect::<Vec<_>>();
        Ok(V1InteriorWideningResult {
            telemetry: reservoir.telemetry.clone(),
            revealed_rows,
            puct_scores,
        })
    }

    fn validate_root_generation(
        &self,
        received: u64,
    ) -> Result<&V1RootIdentity, V1PairSearchError> {
        let identity = self
            .root_identity
            .as_ref()
            .ok_or(V1PairSearchError::InvalidRootState(
                "V1 root must be initialized before this operation",
            ))?;
        if received != identity.generation || received != self.root_generation {
            return Err(V1PairSearchError::StaleRootToken {
                expected: identity.generation,
                received,
            });
        }
        Ok(identity)
    }

    fn record_single_selection(&mut self, cell: Hex, reason: &'static str) {
        self.telemetry.selected_single = Some(cell);
        self.telemetry.selected_pair_key = None;
        self.telemetry.search_performed = false;
        self.telemetry.hardcoded_action = true;
        self.telemetry.hardcoded_reason = Some(reason);
        self.telemetry.admitted_pair_count = 0;
        self.telemetry.simulation_count = 0;
    }

    fn terminal_single_cell(&self, identity: &V1RootIdentity) -> Option<Hex> {
        let legal = identity
            .legal_table
            .rows
            .iter()
            .map(|row| row.cell)
            .collect::<FxHashSet<_>>();
        let mut cells = identity
            .tactical
            .winning_single_cells
            .iter()
            .copied()
            .filter(|cell| legal.contains(cell))
            .collect::<Vec<_>>();
        cells.sort();
        if let Some(cell) = cells.first().copied() {
            return Some(cell);
        }

        identity
            .legal_table
            .rows
            .iter()
            .map(|row| row.cell)
            .find(|&cell| {
                let mut game = self.game.clone();
                game.place(cell.q, cell.r).is_ok()
                    && game.is_over()
                    && game.winner() == Some(identity.legal_table.current_player)
            })
    }

    fn admit_gumbel_root_set(&mut self) -> Result<(), V1PairSearchError> {
        let selectable_count = self
            .root_candidates
            .iter()
            .filter(|candidate| {
                candidate.correction_mode != ProposalCorrectionModeV1::TrainingForbidden
            })
            .count();
        if selectable_count == 0 {
            return Err(V1PairSearchError::InvalidCandidate(
                "V1 root admission has no selectable candidates; training_forbidden rows are diagnostic only"
                    .to_string(),
            ));
        }

        for candidate in &mut self.root_candidates {
            candidate.admitted = false;
        }

        let max_admitted = self
            .config
            .max_root_admitted
            .unwrap_or(selectable_count)
            .max(self.config.min_root_admitted)
            .min(selectable_count);
        let mut admitted = FxHashSet::<usize>::default();
        for (idx, candidate) in self.root_candidates.iter().enumerate() {
            if candidate.forced_exploration_flag
                && candidate.correction_mode != ProposalCorrectionModeV1::TrainingForbidden
            {
                admitted.insert(idx);
            }
        }
        let mut ranked = self
            .root_candidates
            .iter()
            .enumerate()
            .filter(|(idx, candidate)| {
                !admitted.contains(idx)
                    && candidate.correction_mode != ProposalCorrectionModeV1::TrainingForbidden
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|(_, left), (_, right)| compare_candidate_admission(left, right));
        for (idx, _) in ranked {
            if admitted.len() >= max_admitted {
                break;
            }
            admitted.insert(idx);
        }
        if admitted.is_empty() {
            return Err(V1PairSearchError::InvalidRootState(
                "V1 Gumbel admission selected no root candidates",
            ));
        }
        for idx in admitted {
            self.root_candidates[idx].admitted = true;
        }
        self.telemetry.admitted_pair_count = self
            .root_candidates
            .iter()
            .filter(|candidate| candidate.admitted)
            .count() as u32;
        Ok(())
    }

    fn complete_q_values(&mut self) {
        let mut visited_sum = 0.0;
        let mut visited_count = 0u32;
        for candidate in &self.root_candidates {
            if candidate.visit_count > 0 {
                visited_sum += candidate.q_value();
                visited_count += 1;
            }
        }
        let fallback = if visited_count == 0 {
            0.0
        } else {
            visited_sum / visited_count as f32
        };
        for candidate in &mut self.root_candidates {
            candidate.completed_q = if candidate.visit_count > 0 {
                candidate.q_value()
            } else {
                fallback
            };
        }
    }
}

fn validate_apply_generation(received: u64, expected: u64) -> Result<(), V1PairSearchError> {
    if received == expected {
        Ok(())
    } else {
        Err(V1PairSearchError::StaleRootToken { expected, received })
    }
}

fn validate_candidate_metadata(
    pairs: &[(Hex, Hex)],
    model_logits: &[f32],
    correction_weights: &[f32],
    correction_modes: &[ProposalCorrectionModeV1],
) -> Result<(), V1PairSearchError> {
    let n = pairs.len();
    if n == 0 {
        return Err(V1PairSearchError::InvalidCandidate(
            "V1 pair admission requires at least one candidate pair".to_string(),
        ));
    }
    if model_logits.len() != n || correction_weights.len() != n || correction_modes.len() != n {
        return Err(V1PairSearchError::InvalidCandidate(format!(
            "V1 pair candidate metadata length mismatch: pairs={n} logits={} correction_weights={} correction_modes={}",
            model_logits.len(),
            correction_weights.len(),
            correction_modes.len()
        )));
    }
    if let Some(index) = model_logits.iter().position(|value| !value.is_finite()) {
        return Err(V1PairSearchError::InvalidCandidate(format!(
            "V1 pair model logit at {index} is not finite"
        )));
    }
    if let Some(index) = correction_weights
        .iter()
        .position(|value| !value.is_finite())
    {
        return Err(V1PairSearchError::InvalidCandidate(format!(
            "V1 proposal correction weight at {index} is not finite"
        )));
    }
    Ok(())
}

fn assign_priors(candidates: &mut [V1PairCandidate], temperature: f32) {
    let logits = candidates
        .iter()
        .map(|candidate| candidate.prior_logit)
        .collect::<Vec<_>>();
    let priors = softmax(&logits, temperature);
    for (candidate, prior) in candidates.iter_mut().zip(priors) {
        candidate.prior = prior;
    }
}

fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let inv_t = 1.0 / temperature.max(MIN_PRIOR_TEMPERATURE);
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b * inv_t));
    let mut values = Vec::with_capacity(logits.len());
    let mut total = 0.0;
    for &logit in logits {
        let value = (logit * inv_t - max_logit).exp();
        values.push(value);
        total += value;
    }
    if total > 0.0 && total.is_finite() {
        for value in &mut values {
            *value /= total;
        }
    } else {
        let uniform = 1.0 / logits.len() as f32;
        values.fill(uniform);
    }
    values
}

fn tactical_pair_keys(tactical: &TerminalTacticalSetV1) -> FxHashSet<u64> {
    tactical
        .hot_completion_pairs
        .iter()
        .chain(tactical.terminal_equivalent_pairs.iter())
        .chain(tactical.hot_cover_pairs.iter())
        .map(|row| row.pair_key)
        .collect()
}

fn outcome_value(game: &HexGameState, root_player: u8) -> f32 {
    if game.winner() == Some(root_player) {
        1.0
    } else {
        -1.0
    }
}

fn terminal_preferred_pair_order(game: &HexGameState, row: PairRowV1) -> (Hex, Hex) {
    let player = game.current_player();
    let first_wins = immediate_win_if_placed(game, row.first, player);
    let second_wins = immediate_win_if_placed(game, row.second, player);
    if second_wins && !first_wins {
        (row.second, row.first)
    } else {
        (row.first, row.second)
    }
}

fn immediate_win_if_placed(game: &HexGameState, cell: Hex, player: u8) -> bool {
    let mut probe = game.clone();
    probe.place(cell.q, cell.r).is_ok() && probe.is_over() && probe.winner() == Some(player)
}

fn compare_candidate_admission(
    left: &V1PairCandidate,
    right: &V1PairCandidate,
) -> std::cmp::Ordering {
    let left_score = left.prior_logit + left.gumbel;
    let right_score = right.prior_logit + right.gumbel;
    right_score
        .partial_cmp(&left_score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.row.pair_key.cmp(&right.row.pair_key))
}

fn compare_candidate_round(left: &V1PairCandidate, right: &V1PairCandidate) -> std::cmp::Ordering {
    let left_score = left.q_value() + left.prior_logit * 0.01 + left.gumbel * 0.001;
    let right_score = right.q_value() + right.prior_logit * 0.01 + right.gumbel * 0.001;
    right_score
        .partial_cmp(&left_score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.row.pair_key.cmp(&right.row.pair_key))
}

fn compare_candidate_final(left: &V1PairCandidate, right: &V1PairCandidate) -> std::cmp::Ordering {
    left.completed_q
        .partial_cmp(&right.completed_q)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.visit_count.cmp(&right.visit_count))
        .then_with(|| {
            left.prior
                .partial_cmp(&right.prior)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| right.row.pair_key.cmp(&left.row.pair_key))
}

fn gumbel_from_seed(seed: u64, root_generation: u64, pair_key: u64) -> f32 {
    let mut state = seed.max(1) ^ root_generation.rotate_left(17) ^ pair_key.rotate_right(11);
    let uniform = next_uniform(&mut state).clamp(1.0e-6, 1.0 - 1.0e-6);
    -(-uniform.ln()).ln()
}

fn next_uniform(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x as f64 / u64::MAX as f64) as f32
}

fn child_node_key(parent_key: u64, pair_key: u64) -> u64 {
    let mut x = parent_key
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ pair_key.rotate_left(17)
        ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn progressive_widen_limit(
    parent_visits: u32,
    c_pw: f32,
    alpha_pw: f32,
    candidate_count: usize,
) -> usize {
    if candidate_count == 0 {
        return 0;
    }
    let limit = (c_pw.max(1.0) * (parent_visits.max(1) as f32).powf(alpha_pw.clamp(0.1, 1.0)))
        .ceil() as usize;
    limit.max(1).min(candidate_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normal_engine() -> (V1PairSearchEngine, V1RootInit) {
        let mut game = HexGameState::new();
        game.place(0, 0).expect("opening");
        let mut engine = V1PairSearchEngine::new(
            game,
            V1PairSearchConfig {
                num_simulations: 24,
                seed: 7,
                max_root_admitted: Some(4),
                ..V1PairSearchConfig::default()
            },
        );
        let init = engine.init_root();
        (engine, init)
    }

    fn request_reservoir_pairs(request: &V1ExpansionRequest) -> Vec<(Hex, Hex)> {
        assert_eq!(request.legal_table.phase, TurnPhaseV1::NormalTwoPlacement);
        let a = request.legal_table.rows[0].cell;
        let b = request.legal_table.rows[1].cell;
        vec![(a, b)]
    }

    fn complete_request_with_value(
        engine: &mut V1PairSearchEngine,
        request: &V1ExpansionRequest,
        value: f32,
    ) -> V1InteriorReservoirTelemetry {
        let pairs = request_reservoir_pairs(request);
        engine
            .complete_expansion(
                request.node_key,
                value,
                &pairs,
                &[0.0],
                &[1.0],
                &[ProposalCorrectionModeV1::ExactImportance],
            )
            .expect("complete expansion")
    }

    #[test]
    fn v1_root_pair_admission_rejects_stale_duplicate_and_illegal_rows() {
        let (mut engine, init) = normal_engine();
        let a = init.legal_table.rows[0].cell;
        let b = init.legal_table.rows[1].cell;

        let stale = engine.admit_root_pairs(
            init.root_generation + 1,
            &[(a, b)],
            &[1.0],
            &[1.0],
            &[ProposalCorrectionModeV1::ExactImportance],
        );
        assert!(matches!(
            stale,
            Err(V1PairSearchError::StaleRootToken { .. })
        ));

        let duplicate = engine.admit_root_pairs(
            init.root_generation,
            &[(a, b), (b, a)],
            &[1.0, 2.0],
            &[1.0, 1.0],
            &[
                ProposalCorrectionModeV1::ExactImportance,
                ProposalCorrectionModeV1::ExactImportance,
            ],
        );
        assert!(matches!(
            duplicate,
            Err(V1PairSearchError::PairRows(
                PairRowErrorV1::DuplicatePair { .. }
            ))
        ));

        let illegal = engine.admit_root_pairs(
            init.root_generation,
            &[(a, Hex::new(999, 999))],
            &[1.0],
            &[1.0],
            &[ProposalCorrectionModeV1::ExactImportance],
        );
        assert!(matches!(
            illegal,
            Err(V1PairSearchError::PairRows(
                PairRowErrorV1::IllegalCell { .. }
            ))
        ));
    }

    #[test]
    fn v1_neural_expansion_value_backup_changes_root_choice_and_applies_pair() {
        let (mut engine, init) = normal_engine();
        engine.config.num_simulations = 2;
        let a = init.legal_table.rows[0].cell;
        let b = init.legal_table.rows[1].cell;
        let c = init.legal_table.rows[2].cell;
        let d = init.legal_table.rows[3].cell;
        engine
            .admit_root_pairs(
                init.root_generation,
                &[(a, b), (c, d)],
                &[0.0, 8.0],
                &[1.0, 1.0],
                &[
                    ProposalCorrectionModeV1::ExactImportance,
                    ProposalCorrectionModeV1::ExactImportance,
                ],
            )
            .expect("admit pairs");
        let winning_key = engine
            .root_candidates()
            .iter()
            .find(|candidate| candidate.row.first == a && candidate.row.second == b)
            .expect("canonical winning pair")
            .row
            .pair_key;
        let requests = engine.run_search_step(8).expect("expansion requests");
        assert_eq!(requests.len(), 2);
        for request in &requests {
            let value = if request.node_key == winning_key {
                1.0
            } else {
                -1.0
            };
            complete_request_with_value(&mut engine, request, value);
        }
        let selected = engine
            .select_root_action()
            .expect("search")
            .expect("selected pair");
        let V1SelectedAction::Pair {
            row,
            root_generation,
            legal_row_table_hash,
        } = selected
        else {
            panic!("normal root must select a pair");
        };
        assert_eq!(row.first_legal_row_id, 0);
        assert_eq!(row.second_legal_row_id, 1);
        assert_eq!(row.first, a);
        assert_eq!(row.second, b);
        assert!(engine
            .root_candidates()
            .iter()
            .all(|candidate| candidate.visit_count > 0 || !candidate.admitted));

        let applied = engine
            .apply_selected_action(root_generation, legal_row_table_hash, Some(row.pair_key))
            .expect("apply pair");
        assert_eq!(applied.action_kind, "pair");
        assert_eq!(applied.placements_applied, 2);
        assert_eq!(engine.game().move_count(), 3);
        assert_eq!(engine.game().current_player(), 0);
        assert!(engine.game().stones().contains_key(&a));
        assert!(engine.game().stones().contains_key(&b));
    }

    #[test]
    fn v1_opening_and_one_placement_terminal_are_single_action_exceptions() {
        let mut opening = V1PairSearchEngine::new(
            HexGameState::new(),
            V1PairSearchConfig {
                num_simulations: 0,
                ..V1PairSearchConfig::default()
            },
        );
        let init = opening.init_root();
        let selected = opening.run_root_search().unwrap().unwrap();
        let V1SelectedAction::Single {
            cell,
            reason,
            root_generation,
            legal_row_table_hash,
        } = selected
        else {
            panic!("opening must be a single action exception");
        };
        assert_eq!(cell, Hex::ORIGIN);
        assert_eq!(reason, "opening_center");
        let applied = opening
            .apply_selected_action(root_generation, legal_row_table_hash, None)
            .unwrap();
        assert_eq!(applied.placements_applied, 1);
        assert_eq!(opening.game().move_count(), 1);
        assert_eq!(init.legal_pair_count, 0);

        let mut game = HexGameState::new();
        game.set_position(
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)],
            0,
            1,
        )
        .expect("one-placement terminal fixture");
        let mut one = V1PairSearchEngine::new(game, V1PairSearchConfig::default());
        let init = one.init_root();
        assert_eq!(init.legal_table.phase, TurnPhaseV1::OnePlacement);
        let selected = one.run_root_search().unwrap().unwrap();
        let V1SelectedAction::Single { reason, .. } = selected else {
            panic!("one-placement terminal must be a single action exception");
        };
        assert_eq!(reason, "single_placement_terminal_win");
    }

    #[test]
    fn v1_apply_rejects_stale_token_and_pair_key_mismatch() {
        let (mut engine, init) = normal_engine();
        engine.config.num_simulations = 1;
        let a = init.legal_table.rows[0].cell;
        let b = init.legal_table.rows[1].cell;
        engine
            .admit_root_pairs(
                init.root_generation,
                &[(a, b)],
                &[3.0],
                &[1.0],
                &[ProposalCorrectionModeV1::ExactImportance],
            )
            .unwrap();
        let requests = engine.run_search_step(1).unwrap();
        assert_eq!(requests.len(), 1);
        complete_request_with_value(&mut engine, &requests[0], 1.0);
        let selected = engine.select_root_action().unwrap().unwrap();
        let V1SelectedAction::Pair {
            row,
            root_generation,
            legal_row_table_hash,
        } = selected
        else {
            panic!("normal root must select pair");
        };
        assert!(matches!(
            engine.apply_selected_action(
                root_generation + 1,
                legal_row_table_hash,
                Some(row.pair_key)
            ),
            Err(V1PairSearchError::StaleRootToken { .. })
        ));
        assert!(matches!(
            engine.apply_selected_action(
                root_generation,
                legal_row_table_hash,
                Some(row.pair_key + 1)
            ),
            Err(V1PairSearchError::ApplyIdentityMismatch(_))
        ));
    }

    #[test]
    fn v1_normal_root_search_rejects_shallow_nonterminal_fallback() {
        let (mut engine, init) = normal_engine();
        engine.config.num_simulations = 1;
        let a = init.legal_table.rows[0].cell;
        let b = init.legal_table.rows[1].cell;
        engine
            .admit_root_pairs(
                init.root_generation,
                &[(a, b)],
                &[3.0],
                &[1.0],
                &[ProposalCorrectionModeV1::ExactImportance],
            )
            .unwrap();
        assert!(matches!(
            engine.run_root_search(),
            Err(V1PairSearchError::InvalidRootState(message))
                if message.contains("run_search_step/complete_expansion")
        ));
    }

    #[test]
    fn v1_expansion_completion_rejects_stale_child_pair_identity() {
        let (mut engine, init) = normal_engine();
        engine.config.num_simulations = 1;
        let a = init.legal_table.rows[0].cell;
        let b = init.legal_table.rows[1].cell;
        engine
            .admit_root_pairs(
                init.root_generation,
                &[(a, b)],
                &[3.0],
                &[1.0],
                &[ProposalCorrectionModeV1::ExactImportance],
            )
            .unwrap();
        let requests = engine.run_search_step(1).unwrap();
        let stale = engine.complete_expansion(
            requests[0].node_key,
            0.0,
            &[(a, b)],
            &[0.0],
            &[1.0],
            &[ProposalCorrectionModeV1::ExactImportance],
        );
        assert!(matches!(
            stale,
            Err(V1PairSearchError::PairRows(PairRowErrorV1::IllegalCell { .. }))
        ));
    }

    #[test]
    fn v1_search_step_builds_one_reservoir_per_expanded_node_and_widens_from_cache() {
        let (mut engine, init) = normal_engine();
        engine.config.num_simulations = 2;
        let a = init.legal_table.rows[0].cell;
        let b = init.legal_table.rows[1].cell;
        let c = init.legal_table.rows[2].cell;
        let d = init.legal_table.rows[3].cell;
        engine
            .admit_root_pairs(
                init.root_generation,
                &[(a, b), (c, d)],
                &[3.0, 2.0],
                &[1.0, 1.0],
                &[
                    ProposalCorrectionModeV1::ExactImportance,
                    ProposalCorrectionModeV1::ExactImportance,
                ],
            )
            .unwrap();
        let requests = engine.run_search_step(2).unwrap();
        assert_eq!(requests.len(), 2);
        for request in &requests {
            let telemetry = complete_request_with_value(&mut engine, request, 0.25);
            assert_eq!(telemetry.reservoir_build_count, 1);
            assert_eq!(telemetry.scoring_pass_count, 1);
            assert_eq!(telemetry.reservoir_refill_events, 0);
            assert!(telemetry.widening_events >= 1);
        }
        assert_eq!(engine.telemetry().interior_expanded_full_turn_nodes, 2);
        assert_eq!(engine.telemetry().interior_reservoir_build_count, 2);
        assert_eq!(engine.telemetry().interior_scoring_pass_count, 2);
    }

    #[test]
    fn v1_recursive_search_descends_past_depth_one_after_expansion() {
        let (mut engine, init) = normal_engine();
        engine.config.num_simulations = 2;
        let a = init.legal_table.rows[0].cell;
        let b = init.legal_table.rows[1].cell;
        engine
            .admit_root_pairs(
                init.root_generation,
                &[(a, b)],
                &[3.0],
                &[1.0],
                &[ProposalCorrectionModeV1::ExactImportance],
            )
            .unwrap();
        let first = engine.run_search_step(1).unwrap();
        assert_eq!(first.len(), 1);
        let first_key = first[0].node_key;
        complete_request_with_value(&mut engine, &first[0], 0.5);

        let second = engine.run_search_step(1).unwrap();
        assert_eq!(second.len(), 1);
        assert_ne!(second[0].node_key, first_key);
        complete_request_with_value(&mut engine, &second[0], 0.25);

        assert_eq!(engine.telemetry().simulation_count, 2);
        assert_eq!(engine.telemetry().interior_expanded_full_turn_nodes, 2);
        assert!(engine.select_root_action().unwrap().is_some());
    }

    #[test]
    fn v1_search_priors_ignore_proposal_correction_weights() {
        let (mut engine, init) = normal_engine();
        let a = init.legal_table.rows[0].cell;
        let b = init.legal_table.rows[1].cell;
        let c = init.legal_table.rows[2].cell;
        let d = init.legal_table.rows[3].cell;
        engine
            .admit_root_pairs(
                init.root_generation,
                &[(a, b), (c, d)],
                &[2.0, 0.0],
                &[0.001, 1000.0],
                &[
                    ProposalCorrectionModeV1::ExactImportance,
                    ProposalCorrectionModeV1::ExactImportance,
                ],
            )
            .unwrap();
        let first = engine
            .root_candidates()
            .iter()
            .find(|candidate| candidate.row.first == a && candidate.row.second == b)
            .unwrap();
        let second = engine
            .root_candidates()
            .iter()
            .find(|candidate| candidate.row.first == c && candidate.row.second == d)
            .unwrap();
        assert!(first.prior > second.prior);
        assert_eq!(first.prior_logit, 2.0);
        assert_eq!(second.prior_logit, 0.0);
    }

    #[test]
    fn v1_terminal_pair_order_applies_winning_cell_before_filler() {
        let mut game = HexGameState::new();
        game.set_position(&[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], 0, 2)
            .expect("terminal-order fixture");
        let table = legal_row_table_v1(&game);
        assert_eq!(table.phase, TurnPhaseV1::NormalTwoPlacement);
        let winning = Hex::new(4, 0);
        let filler = table
            .rows
            .iter()
            .map(|row| row.cell)
            .find(|&cell| cell != winning)
            .expect("filler legal cell");
        let row = canonical_pair_rows_ordered_v1(&table, &[(filler, winning)])
            .unwrap()
            .rows[0];
        let (first, second) = terminal_preferred_pair_order(&game, row);
        assert_eq!(first, winning);
        assert_eq!(second, filler);
    }

    #[test]
    fn v1_interior_reservoir_scores_once_and_widens_from_cache() {
        let (mut engine, init) = normal_engine();
        let a = init.legal_table.rows[0].cell;
        let b = init.legal_table.rows[1].cell;
        let c = init.legal_table.rows[2].cell;
        let d = init.legal_table.rows[3].cell;
        let telemetry = engine
            .cache_interior_reservoir(
                42,
                &[(a, b), (c, d)],
                &[2.0, 0.0],
                &[1.0, 1.0],
                &[
                    ProposalCorrectionModeV1::ExactImportance,
                    ProposalCorrectionModeV1::ExactImportance,
                ],
            )
            .expect("cache interior reservoir");
        assert_eq!(telemetry.reservoir_build_count, 1);
        assert_eq!(telemetry.scoring_pass_count, 1);

        let widened = engine.widen_interior_reservoir(42, 4).unwrap();
        assert!(!widened.revealed_rows.is_empty());
        assert_eq!(widened.telemetry.reservoir_build_count, 1);
        assert_eq!(widened.telemetry.scoring_pass_count, 1);
        assert!(!widened.puct_scores.is_empty());

        assert!(matches!(
            engine.cache_interior_reservoir(
                42,
                &[(a, b)],
                &[1.0],
                &[1.0],
                &[ProposalCorrectionModeV1::ExactImportance],
            ),
            Err(V1PairSearchError::DuplicateInteriorNode { node_key: 42 })
        ));
    }

    #[test]
    fn v1_widened_reservoir_renormalizes_priors_over_revealed_rows() {
        let (mut engine, init) = normal_engine();
        engine.config.c_pw = 1.0;
        let a = init.legal_table.rows[0].cell;
        let b = init.legal_table.rows[1].cell;
        let c = init.legal_table.rows[2].cell;
        let d = init.legal_table.rows[3].cell;
        let telemetry = engine
            .cache_interior_reservoir(
                77,
                &[(a, b), (c, d)],
                &[0.0, 0.0],
                &[1.0, 1.0],
                &[
                    ProposalCorrectionModeV1::ExactImportance,
                    ProposalCorrectionModeV1::ExactImportance,
                ],
            )
            .expect("cache interior reservoir");
        assert_eq!(telemetry.candidate_count, 2);
        let widened = engine.widen_interior_reservoir(77, 1).unwrap();
        assert_eq!(widened.revealed_rows.len(), 1);
        assert_eq!(widened.puct_scores.len(), 1);
        assert!(
            widened.puct_scores[0] > engine.config.c_puct * 0.9,
            "single revealed row should receive nearly all revealed prior mass"
        );
    }
}
