//! V1 row-identity and tactical payload contracts.
//!
//! These types are the Rust-owned boundary objects for V1 pair actions.  They
//! keep full start-of-turn legal rows distinct from sampled/admitted rows and
//! make pair rows reference legal-row IDs instead of raw coordinate order.

use crate::board::HexGameState;
use crate::core::Hex;
use crate::threats::{live_cells, tactical_status, TacticalStatus};
use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;

pub const LEGAL_ROW_SCHEMA_VERSION_V1: u32 = 1;
pub const PAIR_ROW_SCHEMA_VERSION_V1: u32 = 1;
pub const TERMINAL_TACTICAL_SCHEMA_VERSION_V1: u32 = 1;

pub const LEGAL_ROW_SCHEMA_HASH_V1: u64 = 0x4417_6f3a_51dc_9b2d;
pub const PAIR_ROW_SCHEMA_HASH_V1: u64 = 0xa4b2_d6e9_1267_0c53;
pub const TERMINAL_TACTICAL_SCHEMA_HASH_V1: u64 = 0xd91f_0875_6f3b_c2a4;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TurnPhaseV1 {
    OpeningSingle,
    NormalTwoPlacement,
    OnePlacement,
    Terminal,
}

impl TurnPhaseV1 {
    pub const fn as_str(self) -> &'static str {
        match self {
            TurnPhaseV1::OpeningSingle => "opening_single",
            TurnPhaseV1::NormalTwoPlacement => "normal_two_placement",
            TurnPhaseV1::OnePlacement => "one_placement",
            TurnPhaseV1::Terminal => "terminal",
        }
    }

    const fn code(self) -> u64 {
        match self {
            TurnPhaseV1::OpeningSingle => 1,
            TurnPhaseV1::NormalTwoPlacement => 2,
            TurnPhaseV1::OnePlacement => 3,
            TurnPhaseV1::Terminal => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryPhaseV1 {
    TurnStart,
    TurnContinuation,
    Terminal,
}

impl QueryPhaseV1 {
    pub const fn as_str(self) -> &'static str {
        match self {
            QueryPhaseV1::TurnStart => "turn_start",
            QueryPhaseV1::TurnContinuation => "turn_continuation",
            QueryPhaseV1::Terminal => "terminal",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LegalRowV1 {
    pub row_id: u32,
    pub cell: Hex,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegalRowTableV1 {
    pub schema_version: u32,
    pub schema_hash: u64,
    pub table_hash: u64,
    pub phase: TurnPhaseV1,
    pub query_phase: QueryPhaseV1,
    pub current_player: u8,
    pub placements_remaining: u8,
    pub current_placements_remaining: u8,
    pub turn_start_move_count: u32,
    pub current_move_count: u32,
    pub turn_start_state_hash: u64,
    pub current_state_hash: u64,
    pub first_placement_row_id: Option<u32>,
    pub rows: Vec<LegalRowV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairRowV1 {
    pub row_id: u32,
    pub first_legal_row_id: u32,
    pub second_legal_row_id: u32,
    pub first: Hex,
    pub second: Hex,
    pub pair_key: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairRowTableV1 {
    pub schema_version: u32,
    pub schema_hash: u64,
    pub table_hash: u64,
    pub legal_row_schema_version: u32,
    pub legal_row_schema_hash: u64,
    pub legal_row_table_hash: u64,
    pub phase: TurnPhaseV1,
    pub rows: Vec<PairRowV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalTacticalStatusV1 {
    Quiet,
    HotCompletionAvailable,
    HotCoverRequired,
    HotCoverImpossible,
}

impl TerminalTacticalStatusV1 {
    pub const fn as_str(self) -> &'static str {
        match self {
            TerminalTacticalStatusV1::Quiet => "quiet",
            TerminalTacticalStatusV1::HotCompletionAvailable => "hot_completion_available",
            TerminalTacticalStatusV1::HotCoverRequired => "hot_cover_required",
            TerminalTacticalStatusV1::HotCoverImpossible => "hot_cover_impossible",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalTacticalSetV1 {
    pub schema_version: u32,
    pub schema_hash: u64,
    pub legal_row_table_hash: u64,
    pub pair_row_schema_version: u32,
    pub pair_row_schema_hash: u64,
    pub phase: TurnPhaseV1,
    pub status: TerminalTacticalStatusV1,
    pub winning_single_cells: Vec<Hex>,
    pub hot_completion_pairs: Vec<PairRowV1>,
    pub terminal_equivalent_pairs: Vec<PairRowV1>,
    pub opponent_win_requirements: Vec<Hex>,
    pub hot_cover_pairs: Vec<PairRowV1>,
    pub impossible_to_cover: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairRowErrorV1 {
    WrongPhase {
        phase: TurnPhaseV1,
    },
    DuplicateCell {
        cell: Hex,
    },
    IllegalCell {
        cell: Hex,
    },
    DuplicatePair {
        first_legal_row_id: u32,
        second_legal_row_id: u32,
    },
    RowIdOverflow {
        row_count: usize,
    },
}

impl fmt::Display for PairRowErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PairRowErrorV1::WrongPhase { phase } => write!(
                f,
                "V1 pair rows require normal_two_placement phase, got {}",
                phase.as_str()
            ),
            PairRowErrorV1::DuplicateCell { cell } => {
                write!(
                    f,
                    "V1 pair row contains duplicate cell ({}, {})",
                    cell.q, cell.r
                )
            }
            PairRowErrorV1::IllegalCell { cell } => write!(
                f,
                "V1 pair row references illegal cell ({}, {})",
                cell.q, cell.r
            ),
            PairRowErrorV1::DuplicatePair {
                first_legal_row_id,
                second_legal_row_id,
            } => write!(
                f,
                "V1 pair row duplicate canonical pair ({first_legal_row_id}, {second_legal_row_id})"
            ),
            PairRowErrorV1::RowIdOverflow { row_count } => {
                write!(f, "V1 row count {row_count} exceeds u32::MAX")
            }
        }
    }
}

impl std::error::Error for PairRowErrorV1 {}

pub fn legal_row_table_v1(game: &HexGameState) -> LegalRowTableV1 {
    let current_state_hash = state_hash_v1(game);
    let (turn_start, first_placement) = start_of_turn_state_v1(game);
    let mut legal = turn_start.legal_moves();
    legal.sort();

    let rows = legal
        .into_iter()
        .enumerate()
        .map(|(row_id, cell)| LegalRowV1 {
            row_id: row_id as u32,
            cell,
        })
        .collect::<Vec<_>>();

    let phase = turn_phase_v1(&turn_start, game);
    let query_phase = if game.is_over() {
        QueryPhaseV1::Terminal
    } else if first_placement.is_some() {
        QueryPhaseV1::TurnContinuation
    } else {
        QueryPhaseV1::TurnStart
    };
    let first_placement_row_id = first_placement.and_then(|cell| {
        rows.iter()
            .find(|row| row.cell == cell)
            .map(|row| row.row_id)
    });
    let turn_start_state_hash = state_hash_v1(&turn_start);
    let table_hash = legal_table_hash_v1(
        phase,
        turn_start.current_player(),
        turn_start.placements_remaining(),
        turn_start.move_count(),
        turn_start_state_hash,
        &rows,
    );

    LegalRowTableV1 {
        schema_version: LEGAL_ROW_SCHEMA_VERSION_V1,
        schema_hash: LEGAL_ROW_SCHEMA_HASH_V1,
        table_hash,
        phase,
        query_phase,
        current_player: turn_start.current_player(),
        placements_remaining: turn_start.placements_remaining(),
        current_placements_remaining: game.placements_remaining(),
        turn_start_move_count: turn_start.move_count(),
        current_move_count: game.move_count(),
        turn_start_state_hash,
        current_state_hash,
        first_placement_row_id,
        rows,
    }
}

pub fn pair_row_table_v1(table: &LegalRowTableV1) -> Result<PairRowTableV1, PairRowErrorV1> {
    ensure_pair_phase(table)?;
    let n = table.rows.len();
    let pair_count = n.saturating_mul(n.saturating_sub(1)) / 2;
    if pair_count > u32::MAX as usize {
        return Err(PairRowErrorV1::RowIdOverflow {
            row_count: pair_count,
        });
    }

    let mut rows = Vec::with_capacity(pair_count);
    for first_idx in 0..n {
        for second_idx in (first_idx + 1)..n {
            rows.push(pair_row_from_ids(
                rows.len() as u32,
                table,
                first_idx as u32,
                second_idx as u32,
            ));
        }
    }
    Ok(PairRowTableV1 {
        schema_version: PAIR_ROW_SCHEMA_VERSION_V1,
        schema_hash: PAIR_ROW_SCHEMA_HASH_V1,
        table_hash: pair_table_hash_v1(table, &rows),
        legal_row_schema_version: table.schema_version,
        legal_row_schema_hash: table.schema_hash,
        legal_row_table_hash: table.table_hash,
        phase: table.phase,
        rows,
    })
}

pub fn canonical_pair_rows_v1(
    table: &LegalRowTableV1,
    pairs: &[(Hex, Hex)],
) -> Result<PairRowTableV1, PairRowErrorV1> {
    ensure_pair_phase(table)?;
    if pairs.len() > u32::MAX as usize {
        return Err(PairRowErrorV1::RowIdOverflow {
            row_count: pairs.len(),
        });
    }

    let row_lookup = legal_row_lookup(table);
    let mut seen = FxHashSet::default();
    let mut canonical = Vec::<(u32, u32)>::with_capacity(pairs.len());
    for &(a, b) in pairs {
        if a == b {
            return Err(PairRowErrorV1::DuplicateCell { cell: a });
        }
        let a_id = *row_lookup
            .get(&a)
            .ok_or(PairRowErrorV1::IllegalCell { cell: a })?;
        let b_id = *row_lookup
            .get(&b)
            .ok_or(PairRowErrorV1::IllegalCell { cell: b })?;
        let (first_id, second_id) = if a_id < b_id {
            (a_id, b_id)
        } else {
            (b_id, a_id)
        };
        if !seen.insert((first_id, second_id)) {
            return Err(PairRowErrorV1::DuplicatePair {
                first_legal_row_id: first_id,
                second_legal_row_id: second_id,
            });
        }
        canonical.push((first_id, second_id));
    }

    canonical.sort();
    let rows = canonical
        .into_iter()
        .enumerate()
        .map(|(row_id, (first_id, second_id))| {
            pair_row_from_ids(row_id as u32, table, first_id, second_id)
        })
        .collect::<Vec<_>>();

    Ok(PairRowTableV1 {
        schema_version: PAIR_ROW_SCHEMA_VERSION_V1,
        schema_hash: PAIR_ROW_SCHEMA_HASH_V1,
        table_hash: pair_table_hash_v1(table, &rows),
        legal_row_schema_version: table.schema_version,
        legal_row_schema_hash: table.schema_hash,
        legal_row_table_hash: table.table_hash,
        phase: table.phase,
        rows,
    })
}

pub fn canonical_pair_rows_ordered_v1(
    table: &LegalRowTableV1,
    pairs: &[(Hex, Hex)],
) -> Result<PairRowTableV1, PairRowErrorV1> {
    ensure_pair_phase(table)?;
    if pairs.len() > u32::MAX as usize {
        return Err(PairRowErrorV1::RowIdOverflow {
            row_count: pairs.len(),
        });
    }

    let row_lookup = legal_row_lookup(table);
    let mut seen = FxHashSet::default();
    let mut rows = Vec::<PairRowV1>::with_capacity(pairs.len());
    for (row_id, &(a, b)) in pairs.iter().enumerate() {
        if a == b {
            return Err(PairRowErrorV1::DuplicateCell { cell: a });
        }
        let a_id = *row_lookup
            .get(&a)
            .ok_or(PairRowErrorV1::IllegalCell { cell: a })?;
        let b_id = *row_lookup
            .get(&b)
            .ok_or(PairRowErrorV1::IllegalCell { cell: b })?;
        let (first_id, second_id) = if a_id < b_id {
            (a_id, b_id)
        } else {
            (b_id, a_id)
        };
        if !seen.insert((first_id, second_id)) {
            return Err(PairRowErrorV1::DuplicatePair {
                first_legal_row_id: first_id,
                second_legal_row_id: second_id,
            });
        }
        rows.push(pair_row_from_ids(row_id as u32, table, first_id, second_id));
    }

    Ok(PairRowTableV1 {
        schema_version: PAIR_ROW_SCHEMA_VERSION_V1,
        schema_hash: PAIR_ROW_SCHEMA_HASH_V1,
        table_hash: pair_table_hash_v1(table, &rows),
        legal_row_schema_version: table.schema_version,
        legal_row_schema_hash: table.schema_hash,
        legal_row_table_hash: table.table_hash,
        phase: table.phase,
        rows,
    })
}

pub fn terminal_tactical_set_v1(game: &HexGameState) -> TerminalTacticalSetV1 {
    let table = legal_row_table_v1(game);
    let mut winning_single_cells = Vec::new();
    let mut hot_completion_pairs = Vec::new();
    let mut terminal_equivalent_pairs = Vec::new();
    let mut opponent_win_requirements = Vec::new();
    let mut hot_cover_pairs = Vec::new();
    let mut impossible_to_cover = false;

    let status = match tactical_status(game) {
        TacticalStatus::Quiet => TerminalTacticalStatusV1::Quiet,
        TacticalStatus::WinningTurns(turns) => {
            let mut completion_cells = Vec::<(Hex, Hex)>::new();
            for turn in turns {
                if turn.placements() == 1 {
                    winning_single_cells.push(turn.first());
                } else if let Some(second) = turn.second() {
                    completion_cells.push((turn.first(), second));
                }
            }
            sort_dedup_hex(&mut winning_single_cells);
            hot_completion_pairs = pair_rows_from_cells_lossy(&table, &completion_cells);
            if table.phase == TurnPhaseV1::NormalTwoPlacement && !winning_single_cells.is_empty() {
                terminal_equivalent_pairs =
                    pair_rows_containing_cells(&table, &winning_single_cells);
            }
            TerminalTacticalStatusV1::HotCompletionAvailable
        }
        TacticalStatus::MustBlock(block) => {
            opponent_win_requirements.extend(block.cells().iter().copied());
            let mut cover_cells = block.cells().to_vec();
            let mut cover_pairs = Vec::<(Hex, Hex)>::new();
            for &(a, b) in block.pairs() {
                opponent_win_requirements.push(a);
                opponent_win_requirements.push(b);
                cover_pairs.push((a, b));
            }
            sort_dedup_hex(&mut opponent_win_requirements);
            sort_dedup_hex(&mut cover_cells);

            hot_cover_pairs = pair_rows_from_cells_lossy(&table, &cover_pairs);
            if table.phase == TurnPhaseV1::NormalTwoPlacement && !cover_cells.is_empty() {
                let single_cover_pairs = pair_rows_containing_cells(&table, &cover_cells);
                hot_cover_pairs = merge_pair_rows(&table, hot_cover_pairs, single_cover_pairs);
            }
            TerminalTacticalStatusV1::HotCoverRequired
        }
        TacticalStatus::Unblockable => {
            let opponent = 1 - game.current_player();
            live_cells(game, opponent, &mut opponent_win_requirements);
            sort_dedup_hex(&mut opponent_win_requirements);
            impossible_to_cover = true;
            TerminalTacticalStatusV1::HotCoverImpossible
        }
    };

    TerminalTacticalSetV1 {
        schema_version: TERMINAL_TACTICAL_SCHEMA_VERSION_V1,
        schema_hash: TERMINAL_TACTICAL_SCHEMA_HASH_V1,
        legal_row_table_hash: table.table_hash,
        pair_row_schema_version: PAIR_ROW_SCHEMA_VERSION_V1,
        pair_row_schema_hash: PAIR_ROW_SCHEMA_HASH_V1,
        phase: table.phase,
        status,
        winning_single_cells,
        hot_completion_pairs,
        terminal_equivalent_pairs,
        opponent_win_requirements,
        hot_cover_pairs,
        impossible_to_cover,
    }
}

fn ensure_pair_phase(table: &LegalRowTableV1) -> Result<(), PairRowErrorV1> {
    if table.phase != TurnPhaseV1::NormalTwoPlacement {
        Err(PairRowErrorV1::WrongPhase { phase: table.phase })
    } else {
        Ok(())
    }
}

fn start_of_turn_state_v1(game: &HexGameState) -> (HexGameState, Option<Hex>) {
    if game.is_over() {
        return (game.clone(), None);
    }
    if game.placements_remaining() != 1 || game.move_history().is_empty() {
        return (game.clone(), None);
    }

    let Some(last) = game.move_history().last() else {
        return (game.clone(), None);
    };
    if last.player() != game.current_player() || last.placements_remaining_before() != 2 {
        return (game.clone(), None);
    }

    let first = last.cell();
    let mut turn_start = game.clone();
    if turn_start.unplace().is_ok() {
        (turn_start, Some(first))
    } else {
        (game.clone(), None)
    }
}

fn turn_phase_v1(turn_start: &HexGameState, current: &HexGameState) -> TurnPhaseV1 {
    if current.is_over() || turn_start.is_over() {
        TurnPhaseV1::Terminal
    } else if turn_start.move_count() == 0 && turn_start.placements_remaining() == 1 {
        TurnPhaseV1::OpeningSingle
    } else if turn_start.placements_remaining() == 2 {
        TurnPhaseV1::NormalTwoPlacement
    } else {
        TurnPhaseV1::OnePlacement
    }
}

fn legal_row_lookup(table: &LegalRowTableV1) -> FxHashMap<Hex, u32> {
    let mut lookup = FxHashMap::default();
    for row in &table.rows {
        lookup.insert(row.cell, row.row_id);
    }
    lookup
}

fn pair_row_from_ids(
    row_id: u32,
    table: &LegalRowTableV1,
    first_legal_row_id: u32,
    second_legal_row_id: u32,
) -> PairRowV1 {
    let first = table.rows[first_legal_row_id as usize].cell;
    let second = table.rows[second_legal_row_id as usize].cell;
    PairRowV1 {
        row_id,
        first_legal_row_id,
        second_legal_row_id,
        first,
        second,
        pair_key: pair_key_v1(table.table_hash, first_legal_row_id, second_legal_row_id),
    }
}

fn pair_rows_from_cells_lossy(table: &LegalRowTableV1, pairs: &[(Hex, Hex)]) -> Vec<PairRowV1> {
    if table.phase != TurnPhaseV1::NormalTwoPlacement {
        return Vec::new();
    }
    canonical_pair_rows_v1(table, pairs)
        .map(|pair_table| pair_table.rows)
        .unwrap_or_default()
}

fn pair_rows_containing_cells(table: &LegalRowTableV1, cells: &[Hex]) -> Vec<PairRowV1> {
    if table.phase != TurnPhaseV1::NormalTwoPlacement {
        return Vec::new();
    }
    let lookup = legal_row_lookup(table);
    let mut pair_ids = FxHashSet::default();
    for cell in cells {
        let Some(&cell_id) = lookup.get(cell) else {
            continue;
        };
        for row in &table.rows {
            if row.row_id == cell_id {
                continue;
            }
            let (first_id, second_id) = if cell_id < row.row_id {
                (cell_id, row.row_id)
            } else {
                (row.row_id, cell_id)
            };
            pair_ids.insert((first_id, second_id));
        }
    }
    let mut pair_ids = pair_ids.into_iter().collect::<Vec<_>>();
    pair_ids.sort();
    pair_ids
        .into_iter()
        .enumerate()
        .map(|(row_id, (first_id, second_id))| {
            pair_row_from_ids(row_id as u32, table, first_id, second_id)
        })
        .collect()
}

fn merge_pair_rows(
    table: &LegalRowTableV1,
    left: Vec<PairRowV1>,
    right: Vec<PairRowV1>,
) -> Vec<PairRowV1> {
    let mut pair_ids = FxHashSet::default();
    for row in left.into_iter().chain(right) {
        pair_ids.insert((row.first_legal_row_id, row.second_legal_row_id));
    }
    let mut pair_ids = pair_ids.into_iter().collect::<Vec<_>>();
    pair_ids.sort();
    pair_ids
        .into_iter()
        .enumerate()
        .map(|(row_id, (first_id, second_id))| {
            pair_row_from_ids(row_id as u32, table, first_id, second_id)
        })
        .collect()
}

fn sort_dedup_hex(cells: &mut Vec<Hex>) {
    cells.sort();
    cells.dedup();
}

fn state_hash_v1(game: &HexGameState) -> u64 {
    let mut hash = FNV_OFFSET;
    hash = mix_u64(hash, game.zobrist());
    hash = mix_u64(hash, game.current_player() as u64);
    hash = mix_u64(hash, game.placements_remaining() as u64);
    hash = mix_u64(hash, game.move_count() as u64);
    hash = mix_u64(hash, game.winner().map(u64::from).unwrap_or(2));
    hash
}

fn legal_table_hash_v1(
    phase: TurnPhaseV1,
    current_player: u8,
    placements_remaining: u8,
    move_count: u32,
    state_hash: u64,
    rows: &[LegalRowV1],
) -> u64 {
    let mut hash = FNV_OFFSET;
    hash = mix_u64(hash, LEGAL_ROW_SCHEMA_HASH_V1);
    hash = mix_u64(hash, phase.code());
    hash = mix_u64(hash, current_player as u64);
    hash = mix_u64(hash, placements_remaining as u64);
    hash = mix_u64(hash, move_count as u64);
    hash = mix_u64(hash, state_hash);
    hash = mix_u64(hash, rows.len() as u64);
    for row in rows {
        hash = mix_u64(hash, row.row_id as u64);
        hash = mix_i32(hash, row.cell.q);
        hash = mix_i32(hash, row.cell.r);
    }
    hash
}

fn pair_table_hash_v1(table: &LegalRowTableV1, rows: &[PairRowV1]) -> u64 {
    let mut hash = FNV_OFFSET;
    hash = mix_u64(hash, PAIR_ROW_SCHEMA_HASH_V1);
    hash = mix_u64(hash, table.table_hash);
    hash = mix_u64(hash, rows.len() as u64);
    for row in rows {
        hash = mix_u64(hash, row.row_id as u64);
        hash = mix_u64(hash, row.first_legal_row_id as u64);
        hash = mix_u64(hash, row.second_legal_row_id as u64);
        hash = mix_u64(hash, row.pair_key);
    }
    hash
}

fn pair_key_v1(legal_table_hash: u64, first_legal_row_id: u32, second_legal_row_id: u32) -> u64 {
    let mut hash = FNV_OFFSET;
    hash = mix_u64(hash, legal_table_hash);
    hash = mix_u64(hash, first_legal_row_id as u64);
    hash = mix_u64(hash, second_legal_row_id as u64);
    hash
}

fn mix_i32(hash: u64, value: i32) -> u64 {
    mix_u64(hash, value as i64 as u64)
}

fn mix_u64(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::HexGameState;

    #[test]
    fn legal_row_table_exposes_full_start_of_turn_identity() {
        let mut game = HexGameState::new();
        let opening = legal_row_table_v1(&game);
        assert_eq!(opening.schema_version, LEGAL_ROW_SCHEMA_VERSION_V1);
        assert_eq!(opening.schema_hash, LEGAL_ROW_SCHEMA_HASH_V1);
        assert_eq!(opening.phase, TurnPhaseV1::OpeningSingle);
        assert_eq!(
            opening.rows,
            vec![LegalRowV1 {
                row_id: 0,
                cell: Hex::ORIGIN
            }]
        );

        game.place(0, 0).unwrap();
        let table = legal_row_table_v1(&game);
        assert_eq!(table.phase, TurnPhaseV1::NormalTwoPlacement);
        assert_eq!(table.query_phase, QueryPhaseV1::TurnStart);
        assert_eq!(table.current_player, 1);
        assert_eq!(table.placements_remaining, 2);
        assert_eq!(table.rows.len(), game.legal_moves().len());
        assert!(table.table_hash != 0);

        let first = table.rows[0].cell;
        game.place(first.q, first.r).unwrap();
        let continuation = legal_row_table_v1(&game);
        assert_eq!(continuation.phase, TurnPhaseV1::NormalTwoPlacement);
        assert_eq!(continuation.query_phase, QueryPhaseV1::TurnContinuation);
        assert_eq!(continuation.table_hash, table.table_hash);
        assert_eq!(continuation.first_placement_row_id, Some(0));
    }

    #[test]
    fn pair_rows_are_canonical_deterministic_and_reference_legal_ids() {
        let mut game = HexGameState::new();
        game.place(0, 0).unwrap();
        let table = legal_row_table_v1(&game);
        let first = table.rows[0].cell;
        let second = table.rows[1].cell;

        let canonical = canonical_pair_rows_v1(&table, &[(second, first)]).unwrap();
        assert_eq!(canonical.schema_version, PAIR_ROW_SCHEMA_VERSION_V1);
        assert_eq!(canonical.legal_row_table_hash, table.table_hash);
        assert_eq!(canonical.rows.len(), 1);
        assert_eq!(canonical.rows[0].first_legal_row_id, 0);
        assert_eq!(canonical.rows[0].second_legal_row_id, 1);
        assert_eq!(canonical.rows[0].first, first);
        assert_eq!(canonical.rows[0].second, second);

        let full_a = pair_row_table_v1(&table).unwrap();
        let full_b = pair_row_table_v1(&table).unwrap();
        assert_eq!(full_a.rows, full_b.rows);
        assert_eq!(full_a.table_hash, full_b.table_hash);
        assert_eq!(
            full_a.rows.len(),
            table.rows.len() * (table.rows.len() - 1) / 2
        );
        assert_eq!(full_a.rows[0].first_legal_row_id, 0);
        assert_eq!(full_a.rows[0].second_legal_row_id, 1);
    }

    #[test]
    fn pair_rows_reject_duplicates_illegal_cells_and_wrong_phase() {
        let opening = legal_row_table_v1(&HexGameState::new());
        assert!(matches!(
            pair_row_table_v1(&opening),
            Err(PairRowErrorV1::WrongPhase {
                phase: TurnPhaseV1::OpeningSingle
            })
        ));

        let mut game = HexGameState::new();
        game.place(0, 0).unwrap();
        let table = legal_row_table_v1(&game);
        let a = table.rows[0].cell;
        let b = table.rows[1].cell;

        assert!(matches!(
            canonical_pair_rows_v1(&table, &[(a, a)]),
            Err(PairRowErrorV1::DuplicateCell { cell }) if cell == a
        ));
        assert!(matches!(
            canonical_pair_rows_v1(&table, &[(a, b), (b, a)]),
            Err(PairRowErrorV1::DuplicatePair { .. })
        ));
        assert!(matches!(
            canonical_pair_rows_v1(&table, &[(a, Hex::new(999, 999))]),
            Err(PairRowErrorV1::IllegalCell { cell }) if cell == Hex::new(999, 999)
        ));
    }

    #[test]
    fn terminal_tactical_payload_reports_v1_shapes_and_statuses() {
        let quiet = terminal_tactical_set_v1(&HexGameState::new());
        assert_eq!(quiet.status, TerminalTacticalStatusV1::Quiet);
        assert!(!quiet.impossible_to_cover);
        assert!(quiet.hot_completion_pairs.is_empty());

        let mut completion = HexGameState::new();
        completion
            .set_position(&[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], 0, 2)
            .unwrap();
        let payload = terminal_tactical_set_v1(&completion);
        assert_eq!(
            payload.status,
            TerminalTacticalStatusV1::HotCompletionAvailable
        );
        assert!(payload
            .hot_completion_pairs
            .iter()
            .any(|row| row.first == Hex::new(-1, 0) && row.second == Hex::new(4, 0)));

        let mut cover = HexGameState::new();
        cover
            .set_position(&[(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)], 0, 2)
            .unwrap();
        let payload = terminal_tactical_set_v1(&cover);
        assert_eq!(payload.status, TerminalTacticalStatusV1::HotCoverRequired);
        assert!(payload.opponent_win_requirements.contains(&Hex::new(-2, 0)));
        assert!(payload
            .hot_cover_pairs
            .iter()
            .any(|row| row.first == Hex::new(-2, 0) && row.second == Hex::new(4, 0)));

        let mut impossible = HexGameState::new();
        impossible
            .set_position(
                &[
                    (0, 0, 1),
                    (1, 0, 1),
                    (2, 0, 1),
                    (3, 0, 1),
                    (4, 0, 1),
                    (10, 0, 1),
                    (11, 0, 1),
                    (12, 0, 1),
                    (13, 0, 1),
                    (14, 0, 1),
                ],
                0,
                2,
            )
            .unwrap();
        let payload = terminal_tactical_set_v1(&impossible);
        assert_eq!(payload.status, TerminalTacticalStatusV1::HotCoverImpossible);
        assert!(payload.impossible_to_cover);
        assert!(!payload.opponent_win_requirements.is_empty());
    }
}
