use hexgame_core::rules::{Hex, MoveRecord};
use numpy::ndarray::ArrayView2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const LEGAL_ROW_BYTES: usize = 8;
const BOARD_PIECE_ROW_BYTES: usize = 12;
const COMPACT_HISTORY_ROW_BYTES: usize = 12;

pub(crate) fn encode_legal_rows(legal: &[Hex]) -> Vec<u8> {
    let mut out = Vec::with_capacity(legal.len() * LEGAL_ROW_BYTES);
    for h in legal {
        push_i32(&mut out, h.q);
        push_i32(&mut out, h.r);
    }
    out
}

pub(crate) fn decode_legal_rows(legal_bytes: &[u8]) -> PyResult<Vec<Hex>> {
    validate_row_bytes("legal_bytes", legal_bytes, LEGAL_ROW_BYTES)?;
    Ok(legal_bytes
        .chunks_exact(LEGAL_ROW_BYTES)
        .map(|chunk| Hex::new(read_i32(chunk, 0), read_i32(chunk, 4)))
        .collect())
}

pub(crate) fn encode_board_piece_rows<I>(pieces: I) -> Vec<u8>
where
    I: IntoIterator<Item = (Hex, u8)>,
{
    let iter = pieces.into_iter();
    let (lower, _) = iter.size_hint();
    let mut out = Vec::with_capacity(lower * BOARD_PIECE_ROW_BYTES);
    for (h, player) in iter {
        push_i32(&mut out, h.q);
        push_i32(&mut out, h.r);
        push_i32(&mut out, player as i32);
    }
    out
}

#[allow(dead_code)]
pub(crate) fn decode_board_piece_rows(board_bytes: &[u8]) -> PyResult<Vec<(i32, i32, u8)>> {
    validate_row_bytes("board_bytes", board_bytes, BOARD_PIECE_ROW_BYTES)?;
    let mut rows = Vec::with_capacity(board_bytes.len() / BOARD_PIECE_ROW_BYTES);
    for chunk in board_bytes.chunks_exact(BOARD_PIECE_ROW_BYTES) {
        let player = read_i32(chunk, 8);
        let player = u8::try_from(player).map_err(|_| {
            PyValueError::new_err(format!("board player {player} is outside u8 range"))
        })?;
        rows.push((read_i32(chunk, 0), read_i32(chunk, 4), player));
    }
    Ok(rows)
}

pub(crate) fn encode_compact_history_rows(history: &[MoveRecord]) -> Vec<u8> {
    let mut out = Vec::with_capacity(history.len() * COMPACT_HISTORY_ROW_BYTES);
    for record in history {
        push_i32(&mut out, record.player() as i32);
        push_i32(&mut out, record.cell().q);
        push_i32(&mut out, record.cell().r);
    }
    out
}

pub(crate) fn decode_compact_history_rows(history_bytes: &[u8]) -> PyResult<Vec<(i32, i32, i32)>> {
    validate_row_bytes("history_bytes", history_bytes, COMPACT_HISTORY_ROW_BYTES)?;
    Ok(history_bytes
        .chunks_exact(COMPACT_HISTORY_ROW_BYTES)
        .map(|chunk| (read_i32(chunk, 0), read_i32(chunk, 4), read_i32(chunk, 8)))
        .collect())
}

pub(crate) fn decode_pair_rows(
    pair_qr: ArrayView2<'_, i32>,
    field_name: &str,
) -> PyResult<Vec<(i32, i32, i32, i32)>> {
    if pair_qr.shape()[1] != 4 {
        return Err(PyValueError::new_err(format!(
            "{field_name} must have shape (N, 4)"
        )));
    }
    Ok(pair_qr
        .outer_iter()
        .map(|row| (row[0], row[1], row[2], row[3]))
        .collect())
}

fn validate_row_bytes(name: &str, bytes: &[u8], row_width: usize) -> PyResult<()> {
    if bytes.len().is_multiple_of(row_width) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} length {} is not a multiple of {row_width}",
            bytes.len()
        )))
    }
}

fn push_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn read_i32(row: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes(
        row[offset..offset + 4]
            .try_into()
            .expect("validated row width"),
    )
}
