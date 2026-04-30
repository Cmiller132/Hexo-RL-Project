use crate::protocol;
use hexgame_core::encoding::{self as encoder, BOARD_SIZE, NUM_CHANNELS, TENSOR_SIZE};
use hexgame_core::rules::HexGameState;
use numpy::{PyArray3, PyArray4, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Replay a compact move history and encode each position as a 13-channel tensor.
///
/// `history_bytes` is a flat byte buffer of little-endian `(player: i32, q: i32, r: i32)` triples.
/// Each triple represents one stone placement.
///
/// Returns a numpy array of shape `(N + 1, 13, 33, 33)` where N is the number of moves.
/// Position `i` is the board state after the first `i` placements. Empty histories are valid
/// and return a single empty-board tensor.
#[pyfunction]
fn encode_compact_record<'py>(
    py: Python<'py>,
    history_bytes: &[u8],
    near_radius: i32,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let history = protocol::decode_compact_history_rows(history_bytes)?;
    let num_moves = history.len();

    let positions = py
        .allow_threads(move || -> Result<Vec<f32>, String> {
            let mut game = HexGameState::new();
            let mut positions = Vec::with_capacity((num_moves + 1) * TENSOR_SIZE);
            for (player, q, r) in history {
                let tensor = encoder::encode_board(&game, near_radius, false).tensor;
                positions.extend_from_slice(&tensor);
                if player != game.current_player() as i32 {
                    return Err(format!(
                        "history player mismatch: stored {}, expected {} before move ({}, {})",
                        player,
                        game.current_player(),
                        q,
                        r
                    ));
                }
                game.place(q, r).map_err(|e| e.to_string())?;
            }
            let tensor = encoder::encode_board(&game, near_radius, false).tensor;
            positions.extend_from_slice(&tensor);
            Ok(positions)
        })
        .map_err(PyValueError::new_err)?;

    let shape = (
        num_moves + 1,
        NUM_CHANNELS,
        BOARD_SIZE as usize,
        BOARD_SIZE as usize,
    );
    let arr = numpy::ndarray::Array4::from_shape_vec(shape, positions)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray4::from_owned_array(py, arr))
}

/// Apply one of 12 hex-grid symmetry transforms to a board tensor.
///
/// `sym_idx` ∈ [0, 11] selects the transform:
/// - 0..5: rotations by 0, 60, 120, 180, 240, 300 degrees
/// - 6..11: same rotations after a horizontal reflection
///
/// The tensor must have shape `(13, 33, 33)` — transforms are applied to the
/// spatial dimensions (axes 1 and 2). The 13 channels are preserved as-is.
///
/// Returns a new numpy array (the input is not modified).
#[pyfunction]
fn apply_d6_symmetry<'py>(
    py: Python<'py>,
    tensor: PyReadonlyArray3<'py, f32>,
    sym_idx: u8,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let sym = sym_idx % 12;
    let arr = tensor.as_array();
    let (ch, h, w) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    if ch != NUM_CHANNELS || h != BOARD_SIZE as usize || w != BOARD_SIZE as usize {
        return Err(PyValueError::new_err(format!(
            "expected shape ({}, {}, {}), got ({}, {}, {})",
            NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE, ch, h, w
        )));
    }

    // Copy input to owned Vec so we can release the GIL during the transform.
    let arr_vec: Vec<f32> = arr.iter().copied().collect();
    let out_vec = py.allow_threads(move || -> Vec<f32> {
        let mut out = vec![0.0f32; ch * h * w];
        let half = BOARD_SIZE / 2;
        for c in 0..ch {
            for i in 0..BOARD_SIZE {
                for j in 0..BOARD_SIZE {
                    let val = arr_vec[c * h * w + i as usize * w + j as usize];
                    let qi = i - half;
                    let rj = j - half;
                    let (qi_t, rj_t) = match sym {
                        0 => (qi, rj),
                        1 => (-rj, qi + rj),
                        2 => (-qi - rj, qi),
                        3 => (-qi, -rj),
                        4 => (rj, -qi - rj),
                        5 => (qi + rj, -qi),
                        6 => (rj, qi),
                        7 => (-qi, qi + rj),
                        8 => (-qi - rj, rj),
                        9 => (-rj, -qi),
                        10 => (qi, -qi - rj),
                        11 => (qi + rj, -rj),
                        _ => unreachable!(),
                    };
                    let ti = (qi_t + half) as usize;
                    let tj = (rj_t + half) as usize;
                    if ti < BOARD_SIZE as usize && tj < BOARD_SIZE as usize {
                        out[c * h * w + ti * w + tj] = val;
                    }
                }
            }
        }
        out
    });

    let out_arr = numpy::ndarray::Array3::from_shape_vec((ch, h, w), out_vec)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray3::from_owned_array(py, out_arr))
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_compact_record, m)?)?;
    m.add_function(wrap_pyfunction!(apply_d6_symmetry, m)?)?;
    Ok(())
}
