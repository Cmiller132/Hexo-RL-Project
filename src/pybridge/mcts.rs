//! PyO3 wrapper for the neural MCTS engine.
//!
//! This module defines `PyMCTSEngine`, the Python-facing class that wraps
//! [`MCTSEngine`](crate::mcts::MCTSEngine).  It handles root expansion,
//! leaf selection, Dirichlet noise injection, back-propagation, and tree-node
//! extraction for training-data generation.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use numpy::{ndarray, PyArray3, PyArray4, PyReadonlyArray1};

use crate::core::Hex;
use crate::encoder::{BOARD_SIZE, NUM_CHANNELS};
use crate::mcts::MCTSEngine;

use super::PyHexGame;

#[pyclass(name = "MCTSEngine")]
pub struct PyMCTSEngine {
    inner: MCTSEngine,
}

#[pymethods]
impl PyMCTSEngine {
    #[new]
    #[pyo3(signature = (game, num_simulations, c_puct=1.4, near_radius=8, c_puct_init=19652.0, constrain_threats=true, arena_sim_hint=None))]
    fn new(
        game: &PyHexGame,
        num_simulations: u32,
        c_puct: f32,
        near_radius: i32,
        c_puct_init: f32,
        constrain_threats: bool,
        arena_sim_hint: Option<u32>,
    ) -> Self {
        let hint = arena_sim_hint.unwrap_or(num_simulations);
        let engine = MCTSEngine::with_arena_sim_hint(
            game.inner.clone(),
            num_simulations,
            hint,
            c_puct,
            near_radius,
            constrain_threats,
            c_puct_init,
        );
        Self { inner: engine }
    }

    fn init_root<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<(Bound<'py, PyArray3<f32>>, i32, i32, Bound<'py, PyBytes>)>> {
        let Some((tensor, oq, or_, legal)) = self.inner.init_root() else {
            return Ok(None);
        };
        let arr = ndarray::Array3::from_shape_vec(
            (NUM_CHANNELS, BOARD_SIZE as usize, BOARD_SIZE as usize),
            tensor,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let arr = PyArray3::from_owned_array(py, arr);
        let mut legal_buf: Vec<u8> = Vec::with_capacity(legal.len() * 8);
        for h in &legal {
            legal_buf.extend_from_slice(&h.q.to_le_bytes());
            legal_buf.extend_from_slice(&h.r.to_le_bytes());
        }
        Ok(Some((arr, oq, or_, PyBytes::new(py, &legal_buf))))
    }

    fn expand_root<'py>(
        &mut self,
        policy: PyReadonlyArray1<'py, f32>,
        value: f32,
        offset_q: i32,
        offset_r: i32,
        legal_bytes: &[u8],
    ) -> PyResult<()> {
        let policy_slice = policy
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("policy array must be contiguous"))?;
        if legal_bytes.len() % 8 != 0 {
            return Err(PyErr::new::<PyValueError, _>(
                format!("legal_bytes length {} is not a multiple of 8", legal_bytes.len())
            ));
        }
        let mut legal = Vec::with_capacity(legal_bytes.len() / 8);
        for chunk in legal_bytes.chunks_exact(8) {
            let q = i32::from_le_bytes(chunk[0..4].try_into().unwrap());
            let r = i32::from_le_bytes(chunk[4..8].try_into().unwrap());
            legal.push(Hex::new(q, r));
        }
        self.inner
            .expand_root(policy_slice, value, offset_q, offset_r, &legal);
        Ok(())
    }

    #[pyo3(signature = (noise, noise_fraction))]
    fn add_dirichlet_noise<'py>(
        &mut self,
        noise: PyReadonlyArray1<'py, f32>,
        noise_fraction: f32,
    ) -> PyResult<()> {
        let noise_slice = noise
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("noise array must be contiguous"))?;
        self.inner.add_dirichlet_noise(noise_slice, noise_fraction);
        Ok(())
    }

    fn done(&self) -> bool {
        self.inner.done()
    }

    fn select_leaves<'py>(
        &mut self,
        py: Python<'py>,
        batch_size: u32,
    ) -> PyResult<(Bound<'py, PyArray4<f32>>, u32)> {
        let (count, tensor_vec) = py.allow_threads(|| {
            let (tensors, count) = self.inner.select_leaves(batch_size);
            (count, tensors.to_vec())
        });
        let view = ndarray::ArrayView4::from_shape(
            (
                count as usize,
                NUM_CHANNELS,
                BOARD_SIZE as usize,
                BOARD_SIZE as usize,
            ),
            &tensor_vec,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let arr = PyArray4::from_array(py, &view);
        Ok((arr, count))
    }

    fn expand_and_backprop<'py>(
        &mut self,
        policies: PyReadonlyArray1<'py, f32>,
        values: PyReadonlyArray1<'py, f32>,
        py: Python<'py>,
    ) -> PyResult<()> {
        let policies_slice = policies
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("policies array must be contiguous"))?;
        let values_slice = values
            .as_slice()
            .map_err(|_| PyErr::new::<PyValueError, _>("values array must be contiguous"))?;
        let p = policies_slice.to_vec();
        let v = values_slice.to_vec();
        py.allow_threads(|| {
            self.inner.expand_and_backprop(&p, &v);
        });
        Ok(())
    }

    fn get_results(&self) -> (Vec<i32>, Vec<i32>, Vec<u32>, f32) {
        self.inner.get_results()
    }

    fn root_child_count(&self) -> u16 {
        self.inner.root_child_count()
    }

    fn root_child_priors(&self) -> Vec<f32> {
        self.inner.root_child_priors()
    }

    fn root_child_q_values(&self) -> Vec<f32> {
        self.inner.root_child_q_values()
    }

    #[pyo3(signature = (min_visits=1))]
    fn extract_tree_node_states<'py>(
        &mut self,
        py: Python<'py>,
        min_visits: u32,
    ) -> PyResult<(Bound<'py, PyArray4<f32>>, Vec<Vec<(i32, i32, i32)>>, usize)> {
        let (packed, histories, count) = self
            .inner
            .extract_tree_node_states(min_visits)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        let arr = ndarray::Array4::from_shape_vec(
            (
                count,
                NUM_CHANNELS,
                BOARD_SIZE as usize,
                BOARD_SIZE as usize,
            ),
            packed,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let arr = PyArray4::from_owned_array(py, arr);
        let py_histories: Vec<Vec<(i32, i32, i32)>> = histories
            .into_iter()
            .map(|history| {
                history
                    .into_iter()
                    .map(|(player, q, r)| (player as i32, q as i32, r as i32))
                    .collect()
            })
            .collect();
        Ok((arr, py_histories, count))
    }

    #[pyo3(signature = (q, r, new_num_simulations))]
    fn re_root(&mut self, q: i32, r: i32, new_num_simulations: u32) -> PyResult<()> {
        let q =
            i16::try_from(q).map_err(|_| PyValueError::new_err("q coordinate out of i16 range"))?;
        let r =
            i16::try_from(r).map_err(|_| PyValueError::new_err("r coordinate out of i16 range"))?;
        self.inner.re_root(q, r, new_num_simulations);
        Ok(())
    }
}
