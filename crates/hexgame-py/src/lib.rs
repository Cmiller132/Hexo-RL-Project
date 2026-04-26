pub mod engine;
pub mod encode;
pub mod buffer;

use pyo3::prelude::*;

/// Python module `_engine` — the compiled Rust extension for Hexo.
#[pymodule]
#[pyo3(name = "_engine")]
fn hexgame_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    engine::register_module(m)?;
    encode::register_module(m)?;
    buffer::register_module(m)?;
    Ok(())
}
