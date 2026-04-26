use pyo3::prelude::*;

/// Buffer kernel stubs — populated in Phase 3.
/// These Rust kernels will handle compact-to-dense decode and recency-weighted sampling.
pub fn register_module(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
