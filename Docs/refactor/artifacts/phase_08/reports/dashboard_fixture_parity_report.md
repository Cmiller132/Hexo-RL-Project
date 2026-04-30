# Dashboard Fixture Parity Report

- Required route/view coverage is asserted by `test_contract_inspector_required_views_and_hash_fields`.
- `model-input` view reports `tensor_hash` and `training_input_hash` from the same canonical Rust-derived model input; tests assert `dashboard_training_parity`.
- Debug bundle and mismatch views expose likely owners instead of generic invalid-input failures.
- `ContractInspector.register()` extension is tested with a fake inspector service.
