# Phase 03 Adversarial Review

Findings:

1. **Resolved: Rust-engine inference integration timeout.**
   `Python/tests/test_inference_server.py::TestInferenceServerWithEngine::test_mcts_round_trip` timed out because the test used the old Rust MCTS Python API. `init_root()` now returns four values and `select_leaves()` returns two values. The assertion failure happened before cleanup, so pytest waited on the spawned inference server. The test now uses the current API, has a bounded loop, and always tears down server/client shared memory in `finally`.

2. **Exact `hexorl\.model` audit pattern overmatches `hexorl.models`.**
   The new runtime package is `hexorl.models`, which contains the literal prefix `hexorl.model`. The semantic audit is clean for `hexorl.model.` and the old directory is deleted, but the exact written pattern needs a documented interpretation or a stricter boundary regex.

3. **Config still exposes `model.architecture` as the legacy user-facing field.**
   Runtime construction now normalizes through `ModelSpec`, but the config schema still validates legacy names. This is acceptable only as registry/spec migration-name validation and should be kept from reintroducing runtime branches.

Resolution status:

- Finding 1 is resolved. `python -m pytest Python/tests/test_inference_server.py -q` passes with `7 passed in 14.84s`, and `python Docs\refactor\artifacts\phase_03\commands\mcts_round_trip_probe.py` exits 0.
- Finding 2 is documented in the import audit.
- Finding 3 is contained by `models/specs.py` runtime normalization and no runtime branch audit matches remain.
