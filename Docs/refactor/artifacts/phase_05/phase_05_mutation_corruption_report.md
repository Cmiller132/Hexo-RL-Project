# Phase 05 Mutation And Corruption Report

Covered corruption classes:

| Class | Evidence |
|---|---|
| mutated `SearchEvaluation.row_priors` | `test_engine_adapter_rejects_mutated_search_evaluation` |
| mutated `PairEvaluation.pair_rows` | `test_engine_adapter_validates_pair_row_identity` |
| stale root token | `test_engine_adapter_rejects_stale_root_token`, `phase_05_mcts_error_trace_samples.json` |
| stale batch token | `test_engine_adapter_rejects_stale_batch_token`, `phase_05_mcts_error_trace_samples.json` |
| malformed priors | `test_engine_adapter_rejects_stale_hashes_duplicate_rows_and_nonfinite_priors`, `phase_05_mcts_error_trace_samples.json` |
| sparse/candidate row mismatch | `test_search_evaluation_rejects_unmapped_model_rows`, `phase_05_mcts_error_trace_samples.json` |
| pair row/prior length mismatch | `phase_05_mcts_error_trace_samples.json` |
| wrong root legal bytes or offset | `Python/tests/test_engine_smoke.py` validating `EngineAdapter` root expansion |

No skipped, xfailed, flaky-only, or manual-only check is claimed complete for this phase.
