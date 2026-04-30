# Phase 05 SearchEvaluation Validation

`SearchEvaluation` is the only policy object accepted by `EngineAdapter`.

Validation coverage:

| Invariant | Evidence |
|---|---|
| prior length equals legal row count | `test_search_evaluation_rejects_prior_length_mismatch` |
| row ids are one-to-one with `LegalActionTable` rows | `test_search_evaluation_rejects_unmapped_model_rows` |
| dense indices match `LegalActionTable.dense_indices` | `SearchEvaluation.__post_init__` and policy provider tests |
| priors/value are finite and non-negative | `test_engine_adapter_rejects_stale_hashes_duplicate_rows_and_nonfinite_priors`, engine smoke non-finite test |
| all-zero mass requires explicit fallback | `SearchEvaluation.__post_init__` |
| validated arrays are immutable after construction | `test_engine_adapter_rejects_mutated_search_evaluation` |
| adapter root legal bytes match the same legal table | `test_engine_adapter_validates_legal_row_identity` |

The single-position debug bundle at `phase_05_policy_search_debug_bundle.json` records raw fixture outputs, decoded row mappings, normalized priors, legal hashes, MCTS input, MCTS output, trace id, and timing fields.
