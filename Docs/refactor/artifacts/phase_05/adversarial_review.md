# Phase 05 Adversarial Review

| Attack or regression | Finding | Resolution |
|---|---|---|
| Enable pair scoring by setting `pair_prior_mix > 0` | Search strategy ignores this field; negative test covers it. | `test_pair_prior_mix_does_not_enable_pair_scoring` |
| Enable pair scoring by exposing graph pair heads | Head names are not read by `PolicyProvider` or `PairStrategy` enablement. | `test_pair_head_presence_does_not_enable_pair_scoring` |
| Route global graph policy by architecture prefix | Worker no longer contains `global_graph_enabled` or `architecture.startswith`; providers are selected by `ModelSpec.kind`. | import audit and `test_selfplay_worker_contains_no_architecture_string_checks` equivalent audit in engine tests |
| Pass raw logits directly to MCTS | `EngineAdapter` rejects non-`SearchEvaluation` inputs. | `test_engine_adapter_rejects_raw_logits` |
| Accept stale root or leaf batch tokens | Adapter validates active root generation and selected batch generation before expansion/backprop. | `test_engine_adapter_rejects_stale_root_token`, `test_engine_adapter_rejects_stale_batch_token` |
| Mutate validated arrays after construction | `SearchEvaluation`, `PairEvaluation`, and canonical tables expose read-only arrays. | mutation tests and report |
| Hide MCTS failures behind unstructured strings | Adapter maps backend exceptions to `EngineAdapterError` with `MCTSTrace`. | `test_engine_adapter_maps_mcts_error_to_structured_python_error` |
| Introduce Python per-node MCTS loop | Worker receives contiguous leaf batches and returns batched evaluations through `commit_leaf_batch`. | performance profile and code audit |

Residual risk:

- Existing eval arena/player modules still use Rust game/legal helpers, not Rust MCTS. Phase 08 owns evaluation provider integration.
- `search/expansion.py` and `search/mcts_runner.py` call `EngineAdapter` public methods; they are not Rust API owners.
