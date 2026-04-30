# Phase 05 Deletion Manifest

Deleted or disconnected runtime paths:

| Old path | Closure |
|---|---|
| worker-owned `MockMCTSEngine` | removed from `Python/src/hexorl/selfplay/worker.py`; mock backend is private to `search/engine_adapter.py` for tests/offline fallback |
| worker-owned `RealMCTSEngine` | removed from worker; Rust MCTS construction occurs only in `create_engine_adapter` |
| direct worker calls to `init_root`, `select_leaves`, `expand_root`, `expand_and_backprop`, `apply_root_pair_*` | worker uses `search.mcts_runner` helpers and `EngineAdapter` |
| worker pair chunk helpers `_score_graph_pair_chunks`, `_score_crop_pair_chunks`, graph pair row chunk projection helpers | removed |
| worker architecture-prefix field `global_graph_enabled` | removed; self-play consumes `ModelSpec.is_global_graph` |
| pair enablement from `pair_prior_mix` or head presence | not present in runtime search/self-play; explicit `PairStrategySpec` required |
| direct policy-output-to-MCTS wiring | replaced by `PolicyProvider -> SearchEvaluation -> EngineAdapter` |

No compatibility shim in `Python/src/hexorl/` keeps the old worker-owned MCTS or pair-scoring path alive.
