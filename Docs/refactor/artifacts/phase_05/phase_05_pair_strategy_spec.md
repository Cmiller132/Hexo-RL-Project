# Phase 05 PairStrategySpec

`Python/src/hexorl/search/pair_strategy.py` owns pair consumption semantics.

Supported strategies:

| Strategy | Required caps | Scope |
|---|---|---|
| `none` | all caps zero | emits zero selected/scored rows |
| `two_stage_root_only` | `root_enabled=True`, `max_root_pair_rows > 0`, `leaf_enabled=False` | capped root-only scoring |
| `tactical_only` | explicit root and/or leaf scope plus matching cap | capped tactical rows |
| `diagnostic_full_root` | `diagnostic=True`, `root_enabled=True`, `leaf_enabled=False`, `max_full_pair_rows > 0` | capped full-pair diagnostic root-only |

Default behavior:

- default pair strategy is `none`
- default `global_xattn` and global graph families score zero pair rows
- head presence, architecture names, and `pair_prior_mix` do not enable scoring
- leaf pair scoring requires explicit enablement and leaf cap
- full pair scoring is diagnostic-only, root-only, capped, and never leaf-scored

Primary tests:

- `Python/tests/search/test_pair_strategy.py`
- `Python/tests/search/test_global_graph_pair_contracts.py`
- `Python/tests/test_config_and_guardrails.py`
