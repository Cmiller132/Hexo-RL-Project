# Phase 05 Default Pair Strategy Evidence

Default no-pair behavior is covered by:

- `test_pair_strategy_none_generates_zero_pair_rows`
- `test_pair_strategy_none_scores_zero_pairs`
- `test_global_xattn_default_pair_strategy_none_zero_rows`
- `test_global_graph_default_pair_strategy_none_zero_rows`
- `test_pair_head_presence_does_not_enable_pair_scoring`
- `test_pair_prior_mix_does_not_enable_pair_scoring`
- `test_architecture_prefix_does_not_enable_pair_scoring`
- `test_global_xattn_pair_strategy_defaults_to_none`
- `test_global_xattn_pair_heads_do_not_enable_pair_scoring_without_strategy`

Runtime telemetry sample:

- `phase_05_mcts_trace_sample.json`

Sample fields:

```text
pair_strategy = none
selected_pair_rows = 0
scored_pair_rows = 0
pair_influence = none
```
