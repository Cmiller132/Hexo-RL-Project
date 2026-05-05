# Global Graph Pair Head Review Benchmarks

Date: 2026-05-05

Scope:
- Global graph pair-head contract fixes.
- Output-head gating.
- `global_pair_twostage_0` pair-branch distinction.
- Relation construction performance.

Commands run:

```powershell
$env:PYTHONPATH='Python/src'
python -m pytest Python/tests/test_global_graph_contract.py Python/tests/test_inference_server.py Python/tests/test_training_data_pipeline.py Python/tests/test_config_and_guardrails.py
```

Result: 151 passed, 1 warning.

Benchmark host:
- GPU: NVIDIA GeForce RTX 4070 Ti
- Torch: 2.10.0+cu126
- CPU benchmark threads: 8

Graph construction timings:

| Case | Tokens | Legal | Pair rows | Mean ms |
|---|---:|---:|---:|---:|
| policy only, no pair rows | 291 | 216 | 0 | 10.187 |
| non-materialized pair refs | 291 | 216 | 256 | 10.613 |
| non-materialized pair refs | 291 | 216 | 4096 | 14.201 |
| materialized pair tokens | 547 | 216 | 256 | 25.733 |

Previous measured graph construction baseline from review:

| Case | Previous mean ms |
|---|---:|
| policy only, no pair rows | 275.893 |
| non-materialized pair refs, 4096 rows | 259.231 |
| materialized pair tokens, 256 rows | 987.025 |

CUDA forward timings, channels=128:

| Architecture | Heads | Pair rows | Mean ms | Peak MB |
|---|---|---:|---:|---:|
| global_xattn_0 | policy/value only | 0 | 2.694 | 29.1 |
| global_xattn_0 | pair heads enabled | 4096 | 3.311 | 43.8 |
| global_xattn_0 | pair heads enabled | 23220 | 3.979 | 160.2 |
| global_pair_twostage_0 | policy/value only | 0 | 3.427 | 31.1 |
| global_pair_twostage_0 | pair heads enabled | 4096 | 4.623 | 51.8 |
| global_pair_twostage_0 | pair heads enabled | 23220 | 6.148 | 196.3 |
| global_line_window_0 | policy/value only | 0 | 4.155 | 30.6 |
| global_line_window_0 | pair heads enabled | 4096 | 4.326 | 45.3 |
| global_line_window_0 | pair heads enabled | 23220 | 5.046 | 161.8 |
| global_graph_full_0 | policy/value only | 0 | 3.420 | 30.6 |
| global_graph_full_0 | pair heads enabled | 4096 | 4.528 | 45.3 |
| global_graph_full_0 | pair heads enabled | 23220 | 5.244 | 161.8 |

CPU forward timings, channels=32:

| Architecture | Heads | Pair rows | Params | Mean ms |
|---|---|---:|---:|---:|
| global_xattn_0 | policy/value only | 0 | 278423 | 2.158 |
| global_xattn_0 | pair heads enabled | 4096 | 278423 | 2.848 |
| global_xattn_0 | pair heads enabled | 23220 | 278423 | 7.717 |
| global_pair_twostage_0 | policy/value only | 0 | 285719 | 1.578 |
| global_pair_twostage_0 | pair heads enabled | 4096 | 285719 | 3.474 |
| global_pair_twostage_0 | pair heads enabled | 23220 | 285719 | 11.772 |
| global_line_window_0 | policy/value only | 0 | 278423 | 1.759 |
| global_line_window_0 | pair heads enabled | 4096 | 278423 | 2.752 |
| global_line_window_0 | pair heads enabled | 23220 | 278423 | 7.262 |
| global_graph_full_0 | policy/value only | 0 | 278423 | 1.815 |
| global_graph_full_0 | pair heads enabled | 4096 | 278423 | 2.775 |
| global_graph_full_0 | pair heads enabled | 23220 | 278423 | 6.661 |

Notes:
- Non-materialized pair rows keep pair scoring out of the attention token sequence.
- `policy_pair_second` is now supervised only by `pair_second_policy_target`, which is nonzero only for known-first second-placement rows.
- Config-built global graph models now compute optional heads only when requested by `model.heads`.
- The pair-two-stage architecture now has a dedicated pair refinement branch and a distinct parameter count from the shared trunk variants.
- The joint pair scorer now uses symmetric unordered-pair features; the second-placement head remains order-conditioned on the known first placement.
