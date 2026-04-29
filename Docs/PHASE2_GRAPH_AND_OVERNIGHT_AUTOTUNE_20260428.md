# Phase 2 Graph And Overnight Autotune Notes - 2026-04-28

## Implementation

- Added `architecture="graph_hybrid_0"` as a config-gated Phase 2 scout family. Legacy `architecture="graph"` configs are accepted as an alias and normalized to `graph_hybrid_0`.
- The graph-hybrid path keeps the existing dense compatibility contract: input `(B,13,33,33)`, dense `policy -> (B,1089)`, and `value -> (B,n_bins)`.
- Added a sparse Hex graph Transformer trunk that selects deterministic tactical cell tokens from the encoded state, mixes them with state/turn/player special tokens, and scatters updated token features back into the shared trunk.
- Kept global action identity in the existing candidate/action-keyed sparse policy head, so MCTS can consume sparse `(q,r)` priors without changing dense default behavior.
- Added graph config fields:
  - `graph_token_set`
  - `graph_token_budget`
  - `graph_layers`
- Updated the Phase 3 supervisor so `graph_hybrid_0` is available and uses the crop-compatible sparse-token ladder with action-keyed priors.
- Naming correction: this implementation is not the true global sparse-token `global_graph_option1` described in the Phase 2 architecture spec. It is a crop-compatible hybrid scout.
- Hardened Phase 3A calibration so a pruned or failed trial cannot crash the whole supervisor.

## Overnight Orchestration

- The restarted run includes Phase 2 graph finalist calibration inside the Phase 3 pool.
- The watchdog checks process health, GPU visibility, the latest supervisor event, and restart status for 8 hours.
- Watchdog output:
  - `runs/phase2_phase3_autotune_overnight_20260428/overnight_monitor.md`
  - `runs/phase2_phase3_autotune_overnight_20260428/overnight_monitor_events.jsonl`

Method-status note:

```text
The overnight supervisor used synchronous ASHA-style static narrowing and
PBT-style exploit/explore. It did not run true BOHB, and it did not run true
PB2. Any report from this run should use ASHA-style/PBT terminology unless the
real BOHB/PB2 scheduler paths are later implemented.
```

## Early Risk

The first Phase 3 attempt found that sparse/candidate trials can go non-finite at the original CNN LR center. The supervisor now starts sparse and graph families at a conservative finite-metric safety rail, then lets PBT explore upward after a valid scorecard exists.

## 00:35 Intervention

- The original graph calibration at 800 sims and 8 workers overloaded inference latency: workers hit 30s inference timeouts and Rust MCTS panicked while re-rooting with pending leaves.
- Added Rust MCTS pending-leaf cleanup before `re_root`, with a regression test proving re-root no longer panics after an abandoned selected batch.
- Changed graph autotuning to start lighter automatically:
  - graph calibration recommended recipe: 256 sims, 96 PCR low sims
  - graph self-play cap: 4 workers for the 256-token/1-layer path, lower max batch latency target
  - graph ASHA candidates: 256/384 sims instead of immediately trying 800/1200
- Added family quarantine after hard calibration failures so a family that generates zero self-play positions cannot keep entering ASHA/PBT.

## Early Results After Fix

| Trial | Stage | Self-play pos/min | Train batches/sec | Candidate recall top8 | Notes |
|---|---|---:|---:|---:|---|
| `cal_best_current_33` | 3A | 469.6 | 2.67 | n/a | Dense CNN baseline healthy. |
| `cal_best_restnet_33` | 3A | 400.9 | 3.58 | n/a | Lower LR fixed the earlier non-finite loss. |
| `cal_candidate_policy_33` | 3A | 490.6 | 0.98 | 1.0 | Sparse target/candidate path healthy. |
| `cal_best_restnet_33_candidate_policy_33` | 3A | 382.6 | 1.47 | 1.0 | Stable but slower self-play. |
| `cal_graph_hybrid_0` | 3A | 801.1 | 1.76 | 1.0 | Hybrid path now stable at lower search pressure. Historical run id was `cal_best_graph_option1`. |
| `asha_00_graph_hybrid_0` | 3B | 1548.5 | 2.84 | 1.0 | First ASHA hybrid trial completed cleanly. Historical run id was `asha_00_best_graph_option1`. |

The graph-hybrid architecture was not broken; the failure was an inference latency/worker pressure problem. With automatic hybrid-specific search/worker scaling, it is currently the fastest self-play path in the suite.
