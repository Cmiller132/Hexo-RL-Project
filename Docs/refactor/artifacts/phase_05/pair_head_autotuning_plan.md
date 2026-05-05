# Pair Head Autotuning Plan

Date: 2026-05-05

## Goal

Tune the global graph pair heads while keeping the comparison pool focused on
plain ResTNet and global graph models only.

## Success Criteria

- Launch family pool contains only:
  - `best_restnet_33`
  - `global_xattn_0`
  - `global_line_window_0`
  - `global_pair_twostage_0`
  - `global_graph_full_0`
- Game cap is 500 moves to reduce artificial draw pressure.
- Pair-capable graph runs train `policy_pair_first`, `policy_pair_joint`, and
  `policy_pair_second`.
- Strategy evolution waits for at least 10 completed epochs.
- ASHA resources advance in 10-epoch units.
- Run artifacts include manifest, events, trial metrics, ASHA/PB2 state, and
  final report.

## Constraints

- Do not include dense CNN, candidate-policy CNN, sparse RestNet, or
  `graph_hybrid_0` in this run.
- Do not treat pair-head presence as search permission. Pair influence remains
  strategy-controlled.
- Do not promote a family on skipped, missing, non-finite, or manual-only
  evidence.
- Do not compare pair-head quality from one metric alone; require training
  losses, tactical metrics, game outcomes, throughput, and failure telemetry.

## Required Evidence

- `manifest.json`: confirms family filter, max game moves, ASHA resources,
  perturb interval, score gates, static space, and host.
- `events.jsonl`: confirms stage transitions, trial starts, scorecards,
  quarantines, prunes, and scheduler decisions.
- `trials/*/metrics.jsonl`: confirms train loss, pair losses, self-play output,
  buffer health, illegal/crash rate, and throughput.
- `asha_rungs.json`: confirms 10/20/30 epoch resource decisions.
- `pb2_scheduler.json` or mutation history: confirms no strategy evolution
  before the 10-epoch perturb interval.
- `report.md`: final ranking and known blockers.

## Stop Rules

- Stop a trial on non-finite train metrics.
- Stop or quarantine a family on policy target mass drop.
- Stop or quarantine a family on illegal/crash rate hard failure.
- Stop the run if global graph output heads are missing required `policy_place`
  or `value` outputs.
- Stop the run if pair-head trials produce pair rows but zero pair target mass
  outside expected opening/first-placement second-head skips.
- Stop the run if monitor sees repeated no-progress checks with no epoch,
  checkpoint, or progress line.

## Methodology

### Stage 3A - Calibration

- Epochs: 1 calibration epoch per family.
- Purpose: validate runtime, memory, throughput, replay shape, graph row
  construction, and pair-head loss presence.
- Gate: failed setup, no positions, non-finite losses, or target mass drops are
  hard failures.

### Stage 3B - Static ASHA

- Epoch resources: 10, 20, 30.
- Purpose: compare static recipe choices before schedule evolution.
- Promotion: ASHA promotes only after a completed rung with finite scorecard.
- Pair emphasis: compare `global_pair_twostage_0` and `global_graph_full_0`
  against `global_xattn_0`, `global_line_window_0`, and `best_restnet_33`.

### Stage 3C - Schedule Search

- Epoch cadence: no PB2/PBT perturbation before 10 completed epochs.
- Purpose: tune learning rate, weight decay, MCTS exploration, recency,
  auxiliary weights, graph auxiliary multiplier, and pair policy loss.
- Pair-head metric handling:
  - track `policy_pair_first`
  - track `policy_pair_joint`
  - track `policy_pair_second`
  - confirm skipped second-head loss only on first-placement unordered rows

### Stage 3D - Champion Protection

- Epoch floor: champion candidates must satisfy the configured champion minimum
  before final selection.
- Purpose: keep the best ResTNet/global graph candidate training long enough to
  distinguish learning from early stochastic wins.

### Stage 3E - Final Evaluation

- Purpose: run final arena/tactical scoring and produce report artifacts.
- Evidence: report includes top trials, scorecards, checkpoints, failures, and
  whether pair-capable graphs actually improved over non-pair global controls.

## Current Launch Defaults

`scripts/launch_phase3_48h_autotune.sh` now defaults to:

```text
MAX_GAME_MOVES=500
ASHA_RESOURCES=10,20,30
PERTURB_INTERVAL=10
STRATEGY_SCORE_MIN_EPOCHS=10
FAMILY_FILTER=best_restnet_33,global_xattn_0,global_line_window_0,global_pair_twostage_0,global_graph_full_0
```

The Python supervisor defaults match the 500-move and 10-epoch cadence when
called directly.

## Issue Watchlist

- Pair-capable graph trial has no pair losses.
- `policy_pair_second` appears in first-placement loss records.
- Pair rows cause graph build time to return to hundreds of milliseconds.
- Full-pair memory crosses expected RTX 4070 Ti envelope.
- ResTNet dominates only because graph trials underfeed GPU or stall on graph
  construction.
- 500-move games produce very low terminal outcome rate despite fewer artificial
  max-move draws.
- Strategy scheduler mutates before 10 completed epochs.
