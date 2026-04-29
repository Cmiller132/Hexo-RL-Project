# Phase 07 — Replay/Training/Eval Convergence

## Purpose
Unify replay and learning/evaluation data flow around shared contracts and canonical projection.

## Target Modules
- `replay/codec.py`, `storage.py`, `sampler.py`, `projector.py`, `fixtures.py`
- `train/adapters.py`, `trainer.py`, `losses.py`, `schedules.py`
- `eval/policy_player.py`, `players.py`, `arena.py`, `scorecard.py`, `league.py`

## V2 Requirements
- `replay/` becomes canonical runtime path; no parallel `buffer/` production path.
- Training projection consumes `PositionContract` + legal/candidate/pair contracts.
- Eval/debug tools inspect same projected contract forms.

## Parallel Subagent Work
- S1: replay contract versioning and migration tools.
- S2: write/read integration from self-play runtime.
- S3: model-facing batch adapters from canonical projection.
- S4: sample->batch->loss and eval-player integration.
- S5: parity and drift checks for learning data quality.

## Mandatory Tests
- End-to-end sample-to-loss smoke.
- Legacy-vs-new projector parity on fixed dataset.
- Eval player correctness with unified policy provider.
- Storage codec compatibility tests and corruption handling.

## Exit Criteria
- Train/eval/replay all consume one contract projection path.
- Legacy `buffer/` runtime path removed from active flow (pending final deletion phase).
