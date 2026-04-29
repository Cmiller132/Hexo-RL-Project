# Phase 06 — Self-Play Decomposition

## Purpose
Refactor self-play worker into composable services that do not own unrelated architecture logic.

## Target Modules
- `selfplay/game_runner.py`, `worker.py`, `orchestrator.py`, `records.py`, `record_writer.py`, `telemetry.py`, `rgsc.py`

## V2 Requirements
- Worker coordinates; it does not privately build legal rows, D6 transforms, candidates, pair rows, graph rows, or checkpoint cleanup.
- Game runner executes lifecycle without model-family special-casing.
- MCTS/search integration uses `search/` interfaces exclusively.

## Parallel Subagent Work
- S1: define self-play contract boundaries and handoff objects.
- S2: lifecycle split and orchestrator wiring.
- S3: search/policy-provider integration cleanup.
- S4: replay record completeness and schema alignment.
- S5: deterministic and robustness regression suites.

## Mandatory Tests
- Seeded deterministic game sequence equivalence.
- Replay record field completeness/validation.
- Tactical/candidate/pair telemetry presence and consistency.
- RGSC restart/service continuity tests.

## Exit Criteria
- Worker responsibilities reduced to orchestration.
- All heavy logic delegated to contracts/engine/search/inference components.
