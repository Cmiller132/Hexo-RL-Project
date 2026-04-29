# Modular Refactor — Master Implementation Plan

Date: 2026-04-29

## Purpose
Translate the redesign proposal into a strict, test-gated, parallelized execution program.

## Sequencing

The program executes in strict order:

0. `phases/PHASE_00.md` — Program Setup and Baseline Freeze
1. `phases/PHASE_01.md` — Contracts Foundation
2. `phases/PHASE_02.md` — Engine Boundary and Rust/Python Parity
3. `phases/PHASE_03.md` — Model Registry and Family Adapters
4. `phases/PHASE_04.md` — Inference Protocol and Adapterization
5. `phases/PHASE_05.md` — Search and Pair Strategy Isolation
6. `phases/PHASE_06.md` — Self-Play Decomposition
7. `phases/PHASE_07.md` — Replay/Training/Eval Convergence
8. `phases/PHASE_08.md` — Dashboard and Debug Convergence
9. `phases/PHASE_09.md` — Deletion Sweep and CI Hardening

## Parallel Delivery Model

Execution is parallelized through five subagents with one orchestrator that owns gate approval.

See: `orchestration/PARALLEL_SUBAGENT_EXECUTION_MODEL.md`.

## Promotion Rule

A phase may advance only when all are true:

- mandatory unit/integration/parity/performance checks pass,
- telemetry + contract source/version assertions are present,
- no partial migrations or hidden legacy fallbacks remain in phase scope,
- rollback tag is created and validated,
- orchestrator signs off with evidence.

## Strictness Policy

The orchestrator must reject signoff if any implementation is feature-incomplete, partially wired, or spec-divergent, even if tests pass superficially.
