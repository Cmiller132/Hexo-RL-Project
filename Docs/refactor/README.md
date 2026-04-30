# Hexo Modular Refactor Program

Date: 2026-04-29

This directory is the execution control plane for the V2 modular architecture redesign.

Source of truth: `Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_V2_20260429.md`

The goal is a breaking refactor into one cohesive project. These docs intentionally reject long-lived legacy support, compatibility facades in runtime code, duplicate old/new paths, and under-specified "convergence" work. A phase is complete only when its V2 requirements are implemented, consumed by runtime where applicable, tested, observable, and cleaned up.

## Rust Refactor Baseline

The Rust engine has completed the Phase 2 hardening slice documented under `rust_review/`. That changes the Python refactor starting point:

- Rust exposes narrower facade modules for rules, encoding, tactics, and classical search.
- The Python extension has centralized byte-protocol helpers for legal rows, compact history rows, board-piece rows, and pair rows.
- MCTS no longer has legacy panic convenience wrappers in the active path; Python-facing search work must route through the robust canonical API and preserve root/batch token checks.
- `TacticalStatus` is the public tactical model; `ThreatStatus` is not a compatibility target.
- Rust now has stronger invariant hooks and WindowKey/eval-bound documentation.

This does not make Rust a trusted oracle. It makes Rust the production rules boundary that must still be checked at every Python contract boundary with semantic parity, stale-token tests, corruption tests, structured errors, and debug bundles.

## Structure

- `orchestration/PARALLEL_SUBAGENT_EXECUTION_MODEL.md`
  - Defines one primary orchestrator + five parallel subagents.
  - Defines strict phase gates and anti-partial-implementation checks.
- `phases/PHASE_00.md` ... `phases/PHASE_09.md`
  - One document per phase with objective, work packages, strict review checks, and exit criteria.
- `PHASE_CHECKLIST.md`
  - Universal and phase-level completion checklist used at signoff.
- `PHASED_IMPLEMENTATION_PLAN.md`
  - Master program overview, sequencing, and promotion rules.
- `V2_REQUIREMENTS_MATRIX.md`
  - Requirement-level tracker. This is the orchestrator's master signoff surface.

## Non-Negotiable Rule

A phase is not complete until all mandatory tests pass, artifacts are attached, the deletion/import audits pass, required telemetry samples exist, and the orchestrator signs off that no feature remains half-implemented, unconsumed, deferred, or spec-incomplete.

## No-Deferral Rule

No phase may defer its own core requirement to a later phase. Later phases can build on prior work, but they cannot rescue missing cutovers, missing deletion gates, missing observability, or old runtime paths that should have been removed earlier.
