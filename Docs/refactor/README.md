# Hexo Modular Refactor Program

Date: 2026-04-29

This directory is the execution control-plane for the modular architecture redesign.

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

## Non-Negotiable Rule

A phase is not complete until all mandatory tests pass, artifacts are attached, and the orchestrator signs off that no feature remains half-implemented or spec-incomplete.
