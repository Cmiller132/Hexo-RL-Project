# Hexo-RL Project Agent Instructions

## Completion Standard

When implementing a request, do not take partial measures.

If the user, phase doc, or requirement matrix provides a list of specs, fixes, design details, or acceptance criteria:

1. Treat every listed item as required acceptance criteria.
2. Build a checklist from the request before editing.
3. Implement every item completely unless it is technically impossible.
4. Do not defer items because they are complex, time-consuming, risky, or require refactoring.
5. If a requirement is ambiguous, infer the most complete interpretation that fits the project.
6. If a requirement truly cannot be completed, stop and clearly explain the blocker before proceeding.
7. Do not leave TODOs, placeholders, temporary shims, "future work", or partial compatibility paths.
8. Remove obsolete code made unnecessary by the change.
9. Prefer one cohesive implementation over layered legacy support.
10. Run the relevant tests, type checks, lint checks, audits, or app verification before calling the work done.

Final responses must include:

- what was fully implemented
- any requested item that could not be completed, with the exact reason
- what verification was run

## Refactor Source Of Truth

The modular refactor is governed by:

- `Docs/refactor/README.md`
- `Docs/refactor/PHASED_IMPLEMENTATION_PLAN.md`
- `Docs/refactor/EXECUTION_QUALITY_GUARDRAILS.md`
- `Docs/refactor/PHASE_CHECKLIST.md`
- `Docs/refactor/V2_REQUIREMENTS_MATRIX.md`
- `Docs/refactor/orchestration/PARALLEL_SUBAGENT_EXECUTION_MODEL.md`
- `Docs/refactor/CI_STRATEGY.md`
- `Docs/refactor/PERFORMANCE_STRATEGY.md`
- the active `Docs/refactor/phases/PHASE_XX.md`

For Rust-facing work, also read:

- `Docs/RUST_API.md`
- `Docs/refactor/rust_review/README.md`
- `Docs/refactor/rust_review/PHASE_2_VERIFICATION_REPORT.md`
- `Docs/refactor/rust_review/CI_AND_PERFORMANCE_BUDGET_PLAN.md`

## Refactor Operating Rules

- A phase is complete only when its matrix rows are implemented, consumed by runtime, tested, observable, documented, and cleaned up.
- Do not keep old and new runtime paths alive together.
- Do not add compatibility facades inside `Python/src/hexorl/` unless a phase doc explicitly approves non-runtime migration tooling.
- Treat Rust as the canonical rules boundary but not as self-validating. Rust-derived legal rows, compact history, pair rows, MCTS tokens, FFI bytes, and tactics need semantic validation plus negative tests.
- Centralize semantic authority, not every operation. Extension must go through registered facets, adapters, projections, payload schemas, inspectors, or contract versions.
- Performance is an acceptance criterion. Hot-path changes require host profile, throughput, latency, queue/backpressure, batching, and regression evidence.
- CI is tiered. A phase-closing invariant cannot be closed by a flaky, skipped, quarantined, or manual-only check without deterministic replacement coverage and an explicit blocking record.

## Agent Execution Contract

Every implementation assignment must be framed as:

```text
Goal
Success criteria
Constraints
Required evidence
Stop rules
```

Each subagent or worker completion packet must report:

```text
closed V2 rows
runtime consumers changed
files changed
legacy paths deleted or quarantined
tests and commands run with exit status
artifacts produced
performance/utilization evidence for hot paths
contract examples/docs added where relevant
known blockers, if any
explicit statement that no skipped/deferred/manual-only requirement is being claimed complete
```

Narrative status is not enough. Evidence must be command-backed, artifact-backed, or reviewable from committed files.

## Before Editing

- Read the active phase doc and relevant source files first.
- Use `rg` and `rg --files` for searches.
- Check `git status --short` and preserve unrelated user changes.
- Build a checklist from the active requirements.
- Identify required tests, import audits, deletion proof, telemetry/debug artifacts, and performance evidence before making changes.

## After Editing

- Run focused tests as soon as they are useful.
- Run required phase checks before signoff.
- Run import/code-search audits for deleted or banned paths.
- Update phase artifacts under `Docs/refactor/artifacts/phase_XX/`.
- Update the V2 matrix only when implementation, runtime consumption, tests, deletion/import proof, telemetry/debug proof, docs/examples, and CI/performance evidence exist.
- Review `git diff` and `git status --short` before final response.
