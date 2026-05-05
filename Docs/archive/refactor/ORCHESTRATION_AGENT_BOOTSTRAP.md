# Orchestration Agent Bootstrap

Date: 2026-04-29

This document defines exactly what to send the first orchestration agent that will execute the V2 modular refactor with GPT-5.5.

The prompt is intentionally outcome-first. GPT-5.5 should receive clear goals, success criteria, constraints, evidence rules, and stop rules, while retaining freedom to choose the best implementation path.

## Recommended Model Settings

- Model: `gpt-5.5`
- Reasoning effort: `high` for normal phase execution
- Reasoning effort: `xhigh` for phase kickoff, phase closure, adversarial review, CI architecture design, or cross-phase integration decisions
- Verbosity: `medium`
- Use prompt caching if running through the API. Put stable repository/process instructions first and dynamic phase state last.

## Initial Document Packet

Send these documents to the orchestration agent at kickoff, in this order:

```text
AGENTS.md
Docs/refactor/README.md
Docs/refactor/PHASED_IMPLEMENTATION_PLAN.md
Docs/refactor/EXECUTION_QUALITY_GUARDRAILS.md
Docs/refactor/PHASE_CHECKLIST.md
Docs/refactor/V2_REQUIREMENTS_MATRIX.md
Docs/refactor/orchestration/PARALLEL_SUBAGENT_EXECUTION_MODEL.md
Docs/refactor/CI_STRATEGY.md
Docs/refactor/PERFORMANCE_STRATEGY.md
Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_V2_20260429.md
Docs/RUST_API.md
Docs/refactor/rust_review/README.md
Docs/refactor/rust_review/PHASE_2_VERIFICATION_REPORT.md
Docs/refactor/rust_review/CI_AND_PERFORMANCE_BUDGET_PLAN.md
Docs/refactor/phases/PHASE_00.md
```

Do not inline every phase doc into the first prompt unless the runner has enough context and prompt caching. Provide the paths to Phases 01-09 and require the orchestrator to read each phase doc before starting that phase:

```text
Docs/refactor/phases/PHASE_01.md
Docs/refactor/phases/PHASE_02.md
Docs/refactor/phases/PHASE_03.md
Docs/refactor/phases/PHASE_04.md
Docs/refactor/phases/PHASE_05.md
Docs/refactor/phases/PHASE_06.md
Docs/refactor/phases/PHASE_07.md
Docs/refactor/phases/PHASE_08.md
Docs/refactor/phases/PHASE_09.md
```

For any active phase after Phase 00, also send:

```text
the active phase doc
the previous phase exit gate report
the previous phase artifact MANIFEST.md
the current V2 matrix
current git status
current CI status or latest command transcripts
any relevant rust_review annex docs
```

## Exact Initial Prompt

Use this as the initial user message to the orchestration agent:

```text
You are the orchestration agent for the Hexo-RL V2 modular refactor.

Goal
Execute the V2 modular refactor phase by phase, beginning with Phase 00. Your job is not to create partial scaffolding. Your job is to deliver a cohesive, robust, high-performance architecture whose old runtime paths are deleted or quarantined outside runtime as each phase closes.

Success Criteria
- Every phase closes only when all owned V2 matrix rows are implemented, consumed by runtime where applicable, tested, observable, documented, and cleaned up.
- Every phase produces the artifact packet required by Docs/refactor/PHASED_IMPLEMENTATION_PLAN.md and Docs/refactor/EXECUTION_QUALITY_GUARDRAILS.md.
- Every implementation assignment uses this structure: Goal, Success criteria, Constraints, Required evidence, Stop rules.
- Every subagent completion packet is reconciled into agent_completion_packet.md and evidence_reconciliation.md.
- Old runtime paths, compatibility shims, duplicate semantic owners, private rebuilds, architecture-string dispatch, Python fallback rules paths, and unsafe Rust/MCTS paths are removed according to the active phase.
- Rust is treated as the canonical rules boundary but not as self-validating. Rust-derived data must have semantic validation, negative tests, structured errors, and debug evidence.
- Performance is preserved or improved through HostProfile budgets, bounded backpressure, CPU/Rust parallelism, GPU batching, hot-path validation discipline, and benchmark artifacts.
- CI is tiered according to Docs/refactor/CI_STRATEGY.md. Phase-closing invariants cannot be manual-only, skipped, flaky-only, or unclassified.

Constraints
- Start with Phase 00 only. Do not implement later phases until Phase 00 is signed off.
- Do not defer phase-owned requirements to later phases.
- Do not leave TODOs, placeholders, temporary shims, future-work notes, or dual old/new runtime paths.
- Do not trust old runtime behavior as the only oracle.
- Do not trust Rust outputs without contract validation and negative tests.
- Do not make a centralized mega-object. Centralize semantic authority while allowing extension through approved facets, adapters, payload schemas, projections, inspectors, or contract versions.
- Do not sacrifice hot-path performance for debug convenience. Full validation belongs at construction/decode/replay/test/debug boundaries; hot paths keep cheap identity, hash, generation, shape, finite, and mutation checks.
- Preserve unrelated user changes in the worktree.

Required Evidence
For each phase, produce:
- phase scope and matrix-row checklist
- interface-freeze notes
- fixture/artifact plan
- CI routing plan
- command transcripts with exit codes
- import/code-search audits
- deletion manifest
- telemetry/debug samples
- performance artifacts for touched hot paths
- contract examples/docs for new public contracts or adapters
- adversarial review findings and resolution
- agent_completion_packet.md
- evidence_reconciliation.md
- exit_gate_report.md

Stop Rules
Stop and report a blocker before coding if:
- a required invariant conflicts with current docs or code
- a requested deletion would remove the only runtime path before replacement is consumed
- a phase-closing test would need to be skipped, xfailed, or made manual-only
- performance evidence cannot be produced for a hot-path change
- implementation would require a runtime compatibility shim or fallback not approved by the phase
- subagent work conflicts in ownership or writes overlapping files without a clear integration plan

First Actions
1. Read AGENTS.md and the refactor source-of-truth documents listed in Docs/refactor/ORCHESTRATION_AGENT_BOOTSTRAP.md.
2. Run git status and summarize dirty/untracked files without reverting anything.
3. Build the Phase 00 acceptance checklist from PHASE_00.md and V2_REQUIREMENTS_MATRIX.md.
4. Create the Phase 00 artifact directory structure.
5. Freeze Phase 00 scope, interfaces, fixtures, CI routing, and subagent assignments before implementation.
6. Use subagents only for concrete non-overlapping tasks and require completion packets.
7. Perform adversarial review before closing Phase 00.

Do not finish your turn until Phase 00 is either completed with evidence or blocked with an exact reason and the next required action.
```

## Orchestrator Process

The orchestrator should run each phase as a closure loop:

```text
1. Load instructions and active phase docs.
2. Check git status and existing artifacts.
3. Build a phase acceptance checklist from matrix rows and phase gates.
4. Freeze public interfaces and allowed implementation latitude.
5. Freeze fixtures, negative cases, artifact locations, and CI tier mapping.
6. Delegate non-overlapping implementation slices to subagents.
7. Integrate subagent changes and completion packets.
8. Run tests, import audits, deletion audits, telemetry checks, and performance probes.
9. Run adversarial review against stale data, malformed data, old paths, fallbacks, and partial wiring.
10. Fix all findings or record a true blocker.
11. Update artifacts, matrix, docs/examples, and exit report.
12. Re-check git status and produce final phase signoff.
```

## Subagent Assignment Template

Use this shape for every worker or explorer:

```text
Workspace
/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project

Goal
[One concrete bounded phase slice.]

Success Criteria
[Exact matrix rows, runtime consumers, deletion targets, tests, artifacts.]

Constraints
[Owned files/modules, no legacy support, no fallback paths, no overlapping write scope, performance/CI constraints.]

Required Evidence
[Commands, artifacts, import audits, telemetry/debug samples, performance proof.]

Stop Rules
[When to stop and report blocker instead of inventing a workaround.]

Completion Packet
Return closed V2 rows, files changed, runtime consumers changed, deleted/quarantined legacy paths, commands with exit status, artifacts, blockers, and explicit statement that no skipped/deferred/manual-only requirement is being claimed complete.
```

## What Not To Send As The First Prompt

Avoid sending:

- all phase docs inline without a clear active phase
- vague requests like "do the refactor"
- long step-by-step implementation recipes that conflict with the phase docs
- model-specific guesses about code structure before inspection
- instructions that allow compatibility, fallbacks, skipped tests, or later cleanup
- requests to implement multiple phases before Phase 00 closes

## Why This Shape

The prompt keeps stable guidance first, dynamic state last, and frames work around outcomes, evidence, and stop rules. This matches the GPT-5.5 guidance to state the expected outcome and success criteria, reduce unnecessary step-by-step process guidance, and be explicit about orchestration, acceptance criteria, and when to continue versus ask for help.
