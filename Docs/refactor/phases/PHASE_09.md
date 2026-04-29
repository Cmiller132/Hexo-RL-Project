# Phase 09 - Final Deletion And CI Enforcement

## Purpose

Perform the final conformance sweep after all runtime cutovers are complete. Phase 09 should not be the first serious deletion phase. It is the final proof that earlier phase deletions held, no compatibility systems crept back in, CI enforces the V2 architecture, and the repo now describes only the new cohesive project.

Source of truth: `Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_V2_20260429.md`.

## Required End State

At Phase 09 exit:

- no runtime compatibility facades remain
- no deprecated architecture aliases preserve old behavior
- no production import relies on old `model/`, `buffer/`, private action-contract, dashboard reconstruction, or worker-owned search paths
- no runtime code infers behavior from architecture strings outside the model registry/spec layer
- no production Python legal/history/D6 fallback remains
- no pair scoring can occur outside `PairStrategy`
- no full pair enumeration can occur without diagnostic strategy, root-only mode, and caps
- dashboard, training, eval, replay, self-play, inference, search, models, tuning, and engine all consume the canonical owners established in Phases 01-08
- CI blocks regressions automatically

## Required Deletions And Audits

Deletion was required throughout the program. Phase 09 must verify and finish any remaining non-runtime cleanup.

Required deletion checks:

```text
Python/src/hexorl/model/ absent or non-runtime migration-only references absent
Python/src/hexorl/buffer/ absent from runtime imports
Python/src/hexorl/action_contract/ absent from runtime imports
private legal/history/D6 helpers absent outside contracts/engine/tests/tools
worker-owned pair chunk helpers absent
dashboard private reconstruction helpers absent
autotune raw-config family mutation paths absent
runtime architecture aliases absent
checkpoint cleanup duplicated outside CheckpointManager absent
old inference submit lifecycle helpers absent
```

If a migration tool remains, it must live outside `Python/src/hexorl/`, be documented as offline-only, and be blocked from production imports.

## CI Policy Gates

Add or update CI so the following are enforced by commands, import audits, or policy tests:

```text
cargo test --workspace
pytest Python/tests/contracts Python/tests/engine Python/tests/models
pytest Python/tests/inference Python/tests/search Python/tests/replay Python/tests/train Python/tests/eval
pytest Python/tests/tuning Python/tests/dashboard
npm run build in Python/dashboard_frontend
tuning recipe dry-run validation
self-play no-pair default smoke
inference protocol mismatch fail-fast test
self-play -> replay -> train -> eval -> dashboard smoke
```

Required policy checks:

```text
no architecture-string gates outside registry/spec tests
no startswith("global_") behavior gates outside registry/spec tests
no pair scoring outside search/pair_strategy.py
no direct Rust MCTS calls outside search/engine_adapter.py
no private legal/history/D6 parsers outside contracts/engine/tests/tools
no runtime imports from old model/buffer/action_contract paths
no dashboard route imports sampler-private builders
no trainer model-class checks
no eval dense-only player assumptions
no autotune raw config mutation for model-family behavior
```

Use AST/import-graph checks where regex would be too brittle.

## Final End-To-End Smoke

Archive one final smoke under:

```text
Docs/refactor/artifacts/phase_09/final_smoke/
```

The smoke must cover:

```text
self-play game generation
canonical replay write
canonical replay read
sample -> projector -> training batch
one train step for every registered family or representative family set approved by matrix
eval through PolicyProvider
dashboard ContractInspector route/view smoke
tuning recipe dry-run and rejected-recipe explanation
structured logs/traces for self-play and autotune
```

## V2 Requirement Matrix Closure

`Docs/refactor/V2_REQUIREMENTS_MATRIX.md` must show every row closed.

No row may remain:

```text
partial
deferred
implemented but not consumed
tested only in unit scope
shim remains
manual verification only
```

## Parallel Subagent Work

- S1: requirement matrix closure, schema/alias removal proof, contract invariant audit.
- S2: runtime import graph, engine/self-play/inference/search deletion proof.
- S3: model/checkpoint/train/eval/tuning deletion proof.
- S4: replay/dashboard/autotune artifact and smoke verification.
- S5: CI policy jobs, final conformance report, documentation cleanup.

## Mandatory Tests

- Full CI matrix green.
- Import graph and banned-path checks green.
- Final end-to-end smoke archived.
- Dashboard build and route smoke pass.
- Tuning dry-run and rejection-reason tests pass.
- Rollback drill from final cut tag documented.

## Required Artifacts

```text
Docs/refactor/artifacts/phase_09/MANIFEST.md
Docs/refactor/artifacts/phase_09/ci/
Docs/refactor/artifacts/phase_09/import_audits/
Docs/refactor/artifacts/phase_09/deletion_manifest/
Docs/refactor/artifacts/phase_09/final_smoke/
Docs/refactor/artifacts/phase_09/telemetry_samples/
Docs/refactor/artifacts/phase_09/final_conformance_report.md
```

## Exit Criteria

- CI automatically enforces all V2 architecture invariants.
- No compatibility shims remain in the main runtime path.
- No banned imports or behavior gates remain.
- Final smoke proves the cohesive runtime flow works end to end.
- V2 requirement matrix is fully closed.
- Final conformance report confirms complete, spec-compliant delivery.
