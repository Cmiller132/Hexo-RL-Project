# Implementation Sequence And Completeness Checklist

Date: 2026-04-29

Purpose: provide an executable sequence and completion checklist for the public API, FFI, invariant, CI, and performance-budget refactor slice.

## Implementation Sequence

1. Narrow `hexgame-core` imports to stable facades.
2. Remove public documentation and root exports for stale `ThreatStatus`.
3. Keep only active public implementation modules that have a current downstream reason, currently `mcts` for `hexgame-py`.
4. Update Rust API docs to show facade imports and the FFI exception.
5. Split CI into fast PR gates and scheduled/manual deep oracle gates.
6. Record performance-budget targets and the exact path to executable hard gates.
7. Run formatting, workspace tests, release tests, clippy, and feasible Python smoke/invariant/inference tests.

## Completeness Checklist

| Section | Required outcome | Status |
| --- | --- | --- |
| Public API | Stable facade modules are the documented import surface; root no longer re-exports implementation details. | Implemented in this slice. |
| FFI protocol | Python extension dependencies on `hexgame-core` are explicit and use owning facades or the active `mcts` module. | Implemented in this slice. |
| Invariants | Rule, tactical, WindowKey, and evaluation-bound invariants are documented with verification paths. | Documented in this slice. |
| WindowKey/eval bounds | Sparse tactical correctness is separated from bounded heuristic evaluation. | Documented in this slice. |
| CI/perf gates | Fast CI includes fmt, tests, release tests, clippy, and Python smoke/invariant/inference; deep oracle tests are separated. | Implemented for CI; hard perf gates planned. |
| Implementation sequence | Ordered steps are recorded for this slice and future perf gate promotion. | Documented in this slice. |
| Completeness checklist | This checklist maps each requested section to an outcome. | Documented in this slice. |

## Acceptance Evidence To Capture

The final report for this slice should include:

- Changed files.
- Rust commands and results.
- Python commands and results, or the exact dependency/environment reason a Python command could not run.
- Any requested item that could not be completed.

## No Legacy Compatibility Path

This slice intentionally does not add a root-level compatibility shim for removed exports. In-repo callers were migrated to facade imports. Downstream callers should make the same import change rather than depending on root convenience exports.
