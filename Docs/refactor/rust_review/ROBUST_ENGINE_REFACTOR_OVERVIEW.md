# Robust Engine Refactor Overview

Date: 2026-04-29

## Goal

Build one robust Rust engine surface for rules, encoding, tactics, classical
search, MCTS, and Python inference.  The preferred outcome is a smaller set of
canonical code paths with explicit validation, no panic-based public protocols,
and no compatibility fallbacks that can bypass the safety model.

## Acceptance Standard

Every section is complete only when:

- the active code path is singular and documented;
- malformed external input returns a typed error before state mutation;
- stale protocol data cannot be submitted successfully;
- release builds do not silently truncate or reinterpret coordinates;
- incremental caches are covered by recompute-based tests;
- CI runs the fast gates needed for pull requests; and
- deep oracle/performance checks have an explicit scheduled or manual gate.

## Section Map

| Section | Scope | Completion Requirement |
|---|---|---|
| 1 | Public API narrowing | Stable facade modules are the documented public surface. Implementation details are private unless an active first-party caller needs them. |
| 2 | FFI protocol centralization | Legal rows, pair rows, board pieces, and compact histories share one Rust protocol helper at the PyO3 boundary. |
| 3 | Consistency and invariant hooks | `HexGameState` can recompute and assert hash, candidates, winner, move history, and eval/hot data in debug/test builds. |
| 4 | `WindowKey` and eval bounds | Runtime window keys cannot truncate in release; bounded eval is clearly separate from full-board tactical correctness. |
| 5 | CI and performance gates | PR CI is fast and complete; scheduled/manual CI carries ignored oracle and benchmark-budget gates. |
| 6 | Implementation sequence | Changes land in dependency order so callers never straddle old and new contracts. |
| 7 | Completeness review | Each slice is audited against the same acceptance standard before final verification. |

## Canonical Architecture

The crate should expose stable facades:

- `rules`: coordinates, turns, board state, and rule errors.
- `encoding`: neural tensor encoding and feature extraction.
- `tactics`: full tactical status, masks, and live-cell helpers.
- `classical`: deterministic alpha-beta search.
- root-level MCTS only if it remains an intentionally stable engine API.

All Python-facing binary protocols must live behind one Rust protocol module.
All public or FFI misuse must be fallible.  Panics and debug assertions are
reserved for unreachable internal invariants after validated boundaries.

## Non-Goals

This refactor does not change game rules, model architecture, or self-play
strategy.  It changes the safety, validation, and maintainability of the engine
paths those systems already depend on.
