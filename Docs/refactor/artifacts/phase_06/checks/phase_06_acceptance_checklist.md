# Phase 06 Acceptance Checklist

Goal

Move game execution out of `SelfPlayWorker` into `GameRunner`, leaving the worker as process, lifecycle, and IPC shell only.

Success criteria

- V2-060: `GameRunner` owns game execution through explicit policy, pair, engine, writer, telemetry, builder, and runtime spec dependencies.
- V2-061: `SelfPlayWorker` contains no game-loop, replay assembly, candidate/pair/graph construction, architecture gates, or MCTS wiring.
- V2-062: Self-play emits structured heartbeat, no-progress, game summary, policy timing, pair summary, backpressure, and validation failure events.
- V2-063: `ContractTrace` carries legal/candidate/pair/token/relation counts and required timing spans.
- V2-064: behavior debug bundles and mutation guards localize owner subsystems.
- V2-065: runtime resource spec and record writer backpressure expose bounded queue behavior.

Constraints

- No later phase requirements are claimed for replay storage cutover.
- No old worker runtime path remains.
- No uniform policy fallback remains in search/self-play.
- Pair scoring remains impossible unless `PairStrategy` explicitly enables it.

Required evidence

- Tests under `Python/tests/selfplay`.
- Pair self-play integration test under `Python/tests/search/test_pair_strategy_selfplay_integration.py`.
- Replay-boundary smoke under `Python/tests/replay`.
- Import audits in `import_audits/phase_06_import_audit.md`.
- Performance smoke in `performance/phase_06_selfplay_smoke_profile.json`.

Stop rules

No stop rule was triggered. The missing `Python/tests/replay` path was resolved by adding deterministic Phase 06 replay-boundary coverage.
