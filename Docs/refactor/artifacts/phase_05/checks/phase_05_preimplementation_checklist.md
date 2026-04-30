# Phase 05 Preimplementation Checklist

## Assignment Frame

Goal

Move policy priors, pair-action scoring, and Rust MCTS calls behind explicit `PolicyProvider`, `PairStrategy`, and `EngineAdapter` interfaces. Self-play and game-running code must consume `SearchEvaluation` and `PairEvaluation` instead of wiring model outputs directly into MCTS.

Success criteria

- V2-050: All self-play model priors flow through registered `PolicyProvider` implementations.
- V2-051: `EngineAdapter` is the only Python caller of Rust MCTS APIs.
- V2-052: `PairStrategySpec` validates root, leaf, full, diagnostic caps independently.
- V2-053: No pair scoring happens from architecture/config/head presence.
- V2-054: Global graph policy heads have row-mapped contracts and telemetry.
- V2-055: Raw model outputs map to intended legal rows before MCTS, and MCTS cannot mutate validated inputs.
- V2-056: Python search uses canonical fallible Rust MCTS APIs with root/batch tokens and structured error ownership.
- V2-057: MCTS leaf selection/backprop remains batched through `EngineAdapter` and Rust hot paths.

Constraints

- No compatibility shims that keep old direct worker-to-MCTS or worker-owned pair scoring runtime paths alive.
- No provider checks `architecture.startswith(...)`.
- No provider enables pair scoring.
- Default pair strategy is `none`, including `global_xattn` and global graph families.
- Leaf pair scoring remains disabled unless explicit caps and strategy allow it.
- Full pair scoring is diagnostic-only, root-only, capped, and never leaf-scored.
- `EngineAdapter` accepts only `SearchEvaluation` and `PairEvaluation`, never raw logits.

Required evidence

- Provider API/registration docs.
- Row-mapped `SearchEvaluation` validation tests for dense, restnet, graph_hybrid, global_graph.
- `PairStrategySpec` validation tests for none/root/leaf/full/diagnostic caps.
- EngineAdapter import audit proving it is the only Rust MCTS caller.
- Worker audits proving no architecture string checks, direct MCTS calls, or worker pair chunk scoring remain.
- MCTS trace and debug bundle with policy provider, pair strategy, rows, priors, hashes, timings, and selected move.
- MCTS performance profile with split timings and batching evidence.

Stop rules

- Stop before coding if a required direct MCTS deletion would remove the only runtime path before `EngineAdapter` consumes it.
- Stop if search context would need to rebuild legal rows, candidates, graph rows, pair rows, compact history, or D6 transforms.
- Stop if an EngineAdapter test would need to use panic wrappers, tokenless APIs, skipped coverage, or manual-only verification.
- Stop if pair scoring requires implicit head/config/architecture behavior to preserve current results.
- Stop if performance evidence cannot be produced for changed MCTS hot paths.

## Matrix Rows

| Row | Requirement | Status |
|---|---|---|
| V2-050 | All self-play model priors flow through `PolicyProvider` | complete: `Python/src/hexorl/search/policy_provider.py`, `Python/src/hexorl/selfplay/worker.py`, `Python/tests/search/test_policy_provider.py` |
| V2-051 | `EngineAdapter` is the only Python caller of Rust MCTS APIs | complete: worker wrappers deleted, MCTS boundary moved to `Python/src/hexorl/search/engine_adapter.py`, audit recorded in `phase_05_import_audit.md` |
| V2-052 | `PairStrategySpec` validates root/leaf/full/diagnostic caps | complete: `Python/src/hexorl/search/pair_strategy.py`, `Python/tests/search/test_pair_strategy.py` |
| V2-053 | No pair scoring from architecture/config/head presence | complete: no-implicit-pair tests and worker audit pass |
| V2-054 | Global graph policy heads have row-mapped contracts and telemetry | complete: `Python/tests/search/test_global_graph_pair_contracts.py` |
| V2-055 | Policy/search verification and mutation guards | complete: `SearchEvaluation`/`PairEvaluation` immutability and corruption tests plus debug bundle |
| V2-056 | Canonical fallible Rust MCTS API with token/error ownership | complete: `EngineAdapter` structured error and stale root/batch token tests |
| V2-057 | Batched MCTS hot path through `EngineAdapter` | complete: `mcts_runner.py`, adapter batch tests, performance profile artifact |

## Initial Runtime Audit Snapshot

Original direct runtime owners replaced:

- `Python/src/hexorl/selfplay/worker.py` no longer defines `MockMCTSEngine` or `RealMCTSEngine`.
- `SelfPlayWorker` calls search runner helpers and consumes `SearchEvaluation`/`PairEvaluation`; direct MCTS prior wiring was removed.
- `SelfPlayWorker` derives global policy usage from `ModelSpec.is_global_graph`, not architecture-prefix checks.
- Pair influence is owned by `PairStrategySpec`/`PairStrategy`; `pair_prior_mix` and head presence do not authorize scoring.

These items are covered by tests, import audits, and deletion manifest artifacts.
