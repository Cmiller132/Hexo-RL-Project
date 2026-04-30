# Phase 01 Agent Completion Packet

## Closed V2 Rows

- `V2-010`: `contracts/` package exists, imports are pure, contract validation and source tests pass.
- `V2-011`: `engine/` package exists and runtime direct `_engine` imports were removed.
- `V2-012`: Rust legal rows flow through `LegalTableProvider` and validate into `LegalActionTable`.
- `V2-013`: `MoveHistory` is the shared compact-history contract and runtime parsers were replaced.
- `V2-014`: D6 transforms live in `contracts/symmetry.py`; old runtime D6 helpers were removed.
- `V2-015`: Schema/source/hash policy is implemented for history/legal and covered by tests.
- `V2-016`: Engine/legal/history/D6 validation, mutation safety, malformed byte checks, debug sample, and invariant smoke tests exist.

## Runtime Consumers Changed

Graph batch construction, replay sampler, buffer target processing, dashboard replay/app/model cache/fixtures, eval arena/players/classical, tactical oracle, RGSC, epoch bootstrap, and self-play worker now consume `contracts/` or `engine/` boundary helpers for Phase 01-owned semantics.

## Files Changed

Primary runtime additions are under `Python/src/hexorl/contracts/` and `Python/src/hexorl/engine/`. Runtime cutover touched `action_contract/tactical_oracle.py`, `buffer/sampler.py`, `buffer/targets.py`, `dashboard/*`, `eval/*`, `graph/batch.py`, `selfplay/rgsc.py`, `selfplay/worker.py`, and `epoch/pipeline.py`. Tests were added under `Python/tests/contracts/` and `Python/tests/engine/`, with focused updates to existing tests.

## Legacy Paths Deleted Or Quarantined

See `deletion_manifest/phase01_deletion_manifest.md`.

## Tests And Commands

- `focused_phase01_pytest.txt`: `169 passed`, exit `0`.
- `phase01_py_compile.txt`: exit `0`.
- Import/deletion audits under `import_audits/`.
- Full `Python/tests` was attempted but is not a Phase 01 closing gate; see `checks/adversarial_review.md`.

## Artifacts Produced

Telemetry/debug sample, performance timing JSON, contract examples, deletion manifest, import audits, CI routing plan, and exit-gate report are present under `Docs/refactor/artifacts/phase_01/`.

## Performance/Utilization Evidence

`performance/phase01_contract_engine_perf.json` records local timing for `MoveHistory.decode`, `decode_legal_bytes`, D6 history transform, and `LegalTableProvider.from_history`.

## Known Blockers

None for Phase 01-owned rows. The current Rust MCTS Python API lacks root/batch tokens; Phase 05 owns canonical MCTS adapter/token policy.

## No Deferred Claim

No skipped, xfailed, flaky-only, manual-only, or deferred Phase 01 requirement is claimed complete.
