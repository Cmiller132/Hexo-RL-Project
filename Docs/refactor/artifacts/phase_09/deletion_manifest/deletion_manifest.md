# Phase 09 Deletion Manifest

Deleted runtime packages:
- `Python/src/hexorl/action_contract/`
- `Python/src/hexorl/buffer/`

Deleted stale non-runtime scripts/tests/bench probes:
- `Python/scripts/fresh_full_probe.py`
- `Python/tests/test_training_data_pipeline.py`
- `benches/selfplay_benchmark.py`
- `runs/axis_delta_prod_probe/run_probe.py`

Updated stale callers:
- `hexorl.action_contract.tactical_oracle` imports moved to `hexorl.engine.tactical`.
- Long-run scripts now use `hexorl.replay.storage.ReplayStorage`, not `hexorl.buffer`.
- Bench imports now use `hexorl.models.network`, not `hexorl.model.network`.
- `Docs/SYSTEM_DESIGN.md` now describes `hexorl.replay` as the canonical replay owner.

Deletion proof:
- `tools/refactor/phase09_policy_audit.py` enforces absence of `Python/src/hexorl/model`, `Python/src/hexorl/buffer`, and `Python/src/hexorl/action_contract`.
- `import_audits/phase09_policy_audit.json` reports zero findings.
