# Inference/Search/Self-Play Findings Fix - 2026-04-30

## Closed Rows

- V2-040, V2-041, V2-042, V2-046
- V2-050, V2-052, V2-053, V2-055, V2-056
- V2-060, V2-061, V2-062

## Evidence

- `./.venv/bin/python -m pytest -q Python/tests/search/test_policy_provider.py Python/tests/search/test_pair_strategy.py Python/tests/search/test_pair_strategy_selfplay_integration.py Python/tests/selfplay/test_game_runner_interface.py Python/tests/inference/test_protocol_handshake.py Python/tests/inference/test_protocol_mismatch.py Python/tests/inference/test_batching_policy.py Python/tests/inference/test_server_dispatch_by_request_kind.py` exited 0.
- `./.venv/bin/python -m pytest -q Python/tests/search/test_policy_provider.py Python/tests/search/test_pair_strategy.py Python/tests/search/test_pair_strategy_selfplay_integration.py Python/tests/selfplay/test_game_runner_interface.py Python/tests/inference/test_protocol_handshake.py Python/tests/inference/test_protocol_mismatch.py Python/tests/inference/test_batching_policy.py Python/tests/inference/test_server_dispatch_by_request_kind.py Python/tests/test_inference_server.py` exited 0.
- `./.venv/bin/python -m pytest -q Python/tests/inference Python/tests/search Python/tests/selfplay` exited 1 due to `Python/tests/search/test_global_graph_pair_contracts.py::test_policy_pair_joint_one_logit_per_pair_action_row`, failing in `Python/src/hexorl/graph/tensorize.py` graph pair projection validation outside this task's write scope.
- `PYTHONPATH=Python/src ./.venv/bin/python - <<'PY' ... BatchingPolicy 10000x synthetic selection ... PY` exited 0 with `avg_us=43.326`, `selected_workers=32`, `total_positions=128`, `high_watermark_hit=True`.

## Integrated Verification Update

- The concurrent graph pair projection blocker was fixed in the train/replay/graph integration slice.
- Reran the combined inference/search/self-play/dashboard/eval/tuning/phase09 suite:
  - `PYTHONPATH=Python/src ./.venv/bin/python -m pytest -q Python/tests/inference Python/tests/search Python/tests/selfplay Python/tests/dashboard/test_phase08_contract_inspector.py Python/tests/eval/test_phase08_eval_policy_provider.py Python/tests/tuning/test_phase08_typed_autotune.py Python/tests/phase09/test_phase09_policy_audit.py Python/tests/test_inference_server.py`
  - Exit status: 0
  - Result: `105 passed in 77.81s`.

## Audit Results

- Provider switch / architecture-prefix source audit exited 1 with no matches in `Python/src/hexorl/{inference,search,selfplay}`.
- Client self-manifest / hard-coded dense handshake source audit exited 1 with no matches in `Python/src/hexorl/inference`.
- Dense fallback audit found only loud missing-contract checks in `GraphHybridPolicyProvider` and pair scoring.
- Pair strategy `PairActionTable.target` scoring audit exited 1 with no matches in `Python/src/hexorl/{search,selfplay}`.
- Direct Rust lifecycle/fallback audit found Rust ownership only in `search/engine_adapter.py`; `selfplay/game_runner.py` consumes factories and has no `ENGINE_AVAILABLE`, `HAS_ENGINE`, `_engine`, or `force_mock=True` fallback path.

No skipped, deferred, flaky, or manual-only requirement is claimed complete by this artifact.
