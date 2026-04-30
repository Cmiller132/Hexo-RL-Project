# Phase 04 Inference Package Split - 2026-04-30

## Scope

The flat inference client/server modules were removed and replaced with
canonical package owners:

```text
Python/src/hexorl/inference/client/
Python/src/hexorl/inference/server/
```

Shared protocol, adapters, queue layout, and telemetry remain at
`hexorl.inference` root.

## Deleted Flat Modules

```text
Python/src/hexorl/inference/client.py
Python/src/hexorl/inference/server.py
Python/src/hexorl/inference/batching.py
Python/src/hexorl/inference/shm_transport.py
```

## New Ownership

- `inference/client/api.py`: public `InferenceClient` methods.
- `inference/client/handshake.py`: manifest loading, request-kind selection, negotiation.
- `inference/client/transport.py`: shared-memory request write, bounded wait, response read, transport state.
- `inference/client/static_response.py`: zero-count response helper.
- `inference/server/process.py`: `InferenceServer` process facade.
- `inference/server/runtime.py`: device/model/compile/weight update setup.
- `inference/server/scheduler.py`: event loop and request-kind dispatch.
- `inference/server/batching.py`: compatible-kind batching/backpressure policy.
- `inference/server/collation.py`: dense/sparse/pair/graph slot collation.
- `inference/server/execution.py`: model forward execution.
- `inference/server/outputs.py`: finite checks, clamping, scalar conversion.
- `inference/server/scatter.py`: dense/sparse/pair/graph response scatter.
- `inference/server/metrics.py`: counters and timing summaries.

## Verification

```text
PYTHONPATH=Python/src python3 -m py_compile $(find Python/src/hexorl/inference -name '*.py' -not -path '*/__pycache__/*') Python/tests/test_inference_server.py Python/tests/inference/*.py
exit=0
```

```text
PYTHONPATH=Python/src ./.venv/bin/python -m pytest -q Python/tests/inference Python/tests/test_inference_server.py
exit=0
33 passed in 5.63s
```

```text
PYTHONPATH=Python/src ./.venv/bin/python -m pytest -q Python/tests/inference Python/tests/test_inference_server.py Python/tests/search/test_policy_provider.py Python/tests/search/test_pair_strategy.py Python/tests/search/test_pair_strategy_selfplay_integration.py Python/tests/selfplay/test_game_runner_interface.py Python/tests/selfplay/test_no_worker_architecture_logic.py
exit=0
63 passed in 5.16s
```

## Import/Deletion Audit

```text
test ! -e Python/src/hexorl/inference/client.py
test ! -e Python/src/hexorl/inference/server.py
test ! -e Python/src/hexorl/inference/batching.py
test ! -e Python/src/hexorl/inference/shm_transport.py
exit=0
```

```text
rg -n "from hexorl\.inference\.(batching|shm_transport)|import hexorl\.inference\.(batching|shm_transport)|hexorl\.inference\.(batching|shm_transport)" Python/src scripts tools -S
exit=1
no matches
```

No skipped, deferred, quarantined, flaky, or manual-only requirement is claimed
complete by this package split.
