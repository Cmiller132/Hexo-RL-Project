# Baseline Command Report

Captured on 2026-05-05 in `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project`.

## Environment Notes

- `python` is not on `PATH`.
- `pytest` is not on `PATH`.
- `python3` is available.
- `_engine` is not importable in this environment:

```text
PYTHONPATH=Python/src python3 -c "import importlib.util; print('engine-spec', importlib.util.find_spec('_engine'))"
exit 0
engine-spec None
```

The missing Rust extension is the dominant expected WIP failure for graph and
replay tests that require the production tactical oracle.

## Commands

| Command | Exit | Result |
|---|---:|---|
| `PYTHONPATH=Python/src python3 -c "import hexorl; from hexorl.config import Config; from hexorl.model.network import build_model_from_config; from hexorl.model.global_graph import GlobalHexGraphNet; print('import-ok', Config().model.architecture, len(GlobalHexGraphNet.ARCHITECTURES))"` | 0 | `import-ok cnn 7` |
| `PYTHONPATH=Python/src python3 -m pytest -q Python/tests/test_config_and_guardrails.py` | 0 | `31 passed in 1.46s` |
| `PYTHONPATH=Python/src python3 -m pytest -q Python/tests/test_engine_smoke.py Python/tests/test_engine_invariants.py` | 5 | `2 skipped in 0.11s`; pytest exit 5 because all selected tests were skipped due missing `_engine` |
| `PYTHONPATH=Python/src python3 -m pytest -q Python/tests/test_tactical_oracle.py` | 0 | `5 passed, 3 skipped in 0.08s` |
| `PYTHONPATH=Python/src python3 -m pytest -q Python/tests/test_config_and_guardrails.py Python/tests/test_training_data_pipeline.py` | 1 | `7 failed, 97 passed, 5 skipped in 9.52s` |
| `PYTHONPATH=Python/src python3 -m pytest -q Python/tests/test_global_graph_contract.py` | 1 | `30 failed, 5 passed in 5.70s` |
| `PYTHONPATH=Python/src python3 -m pytest -q Python/tests/test_inference_server.py` | 1 | `1 failed, 5 passed, 1 skipped in 6.33s` |

## Known WIP Failures

All observed failures in the focused model/replay/inference tests are caused by
the production tactical oracle requiring `_engine` while `_engine` is not
available in this shell.

Representative exception:

```text
RuntimeError: engine tactical oracle is required but the Rust engine is unavailable
```

Affected areas:

- `ReplayDataset` candidate and graph batch creation when sparse/graph policy is
  enabled.
- `build_graph_batch_from_history` in most graph contract tests.
- `InferenceServer` graph forward test that builds a graph batch.

## Baseline Trust Implication

Passing tests can be used as local guardrail evidence. Failing graph/replay tests
are not regressions in this Stage 1 run; they require either a built Rust
extension or a test harness that explicitly opts into a Python tactical fallback.
Stage 2 and Stage 3 should not claim graph contract closure from these failing
tests until the engine dependency is resolved or the tests are rewritten around
contract-local fixtures.
