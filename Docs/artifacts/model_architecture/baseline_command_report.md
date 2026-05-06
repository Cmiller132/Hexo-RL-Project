# Baseline Command Report

Captured on 2026-05-06 in `D:\Hexo\Hexo-RL-Project` after fast-forwarding
`main` to `origin/main` and reapplying the local autotune/runtime work.

Stage 1 is an inventory and design stage. These commands establish the local
proof harness before runtime model modularity work begins. This local refresh
supersedes the upstream remote-machine capture where the Rust `_engine`
extension was unavailable.

## Environment Notes

- Platform: Windows PowerShell workspace.
- Python command: `python`.
- `PYTHONPATH`: `Python/src`.
- Rust tactical extension: `_engine` importable locally.
- Expected warning: Triton reports CUDA auto-detection trouble in some pytest
  sessions; this did not block the focused Stage 1 checks.

## Import Smoke

```text
python 3.14.0
engine-spec ModuleSpec(...)
import-ok cnn 7
```

## Commands

| Command | Exit | Result |
|---|---:|---|
| `$env:PYTHONPATH='Python/src'; python -c "import sys, importlib.util; print('python', sys.version.split()[0]); print('engine-spec', importlib.util.find_spec('_engine')); from hexorl.config import Config; from hexorl.model.network import build_model_from_config; from hexorl.model.global_graph import GlobalHexGraphNet; print('import-ok', Config().model.architecture, len(GlobalHexGraphNet.ARCHITECTURES))"` | 0 | `_engine` importable; `import-ok cnn 7` |
| `$env:PYTHONPATH='Python/src'; python -m pytest -q Python/tests/test_config_and_guardrails.py Python/tests/test_training_data_pipeline.py` | 0 | `116 passed, 1 warning in 5.48s` |
| `$env:PYTHONPATH='Python/src'; python -m pytest -q Python/tests/test_tactical_oracle.py Python/tests/test_engine_smoke.py Python/tests/test_engine_invariants.py` | 0 | `35 passed in 2.96s` |
| `$env:PYTHONPATH='Python/src'; python -m pytest -q Python/tests/test_inference_server.py Python/tests/test_global_graph_contract.py` | 0 | `45 passed, 1 warning in 21.89s` |

## Harness Adjustment

The replay-pipeline golden test for first-placement graph pair rows was updated
to match the current performance contract: graph replay emits a budgeted pair
row table by default (`256` rows), while still preserving first-placement pair
target mass and suppressing second-placement target mass when no first move is
known.

This keeps Stage 1 honest about the production training contract after the pair
row materialization bottleneck fix. Tests that require exhaustive pair rows must
request that behavior explicitly instead of assuming it is the replay default.

## Stage 1 Interpretation

Stage 1 remains an inventory and design closure, not runtime cutover. The local
checks above prove that the current codebase can import the legacy model
authority, build config defaults, use the Rust tactical extension, and exercise
the focused graph, replay, inference, tactical, engine, and configuration
guardrails. Stage 2 still needs its own contract tests for registry, row
identity, output contracts, target contracts, loss plans, and inference protocol
boundaries.
