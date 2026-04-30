# Phase 05 Command Transcripts

## Final Verification

```text
python -m maturin develop --manifest-path crates\hexgame-py\Cargo.toml
exit 0
Built and installed hexgame-py-0.2.0
```

```text
Copy-Item -LiteralPath target\debug\_engine.dll -Destination C:\Users\epicm\AppData\Roaming\Python\Python314\site-packages\_engine\_engine.pyd -Force
exit 0
```

```text
python -m pytest Python\tests\search -q
exit 0
45 passed in 0.09s
```

```text
python -m pytest Python\tests\test_config_and_guardrails.py Python\tests\test_engine_smoke.py Python\tests\test_production_smoke.py -q
exit 0
46 passed, 1 warning in 3.00s
```

```text
python -m pytest Python\tests\search Python\tests\test_config_and_guardrails.py Python\tests\test_engine_smoke.py Python\tests\test_production_smoke.py -q
exit 0
91 passed, 1 warning in 1.80s
```

```text
python -m compileall Python\src\hexorl
exit 0
```

```text
cargo test -p hexgame-core mcts_stale -- --nocapture
exit 0
2 passed; 0 failed
```

## Tool Fallback

```text
rg --version
exit 1
Program 'rg.exe' failed to run: Access is denied
```

`git grep --untracked` was used for phase audits.

## Performance Artifact

```text
python <inline phase_05_mcts_performance_profile generator>
exit 0
{"artifact": "Docs\\refactor\\artifacts\\phase_05\\phase_05_mcts_performance_profile.json", "leaf_batch_count": 8, "positions_per_sec_proxy": 154.51890872087526}
```

## Debug And Error Artifacts

```text
python <inline phase_05_mcts_trace/debug generator>
exit 0
{"debug_trace_id": "220b462f84fd4fa48429fc98f546c9c7", "trace_rows": 12}
```

```text
python <inline phase_05_mcts_error_trace_samples generator>
exit 0
{"artifact": "Docs/refactor/artifacts/phase_05/phase_05_mcts_error_trace_samples.json", "cases": 5}
```
