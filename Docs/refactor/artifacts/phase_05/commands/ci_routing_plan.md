# Phase 05 CI Routing Plan

Focused local commands:

```text
python -m pytest Python\tests\search -q
python -m pytest Python\tests\test_inference_server.py -q
python -m pytest Python\tests\test_training_data_pipeline.py -q
python -m compileall Python\src\hexorl
```

Phase-closing commands:

```text
python -m pytest Python\tests\search Python\tests\inference Python\tests\test_inference_server.py -q
python -m pytest Python\tests\test_production_smoke.py Python\tests\test_engine_smoke.py -q
```

No skipped, xfailed, flaky-only, or manual-only check may close a Phase 05 invariant.
