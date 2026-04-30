# Phase 07 Test Output

```text
python -m pytest Python\tests\replay\test_phase07_codec_storage_projector.py Python\tests\replay\test_phase07_import_audit.py -q
..........                                                               [100%]
10 passed in 1.56s
```

```text
python -m pytest Python\tests\selfplay\test_record_writer.py Python\tests\test_config_and_guardrails.py -q
.................................                                        [100%]
33 passed, 1 warning in 1.23s
```

```text
python -m pytest Python\tests\selfplay Python\tests\replay Python\tests\search\test_pair_strategy_selfplay_integration.py Python\tests\test_production_smoke.py -q
..........................                                               [100%]
26 passed in 0.83s
```

```text
python -m pytest Python\tests\search Python\tests\test_config_and_guardrails.py -q
........................................................................ [ 92%]
......                                                                   [100%]
78 passed, 1 warning in 1.48s
```
