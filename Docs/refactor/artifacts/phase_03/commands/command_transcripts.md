# Phase 03 Command Transcripts

## Git and Setup

```text
git status --short --untracked-files=all
exit=0
Observed Phase 03 working tree edits only; Phase 02 was already committed at 5638e8b.
```

```text
git switch -c codex/phase-03-model-registry-specs
exit=0
Switched to a new branch 'codex/phase-03-model-registry-specs'
```

```text
rg "hexorl\.model|from hexorl import model|Python/src/hexorl/model" Python/src Python/tests
exit=1
Program 'rg.exe' failed to run: Access is denied
Fallback used: git grep.
```

## Focused Tests

```text
python -m compileall -q Python/src/hexorl
exit=0
```

```text
python -m pytest Python/tests/models/test_phase03_model_registry.py Python/tests/train/test_phase03_train_adapter_checkpoint.py -q
exit=0
30 passed, 1 warning in 2.82s
```

```text
python -m pytest Python/tests/test_inference_server.py -q
exit=0
7 passed in 14.84s
```

```text
python -m pytest Python/tests/test_config_and_guardrails.py Python/tests/test_global_graph_contract.py Python/tests/test_production_smoke.py Python/tests/test_training_data_pipeline.py -q
exit=0
139 passed, 1 warning in 53.32s
```

```text
python Docs\refactor\artifacts\phase_03\commands\mcts_round_trip_probe.py
exit=0
Server start, client submit, Rust MCTS backprop, results, and cleanup all completed.
```

```text
git diff --check
exit=0
```

```text
git grep --untracked -n "hexorl\.model\.\|from hexorl import model\|Python/src/hexorl/model" -- Python/src Python/tests
exit=1
no matches
```

```text
git grep --untracked -n "architecture\.startswith\|architecture ==\|isinstance(.*GlobalHexGraphNet\|build_model_from_config" -- Python/src/hexorl
exit=1
no matches
```

```text
git grep --untracked -n "_orig_mod\|strip.*prefix\|state_dict.*cleanup\|strict=False" -- Python/src/hexorl
exit=1
no matches
```

```text
git grep --untracked -n "pair_prior_mix\|pair_head_present" -- Python/src/hexorl/models Python/src/hexorl/train
exit=1
no matches
```

## Historical Timeout During Debug

```text
python -m pytest Python/tests/test_config_and_guardrails.py Python/tests/test_global_graph_contract.py Python/tests/test_inference_server.py Python/tests/test_production_smoke.py Python/tests/test_training_data_pipeline.py -q
exit=124
Timed out after 604027 ms. Left orphan pytest/server processes, which were terminated after inspection.
```

```text
python -m pytest Python/tests/test_inference_server.py -q
exit=124
Timed out after 244041 ms.
```

```text
python -m pytest Python/tests/test_inference_server.py::TestInferenceServerWithEngine::test_mcts_round_trip -q -s
exit=124
Timed out after 124032 ms. Root cause was stale Rust MCTS test API and missing cleanup on assertion failure. Resolved by updating the test to the current Rust API and adding bounded cleanup; current `Python/tests/test_inference_server.py` passes.
```

## Process Cleanup

```text
Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | Select-Object ProcessId,CommandLine
exit=0
Identified orphan pytest/server children from timed-out inference runs.
```

```text
Stop-Process -Id <pytest parent>,<spawn child> -Force
exit=0
Terminated only pytest/server children from the timed-out commands. The dashboard LAN proxy process was left running.
```
