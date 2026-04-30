# Command Transcripts

```text
git status --short
exit 0
output: clean at phase entry
```

```text
rg --version
exit 1
output: Access is denied. Fallback audits used PowerShell Select-String.
```

```text
python -m pytest Python\tests\selfplay Python\tests\search\test_pair_strategy_selfplay_integration.py -q
exit 0
14 passed in 0.06s
```

```text
python -m pytest Python\tests\inference Python\tests\replay -q
exit 0
15 passed in 1.40s
```

```text
python -m pytest Python\tests\selfplay Python\tests\search\test_pair_strategy_selfplay_integration.py Python\tests\inference Python\tests\replay -q
exit 0
29 passed in 0.11s
```

```text
python -m pytest Python\tests\selfplay Python\tests\search Python\tests\test_config_and_guardrails.py -q
exit 0
90 passed, 1 warning in 1.11s
```

```text
python -m compileall Python\src\hexorl
exit 0
```
