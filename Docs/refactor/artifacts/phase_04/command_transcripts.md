# Phase 04 Command Transcripts

```text
python -m pytest Python\tests\inference Python\tests\test_inference_server.py -q
Exit: 0
21 passed in 14.62s
```

```text
python -m pytest Python\tests\inference -q
Exit: 0
14 passed in 1.30s
```

```text
python -m pytest Python\tests\test_inference_server.py -q
Exit: 0
7 passed in 14.50s
```

```text
python -m compileall Python\src\hexorl
Exit: 0
```

```text
rg --version
Exit: 1
Program 'rg.exe' failed to run: Access is denied
```

```text
git grep -n -E 'submit_sparse|submit_sparse_pair|submit_graph|submit_regret_rank' -- Python/src/hexorl Python/tests
Exit: 1
No matches
```

```text
git grep -n -E 'req_mode|submit_.*global|submit_.*dense|submit_.*sparse|submit_.*pair|architecture\.startswith|startswith\("global_"|pair_head_present|pair_prior_mix' -- Python/src/hexorl/inference
Exit: 1
No matches
```
