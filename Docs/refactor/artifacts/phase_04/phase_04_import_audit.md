# Phase 04 Import And Code-Search Audit

`rg` was attempted but the local `rg.exe` fails with `Access is denied`, so `git grep` was used as the deterministic fallback.

Commands:

```text
rg --version
Exit: 1
Program 'rg.exe' failed to run: Access is denied

git grep -n -E 'req_mode|submit_.*global|submit_.*dense|submit_.*sparse|submit_.*pair|architecture\.startswith|startswith\("global_"|pair_head_present|pair_prior_mix' -- Python/src/hexorl/inference
Exit: 1
No matches

git grep -n -E 'submit_sparse|submit_sparse_pair|submit_graph|submit_regret_rank' -- Python/src/hexorl Python/tests
Exit: 1
No matches
```

Scoped note: `pair_prior_mix` remains in `Python/src/hexorl/selfplay/worker.py` as self-play pair-prior blending configuration. Phase 04 removed inference-boundary implicit pair scoring and mode dispatch; pair-strategy semantics are Phase 05 scope.
