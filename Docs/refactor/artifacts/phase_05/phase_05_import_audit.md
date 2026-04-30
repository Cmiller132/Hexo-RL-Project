# Phase 05 Import And Code Audit

`rg` status:

```text
rg --version
exit 1
Program 'rg.exe' failed to run: Access is denied
```

Fallback audit tools: `git grep --untracked` and scoped PowerShell `Select-String` where the Windows shell made quoted `git grep -E` pathspecs unreliable.

## Architecture Gates

Command:

```text
Get-ChildItem -Path Python/src/hexorl/selfplay,Python/src/hexorl/search -Recurse -File | Select-String -Pattern 'global_graph_enabled'
```

Exit: `0`; no output.

Command:

```text
Get-ChildItem -Path Python/src/hexorl/selfplay,Python/src/hexorl/search -Recurse -File | Select-String -Pattern 'architecture\.startswith|startswith\(\"global_|pair_head_present'
```

Exit: `0`; no output.

Scoped note: `global_xattn` remains as a registered model kind/spec name. It is not used as a runtime architecture-prefix dispatch gate in search or self-play.

## Pair Enablement

Command:

```text
Get-ChildItem -Path Python/src/hexorl/selfplay,Python/src/hexorl/search -Recurse -File | Select-String -Pattern 'pair_prior_mix'
```

Exit: `0`; no output in runtime search/self-play code.

Negative test coverage exists in `Python/tests/search/test_pair_strategy.py`.

Command:

```text
Get-ChildItem -Path Python/src/hexorl/selfplay,Python/src/hexorl/search,Python/src/hexorl/inference -Recurse -File | Select-String -Pattern 'score_pair|pair_chunk|_score_graph_pair_chunks|_score_crop_pair_chunks'
```

Exit: `0`; no output.

## Rust MCTS Boundary

Command:

```text
Get-ChildItem -Path Python/src/hexorl/search -Recurse -File | Select-String -Pattern 'args\[:|tokenless|compat'
```

Exit: `0`; no output. `EngineAdapter` no longer contains tokenless argument fallback.

Command:

```text
git grep --untracked -n -E "init_root\(|select_leaves\(|expand_root\(|expand_and_backprop|apply_root_pair|PyMCTSEngine|mcts_engine_class" -- Python/src/hexorl/selfplay Python/src/hexorl/search Python/src/hexorl/eval
```

Exit: `0`.

Allowed matches:

- `Python/src/hexorl/search/engine_adapter.py` owns `mcts_engine_class` and backend lifecycle calls.
- `Python/src/hexorl/search/mcts_runner.py` and `expansion.py` call only `EngineAdapter` public methods, not Rust APIs.

No self-play or eval runtime module calls Rust MCTS APIs directly.

Command:

```text
Get-ChildItem -Path Python/src/hexorl -Recurse -File | Select-String -Pattern 'RealMCTSEngine|MockMCTSEngine'
```

Exit: `0`; no output.
