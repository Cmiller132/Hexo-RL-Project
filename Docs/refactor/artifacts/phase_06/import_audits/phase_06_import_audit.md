# Phase 06 Import And Code Audits

`rg.exe` was unavailable with `Access is denied`, so the exact audit patterns were run with PowerShell `Select-String`.

```text
Select-String -Path Python\src\hexorl\selfplay\worker.py -Pattern 'architecture|startswith\("global_|pair_prior_mix|pair_head|GlobalHexGraphNet|build_model_from_config'
exit 0
output: no matches
```

```text
Select-String -Path Python\src\hexorl\selfplay\worker.py -Pattern 'Candidate|PairAction|PAIR_ACTION|graph_token|graph_relation|chunk|MCTS|prior'
exit 0
output: no matches
```

```text
Select-String -Path Python\src\hexorl\selfplay\worker.py -Pattern 'Replay|record|writer|json|np.save|open\('
exit 0
output: no matches
```

```text
Get-ChildItem -Path Python\src\hexorl\selfplay,Python\src\hexorl\search -Recurse -File | Select-String -Pattern 'HAS_ENGINE|MockMCTSEngine|RealMCTSEngine|_score_graph_pair_chunks|_score_crop_pair_chunks|_align_global_logits_to_rust_legal|_engine\.MCTSEngine|client\.submit_|process_game_record|uniform.*fallback'
exit 0
output: no matches
```

```text
Get-ChildItem over existing source paths from Python\src\hexorl\search, Python\src\hexorl\contracts, Python\src\hexorl\graph, Python\src\hexorl\replay | Select-String -Pattern 'hexorl\.selfplay\.worker'
exit 0
output: no matches. `Python\src\hexorl\replay` is not present in this repository at Phase 06, so the fallback audit filtered missing paths instead of adding a runtime package outside this phase's ownership.
```
