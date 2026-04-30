# Phase 05 Import Audit Command Plan

`rg` is preferred but currently unavailable in this environment (`Access is denied`), so use `git grep` fallback and record both attempts.

Required audits:

```text
rg -n "architecture\\.startswith|startswith\\(\"global_|global_graph_enabled|global_xattn|pair_prior_mix|pair_head_present" Python/src/hexorl
rg -n "score_pair|pair_chunk|pair.*forward|policy_pair_first|policy_pair_second|policy_pair_joint|PairPolicyHead" Python/src/hexorl/selfplay Python/src/hexorl/search Python/src/hexorl/inference
rg -n "mcts|MCTS|expand_root|expand_and_backprop|apply_pair_priors" Python/src/hexorl
rg -n "MockMCTSEngine|RealMCTSEngine|_engine\\.MCTSEngine|try_|tokenized|init_root\\(|select_leaves\\(|Result<.*, String>|panic wrapper|unwrap\\(" Python/src/hexorl crates/hexgame-py/src crates/hexgame-core/src/mcts.rs
rg -n "from hexorl\\.engine|import hexorl\\.engine" Python/src/hexorl/search Python/src/hexorl/selfplay Python/src/hexorl/eval
```

Expected final outcomes:

- Runtime search/self-play has no architecture string gates.
- Pair scoring names in runtime are owned by `search/pair_strategy.py` and inference adapters only.
- Rust MCTS API imports/calls are owned by `search/engine_adapter.py` only.
- No production runtime `MockMCTSEngine` or `RealMCTSEngine` remains in worker.
