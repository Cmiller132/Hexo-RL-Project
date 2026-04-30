# Phase 05 Interface Freeze Notes

Frozen target modules:

- `Python/src/hexorl/search/context.py`
- `Python/src/hexorl/search/policy_provider.py`
- `Python/src/hexorl/search/pair_strategy.py`
- `Python/src/hexorl/search/priors.py`
- `Python/src/hexorl/search/expansion.py`
- `Python/src/hexorl/search/mcts_runner.py`
- `Python/src/hexorl/search/engine_adapter.py`

Frozen public contracts:

- `SearchContext`: position/history identity, phase, legal table, root/batch generation, optional candidate/graph/pair contract handles, model/spec identity, recipe/search/pair identity, trace id.
- `SearchEvaluation`: context identity, trace id, value, legal row ids, dense indices, row-mapped priors, source labels, provider identity, model identity, inference protocol identity, warnings, timings.
- `PairEvaluation`: strategy name, phase/scope, pair action table identity, pair rows, row-mapped pair priors, source labels, known-first, caps, counts, warnings, timings.
- `PolicyProvider`: `evaluate_root(context)` and `evaluate_leaves(contexts)`.
- `PairStrategy`: `score_root(context, base_eval)` and `score_leaves(contexts, base_evals)`.
- `EngineAdapter`: only Python boundary for Rust MCTS root expansion, pair-prior application, leaf selection, backprop, sample, and re-root.

Frozen deletion targets:

- Worker-owned MCTS wrappers.
- Worker-owned pair chunk helpers.
- Direct MCTS prior wiring in worker.
- Runtime pair enablement from architecture prefix, head presence, or `pair_prior_mix`.
