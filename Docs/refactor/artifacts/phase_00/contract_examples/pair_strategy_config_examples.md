# Phase 00 Pair Strategy Config Examples

- Created: `2026-04-30T03:31:49Z`
- Git SHA: `9d7a24ca196e2c3343d34cbd6721ec96bb195d96`

Phase 00 introduced the minimal explicit pair-scoring guard fields, not the final Phase 05 `PairStrategySpec`.

## Accepted Default

```toml
model.architecture = "global_xattn_0"
model.pair_strategy = "none"
model.pair_strategy_max_pairs = 0
```

Expected behavior: self-play reports `pair_strategy=none` and `pair_rows_scored=0` even when pair-capable heads or nonzero `pair_prior_mix` exist.

## Rejected Diagnostic Without Cap

```toml
model.pair_strategy = "diagnostic_full_pair"
model.pair_strategy_max_pairs = 0
```

Expected behavior: config validation rejects the setting before pair scoring can run.

## Required Evidence

- `Python/tests/test_config_and_guardrails.py` covers default `global_xattn`, pair-head presence, nonzero `pair_prior_mix`, and diagnostic cap requirements.
- `commands/pytest_full_python_tests.txt` records `258 passed`.
- `commands/phase00_selfplay_smoke.txt` records `pair_strategy=none` and `pair_rows_scored=0`.
