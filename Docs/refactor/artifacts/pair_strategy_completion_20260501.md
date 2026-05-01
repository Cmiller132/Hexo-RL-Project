# Pair Strategy Registry Completion

Date: 2026-05-01

## Goal

Replace pair-strategy string dispatch sprawl with a single descriptor registry, reconcile pair generation `mode` naming, add tests/gates, and preserve current strategy behavior.

## Success Criteria Status

| Item | Status | Evidence |
| --- | --- | --- |
| One registry owns pair strategy descriptors | Complete | `Python/src/hexorl/contracts/pair_strategy/registry.py`, `descriptors.py`, `__init__.py` |
| `GameRunner` pair-strategy ladders removed | Complete | `_pair_strategy_spec()` and module helper call `PAIR_STRATEGY_REGISTRY`; `_pair_table_for_search()` calls descriptor table strategy |
| `search/pair_strategy.py` ladder removed | Complete | `PairStrategySpec` moved to registry; search validates through registry |
| `PairStrategy.mode` renamed to `generation_mode` | Complete | `Python/src/hexorl/contracts/pairs.py` and all production/tests call sites updated |
| Literal validation derives from `PairGenerationMode` | Complete | `_VALID_MODES = frozenset(get_args(PairGenerationMode))` |
| Alias sets live on descriptors | Complete | aliases only in `descriptors.py` |
| CI grep gate added | Complete | `Python/tests/contracts/test_pair_strategy_registry.py::test_pair_strategy_grep_gate_has_no_runtime_ladders` |
| Positive fake-strategy extension test | Complete | `Python/tests/selfplay/test_pair_strategy_registry_extension.py` |
| Requested broad pytest command | Blocked by unrelated failure | 123 passed, 1 failed in replay model fixture; details below |

## Registered Strategies

```text
diagnostic_full_root: aliases=['diagnostic_full_pair'] generation_mode=full_capped root=True leaf=False diagnostic=True cap_field=max_full_pair_rows chunk_cap=4096 allow_full=True
none: aliases=[] generation_mode=none root=False leaf=False diagnostic=False cap_field=max_root_pair_rows chunk_cap=0 allow_full=False
tactical_only: aliases=['tactical'] generation_mode=capped_fill root=True leaf=False diagnostic=False cap_field=max_root_pair_rows chunk_cap=512 allow_full=False
two_stage_root_only: aliases=['two_stage_root'] generation_mode=capped_fill root=True leaf=False diagnostic=False cap_field=max_root_pair_rows chunk_cap=4096 allow_full=False
```

## Worked Extension Example

`Python/tests/selfplay/test_pair_strategy_registry_extension.py` creates a throwaway registry with:

```python
PairStrategyDescriptor(
    name="experimental_root",
    aliases=frozenset({"experimental"}),
    generation_mode="capped_fill",
    root_enabled=True,
    leaf_enabled=False,
    diagnostic=False,
    max_pair_rows_field="max_root_pair_rows",
    chunk_cap=4,
    allow_full=False,
    requires_pair_head=True,
)
```

The test builds a `GameRunner` with `pair_strategy_name="experimental"` and asserts the runner resolves `experimental_root` and builds its spec without edits to `game_runner.py`, `search/pair_strategy.py`, or `contracts/pairs.py`.

## Runtime Consumers Changed

- `Python/src/hexorl/selfplay/game_runner.py`
  - Resolves pair strategy once via registry.
  - Uses descriptor-built `PairStrategySpec`.
  - Uses descriptor-built pair table generation request.
  - Module helper `_pair_strategy_spec_from_config()` calls registry.
- `Python/src/hexorl/search/pair_strategy.py`
  - Imports `PairStrategySpec` from registry.
  - Uses registry validation in `create_pair_strategy()`.
  - Uses generic spec cap property instead of strategy-name checks.
- `Python/src/hexorl/config/schema.py`
  - Validates and canonicalizes `model.pair_strategy` through registry.
- `Python/src/hexorl/tuning/*`
  - Recipes, dry-run validation, and family spaces query registry.
- `Python/src/hexorl/train/adapters.py`
  - Uses registry descriptor to detect enabled pair strategy.
- `Python/src/hexorl/models/checkpoint.py`
  - Stores canonical `pair_strategy_used`.
- `Python/src/hexorl/replay/projector.py`, `graph/semantic_builder.py`, `eval/position_services.py`
  - Use `PairStrategy(generation_mode=...)`.

## Legacy Paths Deleted Or Quarantined

- Deleted duplicate pair-strategy spec ladders from `GameRunner`.
- Deleted strategy-name validation ladder from `search/pair_strategy.py`.
- Deleted inline pair strategy allowed sets from config and recipes.
- Deleted `PairStrategy.mode` field. No `mode` property shim was added.
- Inline pair generation mode sets in `contracts/pairs.py` were replaced by `get_args(PairGenerationMode)`.

## Verification Commands

```text
PYTHONPATH=Python/src ./.venv/bin/python -m py_compile \
  Python/src/hexorl/contracts/pair_strategy/registry.py \
  Python/src/hexorl/contracts/pair_strategy/descriptors.py \
  Python/src/hexorl/contracts/pairs.py \
  Python/src/hexorl/search/pair_strategy.py \
  Python/src/hexorl/selfplay/game_runner.py \
  Python/src/hexorl/config/schema.py \
  Python/src/hexorl/tuning/recipes.py \
  Python/src/hexorl/tuning/validation.py \
  Python/src/hexorl/tuning/family_spaces.py
```

Exit status: 0.

```text
PYTHONPATH=Python/src ./.venv/bin/python -m pytest -q \
  Python/tests/contracts/test_pair_strategy_registry.py \
  Python/tests/contracts/test_phase02_builders.py \
  Python/tests/search/test_pair_strategy.py \
  Python/tests/search/test_pair_strategy_selfplay_integration.py \
  Python/tests/selfplay/test_pair_strategy_registry_extension.py
```

Exit status: 0. Summary: 29 passed in 0.85s.

```text
! rg -n 'pair_strategy\s*(==|in\s*\{)' \
  Python/src/hexorl/selfplay Python/src/hexorl/search Python/src/hexorl/contracts \
  -g '*.py' | rg -v 'pair_strategy/registry.py|pair_strategy/descriptors.py'
```

Exit status: 0. No matches.

```text
PYTHONPATH=Python/src ./.venv/bin/python -m pytest -q \
  Python/tests/contracts/test_phase01_contract_import_boundaries.py \
  Python/tests/contracts/test_pair_strategy_registry.py
```

Exit status: 0. Summary: 5 passed in 0.15s.

Requested command:

```text
PYTHONPATH=Python/src ./.venv/bin/python -m pytest -q \
  Python/tests/selfplay Python/tests/search Python/tests/contracts \
  Python/tests/replay Python/tests/train Python/tests/eval
```

Exit status: 1. Summary: 123 passed, 1 failed in 2.85s.

Remaining failure:

```text
Python/tests/replay/test_phase07_codec_storage_projector.py::test_sample_to_loss_uses_projected_replay_batch
ValueError: DenseCnnParams does not accept model fields: ['candidate_budget', 'sparse_policy']
```

This failure is outside the pair-strategy refactor path. The test mutates a default dense `Config()` after validation into a sparse-policy shape, then `build_model()` resolves the still-dense architecture and rejects graph-hybrid-only fields. No pair-strategy code is on this traceback.

## Diff Stats

Tracked modified files:

```text
Python/src/hexorl/config/schema.py                 +9  -24
Python/src/hexorl/contracts/pairs.py               +14 -13
Python/src/hexorl/eval/position_services.py        +1  -1
Python/src/hexorl/graph/semantic_builder.py        +1  -1
Python/src/hexorl/models/checkpoint.py             +2  -1
Python/src/hexorl/replay/projector.py              +2  -2
Python/src/hexorl/search/pair_strategy.py          +12 -52
Python/src/hexorl/selfplay/game_runner.py          +14 -78
Python/src/hexorl/train/adapters.py                +3  -1
Python/src/hexorl/tuning/family_spaces.py          +6  -1
Python/src/hexorl/tuning/recipes.py                +4  -4
Python/src/hexorl/tuning/validation.py             +8  -2
Python/tests/contracts/test_phase02_builders.py    +29 -6
Python/tests/search/test_global_graph_pair_contracts.py +4 -3
Python/tests/search/test_pair_strategy.py          +23 -22
Python/tests/search/test_pair_strategy_selfplay_integration.py +5 -9
Python/tests/selfplay/conftest.py                  +6  -8
Python/tests/test_config_and_guardrails.py         +19 -3
Python/tests/test_global_graph_contract.py         +1  -1
```

New files:

```text
Python/src/hexorl/contracts/pair_strategy/__init__.py              28 lines
Python/src/hexorl/contracts/pair_strategy/registry.py             231 lines
Python/src/hexorl/contracts/pair_strategy/descriptors.py           79 lines
Python/tests/contracts/test_pair_strategy_registry.py              59 lines
Python/tests/selfplay/test_pair_strategy_registry_extension.py     34 lines
Docs/refactor/artifacts/pair_strategy_investigation.md            221 lines
```

## Performance And Utilization Evidence

No pair generation/scoring algorithm changed. The hot path now resolves the descriptor once at `GameRunner` construction and reuses it for spec/table-strategy construction. No runtime benchmark was run for this registry-only dispatch refactor.

## Contract Examples And Docs

- Descriptor unit tests document the built-in descriptor behavior.
- Throwaway registry test documents the extension path.
- This artifact lists registered strategies and aliases.

## Known Blockers

- The requested broad pytest command is not fully green because of the replay/model-family failure shown above.
- No skipped, deferred, or manual-only requirement is being claimed complete.
- No V2 matrix row was marked complete in this change.
