# Pair Strategy Dispatch Investigation

Phase: 0  
Scope: investigation only; no implementation refactor performed.

## Assignment Frame

Goal: identify every current string-dispatch site for pair strategies and pair generation modes before replacing them with one registry.

Success criteria:
- List every production branch on `pair_strategy`, `PairStrategySpec.name`, `PairStrategy.mode`, or `PairActionTable.generation_mode`.
- List every accepted canonical pair strategy and alias.
- Trace every `mode` and `generation_mode` read/write and identify the downstream source of truth.
- Inventory current tests by strategy and generation mode.
- Identify uncalled, untested, or recipe-unreachable strategies.

Constraints:
- Stop after Phase 0 and wait for confirmation.
- Do not change runtime code.
- Preserve unrelated worktree changes.

Required evidence:
- `git status --short`
- `rg` inventories for `pair_strategy`, `generation_mode`, `PairStrategy(`, `PairActionTable(`, and known strategy strings.
- Direct line reads with `nl -ba` for production call sites and tests.

Stop rules:
- If a strategy has no caller and no test, report it as a deletion candidate and wait.
- If `mode`/`generation_mode` touches serialized replay/checkpoint schema, report the schema impact before renaming.

## Commands Used

- `git status --short`
- `rg -n "if .*pair_strategy|elif .*pair_strategy|pair_strategy not in|pair_strategy in|pair_strategy !=|pair_strategy ==|valid_pair_strategies|pair_strategy_choices" Python/src/hexorl Python/tests -g '*.py'`
- `rg -n "pair_strategy\s*(==|in\s*\{)|\.pair_strategy|pair_strategy_name|PairStrategy\(|PairActionTable\(|generation_mode|\.mode\b|mode=" Python/src/hexorl Python/tests Docs -g '*.py' -g '*.md' -g '*.json'`
- `rg -n "two_stage_root_only|two_stage_root|tactical_only|tactical\b|diagnostic_full_pair|diagnostic_full_root|pair_strategy\s*[:=]|pair_strategy_name" Python/src Python/tests configs Docs -g '*.py' -g '*.yaml' -g '*.yml' -g '*.json' -g '*.md'`
- `rg -n "mode\s*==\s*['\"](none|selected|capped_fill|full_capped)|generation_mode\s*==|generation_mode\s+in|mode\s+in\s*\{" Python/src/hexorl Python/tests -g '*.py'`

## Worktree Note

`git status --short` showed many pre-existing modified/deleted/untracked files, including `Python/src/hexorl/selfplay/game_runner.py`, `Python/src/hexorl/search/pair_strategy.py`, and several tests. This report describes the current worktree state and does not claim those changes were made during Phase 0.

## Phase 0 Checklist

- [x] Read refactor source of truth relevant to PairStrategy: `Docs/refactor/README.md`, `PHASED_IMPLEMENTATION_PLAN.md`, `EXECUTION_QUALITY_GUARDRAILS.md`, `PHASE_CHECKLIST.md`, and `phases/PHASE_05.md`.
- [x] Check git status before touching files.
- [x] Locate production branches on pair strategy names and pair generation modes.
- [x] Locate tests and fixtures that exercise strategies and generation modes.
- [x] Trace `mode` and `generation_mode` reads/writes through contracts, graph projection, replay, self-play, training, and checkpoint manifest.
- [x] Identify aliases and recipe/config reachability.
- [x] Write this report and stop.

## Canonical Pair Strategies And Aliases

Current canonical search strategy names are declared in `Python/src/hexorl/search/pair_strategy.py:19`:

| Canonical strategy | Aliases accepted anywhere | Accepting sites |
| --- | --- | --- |
| `none` | none observed | `PairStrategySpec`, config, recipes, family spaces, `GameRunner` |
| `two_stage_root_only` | `two_stage_root` | `GameRunner._pair_strategy_spec`, `GameRunner._pair_table_for_search`, module `_pair_strategy_spec_from_config` |
| `tactical_only` | `tactical` | `GameRunner._pair_strategy_spec`, `GameRunner._pair_table_for_search`, module `_pair_strategy_spec_from_config` |
| `diagnostic_full_root` | `diagnostic_full_pair` | `GameRunner._pair_strategy_spec`, `GameRunner._pair_table_for_search`, module `_pair_strategy_spec_from_config`; config/recipes accept only `diagnostic_full_pair` |

Important mismatch: `Python/src/hexorl/config/schema.py:105` and `Python/src/hexorl/tuning/recipes.py:54` allow only `{"none", "diagnostic_full_pair"}`. The search layer accepts `diagnostic_full_root` as canonical, while the public config/recipe layer currently exposes `diagnostic_full_pair`.

## Branch Sites

### Pair Strategy Dispatch

| File:lines | Branches on | Produces | Handles |
| --- | --- | --- | --- |
| `Python/src/hexorl/config/schema.py:104-126` | `self.pair_strategy` | Validates model config and cap/mix requirements. | Accepts `none`, `diagnostic_full_pair`; rejects all others. |
| `Python/src/hexorl/config/schema.py:249-258` | `self.model.pair_strategy != "none"` | Requires a pair-capable head for non-none config. | Non-`none` as one bucket; current earlier validator limits that to `diagnostic_full_pair`. |
| `Python/src/hexorl/tuning/recipes.py:54-63` | `self.pair_strategy` | Validates recipe strategy and zero cap for `none`. | Accepts `none`, `diagnostic_full_pair`; rejects all others. |
| `Python/src/hexorl/tuning/validation.py:26-29` | `recipe.pair_strategy == "diagnostic_full_pair"` | Dry-run validation result for pair cap. | Special-cases `diagnostic_full_pair`; all other valid recipe strategies fall through to OK. |
| `Python/src/hexorl/tuning/family_spaces.py:72` | family booleans, emits strategy tuple | Recipe search-space options. | Emits `("none", "diagnostic_full_pair")` for global/graph-hybrid, otherwise `("none",)`. |
| `Python/src/hexorl/train/adapters.py:180-181` | `cfg.model.pair_strategy != "none"` | Rejects opening-position pair loss when pair strategy is non-none and no pair rows exist. | Non-`none` as one bucket. This is downstream training behavior, not strategy construction. |
| `Python/src/hexorl/selfplay/game_runner.py:381-383` | `self.pair_strategy != PAIR_STRATEGY_NONE` | Sets `pair_policy_enabled`. | Non-`none` as one bucket. |
| `Python/src/hexorl/selfplay/game_runner.py:433-461` | `self.pair_strategy` | Builds `PairStrategySpec`. | `none`; `two_stage_root_only`/`two_stage_root`; `tactical_only`/`tactical`; `diagnostic_full_pair`/`diagnostic_full_root`. |
| `Python/src/hexorl/selfplay/game_runner.py:523-540` | `self.pair_strategy` | Builds pair table generation request (`PairTableStrategy`) for non-global root pair rows. | `two_stage_root_only`/`two_stage_root`/`tactical_only`/`tactical` -> `capped_fill`; `diagnostic_full_pair`/`diagnostic_full_root` -> `full_capped`. |
| `Python/src/hexorl/selfplay/game_runner.py:569-575` | `self.pair_policy_enabled` derived from strategy string | Global graph root graph-building includes/excludes pair rows and cap. | Non-`none` as one bucket. |
| `Python/src/hexorl/selfplay/game_runner.py:598-599` | `self.pair_policy_enabled` derived from strategy string | Non-global root pair table is built only when enabled. | Non-`none` as one bucket. |
| `Python/src/hexorl/selfplay/game_runner.py:1545-1575` | local `pair_strategy` from config | Module-level duplicate of `_pair_strategy_spec`. | Same as `433-461`. |
| `Python/src/hexorl/search/pair_strategy.py:37-62` | `PairStrategySpec.name` | Validates strategy-specific caps/scope/diagnostic invariants. | `none`, `two_stage_root_only`, `tactical_only`, `diagnostic_full_root`. |
| `Python/src/hexorl/search/pair_strategy.py:197-200` | `self.spec.name == "diagnostic_full_root"` | Chooses root cap field. | `diagnostic_full_root` uses `max_full_pair_rows`; all other explicit strategies use `max_root_pair_rows`. |
| `Python/src/hexorl/search/pair_strategy.py:240-244` | `spec.name == "none"` | Selects `NoPairStrategy` vs `ExplicitPairStrategy`. | `none` vs all explicit strategies. |

### Pair Generation Mode Dispatch

| File:lines | Branches on | Produces | Handles |
| --- | --- | --- | --- |
| `Python/src/hexorl/contracts/pairs.py:33-43` | `PairStrategy.mode` | Validates generation request and full-generation permission. | `none`, `selected`, `capped_fill`, `full_capped`; inline set duplicates `PairGenerationMode`. |
| `Python/src/hexorl/contracts/pairs.py:79-82` | `PairActionTable.phase`, `PairActionTable.generation_mode` | Validates output table phase and generation mode. | `generation_mode` inline set duplicates `PairGenerationMode`. |
| `Python/src/hexorl/contracts/pairs.py:187-188` | `strategy.mode == "none"` | Returns empty `PairActionTable` with `generation_mode="none"`. | `none`. |
| `Python/src/hexorl/contracts/pairs.py:230-242` | `strategy.mode` | Selects full, selected-only, or capped-fill pair row materialization. | `full_capped`, `selected`, all other non-none modes as capped-fill. |
| `Python/src/hexorl/graph/semantic_builder.py:679-692` | `allow_pair_truncation` chooses mode string | Constructs `PairStrategy(mode="capped_fill")` or `PairStrategy(mode="full_capped")`. | `capped_fill`, `full_capped`. |
| `Python/src/hexorl/eval/position_services.py:93-100` | `include_pair_rows` and row count choose pair generation | Constructs `PairStrategy(mode="capped_fill")` for eval position contracts. | `capped_fill`. |
| `Python/src/hexorl/replay/projector.py:260-279` | no branch; hardcoded mode | Projects crop pair targets with `PairStrategy(mode="capped_fill")`. | `capped_fill`. |
| `Python/src/hexorl/replay/projector.py:290-313` | no branch; hardcoded mode | Projects global graph pair targets with `PairStrategy(mode="full_capped", allow_full=True)`. | `full_capped`. |

Test-only generation-mode branching exists in `Python/tests/search/test_global_graph_pair_contracts.py:31-35`, where helper parameter `mode` sets `allow_full=mode == "full_capped"`.

## `mode` Vs `generation_mode` Trace

### `PairStrategy.mode`

Read/write sites:
- Declared as the input field in `Python/src/hexorl/contracts/pairs.py:29`.
- Validated in `Python/src/hexorl/contracts/pairs.py:34-43` against an inline set.
- Read by `PairActionTableBuilder.build` at `Python/src/hexorl/contracts/pairs.py:187`, `230`, `234`, `240`, and written into `PairActionTable.generation_mode` at `275`.
- Constructed in production:
  - `Python/src/hexorl/selfplay/game_runner.py:528-538`
  - `Python/src/hexorl/graph/semantic_builder.py:687-692`
  - `Python/src/hexorl/eval/position_services.py:97-100`
  - `Python/src/hexorl/replay/projector.py:275`
  - `Python/src/hexorl/replay/projector.py:308`
- Constructed in tests:
  - `Python/tests/contracts/test_phase02_builders.py:56,69,88,101,137,163`
  - `Python/tests/search/test_pair_strategy.py:68,75,123,130,137,144,156`
  - `Python/tests/search/test_pair_strategy_selfplay_integration.py:65`
  - `Python/tests/search/test_global_graph_pair_contracts.py:31-35,96,120`
  - `Python/tests/test_global_graph_contract.py:111`

Current role: `mode` is only the pair table builder input/selection request. It is not serialized directly and is not the downstream identity-bearing field.

### `PairActionTable.generation_mode`

Read/write sites:
- Declared as output field in `Python/src/hexorl/contracts/pairs.py:57`.
- Validated in `Python/src/hexorl/contracts/pairs.py:81-82` against an inline set.
- Included in `PairActionTable.table_hash` at `Python/src/hexorl/contracts/pairs.py:136`.
- Included in debug payload at `Python/src/hexorl/contracts/pairs.py:164`.
- Written from `PairStrategy.mode` in `PairActionTableBuilder.build` at `Python/src/hexorl/contracts/pairs.py:275`.
- Written as an explicit parameter in `_empty` at `Python/src/hexorl/contracts/pairs.py:289-300`.
- Copied in stale-reference test fixture at `Python/tests/contracts/test_phase02_builders.py:178`.
- Present in artifact sample `Docs/refactor/artifacts/phase_02/telemetry_samples/single_position_debug_bundle.json`.

Current role and source of truth:
- For contract identity and debug/inspection, `PairActionTable.generation_mode` is the source of truth because it is included in `table_hash` and `debug_payload`.
- For graph tensors/training, downstream consumers use `pair_rows`, `pair_table_mask`, `pair_phase`, and pair indices. `generation_mode` is not propagated into `GraphBatch` by `Python/src/hexorl/graph/tensorize.py:218-222`.
- For replay records, neither `mode` nor `generation_mode` is serialized. Replay stores pair targets (`pair_policy_target_v2`) and pair prior counts, not pair generation mode.
- For self-play records, neither field is stored directly. `PositionRecord` stores `pair_policy_target_v2` and `pair_prior_candidate_count`; game summary stores `pair_strategy_used`.
- For checkpoint manifests, `pair_strategy_used` is serialized from `cfg.model.pair_strategy` at `Python/src/hexorl/models/checkpoint.py:163`, but `mode`/`generation_mode` are not serialized.

Schema impact of renaming `PairStrategy.mode` to `PairStrategy.generation_mode`:
- No replay codec field named `mode` or `generation_mode` was found.
- `PairActionTable.generation_mode` is already public contract/debug identity. Keeping that name as the input field aligns the builder request with the output identity.
- Checkpoint schema records `pair_strategy_used`, not generation mode. The strategy-name rename/alias reconciliation may affect checkpoint manifest values if config canonicalization changes, but the `mode` -> `generation_mode` field rename itself does not appear to require a replay codec schema bump.

Ambiguity:
- `PairActionTable.table_hash` includes `generation_mode`, but graph tensor/replay projections drop it. If downstream needs to prove the exact table generation request after projection, current tensors do not carry that field. This is not a rename blocker, but it is a provenance gap.

## Runtime Consumers

- `GameRunner` constructs root `SearchContext` with `pair_strategy_id=self.pair_strategy` and optional `pair_table`/global graph pair rows (`Python/src/hexorl/selfplay/game_runner.py:600-610`).
- `GameRunner` calls `self.pair_strategy_impl.score_root` and emits `pair_strategy_summary` telemetry from the resulting `PairEvaluation` (`Python/src/hexorl/selfplay/game_runner.py:641-696`).
- `commit_root` applies pair priors only when `PairEvaluation.scored_pair_rows > 0` (`Python/src/hexorl/search/mcts_runner.py:33-44`).
- `EngineAdapter._apply_pair` consumes `PairEvaluation` and ignores strategy names except for telemetry/influence (`Python/src/hexorl/search/engine_adapter.py:343-355`).
- Training validates pair targets using config `pair_strategy != "none"` as a generic signal for opening-position rejection (`Python/src/hexorl/train/adapters.py:172-181`).

## Test Coverage By Strategy

| Strategy | Tests | Assertions |
| --- | --- | --- |
| `none` | `Python/tests/search/test_pair_strategy.py:67-109`; `Python/tests/search/test_pair_strategy_selfplay_integration.py:44-49`; `Python/tests/test_config_and_guardrails.py:365-420` | Zero selected/scored rows; default global families do not score pairs; pair head presence, pair prior mix, and architecture prefix do not enable pair scoring; runner defaults to disabled pair policy. |
| `two_stage_root_only` | `Python/tests/search/test_pair_strategy.py:122-127,143-159`; `Python/tests/selfplay/test_game_runner_interface.py:31-44`; `Python/tests/search/test_global_graph_pair_contracts.py:133-150` | Root cap is enforced; explicit scorer is required and used; pair prior source/influence is reported; runner consumes fake pair scorer; MCTS telemetry reports pair influence. |
| `tactical_only` | `Python/tests/search/test_pair_strategy.py:112-114,129-133` | Leaf scoring requires explicit cap; leaf cap is enforced when enabled. |
| `diagnostic_full_root` | `Python/tests/search/test_pair_strategy.py:117-140`; `Python/tests/search/test_pair_strategy_selfplay_integration.py:52-84`; `Python/tests/test_config_and_guardrails.py:425-471`; `Python/tests/search/test_engine_adapter.py:120-132` | Diagnostic/root/full cap validation; leaves score zero; explicit full-root strategy scores through fake scorer; config alias `diagnostic_full_pair` requires cap and enables runner pair policy; `PairEvaluation` immutability/engine interaction coverage. |
| `diagnostic_full_pair` alias | `Python/tests/test_config_and_guardrails.py:425-466`; `Python/tests/train/test_phase03_train_adapter_checkpoint.py:128-142` | Public config alias requires cap; train adapter rejects opening pair loss for non-none config. |

## Test Coverage By Pair Generation Mode

| Generation mode | Tests | Assertions |
| --- | --- | --- |
| `none` | `Python/tests/search/test_pair_strategy.py:67-78` | Empty pair table works with `NoPairStrategy`; zero root and leaf scores. |
| `selected` | `Python/tests/contracts/test_phase02_builders.py:97-105` | Selected target with wrong known-first fails validation. No positive selected-mode materialization test found. |
| `capped_fill` | `Python/tests/search/test_pair_strategy.py:122-159`; `Python/tests/search/test_global_graph_pair_contracts.py:31-40`; replay/eval projector construction paths indirectly | Root/leaf cap behavior and scorer flow. |
| `full_capped` | `Python/tests/contracts/test_phase02_builders.py:44-71,74-96,120-186`; `Python/tests/search/test_global_graph_pair_contracts.py:95-123`; `Python/tests/search/test_pair_strategy_selfplay_integration.py:52-84`; `Python/tests/test_global_graph_contract.py:89-116` | Full cap must cover possible pairs; phase-aware known-first rows; graph projection derives from canonical `PairActionTable`; stale references rejected; global pair logits align to pair rows. |

Coverage gap: `selected` has negative coverage but no positive test proving selected-only rows are materialized and hashed as intended.

## Recipe/Caller Reachability

| Strategy | Public config/recipe entry | Direct runtime caller | Test coverage | Candidate status |
| --- | --- | --- | --- | --- |
| `none` | Yes: config, recipes, family spaces | Yes: default everywhere | Strong | Migrate. |
| `diagnostic_full_root` | Indirect only via alias `diagnostic_full_pair` | Yes: `GameRunner` maps alias/canonical to spec | Strong | Migrate, but reconcile alias/canonical. |
| `two_stage_root_only` | No current config/schema/recipe entry found | Yes only through direct `GameRunnerConfig`/test factory or code caller that bypasses public config | Moderate: search + self-play fixture | Ambiguous. Phase 05 requires it, so likely migrate, but public recipe reachability is currently missing. |
| `tactical_only` | No current config/schema/recipe entry found | Yes only through direct `GameRunnerConfig` or code caller that bypasses public config | Narrow: validation/cap only, no self-play runner positive test | Ambiguous. Phase 05 requires it, but no public recipe entry and no observed production config route. |

No strategy was found with both no caller and no tests. No immediate deletion candidate satisfies the task's stop rule. The ambiguous candidates are `two_stage_root_only` and `tactical_only`: they are required by Phase 05 and have some tests, but current public config/recipe validation makes them unreachable from normal `Config`/recipe construction.

## Ambiguities To Resolve Before Phase 1

1. Canonical external name for diagnostic full strategy:
   - Search canonical: `diagnostic_full_root`.
   - Config/recipe public name: `diagnostic_full_pair`.
   - The proposed registry should pick one canonical name and keep the other only as descriptor alias if compatibility with existing configs/manifests is required.

2. Config/recipe allowed strategy set:
   - The north-star says new strategies should require one descriptor file and zero edits elsewhere.
   - Current `ModelConfig`, `ModelRecipe`, `FamilySpace`, and dry-run validation contain their own allowed-name branches/sets. Phase 1 must either move these to registry resolution or deliberately define a separate recipe policy.

3. `two_stage_root_only` and `tactical_only` product reachability:
   - Both exist in search/GameRunner and tests.
   - Neither is accepted by public config/recipe validation.
   - The refactor should decide whether to expose them via registry-driven config/recipe validation or remove them only if Phase 05 requirements are superseded. Current docs still list both as required strategies.

4. `selected` generation mode positive behavior:
   - The mode is part of `PairGenerationMode`, and the builder branches on it.
   - Only negative coverage was found.

5. Generation provenance after graph/replay projection:
   - `PairActionTable.generation_mode` is identity-bearing in the table hash/debug payload.
   - Graph tensors and replay targets do not carry it, so downstream training can validate pair rows/phases but not recover the generation mode after projection.

## Phase 0 Conclusion

The report matches the current worktree evidence: pair-strategy dispatch is duplicated in `GameRunner`, `search/pair_strategy.py`, config, recipes, tuning family spaces, and training validation. Pair generation mode dispatch is centralized in `contracts/pairs.py` for the builder, but the request field is still named `mode` while the identity/output field is `generation_mode`, and both validators duplicate the `PairGenerationMode` literal values as inline sets.

Recommended next step after confirmation: create the registry under `hexorl/contracts/pair_strategy/`, migrate config/recipe/search/self-play validation to registry resolution, and then perform the `PairStrategy.mode` -> `PairStrategy.generation_mode` rename with a grep gate.
