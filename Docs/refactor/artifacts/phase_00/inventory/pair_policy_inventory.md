# Phase 00 S3 Pair Policy Inventory

Date: 2026-04-29

Owned artifact: `Docs/refactor/artifacts/phase_00/inventory/pair_policy_inventory.md`

Scope: S3 Models/Search inventory for `global_xattn` pair policy, `pair_prior_mix`, pair heads, architecture-string gates, and accidental pair scoring.

## Assignment Frame

Goal:

- Inspect model/search/config coupling for `global_xattn` pair policy, `pair_prior_mix`, pair heads, architecture-string gates, and accidental pair scoring.
- Identify the smallest likely Phase 00 runtime guard/test surface without implementing it in this inventory pass.

Success criteria:

- List all found pair enablement/coupling paths with file path, symbol or pattern, current behavior, risk, owner phase, expected deletion or replacement, and blocking tests/evidence.
- Use command-backed search evidence.

Constraints:

- Do not edit runtime code or tests.
- Do not write outside this owned file.
- Preserve unrelated changes.

Required evidence:

- Commands run with exit status.
- Artifact path changed.
- Blocked searches and ambiguity.

Stop rules:

- Stop if determining whether pair scoring can occur requires long self-play; record the exact runtime path to test instead.
- Stop if code edits would be needed.

## Checklist

- [x] Read `AGENTS.md`.
- [x] Read `Docs/refactor/V2_REQUIREMENTS_MATRIX.md`.
- [x] Read `Docs/refactor/phases/PHASE_00.md`.
- [x] Read `Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_V2_20260429.md`.
- [x] Checked worktree status before editing.
- [x] Searched for `pair_prior_mix`, pair heads, `pair_strategy`, pair scoring helpers, `PAIR_ACTION`, and global architecture gates.
- [x] Inspected focused source ranges for config, self-play/search, graph, model, inference, replay/training, autotune, tests, and Rust pair APIs.
- [x] Identified smallest likely Phase 00 guard/test surface.
- [x] Did not edit runtime code or tests.
- [x] Did not execute long self-play.

## V2 Rows Touched By This Inventory

This inventory supports, but does not close, these rows:

| V2 row | Relevance | Status from this artifact |
|---|---|---|
| V2-001 | `global_xattn` default pair strategy is `none` and emits zero pair rows unless explicitly opted in. | Inventory evidence only. Current worktree contains a candidate guard, but this pass did not implement or run it. |
| V2-002 | Accidental pair scoring guarded before broad refactor. | Inventory evidence only. Guard/test surface identified below. |
| V2-004 | Legacy fallback, architecture-string, duplicate helper, stale runtime path inventory. | This file contributes S3 pair-policy inventory. |
| V2-021 | Pair action construction central owner. | Later owner for `PAIR_ACTION`, crop pair candidates, and graph pair row builders. |
| V2-053 | No pair scoring from architecture/config/head presence. | Later search owner; Phase 00 needs a narrow guard. |
| V2-054 | Global graph policy heads have row-mapped contracts and telemetry. | Later model/inference/search owner. |

No V2 row is claimed complete by this artifact.

## Worktree Note

Initial `git status --short` returned exit 0 with no output. During inventory, `git status --short` later returned exit 0 with these pre-existing/unowned modifications:

```text
 M Python/src/hexorl/config/schema.py
 M Python/src/hexorl/selfplay/worker.py
 M Python/tests/test_config_and_guardrails.py
```

This S3 inventory pass did not edit those files. Because they affect the pair-policy surface, this inventory records the current worktree behavior and identifies those runtime/test changes as observed context, not as S3-produced implementation.

Final `git status --short --untracked-files=all` also showed multiple untracked Phase 00 artifacts and `scripts/phase00_capture_baseline.py`. Those files were not created or edited by this S3 pass, except for this owned `pair_policy_inventory.md` artifact.

## Command Evidence

Blocked or failed search attempts:

| Command | Exit | Evidence/result |
|---|---:|---|
| `Get-Content -Raw Docs/refactor/artifacts/phase_00/inventory/pair_policy_inventory.md` | 1 | Owned artifact did not exist before this pass. |
| `rg --files` | 1 | `rg.exe` failed with `Access is denied`. |
| `rg -n "pair_prior_mix|pair_policy_enabled|pair_head|PairPolicyHead|policy_pair_(first|second|joint)|pair_strategy|PAIR_ACTION|pair_rows|pair_scoring|global_xattn|global_graph_enabled|pair_prior" Python Docs scripts configs .github crates` | 1 | PowerShell parser error from the quoted regex attempt. |
| `rg -n "architecture|model_arch|arch\\b|global_" Python/src Python/tests scripts configs Docs/refactor -g "*.py" -g "*.md" -g "*.toml" -g "*.yaml" -g "*.yml" -g "*.json"` | 1 | `rg.exe` failed with `Access is denied`. |

Successful evidence commands:

| Command | Exit | Evidence/result |
|---|---:|---|
| `Get-Content -Raw AGENTS.md` | 0 | Read local agent instructions. |
| `Get-Content -Raw Docs/refactor/V2_REQUIREMENTS_MATRIX.md` | 0 | Read V2 rows V2-001, V2-002, V2-004, V2-021, V2-053, V2-054. |
| `Get-Content -Raw Docs/refactor/phases/PHASE_00.md` | 0 | Read Phase 00 pair guard, accidental pair scoring test, inventory requirements, and hard gates. |
| `Get-Content -Raw Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_V2_20260429.md` | 0 | Read V2 design pair-policy and no architecture-string inference rules. |
| `Get-ChildItem -Path Docs/refactor/artifacts/phase_00/inventory -Force` | 0 | Inventory directory existed; no files were printed at that moment. |
| `Get-Command rg` | 0 | `rg.exe` exists under WindowsApps, but invocation was blocked. |
| `Get-ChildItem -Path Python/src,Python/tests,scripts,configs,Docs/refactor,Docs,crates -Recurse -File -Include *.py,*.md,*.toml,*.yaml,*.yml,*.json,*.rs | Select-String -Pattern 'pair_prior_mix|pair_policy_enabled|pair_head|PairPolicyHead|policy_pair_(first|second|joint)|pair_strategy|PAIR_ACTION|pair_rows|pair_scoring|global_xattn|global_graph_enabled|pair_prior'` | 0 | Broad pair-policy fallback search. |
| `Get-ChildItem -Path Python/src,Python/tests,scripts,configs -Recurse -File -Include *.py,*.toml,*.yaml,*.yml,*.json | Select-String -Pattern 'architecture|model_arch|arch\\b|global_'` | 0 | Broad architecture/global fallback search. |
| `Select-String -Path Python/src/hexorl/config/schema.py -Pattern 'pair_prior_mix|pair_policy|policy_pair_|global_xattn|global_architectures|heads'` | 0 | Config pair fields, validators, global defaults. |
| `Select-String -Path Python/src/hexorl/selfplay/worker.py -Pattern 'global_graph_enabled|pair_prior_mix|pair_policy_enabled|apply_root_pair_priors|policy_pair_first|policy_pair_second|policy_pair_joint|include_pair_rows|max_pair_rows|_score_root_pair_actions|_score_second_pair_actions|_graph_batch_with_pair_rows'` | 0 | Self-play/search pair enablement and scoring call sites. |
| `Select-String -Path Python/src/hexorl/selfplay/worker.py -Pattern '_score_crop_pair_chunks|pair_policy_enabled|pair_strategy' -Context 3,5` | 0 | Current worktree guard consumes `pair_strategy` and requires `pair_strategy_max_pairs`. |
| `Select-String -Path Python/src/hexorl/graph/batch.py -Pattern 'PAIR_ACTION|include_pair_rows|max_pair_rows|total_pair_rows|pair_limit|pair_qr|graph_batch_with_reference_pair_rows'` | 0 | Graph pair row materialization and reference pair rows. |
| `Select-String -Path Python/src/hexorl/model/global_graph.py -Pattern 'ARCHITECTURES|global_xattn_0|policy_pair_first|policy_pair_second|policy_pair_joint|PAIR_ACTION|pair_token|architecture_family|architecture =='` | 0 | Global graph architecture gates and pair heads. |
| `Select-String -Path Python/src/hexorl/model/network.py -Pattern 'PairPolicyHead|pair_policy|pair_rows|pair_candidate_row_indices|heads|graph_token_set|build_model_from_config|global_architectures'` | 0 | Crop-compatible pair head and model factory gates. |
| `Select-String -Path Python/src/hexorl/inference/server.py -Pattern '_global_graph_mode|architecture|policy_pair_first|policy_pair_second|policy_pair_joint|pair_count|res_graph_pair'` | 0 | Inference architecture dispatch and graph pair logit scatter. |
| `Select-String -Path Python/src/hexorl/inference/client.py -Pattern 'policy_pair_first|policy_pair_second|policy_pair_joint|pair_count|res_graph_pair|prior_source'` | 0 | Graph request pair counts and response pair arrays. |
| `Select-String -Path Python/src/hexorl/buffer/ring.py -Pattern 'PAIR_POLICY_HEADS|GLOBAL_GRAPH_ARCHITECTURES|architecture|pair_policy|pair_prior_candidate_count|pair_prior_hit_frac'` | 0 | Replay feature flags tied to heads and architectures. |
| `Select-String -Path Python/src/hexorl/buffer/sampler.py -Pattern 'pair_policy|policy_pair_first|policy_pair_second|policy_pair_joint|include_pair_rows|graph_batch_with_reference_pair_rows|complete search-observed|pair_rows'` | 0 | Training sampler pair target and graph pair-row projection path. |
| `Select-String -Path Python/src/hexorl/epoch/pipeline.py -Pattern 'GRAPH_PAIR_POLICY_HEADS|include_graph_policy|startswith\\(\"global_|architecture|pair_policy'` | 0 | Epoch sampler switches tied to heads and architecture string. |
| `Select-String -Path scripts/run_phase3_48h_autotune.py -Pattern 'GLOBAL_GRAPH_PAIR_HEADS|pair_prior_mix|policy_pair_|global_xattn_0|global_pair_twostage_0|global_graph_full_0|_make_config|_make_replay_buffer|pair_scout|head_bundle|family.global_graph|family.architecture'` | 0 | Autotune family/head/raw config coupling. |
| `Select-String -Path Python/tests/test_config_and_guardrails.py -Pattern 'global_graph_selfplay_consumes_pair_priors_by_default|pair_prior_mix|pair_policy_enabled|policy_pair_|pair_strategy|global_xattn'` | 0 | Current worktree has focused Phase 00 guard tests. |
| `Select-String -Path Python/tests/test_global_graph_contract.py -Pattern 'include_pair_rows|max_pair_rows|policy_pair_|PAIR_ACTION|global_xattn_0|global_pair_twostage_0|pair_rows|pair_logits|pair_chunks'` | 0 | Positive global pair row/model capability tests. |
| `Select-String -Path Python/tests/test_phase3_autotune.py -Pattern 'pair_policy|GLOBAL_GRAPH_PAIR_HEADS|global_xattn_0|global_pair_twostage_0|pair_prior_mix|policy_pair_'` | 0 | Autotune pair storage tests tied to global pair heads. |
| `Select-String -Path crates/hexgame-core/src/mcts.rs,crates/hexgame-py/src/engine.rs,crates/hexgame-py/src/protocol.rs -Pattern 'apply_root_pair_priors|apply_root_pair_second_priors|decode_pair_rows|pair_logits|pair_actions'` | 0 | Rust pair prior APIs accept calls; Python/search must own strategy gating. |
| `git diff -- Python/src/hexorl/config/schema.py` | 0 | Observed unowned worktree guard additions. |
| `git diff -- Python/src/hexorl/selfplay/worker.py` | 0 | Observed unowned worktree guard additions and capped pair scoring helpers. |
| `git diff -- Python/tests/test_config_and_guardrails.py` | 0 | Observed unowned focused guard tests. |
| `Get-ChildItem -Path Python/src,Python/tests,scripts,configs -Recurse -File -Include *.py,*.toml,*.yaml,*.yml,*.json | Select-String -Pattern 'pair_strategy|pair_prior_mix|policy_pair_first|policy_pair_second|policy_pair_joint|pair_policy_enabled|diagnostic_full_pair'` | 0 | Current pair-strategy and pair-head coupling inventory. |
| `Get-ChildItem -Path Python/src,Python/tests,scripts,configs -Recurse -File -Include *.py,*.toml,*.yaml,*.yml,*.json | Select-String -Pattern 'startswith\\(\"global_|_is_global_graph_architecture|GLOBAL_GRAPH_ARCHITECTURES|GlobalHexGraphNet\\.ARCHITECTURES|architecture in global_architectures|architecture == \"global_xattn_0\"'` | 0 | Current architecture-string coupling inventory. |
| `git status --short --untracked-files=all` | 0 | Final status showed unowned runtime/test modifications and untracked Phase 00 artifacts; this pass only owns `pair_policy_inventory.md`. |

Focused line-range reads of the files listed in the inventory below also exited 0 and were used to confirm behavior around the matched symbols.

## Pair Enablement And Coupling Inventory

| ID | File path | Symbol/pattern | Current behavior | Risk | Owner phase | Expected deletion/replacement | Blocking tests/evidence |
|---|---|---|---|---|---|---|---|
| S3-001 | `Python/src/hexorl/config/schema.py` | `ModelConfig.pair_prior_mix`, `pair_strategy`, `pair_strategy_max_pairs` | Current worktree has `pair_strategy="none"` and `pair_strategy_max_pairs=0` defaults. Validator accepts `none` only with zero cap and requires positive mix/cap for `diagnostic_full_pair`. `pair_prior_mix` still defaults to `0.35`. | `pair_prior_mix` remains a tempting implicit enablement flag if any runtime path ignores `pair_strategy`. | Phase 00 for immediate guard, Phase 05 for final `PairStrategySpec`. | Replace raw string fields with typed `PairStrategySpec`; `pair_prior_mix` becomes a strategy parameter, not model behavior. | Fast config tests proving `global_xattn_0` default is `none`, nonzero `pair_prior_mix` alone does not enable scoring, and non-none strategy needs a cap. |
| S3-002 | `Python/src/hexorl/config/schema.py` | `global_architectures`, graph train loss defaults for `policy_pair_first`, `policy_pair_second`, `policy_pair_joint` | All global architectures get default loss weights for graph pair heads if the heads appear in `model.heads`; architecture name drives graph default behavior. | Architecture-string family membership still controls training defaults and can blur "model can output a pair head" with "search should consume a pair head." | Phase 03, Phase 05. | Model registry/family descriptors declare train adapter/loss plan/capabilities; `PairStrategy` decides search consumption. | Registry matrix tests; trainer one-batch tests for each family; audit that config no longer owns global family behavior. |
| S3-003 | `Python/src/hexorl/selfplay/worker.py` | `global_graph_enabled = architecture.startswith("global_")` | Self-play still enters the global graph path by architecture prefix. Current worktree pair enablement is no longer tied to this prefix, but graph mode still is. | Architecture-string gate remains in runtime. It can indirectly select graph inference and pair-capable response paths. | Phase 05, Phase 06. | `PolicyProvider`/model capability object selects graph provider; `GameRunner` consumes providers without architecture checks. | Import/code audit showing `SelfPlayWorker` contains no architecture string checks; provider integration tests for dense, graph hybrid, and global graph. |
| S3-004 | `Python/src/hexorl/selfplay/worker.py` | `self.pair_policy_enabled = self.pair_strategy != PAIR_STRATEGY_NONE` | Current worktree consumes `cfg.model.pair_strategy` and ignores head presence and `pair_prior_mix` for `pair_policy_enabled`. | This is the smallest plausible Phase 00 guard, but it is still a string guard inside worker rather than final strategy ownership. | Phase 00 immediate guard, Phase 05 final strategy. | Move to `search/pair_strategy.py` with typed strategy validation and independent root/leaf/full/diagnostic caps. | Focused guard tests plus trace/log sample showing default `global_xattn_0` has `pair_strategy=none`, `pair_rows_possible` recorded, and `pair_rows_scored=0`. |
| S3-005 | `Python/src/hexorl/selfplay/worker.py` | `_score_graph_pair_chunks(..., max_pair_rows=...)` | Current worktree helper refuses `max_pair_rows <= 0` and scores graph pair rows in bounded chunks when called. It can enumerate unordered first-placement pairs or known-first second placements up to the cap. | Hot path can still perform expensive pair work if strategy/cap are set accidentally or leaf scoring shares the same broad cap. | Phase 05, Phase 02. | Replace helper with `PairStrategy` scoring over canonical `PairActionTable`, with independent root/leaf/diagnostic caps. | Negative test that calling scoring without explicit nonzero cap fails; diagnostic test that full pair enumeration cannot run without explicit diagnostic strategy and cap; performance budget evidence for any enabled strategy. |
| S3-006 | `Python/src/hexorl/selfplay/worker.py` | `_score_crop_pair_chunks(..., max_pair_rows=...)` | Current worktree helper refuses `max_pair_rows <= 0` and scores crop-compatible `pair_policy` rows through candidate chunks when called. | Crop-compatible pair head can still influence root priors if a strategy enables it; semantics are separate from graph `PAIR_ACTION` rows. | Phase 05, Phase 02, Phase 03. | Demote crop pair scorer to a `PairActionTable` projection or auxiliary train adapter, not implicit search behavior. | No-implicit-pair tests for crop models with `pair_policy` head and `pair_strategy=none`; projection equality tests after `PairActionTable` exists. |
| S3-007 | `Python/src/hexorl/selfplay/worker.py` | Root global graph path: `_score_graph_pair_chunks`, `apply_root_pair_first_priors`, `apply_root_pair_priors`, `apply_root_pair_second_priors` | Current worktree wraps scoring and Rust pair prior calls in `if self.pair_policy_enabled`; scoring gets `self.pair_strategy_max_pairs`. Prior to the observed unowned worktree changes, search output showed this path was coupled to `global_graph_enabled and pair_prior_mix > 0` or pair heads. | This is the primary accidental pair-scoring trap for `global_xattn_0`: a root request can call model pair heads and Rust pair-prior APIs. | Phase 00 immediate guard, Phase 05 final search strategy. | `PolicyProvider` supplies place priors; `PairStrategy` owns any pair scoring and Rust pair-prior application. | Fast worker construction tests, plus a narrow runtime smoke or monkeypatched inference test that fails if `_score_graph_pair_chunks` or Rust pair APIs are reached with strategy `none`. |
| S3-008 | `Python/src/hexorl/selfplay/worker.py` | Non-global root path: `use_action_keyed_root = sparse... or self.pair_policy_enabled`, `_score_crop_pair_chunks`, Rust pair prior calls | Pair strategy can select crop pair scoring for non-global/crop-compatible pair heads. Current worktree guard/cap applies. | Pair head presence in crop model must not activate action-keyed pair scoring; pair strategy must not be confused with sparse policy. | Phase 05. | Search providers separate sparse candidate priors from pair strategy. | No-implicit-pair test for `"pair_policy"` head with `pair_strategy=none`; search audit proving `pair_policy_enabled` is not derived from head presence. |
| S3-009 | `Python/src/hexorl/selfplay/worker.py` | Leaf global graph path: pair blending inside pending leaf expansion | Current worktree guards leaf pair scoring with `self.pair_policy_enabled` and passes `pair_strategy_max_pairs`. It can score pairs at leaf expansion and blend pair logits into action logits. | Leaf pair scoring is a separate performance trap; Phase 00 can guard it, but Phase 05 needs separate leaf caps and telemetry. | Phase 05, Phase 06. | `PairStrategySpec` independently validates root, leaf, full, and diagnostic caps; leaf pair scoring disabled by default. | No-implicit-leaf-pair test with monkeypatched `_score_graph_pair_chunks`; throughput/latency evidence for any enabled leaf pair strategy. |
| S3-010 | `Python/src/hexorl/selfplay/worker.py` | `pair_strategy_summary` log event | Current worktree has a summary payload with `pair_strategy`, `pair_prior_mix`, `pair_rows_possible`, and `pair_rows_scored`, emitted on worker start. | Summary start log alone does not prove zero scored rows during a game or leaf expansion. | Phase 00, Phase 06. | Structured telemetry owner emits per-position/game pair summaries and `ContractTrace` row counts. | Real or narrow smoke log sample showing default `global_xattn_0` reports `pair_rows_scored=0`; trace sample includes pair counts. |
| S3-011 | `Python/src/hexorl/graph/batch.py` | `build_graph_batch_from_history(..., include_pair_rows=True, max_pair_rows=PAIR_CHUNK_LIMIT)` | Graph builder materializes `PAIR_ACTION` rows by default. First-placement states build unordered `A * (A - 1) / 2` pair rows up to `max_pair_rows`; second-placement states build known-first legal seconds. Callers can pass `include_pair_rows=False`. | Accidental callers can build pair rows even when search strategy is `none`. This is graph-construction cost, even before model scoring. | Phase 02, Phase 05. | Pair rows come only from canonical `PairActionTable` selected by `PairStrategy`; graph semantic builder default emits no pair rows unless requested by a strategy/projection. | Code-search audit that default self-play/global requests use zero pair rows; unit test that `include_pair_rows=False` produces empty pair table for global default; deletion/import audit after builder split. |
| S3-012 | `Python/src/hexorl/graph/batch.py` | `graph_batch_with_reference_pair_rows` | Replay/training can attach full legal pair references without materializing `PAIR_ACTION` tokens, allowing pair heads to train over complete pair tables. | Useful training projection but duplicates pair table semantics outside a canonical contract. | Phase 02, Phase 03, Phase 07. | Projection from `PairActionTable`; no private full pair reconstruction in sampler/graph. | Pair table D6/phase tests; projection-only tests; import audit proving no duplicate pair mini-contract remains. |
| S3-013 | `Python/src/hexorl/model/global_graph.py` | `GlobalHexGraphNet.ARCHITECTURES`, `architecture == "global_xattn_0"` | Model family selection and internal block/context behavior use architecture strings. `global_xattn_0` uses cross-attention over non-LEGAL context. | Model behavior is still stringly; search/inference can key behavior from architecture rather than model capabilities. | Phase 03. | Registered `global_xattn` family descriptor/facet owns trunk and capabilities. | Registry build tests; architecture-string audit for model construction after Phase 03. |
| S3-014 | `Python/src/hexorl/model/global_graph.py` | `policy_pair_first`, `policy_pair_joint`, `policy_pair_second` | `policy_pair_first` is returned for every legal row. Joint/second pair logits are returned whenever pair index tensors are supplied. | Head existence/output is first-class capability but must not imply MCTS consumption. Consumers can accidentally use these outputs. | Phase 03, Phase 05, V2-054 owner surface. | Model family declares output contracts; inference adapter validates row mapping; `PairStrategy` decides consumption. | Shape/row-mapping tests for all graph pair heads; no-implicit-pair search tests; telemetry proving whether pair heads influenced MCTS. |
| S3-015 | `Python/src/hexorl/model/network.py` | `PairPolicyHead`, `HexNet.pair_policy_head` | Crop-compatible pair head is built if `"pair_policy"` is in `model.heads` and emits logits when pair candidate inputs are supplied. | Auxiliary head can be mistaken for global pair strategy or MCTS pair consumption. | Phase 03, Phase 05. | Train adapter owns auxiliary pair loss; search consumes only explicit `PairStrategy` outputs. | Trainer adapter tests for pair targets; no search path can branch only on `"pair_policy"` head. |
| S3-016 | `Python/src/hexorl/model/network.py` | `build_model_from_config`, `arch in GlobalHexGraphNet.ARCHITECTURES` | Central model factory dispatches global graph model by architecture string and passes `output_heads` from config. | Model assembly still combines family selection, head declaration, and config-specific behavior. | Phase 03. | `models/registry.py` and family modules replace factory switch. | Build tests for all registered families; code audit that runtime no longer imports `hexorl/model` factory. |
| S3-017 | `Python/src/hexorl/inference/server.py` | `_global_graph_mode = architecture.startswith("global_")` | Inference server dispatches graph mode by architecture string and requires only `value` for graph models. | Violates V2 request-kind/protocol dispatch target; graph mode can be selected without manifest. | Phase 04. | `InferenceProtocolManifest` and request kind dispatch. | Protocol handshake tests and architecture-string dispatch audit. |
| S3-018 | `Python/src/hexorl/inference/server.py` | `_forward_graph_batch`, `_scatter_graph_results`, pair logit scatter | Server sanitizes pair heads if model returns them. It requires `policy_pair_first` whenever `legal_count` is nonzero and requires joint/second pair logits when `pair_count` is nonzero. | Transport is pair-capable regardless of search strategy; pair_first is treated as required graph output. | Phase 04, Phase 05. | Inference adapter declares output contracts and only returns pair payloads for request kinds that include pair rows. | Response telemetry assertions with request kind, pair counts, and strategy; mismatch tests for pair head absence only when pair request expects it. |
| S3-019 | `Python/src/hexorl/inference/client.py` | `submit_graph`, `pair_count`, `policy_pair_*` response arrays | Client copies graph pair row indices into IPC only when `graph_batch.pair_token_indices` has rows. Response always exposes `policy_pair_first`; joint/second arrays are empty when `pair_count=0`. | Capability path is safe only if upstream graph batches do not contain accidental pair rows. Client metadata does not record strategy. | Phase 04, Phase 05. | Request schema carries pair strategy/request kind and pair row contract hash. | Tests for zero pair rows in default graph requests; telemetry showing `pair_count=0` for default `global_xattn`. |
| S3-020 | `Python/src/hexorl/buffer/ring.py` | `PAIR_POLICY_HEADS`, `replay_feature_flags` | Replay storage enables pair policy blobs when model heads contain graph pair heads. Sparse diagnostics also enable for global architectures. | Storage/training behavior is head/architecture-coupled. This is not search scoring, but it can preserve pair policy assumptions into replay. | Phase 07, Phase 03. | Replay record schema records explicit target contracts and model recipe identity; pair target storage follows train adapter/recipe, not raw head names. | Replay import audit; storage round-trip tests for explicit pair target contracts. |
| S3-021 | `Python/src/hexorl/buffer/sampler.py` | `include_pair_policy`, `graph_batch_with_reference_pair_rows` | Sampler builds crop pair targets and graph reference pair rows when `include_pair_policy` is true. It rejects graph first-placement pair training if search-observed pair targets are incomplete. | Pair target generation is duplicated across sampler/graph and is driven by train head inclusion. | Phase 02, Phase 03, Phase 07. | `PairActionTable` plus train adapter projection; sampler uses replay/projector contracts only. | Golden equality tests across self-play, replay sampler, training, dashboard; projection corruption tests. |
| S3-022 | `Python/src/hexorl/epoch/pipeline.py` | `_uses_pair_policy_targets`, `include_graph_policy=architecture.startswith("global_")` | Epoch pipeline includes pair policy targets when heads contain pair names and graph policy when architecture starts with `global_`. | Training data path still infers behavior from head names and architecture strings. | Phase 03, Phase 07. | `TrainAdapter` and recipe/model spec drive target requirements. | Trainer no-branch audit; one-batch every family; sampler/projector import audit. |
| S3-023 | `scripts/run_phase3_48h_autotune.py` | `GLOBAL_GRAPH_PAIR_HEADS`, `_heads_for_recipe`, `_make_config`, `pair_prior_mix = 0.0` for global without pair heads | Autotune uses raw config mutation and architecture names to choose pair heads. It zeros `pair_prior_mix` for global families without pair heads, but leaves default mix when pair heads exist. It does not set a non-none `pair_strategy` in the observed code. | Autotune can create configs with pair heads and nonzero mix. Current worktree self-play guard should still disable scoring by default, but recipe semantics remain implicit. | Phase 08, Phase 05. | Typed `ModelRecipe` and `PairStrategySpec`; family spaces declare pair-capable trials explicitly. | Recipe dry-run tests proving `global_xattn_0` keeps `pair_strategy=none`; no raw-config mutation audit. |
| S3-024 | `Python/src/hexorl/train/losses.py` | `policy_pair_first`, `policy_pair_second`, `policy_pair_joint` loss branches | Loss code supports graph pair heads. | Training support can be mistaken for active search strategy; pair target semantics need turn/phase validation. | Phase 03. | Train adapter validates pair targets and loss plan per family/spec. | Pair target training validation: first, known-first second, joint pair, opening no-pair semantics. |
| S3-025 | `Python/src/hexorl/train/trainer.py` | Global graph pair loss weight defaults | Trainer defaults pair graph head weights for global graph models. | Trainer branches on model class/global graph behavior rather than adapters. | Phase 03. | `TrainAdapter` and loss plan per family. | No trainer model-class branch audit; one-batch tests for every family. |
| S3-026 | `Python/tests/test_config_and_guardrails.py` | Current worktree tests: `test_global_xattn_pair_strategy_defaults_to_none`, `test_global_xattn_pair_heads_do_not_enable_pair_scoring_without_strategy`, `test_pair_scoring_requires_explicit_diagnostic_strategy_and_cap` | Unowned current worktree has the likely Phase 00 fast guard tests. This pass did not edit or run them. | Tests are not evidence until executed and archived in command transcripts; they also do not prove long self-play cannot score pairs. | Phase 00. | Keep as focused Phase 00 guard tests or equivalent under final test layout. | Run focused tests with cache/bytecode writes disabled or record command transcript under Phase 00 commands. |
| S3-027 | `Python/tests/test_global_graph_contract.py` | Pair row materialization/chunking/model pair output positive tests | Existing tests prove pair-capable graph/model paths. They do not prove default no-pair search behavior. | Positive capability tests can mask accidental consumption if no negative no-implicit-pair test exists. | Phase 00, Phase 05. | Add no-implicit-pair negative tests beside positive graph pair tests. | Negative tests for default `global_xattn`, pair heads with `pair_strategy=none`, nonzero mix alone, and full pair diagnostic cap. |
| S3-028 | `Python/tests/test_phase3_autotune.py` | `test_phase3_pair_policy_storage_is_only_for_global_pair_heads` | Autotune tests assert pair storage follows global pair heads. | Storage behavior remains head-driven and does not encode pair strategy. | Phase 08, Phase 07. | Recipe/family-space tests assert explicit pair strategy and replay contract storage. | Autotune recipe dry-run evidence and replay feature flag tests after recipe refactor. |
| S3-029 | `crates/hexgame-core/src/mcts.rs`, `crates/hexgame-py/src/engine.rs`, `crates/hexgame-py/src/protocol.rs` | `apply_root_pair_priors`, `apply_root_pair_second_priors`, `decode_pair_rows` | Rust validates pair row shape/semantics and applies pair priors when Python calls it. Rust does not own whether a strategy should call it. | If Python calls Rust pair APIs accidentally, Rust will perform pair prior application; no strategy guard exists at this boundary. | Phase 05, with Phase 01 boundary suspicion tests. | `EngineAdapter` is the only Python caller; `PairStrategy` owns calls and tokens/caps. | Import/code audit for Rust MCTS APIs; stale token and invalid prior tests; no-implicit-pair tests before Rust calls. |

## Smallest Likely Phase 00 Guard/Test Surface

The smallest Phase 00 runtime guard surface is:

1. Config-level declaration:
   - `pair_strategy` defaults to `none`.
   - `pair_strategy_max_pairs` defaults to `0`.
   - Non-none strategy requires explicit pair head, positive `pair_prior_mix`, and positive cap.

2. Self-play/search guard:
   - `SelfPlayWorker` derives `pair_policy_enabled` only from `pair_strategy != "none"`.
   - No branch may derive pair scoring from `global_graph_enabled`, `pair_prior_mix`, `policy_pair_*` head presence, or `"pair_policy"` head presence.
   - `_score_graph_pair_chunks` and `_score_crop_pair_chunks` fail fast when called without a positive cap.
   - Root and leaf pair scoring call sites pass the cap and remain behind `pair_policy_enabled`.

3. Minimal logging:
   - A `pair_strategy_summary` or equivalent event reports `pair_strategy`, `pair_prior_mix`, `pair_rows_possible`, and `pair_rows_scored`.
   - Default `global_xattn_0` evidence must show `pair_rows_scored=0`.

4. Fast tests, no long self-play required:
   - `global_xattn_0` default config creates a worker with `pair_strategy=none` and `pair_policy_enabled=False`.
   - `global_xattn_0` with pair heads and nonzero `pair_prior_mix` still has `pair_policy_enabled=False` when strategy is `none`.
   - Non-none diagnostic strategy without a positive cap is rejected.
   - Pair scoring helper called with cap `0` raises before inference/model scoring.

Observed current worktree already contains a candidate version of this guard and test surface in `Python/src/hexorl/config/schema.py`, `Python/src/hexorl/selfplay/worker.py`, and `Python/tests/test_config_and_guardrails.py`. This inventory did not implement it and did not run it.

## Runtime Paths To Test If End-To-End Pair Scoring Must Be Proven

Executing full self-play was not required for this inventory and would violate the stop rule if needed just to determine whether pair scoring can occur. The exact runtime paths to test instead are:

- Root global path: `Python/src/hexorl/selfplay/worker.py` around `_score_graph_pair_chunks(...)` and subsequent `engine.apply_root_pair_first_priors`, `engine.apply_root_pair_priors`, `engine.apply_root_pair_second_priors`.
- Root crop/action-keyed path: `Python/src/hexorl/selfplay/worker.py` around `_score_crop_pair_chunks(...)` and Rust pair prior calls.
- Leaf global path: `Python/src/hexorl/selfplay/worker.py` pending leaf metadata loop where graph pair chunks are scored and pair logits are blended into leaf logits.
- Graph construction precondition: `Python/src/hexorl/graph/batch.py` `build_graph_batch_from_history(..., include_pair_rows=False, max_pair_rows=0)` for default self-play graph requests.
- Inference transport precondition: `Python/src/hexorl/inference/client.py` `submit_graph` metadata `pair_count` should be `0` for default `global_xattn_0`.

Suggested narrow Phase 00 tests should monkeypatch the pair scoring helpers or Rust pair prior methods to fail if reached under `pair_strategy=none`, rather than running long self-play.

## Blockers And Ambiguity

- `rg` was blocked by the local WindowsApps shim with `Access is denied`; fallback `Get-ChildItem | Select-String` searches succeeded.
- The owned inventory file did not exist before this pass and was created here.
- The worktree changed during inventory or contained unowned runtime/test modifications not present in the initial status check. This inventory preserves those changes and does not attribute them to S3.
- This pass did not run pytest or self-play. Running tests could write caches/bytecode outside the owned artifact unless explicitly configured, and long self-play is outside the requested inventory-only scope.
- End-to-end proof that no pair scoring occurs in live self-play remains blocked on focused runtime tests or a short smoke command recorded under Phase 00 commands.

## Artifact Change

Changed by this pass:

- `Docs/refactor/artifacts/phase_00/inventory/pair_policy_inventory.md`

Runtime consumers changed:

- None by this pass.

Legacy paths deleted or quarantined:

- None by this pass.

Performance evidence:

- None. This was an inventory-only pass. Hot-path performance evidence is required before enabling any non-none pair strategy.

Statement of scope:

- No skipped, deferred, or manual-only requirement is claimed complete by this artifact.
