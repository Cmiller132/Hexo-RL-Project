# Architecture Inventory

Historical Stage 1 inventory for the architecture-authority cutover. The
implemented authority now lives in `Python/src/hexorl/models/`; the old
`Python/src/hexorl/model/` package has been deleted.

## Baseline Architecture Table Captured Before Cutover

| Architecture id | Legacy class or constructor | Required input tensors | Optional input tensors | Supported heads | Default heads | Loss defaults | Inference adapter needed | Policy provider needed | Pair capabilities | Current runtime consumers | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `cnn` | `HexNet` via `build_model_from_config` | dense crop tensor `(B,13,33,33)` | candidate rows for sparse/pair heads | `policy`, `value`, `lookahead_*`, `opp_policy`, `axis`, `axis_delta_norm`, `regret_rank`, `regret_value`, `moves_left`, optional `sparse_policy`, optional `pair_policy` | config default `policy`, `value`; default configs add lookahead and auxiliaries | `TrainConfig.loss_weights`; sparse/pair weights require explicit active entries | dense crop adapter with optional candidate/pair transport | dense policy provider; optional sparse provider | optional crop `pair_policy` only when head is enabled and a pair strategy requests it | trainer, inference server/client, self-play worker, dashboard/eval players, epoch pipeline | keep family, replace authority and adapters |
| `restnet` | `HexNet` with `SpatialTransformerBlock` at configured positions | dense crop tensor `(B,13,33,33)` | same as `cnn` | same as `cnn` | config-selected heads | same as `cnn` | dense crop adapter with restnet spec metadata | dense/sparse provider | same as `cnn` | config guardrails, trainer, inference, self-play, tuning | keep family, replace authority and assembly |
| `graph_hybrid_0` | `HexNet` with `SparseHexGraphHybrid0Encoder` | dense crop tensor `(B,13,33,33)` | candidate rows for `sparse_policy` and crop `pair_policy` | same crop heads plus sparse/pair candidate heads | config-selected heads; no `graph` alias at runtime | same as `cnn` | dense/candidate adapter; not true global graph | dense/sparse provider | optional crop pair candidate scorer | config, model, trainer, inference, replay feature flags, tuning | keep as crop-compatible scout, rename metadata as crop-sparse hybrid |
| `global_graph_option1` | `GlobalHexGraphNet` | graph tokens, token types, token qr, token mask, legal token indices, legal mask, relation type, relation bias | opponent legal rows, pair rows, crop tensor | `policy_place`, `value`, `lookahead_*`, `opp_policy`, `policy_pair_first`, `policy_pair_joint`, `policy_pair_second`, `regret_rank`, `regret_value`, `moves_left`, `tactical`, `axis`, `axis_delta_norm`, `legal_token_quality` | config heads plus automatic lookahead horizons | config mutates graph default loss weights | global graph adapter with relation schema and row hashes | global legal-row provider | first/joint/second pair outputs; strategy gated | config, model, trainer, inference, self-play, replay, tests | keep family, move architecture membership to registry |
| `global_xattn_0` | `GlobalHexGraphNet` family `context_cross_attention` | graph tokens and legal rows; relation tensors accepted but not required by model family | opponent legal rows, pair rows | same as global graph | config heads plus lookahead | same as global graph | global graph adapter without relation-required flag | global legal-row provider | same pair head surface | same as global graph | keep as registered recipe variant |
| `global_line_window_0` | `GlobalHexGraphNet` family `line_window_cover` | graph tensors plus relation type/bias | opponent legal rows, pair rows | same as global graph | config heads plus lookahead | same as global graph | global graph adapter with relation-required flag | global legal-row provider | same pair head surface | same as global graph | keep as registered recipe variant |
| `global_pair_twostage_0` | `GlobalHexGraphNet` family `pair_two_stage` | graph tensors | opponent legal rows, pair rows | same as global graph, with refined pair scorers | config heads plus lookahead | same as global graph | global graph pair-aware adapter | global provider plus pair strategy | primary pair-capable global family | same as global graph | keep; pair strategy must request pair behavior |
| `global_graph_full_0` | `GlobalHexGraphNet` family `full_relation_graph` | graph tensors plus relation type/bias | opponent legal rows, pair rows | same as global graph | config heads plus lookahead | same as global graph | global graph adapter with full relation schema | global legal-row provider | same pair head surface | same as global graph | keep as registered recipe variant |
| `global_hybrid_action_0` | `GlobalHexGraphNet` family `crop_diagnostic_global_action` | graph tensors | optional crop tensor for action gate | same as global graph | config heads plus lookahead | same as global graph | graph adapter that can carry optional crop context | global legal-row provider | same pair head surface | same as global graph | keep; crop tensor must be declared optional input contract |
| `global_graph768_champion` | `GlobalHexGraphNet` family `scaled_relation_graph` | graph tensors plus relation type/bias | opponent legal rows, pair rows | same as global graph | config heads plus lookahead | same as global graph | global graph adapter with scaled capacity hints | global legal-row provider | same pair head surface | same as global graph | keep as registered recipe variant |
| `graph` | alias in `ModelConfig` and `HexNet.__init__` | same as `graph_hybrid_0` after mutation | same as `graph_hybrid_0` | same as `graph_hybrid_0` | none as future id | inherited after normalization | no future adapter | no future provider | inherited only during legacy config normalization | config alias, model alias | delete as future id; Stage 2 may retain a non-runtime config migration error or one-shot alias decision only |

## Future Metadata Decisions

- Architecture ids are exact strings in the registry. Prefix checks such as
  `architecture.startswith("global_")` are deleted.
- `sparse_policy` is not an architecture. It is an optional output capability
  on crop-compatible families.
- `pair_policy` is not a pair strategy. It is an output capability. Runtime pair
  influence is owned by an executable pair strategy.
- Graph families share graph row contracts but may differ in recipe, relation
  requirements, optional crop context, and pair scorer structure.
- Every self-play architecture must resolve:
  - `search_policy`: dense board, candidate rows, or global legal rows.
  - `search_value`: binned value decoded to scalar in current-player
    perspective.

## Architecture Defaults And Overrides

- Specs own default outputs and default loss plan participation.
- Config may explicitly enable or disable supported optional outputs.
- Config may not disable outputs required by self-play:
  - dense/crop self-play requires `policy` and `value`;
  - sparse self-play requires `policy`, `value`, and `sparse_policy` when
    sparse prior stage is active;
  - global graph self-play requires `policy_place` and `value`.
- `lookahead_*` expands from `buffer.lookahead_horizons` during spec resolution.
  The preserved default horizons are `4`, `12`, and `36`; configs may select
  other integer horizons only if both target builder and loss plan resolve them.

## Scattered Architecture Authority Audit

Stage 2 must replace every source below with the registry/spec authority. These
are not equivalent today.

| Current source | Authority encoded | Exact ids or rule | Stage 2 action |
|---|---|---|---|
| `Python/src/hexorl/model/global_graph.py:136-144` | `GlobalHexGraphNet.ARCHITECTURES` runtime constructor allow-list | `global_graph_option1`, `global_xattn_0`, `global_line_window_0`, `global_pair_twostage_0`, `global_graph_full_0`, `global_hybrid_action_0`, `global_graph768_champion` | Move ids to registry recipes; constructor receives resolved spec only. |
| `Python/src/hexorl/model/global_graph.py:145-149` | relation-required family list | `global_graph_option1`, `global_line_window_0`, `global_graph_full_0`, `global_graph768_champion` | Move relation requirement to graph input contract metadata. |
| `Python/src/hexorl/model/global_graph.py:180-188` | architecture-to-family mapping | all seven global graph ids | Move family/recipe name to registry metadata and tests. |
| `Python/src/hexorl/config/schema.py:44-52` | config validation global architecture set | all seven global graph ids | Config validates by asking registry for known ids and capabilities. |
| `Python/src/hexorl/config/schema.py:289-297` | graph default loss mutation set | all seven global graph ids | Default losses come from resolved spec/loss plan, not config mutation. |
| `Python/src/hexorl/buffer/ring.py:22-28` | replay feature flag global graph set | only five ids: missing `global_hybrid_action_0` and `global_graph768_champion` | Delete list; replay flags come from resolved spec target/output requirements. |
| `Python/src/hexorl/buffer/ring.py:75-96` | sparse/graph replay diagnostics decision | graph flag, `graph`, `graph_hybrid_0`, and incomplete global set | Replace with explicit row/target storage requirements from spec. |
| `Python/src/hexorl/epoch/pipeline.py:184` and `Python/src/hexorl/epoch/pipeline.py:323` | graph replay inclusion by prefix | `architecture.lower().startswith("global_")` | Replace with `spec.requires_graph_policy_rows`. |
| `Python/src/hexorl/buffer/process.py:89` | self-play replay graph flag by prefix | `architecture.lower().startswith("global_")` | Replace with spec replay feature flags. |
| `Python/src/hexorl/dashboard/app.py:1103-1114` | dashboard architecture summary | special-cases only `graph` and `graph_hybrid_0`; all global graph ids fall through as CNN | Dashboard reads registry display metadata. |
| `Python/src/hexorl/dashboard/app.py:1117-1121` | trial model summary | special-cases only `graph` and `graph_hybrid_0`; all global graph ids fall through to generic `channels x blocks` | Dashboard reads registry display metadata. |
| `Python/src/hexorl/runtime.py:224-234` | memory estimate architecture scaling | special-cases `restnet`, `graph`, and `graph_hybrid_0`; global graph ids receive no graph token/layer scale | Runtime resource estimates use spec-declared tensor and token budgets. |
| `Python/tests/test_global_graph_contract.py:628-629` | test coverage driven by legacy model class set | `sorted(GlobalHexGraphNet.ARCHITECTURES)` | Parametrize from registry ids after Stage 2. |
| `Python/tests/test_global_graph_contract.py:749-758` | family distinctness assertion | checks four family labels but not every id by name | Assert every registered recipe id resolves expected family, requirements, and outputs. |
| `Python/tests/test_phase3_autotune.py` global graph assertions | tuning search assumptions about global graph ids | includes an assertion that `global_graph768_champion` is absent from a search result | Rewrite tuning assertions to use registry capability filters and explicit exclusion reasons. |

The original mismatch that blocked a blind cutover was:

- `global_hybrid_action_0` and `global_graph768_champion` are accepted by
  `GlobalHexGraphNet` and `Config`.
- `buffer/ring.py` omits both from `GLOBAL_GRAPH_ARCHITECTURES`, so replay
  feature flags can diverge from model/config authority.
- dashboard and runtime summaries still know only the crop graph alias pair,
  so global graph variants are under-reported unless Stage 2 adds spec metadata
  before removing legacy lists.

Stage 2/4 cleanup resolves this by routing architecture membership, replay
feature flags, dashboard summaries, runtime memory estimates, and family
construction through `hexorl.models.registry` and `hexorl.models.assembly`.
