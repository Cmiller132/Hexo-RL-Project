# Modular Hexo Architecture Redesign

Date: 2026-04-29

Status: design proposal

Scope: model loading, model assembly, inference, self-play, MCTS integration, pair policy, global graph support, replay contracts, autotuning, observability, dashboard/debugging, and test structure.

Research inputs: this writeup synthesizes three read-only investigations:

- Lorentz: model construction, config, checkpoint loading, inference contracts.
- Curie: self-play, search, pair scoring, Rust MCTS bridge, autotuning.
- Kuhn: replay/data contracts, D6 transforms, graph/candidate/pair contracts, dashboard/debugging, observability.

Assumption: implementation cost is free. The goal is the best possible long-term structure for maintainability, expandability, code quality, and modularity.

## Executive Summary

The project has strong technical pieces, but the current boundaries are too blurry. The biggest issue is not any one model. The issue is that model-family behavior leaks into self-play, inference, replay sampling, dashboard, training, and autotuning through string checks and shared helper code.

The clean target is:

```text
ModelFamily declares what a model is.
InferenceAdapter declares how to run it.
PolicyProvider declares how it supplies MCTS priors.
PairStrategy declares whether and how pair actions are scored.
EngineAdapter declares how priors enter Rust MCTS.
GameRunner plays games without knowing architecture details.
Replay/Contract builders own legal rows, D6, targets, candidates, pairs, and graph rows.
Autotuner mutates typed recipes, not raw config soup.
```

In the best end state, `cnn`, `restnet`, `graph_hybrid_0`, and all true global graph variants are peers registered in one model-family registry. Self-play should not contain architecture string checks. Pair scoring should never happen implicitly. D6 transforms should exist in exactly one place. The dashboard should inspect the same contracts used by self-play and training, not reconstruct its own partial view.

## Current Diagnosis

### The Worker Owns Too Much

`Python/src/hexorl/selfplay/worker.py` currently mixes:

- game loop orchestration
- dense inference
- sparse candidate construction
- global graph construction
- online pair chunk scoring
- tactical oracle integration
- Rust MCTS calls
- replay record construction
- runtime telemetry
- model-family feature gates

This creates a brittle center of gravity. Every new model family or policy head risks becoming another branch in the worker.

Specific current pain:

```python
self.global_graph_enabled = architecture.startswith("global_")
self.pair_policy_enabled = (
    global_graph_enabled and pair_prior_mix > 0
) or pair_head_present
```

That shape makes `global_xattn_0` vulnerable to accidentally doing expensive online pair scoring simply because `pair_prior_mix` is nonzero. Pair behavior should be explicit search policy, not inferred from the architecture string.

### Model Construction Is Too Centralized

`Python/src/hexorl/model/network.py` currently contains shared blocks, dense CNN, RestNet, graph hybrid pieces, sparse heads, pair heads, factory logic, and checkpoint compatibility. `Python/src/hexorl/model/global_graph.py` contains the true global graph variants as modes inside one model class.

This is workable for small iteration, but it becomes fragile when adding:

- global graph variants with different token families
- models with different inference request shapes
- models with optional pair heads
- checkpoint migrations
- family-specific autotune search spaces
- model-specific dashboard inspection

The model registry should become the single source of truth.

### Config Is A Single Bucket

`Python/src/hexorl/config/schema.py` carries dense, RestNet, graph hybrid, global graph, sparse policy, and pair policy knobs in one `ModelConfig`. Global graph names and validation rules appear in multiple places. Loss defaults and model-family assumptions are also partly embedded in schema validation.

This makes config easy to mutate but hard to reason about. A better design is a discriminated `ModelSpec` plus explicit `SearchSpec`, `PairStrategySpec`, `RuntimeSpec`, and `TrainingSpec`.

### Inference Uses Method Growth Instead Of Contracts

The client currently exposes separate calls like:

```text
submit
submit_sparse
submit_sparse_pair
submit_graph
```

The server branches by architecture and request shape. This works, but it couples transport, model semantics, batching, and decoding. It also makes logging too coarse. A stuck calibration can look like "started sweep, then silence" because request-level telemetry is not strong enough.

### Pair Policy Is The Main Performance Trap

There are currently separate full-pair scoring paths for graph and crop/sparse models. Both can fall into:

```text
A * (A - 1) / 2
```

This is fine as a diagnostic or root-only experiment with explicit caps. It is not fine as an implicit default. Full pair scoring should be opt-in and budgeted. Two-stage pair scoring should be the normal pair-aware path.

### Replay, D6, Candidates, Legal Tables, And Graphs Are Split Across Layers

The project already has useful contract pieces:

- replay records
- binary replay codec versioning
- ring buffer projections
- candidate contracts
- tactical oracle results
- global graph batches
- graph schema versions
- dashboard replay/debug views

The problem is that the same concepts are rebuilt in multiple places:

- compact history parsing
- legal move generation
- D6 transforms
- candidate construction
- pair action canonicalization
- graph row construction
- dashboard model-input reconstruction

The desired rule is simple:

```text
self-play, replay sampling, graph building, dashboard, and tests consume the same contract objects.
No layer should parse history, transform D6, build legal tables, or canonicalize pairs by hand.
```

## North Star Architecture

The clean system has these layers:

```text
hexorl/contracts
  Canonical game, action, replay, target, graph, and telemetry contracts.

hexorl/models
  Registered model families, model specs, trunks, heads, checkpoint manifests.

hexorl/inference
  Typed requests, adapters, batching, shared-memory transport, response decoding.

hexorl/search
  Policy providers, pair strategies, MCTS engine adapter, search priors.

hexorl/selfplay
  Game runner, process worker, orchestration, record writing.

hexorl/replay
  Versioned replay records, storage, sampling, training-batch projection.

hexorl/tuning
  Model recipes, search spaces, schedulers, runtime sweeps, scoring.

hexorl/dashboard
  Contract inspector and model-family-aware debug views.
```

This is not just package reshuffling. The important shift is ownership:

| Concern | Current Shape | Target Owner |
|---|---|---|
| Architecture names | String checks in many layers | `ModelFamilyRegistry` |
| Model build | `build_model_from_config()` switch | `ModelFamily.build_model()` |
| Checkpoint compatibility | Scattered load helpers | `CheckpointManager` |
| Inference packing | Client/server methods | `InferenceAdapter` |
| Search priors | Worker branches | `PolicyProvider` |
| Pair scoring | Worker helpers and config side effects | `PairStrategy` |
| Legal rows | Rebuilt in multiple paths | `LegalTableProvider` |
| D6 | Duplicated helpers | `contracts/symmetry.py` |
| Candidates | Central helper plus caller-specific assembly | `CandidateContractBuilder` |
| Graph construction | One large graph batch module | Semantic graph builder plus tensorizer |
| Autotune | Large script owns family behavior | `ModelRecipe` and family search spaces |
| Dashboard | Recomputed partial views | `ContractInspector` |

## Proposed Package Layout

```text
Python/src/hexorl/
  contracts/
    __init__.py
    identity.py
    history.py
    coordinates.py
    symmetry.py
    legal.py
    actions.py
    targets.py
    tactical.py
    candidates.py
    pairs.py
    graph.py
    replay.py
    inference.py
    search.py
    telemetry.py
    validation.py
    debug.py

  models/
    __init__.py
    registry.py
    specs.py
    capabilities.py
    checkpoint.py
    factory.py
    heads/
      __init__.py
      policy.py
      value.py
      sparse_policy.py
      pair_policy.py
      regret.py
    trunks/
      __init__.py
      dense_cnn.py
      restnet.py
      graph_hybrid.py
      global_graph.py
    families/
      __init__.py
      dense_cnn.py
      restnet.py
      graph_hybrid.py
      global_xattn.py
      global_line_window.py
      global_pair_twostage.py
      global_graph_full.py
      global_graph_champion.py

  graph/
    __init__.py
    schema.py
    tokens.py
    token_families.py
    relations.py
    features.py
    semantic_builder.py
    tensorize.py
    collate.py
    debug.py

  inference/
    __init__.py
    protocol.py
    client.py
    server.py
    batching.py
    shm_transport.py
    telemetry.py
    adapters/
      __init__.py
      dense.py
      sparse.py
      sparse_pair.py
      global_graph.py
      pair_scoring.py

  search/
    __init__.py
    context.py
    priors.py
    policy_provider.py
    pair_strategy.py
    engine_adapter.py
    expansion.py
    mcts_runner.py
    telemetry.py

  selfplay/
    __init__.py
    game_runner.py
    worker.py
    orchestrator.py
    records.py
    record_writer.py
    rgsc.py
    telemetry.py

  replay/
    __init__.py
    codec.py
    storage.py
    sampler.py
    projector.py
    migrations.py

  train/
    trainer.py
    adapters.py
    losses.py
    schedules.py

  tuning/
    __init__.py
    recipes.py
    family_spaces.py
    scheduler.py
    runtime_sweep.py
    scoring.py
    manifests.py
    reporting.py

  dashboard/
    contract_inspector.py
    model_inspector.py
    graph_inspector.py
    replay_views.py
```

The existing modules do not need to disappear immediately. The ideal migration is to add the target interfaces first, adapt old code behind them, then gradually delete old direct paths.

## Core Contracts

Contracts are the most important foundation. They should be plain, versioned, hashable where useful, and shared by all layers.

### MoveHistory

```python
@dataclass(frozen=True)
class MoveHistory:
    raw: bytes
    moves: tuple[Move, ...]
    current_player: int
    placements_remaining: int
    radius: int
    history_hash: str
```

This replaces local compact-history parsing in graph, tactical oracle, dashboard, sampler, and worker helpers.

### LegalActionTable

```python
@dataclass(frozen=True)
class LegalActionTable:
    rows: np.ndarray          # shape (A, 2), int32 q/r
    dense_indices: np.ndarray # shape (A,), int32, if applicable
    source: str               # rust_engine | python_fallback | fixture
    radius: int
    occupied_count: int
    table_hash: str
```

The legal table should come from one provider:

```python
class LegalTableProvider:
    def from_history(self, history: MoveHistory) -> LegalActionTable:
        ...
```

Rules:

- Use Rust engine replay first.
- Use Python fallback only in tests or explicit degraded mode.
- Emit telemetry when fallback is used.
- Never let graph, sampler, tactical oracle, or dashboard hand-roll legal rows.

### PositionContract

```python
@dataclass(frozen=True)
class PositionContract:
    identity: PositionIdentity
    history: MoveHistory
    legal: LegalActionTable
    policy_target: PolicyTarget | None
    opponent_policy_target: PolicyTarget | None
    opponent_legal: LegalActionTable | None
    pair_target: PairPolicyTarget | None
    tactical: TacticalReport
    crop: CropWindowContract | None
    telemetry: ContractTelemetry
```

This becomes the root object for training projection, graph building, candidate building, dashboard inspection, and replay validation.

### D6 Symmetry

All D6 code should live in:

```text
hexorl/contracts/symmetry.py
```

It owns:

```text
transform_qr
transform_history
transform_legal_table
transform_policy_target
transform_pair_policy_target
transform_dense_policy
transform_axis_label
transform_axis_maps
apply_tensor_symmetry
compose_symmetries
inverse_symmetry
```

Non-negotiable tests:

```text
transform(a then b) == transform(compose(a, b))
target mass is preserved
legal-table hashes match transformed legal rows
pair identity is preserved for unordered first-placement pairs
ordered second-placement pairs preserve known-first semantics
dashboard inputs match sampler inputs after the same symmetry
```

### CandidateTable

```python
@dataclass(frozen=True)
class CandidateTable:
    rows: np.ndarray           # shape (C, 2), q/r
    dense_indices: np.ndarray  # shape (C,)
    features: np.ndarray       # shape (C, F)
    feature_names: tuple[str, ...]
    mask: np.ndarray           # shape (C,)
    target: np.ndarray | None
    missing_mass: float
    recall: CandidateRecall
    diagnostics: CandidateDiagnostics
    contract_hash: str
```

Built only by:

```python
class CandidateContractBuilder:
    def build(self, position: PositionContract, spec: CandidateSpec) -> CandidateTable:
        ...
```

The builder should be used by self-play, replay sampler, graph builder, dashboard, and model cache.

### PairActionTable

```python
@dataclass(frozen=True)
class PairActionTable:
    rows: np.ndarray              # shape (P, 4): q1,r1,q2,r2
    first_row_indices: np.ndarray # shape (P,)
    second_row_indices: np.ndarray
    phase: str                    # first_placement | second_placement
    known_first: tuple[int, int] | None
    generation: str               # full | topk_first | tactical_only | diagnostic
    total_possible_pairs: int
    selected_pair_rows: int
    table_hash: str
```

This is the canonical representation of pair actions. Crop pair candidates, graph `PAIR_ACTION` rows, pair targets, search pair priors, and dashboard views should all use this type.

## Model Family Registry

Model construction should be owned by a registry of model families.

```python
class ModelFamily(Protocol):
    name: str
    aliases: set[str]
    capabilities: CapabilitySet

    def validate_spec(self, spec: ModelSpec) -> None: ...
    def build_model(self, spec: ModelSpec) -> nn.Module: ...
    def build_train_adapter(self, spec: ModelSpec) -> TrainAdapter: ...
    def build_inference_adapter(self, spec: ModelSpec) -> InferenceAdapter: ...
    def build_policy_provider(self, spec: ModelSpec, runtime: RuntimeSpec) -> PolicyProvider: ...
    def default_loss_plan(self, spec: ModelSpec) -> LossPlan: ...
    def default_recipe(self, host: HostProfile) -> ModelRecipe: ...
    def tune_space(self, host: HostProfile) -> RecipeSearchSpace: ...
```

### ModelSpec

Replace one large model config bucket with discriminated specs.

```python
@dataclass(frozen=True)
class DenseCnnSpec:
    kind: Literal["cnn"]
    channels: int
    blocks: int
    input_contract: str
    heads: HeadBundleSpec

@dataclass(frozen=True)
class RestNetSpec:
    kind: Literal["restnet"]
    channels: int
    blocks: int
    attention_positions: tuple[int, ...]
    input_contract: str
    heads: HeadBundleSpec

@dataclass(frozen=True)
class GraphHybridSpec:
    kind: Literal["graph_hybrid_0"]
    channels: int
    blocks: int
    sparse_token_budget: int
    candidate_contract: CandidateSpec
    heads: HeadBundleSpec

@dataclass(frozen=True)
class GlobalGraphSpec:
    kind: Literal[
        "global_xattn_0",
        "global_line_window_0",
        "global_pair_twostage_0",
        "global_graph_full_0",
        "global_graph768_champion",
    ]
    d_model: int
    layers: int
    heads: int
    token_spec: GraphTokenSpec
    relation_spec: GraphRelationSpec
    legal_action_contract: str
    heads: HeadBundleSpec
```

### Capabilities

Capabilities make behavior explicit.

```python
class Capability(Enum):
    DENSE_PLACE_POLICY = "dense_place_policy"
    SPARSE_PLACE_POLICY = "sparse_place_policy"
    GLOBAL_PLACE_POLICY = "global_place_policy"
    PAIR_FIRST_POLICY = "pair_first_policy"
    PAIR_SECOND_POLICY = "pair_second_policy"
    JOINT_PAIR_POLICY = "joint_pair_policy"
    REGRET_HEAD = "regret_head"
    GLOBAL_GRAPH_INPUT = "global_graph_input"
    CROP_INPUT = "crop_input"
```

Example family declarations:

```text
DenseCnnFamily:
  input = crop_tensor
  outputs = dense_policy, value
  capabilities = dense_place_policy

RestNetFamily:
  input = crop_tensor
  outputs = dense_policy, value, optional aux heads
  capabilities = dense_place_policy

GraphHybridFamily:
  input = crop_tensor + sparse candidate rows
  outputs = dense_policy, sparse_policy, optional pair policy
  capabilities = dense_place_policy, sparse_place_policy, optional crop_pair_rows

GlobalXattnFamily:
  input = global_graph
  outputs = global place policy, value
  capabilities = global_place_policy
  default pair_strategy = none

GlobalPairTwoStageFamily:
  input = global_graph
  outputs = place policy, pair first, pair second
  capabilities = global_place_policy, pair_first_policy, pair_second_policy
  default pair_strategy = two_stage_root_only
```

This prevents accidental behavior. A model can have a pair-capable head, but pair scoring only occurs when a `PairStrategy` explicitly asks for it.

## Model Assembly

Model assembly should be split into:

```text
trunk
heads
family wrapper
adapter
recipe
checkpoint manifest
```

### Trunks

```text
DenseCnnTrunk
RestNetTrunk
GraphHybridTrunk
GlobalGraphEncoder
```

These only compute features. They do not know about self-play, replay, MCTS, or inference transport.

### Heads

```text
DensePolicyHead
SparsePolicyHead
ValueHead
PairFirstHead
PairSecondHead
JointPairHead
RegretHead
MovesLeftHead
```

Heads define:

```text
input feature type
output contract
loss contract
masking behavior
dashboard debug labels
```

### Family Wrapper

The family wrapper composes trunk and heads and exposes one clean `forward()` for its input contract.

Example:

```python
class GlobalGraphModel(nn.Module):
    def __init__(self, encoder: GlobalGraphEncoder, heads: HeadBundle):
        ...

    def forward(self, batch: GlobalGraphTensorBatch) -> GlobalGraphOutput:
        ...
```

Avoid passing generic dicts through the model core. Dicts are fine at process boundaries, but internal contracts should be typed.

## Checkpoint Redesign

Introduce:

```python
class CheckpointManager:
    def save(self, bundle: CheckpointBundle, path: Path) -> None: ...
    def load(self, path: Path, *, purpose: LoadPurpose, device: str) -> LoadedCheckpoint: ...
    def inspect(self, path: Path) -> CheckpointManifest: ...
    def migrate(self, path: Path, target_version: int) -> Path: ...
```

Every checkpoint should include a manifest:

```yaml
checkpoint_schema_version: 3
model_family: global_xattn_0
model_spec_version: 1
model_spec:
  kind: global_xattn_0
  d_model: 384
  layers: 4
  heads: 6
input_contract: global_graph_v1
output_contract: global_place_value_v1
action_contract: legal_action_table_v1
graph_schema_version: 1
relation_schema_version: 1
heads:
  - policy_place
  - value
pair_strategy_used: none
training_target_contract: policy_v2_pair_optional_v1
state_dict_key_format: torch_compile_stripped_v1
created_by:
  git_sha: ...
  command: ...
  config_hash: ...
```

Loading rules:

- Strict load by default.
- Partial load only through an explicit migration or transfer-learning path.
- `_orig_mod` stripping is a named compatibility transform, not silent magic.
- Runtime refuses to load a checkpoint if the requested `PolicyProvider` cannot satisfy the checkpoint output contract.
- Dashboard and arena evaluation use the same `CheckpointManager` as training.

## Inference Redesign

Inference should be divided into semantic adapters and transport.

### Typed Requests

```python
@dataclass(frozen=True)
class EvalRequest:
    trace_id: str
    model_family: str
    request_kind: str
    schema_version: int
    phase: str               # root | leaf | pair_chunk | calibration
    position: PositionIdentity
    payload: EvalPayload
    caps: RequestCaps
```

Concrete request payloads:

```text
DenseEvalPayload
SparseCandidateEvalPayload
GlobalGraphEvalPayload
PairScoringEvalPayload
RegretEvalPayload
```

### Typed Responses

```python
@dataclass(frozen=True)
class EvalResponse:
    trace_id: str
    model_family: str
    output_contract: str
    value: float
    place_logits: PlaceLogits | None
    sparse_logits: SparseLogits | None
    pair_logits: PairLogits | None
    telemetry: EvalTelemetry
    warnings: tuple[str, ...]
```

### InferenceAdapter

```python
class InferenceAdapter(Protocol):
    request_kind: str
    input_contract: str
    output_contract: str

    def validate_request(self, request: EvalRequest) -> None: ...
    def pack(self, request: EvalRequest, slot: ShmSlot) -> None: ...
    def collate(self, slots: list[ShmSlot]) -> ModelBatch: ...
    def forward(self, model: nn.Module, batch: ModelBatch) -> ModelOutput: ...
    def scatter(self, output: ModelOutput, slots: list[ShmSlot]) -> None: ...
    def decode(self, slot: ShmSlot) -> EvalResponse: ...
```

The server should route by `request_kind`, not by `architecture.startswith("global_")`.

### Inference Telemetry

Every request should emit or aggregate:

```text
request_kind
model_family
phase
worker_id
game_id
move_idx
legal_count
candidate_count
pair_rows_total
pair_rows_scored
graph_token_count
graph_relation_count
graph_edge_density
payload_bytes
queue_wait_ms
ipc_pack_ms
server_wait_ms
collate_ms
model_forward_ms
scatter_ms
client_wait_ms
postprocess_ms
```

This is the difference between "xattn froze" and "xattn spent 97 percent of time scoring 48,000 pair rows in pair chunks".

## Search And Self-Play Redesign

Self-play should be mostly process lifecycle and game-loop control. It should not know model internals.

### SearchContext

```python
@dataclass(frozen=True)
class SearchContext:
    position: PositionContract
    phase: str                    # root | leaf
    move_idx: int
    engine_snapshot: EngineSnapshot
    search_limits: SearchLimits
    trace_id: str
```

### SearchEvaluation

```python
@dataclass(frozen=True)
class SearchEvaluation:
    legal: LegalActionTable
    value: float
    place_logits: np.ndarray
    place_source: str
    dense_policy: np.ndarray | None = None
    sparse_rows: CandidateTable | None = None
    pair_priors: PairPriorResult | None = None
    telemetry: EvaluationTelemetry = field(default_factory=EvaluationTelemetry)
```

### PolicyProvider

```python
class PolicyProvider(Protocol):
    def evaluate_root(self, ctx: SearchContext) -> SearchEvaluation:
        ...

    def evaluate_leaves(self, ctxs: list[SearchContext]) -> list[SearchEvaluation]:
        ...
```

Concrete providers:

```text
DensePolicyProvider
RestNetPolicyProvider
GraphHybridPolicyProvider
GlobalGraphPolicyProvider
```

The provider owns model-specific details:

- input building
- inference request type
- action-row decoding
- fallback behavior
- output contract validation
- telemetry

The worker only receives `SearchEvaluation`.

### EngineAdapter

Keep the existing Rust primitives, but hide them behind one Python adapter.

```python
class EngineAdapter:
    def start_position(self, history: MoveHistory, limits: SearchLimits) -> SearchContext:
        ...

    def expand_root(
        self,
        ctx: SearchContext,
        evaluation: SearchEvaluation,
        pair_result: PairPriorResult | None,
    ) -> None:
        ...

    def select_leaf_contexts(self, max_batch: int) -> list[SearchContext]:
        ...

    def expand_leaves(
        self,
        ctxs: list[SearchContext],
        evaluations: list[SearchEvaluation],
    ) -> None:
        ...

    def collect_result(self) -> RootSearchResult:
        ...
```

The Rust MCTS bridge already has useful primitives for dense, sparse, global, pair-first, pair-joint, and pair-second priors. The cleanup is mostly on the Python side: the worker should not call different Rust methods based on model architecture.

### GameRunner

Target shape:

```python
class GameRunner:
    def play_one_game(self) -> ReplayGame:
        state = self.engine.start_new_game()

        while not state.done:
            root_ctx = self.engine.start_position(state.history, self.search_limits)
            root_eval = self.policy_provider.evaluate_root(root_ctx)
            pair_eval = self.pair_strategy.score_root(root_ctx, root_eval)
            self.engine.expand_root(root_ctx, root_eval, pair_eval)

            while self.engine.needs_leaf_eval():
                leaf_ctxs = self.engine.select_leaf_contexts(self.runtime.leaf_batch_size)
                leaf_evals = self.policy_provider.evaluate_leaves(leaf_ctxs)
                leaf_pair_evals = self.pair_strategy.score_leaves(leaf_ctxs, leaf_evals)
                self.engine.expand_leaves(leaf_ctxs, leaf_evals, leaf_pair_evals)

            decision = self.engine.collect_result()
            state = state.apply(decision.action)
            self.record_builder.capture(state, decision)

        return self.record_builder.finish()
```

This is the desired dependency direction:

```text
GameRunner -> PolicyProvider -> InferenceAdapter -> ModelFamily
GameRunner -> PairStrategy -> PolicyProvider/InferenceAdapter
GameRunner -> EngineAdapter -> Rust MCTS
```

## Pair Strategy Redesign

Pair behavior must be explicit, independent, and budgeted.

### PairStrategySpec

```python
@dataclass(frozen=True)
class PairStrategySpec:
    name: Literal[
        "none",
        "pair_first_only",
        "two_stage_root_only",
        "tactical_only",
        "diagnostic_full_root",
        "split_turn_search",
    ]
    first_top_k: int = 16
    max_pair_rows_root: int = 8192
    max_pair_rows_leaf: int = 0
    tactical_force: bool = True
    full_pair_when_legal_lte: int = 64
    pair_prior_mix: float = 0.35
```

### Strategies

```text
NoPairStrategy
  Default for CNN, RestNet, graph_hybrid_0, and global_xattn_0.
  Scores no pairs.

PairFirstOnlyStrategy
  Scores first-placement pressure, not joint pairs.
  Useful for line/window global variants.

TwoStageRootOnlyStrategy
  Scores all first moves or first-top-K.
  Keeps tactical first moves protected.
  Scores legal second moves only under selected first moves.
  Caps total pair rows.

TacticalOnlyPairStrategy
  Scores pairs only when tactical oracle says pair structure matters.
  Good for speed-sensitive runs.

DiagnosticFullRootPairStrategy
  Full A * (A - 1) / 2 pair scoring at root only.
  Requires explicit cap and should be used for measurement, not default training.

SplitTurnSearchStrategy
  Long-term best structure.
  MCTS models first placement and same-player second placement as separate decisions.
  Avoids collapsed pair scoring as the primary online mechanism.
```

### Recommended Defaults

| Family | Default Pair Strategy |
|---|---|
| `cnn` | `none` |
| `restnet` | `none` |
| `graph_hybrid_0` | `none`, unless recipe explicitly selects pair head |
| `global_xattn_0` | `none` |
| `global_line_window_0` | `pair_first_only` or `none` |
| `global_pair_twostage_0` | `two_stage_root_only` |
| `global_graph_full_0` | explicit only |
| `global_graph768_champion` | explicit only |

### Critical Invariant

```text
No full A * (A - 1) / 2 online pair scoring happens unless PairStrategy explicitly enables it.
```

This single invariant would prevent the current class of performance surprises.

## Global Graph Redesign

Global graph support should split semantic construction from tensorization.

### Semantic Layer

```python
@dataclass(frozen=True)
class GraphSemanticContract:
    schema_version: int
    position: PositionIdentity
    tokens: TokenTable
    legal_refs: LegalActionTable
    pair_refs: PairActionTable | None
    relations: RelationTable
    debug_labels: GraphDebugLabels
    trace: ContractTrace
```

This layer is human-readable and dashboard-friendly.

### Tensor Layer

```python
@dataclass(frozen=True)
class GlobalGraphTensorBatch:
    token_features: torch.Tensor
    token_type_ids: torch.Tensor
    token_masks: torch.Tensor
    relation_bias: torch.Tensor | None
    legal_action_rows: torch.Tensor
    pair_action_rows: torch.Tensor | None
    batch_index: torch.Tensor
```

This layer is model-friendly.

### Module Split

```text
hexorl/graph/schema.py
  TOKEN_SCHEMA_VERSION
  RELATION_SCHEMA_VERSION
  GRAPH_FEATURE_SCHEMA_VERSION
  feature names and token-family ids

hexorl/graph/tokens.py
  state, turn, player, stone, legal, hot-cell, window6, line, cover-set, component, pair-action tokens

hexorl/graph/relations.py
  relation builders and relation names

hexorl/graph/semantic_builder.py
  PositionContract -> GraphSemanticContract

hexorl/graph/tensorize.py
  GraphSemanticContract -> GlobalGraphTensorBatch

hexorl/graph/collate.py
  batch padding and masks

hexorl/graph/debug.py
  labels and dashboard payloads
```

### Token Family Policy

Token families should be feature-configured, not architecture-hardcoded:

```yaml
token_spec:
  state: true
  turn: true
  player: true
  stones: true
  legal_actions: true
  hot_cells:
    enabled: true
    budget: 96
  window6:
    enabled: false
  lines:
    enabled: true
    max_lines: 128
  cover_sets:
    enabled: true
    max_sets: 64
  components:
    enabled: false
  pair_actions:
    enabled: false
```

This lets global variants differ by declared token contract:

```text
global_xattn_0:
  lean global tokens, legal actions, no pair rows by default

global_line_window_0:
  line/window structure emphasized, optional first-placement pair signal

global_pair_twostage_0:
  pair-first and conditional second outputs, no full joint pair default

global_graph_full_0:
  richer tokens and relations, explicit high-cost recipe

global_graph768_champion:
  large model preset, explicit runtime and memory profile
```

## Replay And Sampling Redesign

The current record structure should evolve into nested, versioned records.

### ReplayPositionV3

```python
@dataclass(frozen=True)
class ReplayPositionV3:
    identity: PositionIdentity
    state: StateSnapshotRef
    search: SearchTargets
    training: TrainingTargets
    diagnostics: PositionDiagnostics
    telemetry: PositionTelemetry
```

### SearchTargets

```python
@dataclass(frozen=True)
class SearchTargets:
    policy_v2: PolicyTarget
    pair_policy_v2: PairPolicyTarget | None
    root_value: float
    selected_action_value: float | None
    root_prior_sources: PriorSourceSummary
```

### TrainingTargets

```python
@dataclass(frozen=True)
class TrainingTargets:
    dense_policy: np.ndarray | None
    opponent_policy: PolicyTarget | None
    value: float
    value_weight: float
    policy_weight: float
    regret: RegretTarget | None
    axis: AxisTarget | None
    moves_left: MovesLeftTarget | None
```

### ReplayStorage

Instead of hand-maintaining a very wide struct-of-arrays for every field, store:

```text
blob_store:
  exact versioned ReplayPosition records

index_arrays:
  game_id
  move_idx
  player
  value_weight
  policy_weight
  is_full_search
  model_family
  flags needed for fast sampling

side_arrays:
  optional variable-length target offsets for hot columns
```

This gives fast sampling without duplicating every semantic field in ring-buffer layout.

### ReplayDataset

The sampler should become a thin orchestrator:

```text
sample record ids
decode ReplayPositionV3
apply SymmetryTransform
build PositionContract
call TargetProjector
call CandidateContractBuilder if needed
call PairActionTableBuilder if needed
call GraphSemanticBuilder if needed
tensorize for selected ModelFamily
return TrainingBatch
```

The sampler should not own private D6, legal fallback, candidate assembly, pair second-placement handling, graph construction, and axis transforms all at once.

## Training Redesign

Training should be model-family-agnostic.

```python
class TrainAdapter(Protocol):
    input_contract: str
    target_contract: str

    def prepare_batch(self, raw: ReplayBatch, device: torch.device) -> ModelTrainBatch:
        ...

    def compute_losses(
        self,
        outputs: ModelOutput,
        targets: TrainingTargets,
        loss_plan: LossPlan,
    ) -> LossBundle:
        ...
```

Trainer loop target:

```python
batch = train_adapter.prepare_batch(raw_batch, device)
outputs = model(batch.inputs)
losses = train_adapter.compute_losses(outputs, batch.targets, loss_plan)
losses.total.backward()
```

The trainer should not branch on global graph model classes. The family adapter owns batch projection and loss wiring.

## Autotuning Redesign

Autotune should operate on typed recipes.

### ModelRecipe

```python
@dataclass(frozen=True)
class ModelRecipe:
    name: str
    family: str
    model_spec: ModelSpec
    input_spec: InputSpec
    head_bundle: HeadBundleSpec
    policy_provider: str
    pair_strategy: PairStrategySpec
    search: SearchSpec
    training: TrainingSpec
    runtime: RuntimeSpec
    telemetry: TelemetrySpec
```

Example:

```yaml
name: global_xattn_small_no_pair
family: global_xattn_0
model_spec:
  kind: global_xattn_0
  d_model: 384
  layers: 4
  heads: 6
input_spec:
  token_preset: global_minimal_256
head_bundle:
  policy_place: true
  value: true
  regret: light
policy_provider: global_graph
pair_strategy:
  name: none
search:
  full_sims: 512
  policy_top_k: 96
runtime:
  profile: graph_low_mem
  workers: 1
```

### Family Search Spaces

Each family contributes:

```python
def default_recipes(host: HostProfile) -> list[ModelRecipe]: ...
def search_space(host: HostProfile) -> RecipeSearchSpace: ...
def validate_recipe(recipe: ModelRecipe, host: HostProfile) -> RecipeValidation: ...
def runtime_candidates(recipe: ModelRecipe, host: HostProfile) -> list[RuntimeCandidate]: ...
```

The Phase 3 script should not hardcode family internals. It should ask the registry for recipes and allowed mutations.

### Search Knobs By Owner

```text
ModelFamily owns:
  channels, layers, d_model, token budgets, head bundle options

PolicyProvider owns:
  candidate budget, policy_top_k, sparse/global prior source

PairStrategy owns:
  pair mode, first_top_k, pair row caps, root/leaf enablement

SearchSpec owns:
  sims, c_puct, temperature, Dirichlet, root exploration

TrainingSpec owns:
  lr, batch size, loss weights, augmentation, replay weighting

RuntimeSpec owns:
  workers, batch size, inference queue sizes, memory mode
```

This makes the autotuner explainable. It also prevents impossible combinations, like a non-pair model silently running a pair-heavy strategy.

## Observability Redesign

### ContractTrace

```python
@dataclass(frozen=True)
class ContractTrace:
    trace_id: str
    history_hash: str
    model_family: str
    phase: str
    legal_count: int
    candidate_count: int
    pair_rows_total: int
    pair_rows_scored: int
    graph_token_count: int
    graph_relation_count: int
    timings_ms: dict[str, float]
    warnings: tuple[str, ...]
```

### Timing Spans

Emit these spans from the shared builders and adapters:

```text
history_parse_ms
engine_replay_ms
legal_table_ms
tactical_oracle_ms
candidate_build_ms
pair_table_build_ms
graph_token_build_ms
graph_relation_build_ms
graph_tensorize_ms
ipc_pack_ms
ipc_wait_ms
queue_wait_ms
collate_ms
model_forward_ms
scatter_ms
decode_ms
pair_chunk_count
pair_chunk_forward_ms
```

### Worker Heartbeat

Every worker should emit periodic progress:

```json
{
  "event": "selfplay_worker_heartbeat",
  "worker_id": 0,
  "game_id": "trial42-w0-g17",
  "move_idx": 37,
  "phase": "leaf_eval",
  "positions_done": 412,
  "last_progress_s": 2.1
}
```

### Policy Evaluation Timing

```json
{
  "event": "policy_eval_timing",
  "family": "global_xattn_0",
  "phase": "root",
  "legal_count": 284,
  "candidate_count": 0,
  "token_count": 512,
  "relation_count": 4096,
  "pair_rows": 0,
  "graph_build_ms": 8.4,
  "ipc_wait_ms": 41.2,
  "forward_ms": 17.6
}
```

### Pair Strategy Summary

```json
{
  "event": "pair_strategy_summary",
  "strategy": "two_stage_root_only",
  "legal_count": 312,
  "full_pair_rows_possible": 48516,
  "first_candidates": 18,
  "pair_rows_scored": 5616,
  "chunks": 2,
  "elapsed_ms": 94.0,
  "skipped_reason": null
}
```

### Runtime Sweep Watchdog

Runtime calibration should abort and explain no-progress probes:

```json
{
  "event": "runtime_sweep_no_progress",
  "trial_id": "phase3-global-xattn-small",
  "candidate": {
    "family": "global_xattn_0",
    "workers": 1,
    "batch_size": 1
  },
  "last_policy_eval": {
    "phase": "root",
    "legal_count": 329,
    "pair_rows_scored": 0,
    "graph_token_count": 768,
    "last_span": "model_forward_ms"
  },
  "worker_states": [],
  "action": "abort_probe"
}
```

This makes slow global graph calibration debuggable.

## Dashboard Redesign

Add:

```python
class ContractInspector:
    def inspect_history(
        self,
        history: MoveHistory,
        *,
        targets: SearchTargets | None,
        model_spec: ModelSpec | None,
    ) -> ContractDebugPayload:
        ...
```

Dashboard routes/views:

```text
/history
/legal-table
/tactical
/candidates
/pairs
/graph
/d6
/model-input
/model-output
/trace
```

For each inspected position, show:

```text
history hash
legal table hash
candidate table hash
pair table hash
graph schema version
token counts by family
relation counts by type
target missing mass
critical overflow examples
pair strategy and row cap
build timings
source: rust engine or fallback
model input contract
model output contract
checkpoint manifest
```

The dashboard should not import private sampler helpers or reconstruct model inputs through special-case paths. It should use the same contract builders and model-family adapters as training/inference.

## Testing Strategy

Organize tests by invariant, not by implementation file.

```text
tests/contracts/test_history_codec.py
tests/contracts/test_symmetry_group.py
tests/contracts/test_legal_table_provider.py
tests/contracts/test_policy_targets.py
tests/contracts/test_pair_targets.py
tests/contracts/test_candidate_contract.py
tests/contracts/test_tactical_contract.py
tests/contracts/test_graph_contract.py
tests/contracts/test_contract_parity.py

tests/models/test_model_registry.py
tests/models/test_checkpoint_manifest.py
tests/models/test_family_capabilities.py

tests/inference/test_dense_adapter.py
tests/inference/test_sparse_adapter.py
tests/inference/test_global_graph_adapter.py
tests/inference/test_pair_scoring_adapter.py

tests/search/test_policy_providers.py
tests/search/test_pair_strategies.py
tests/search/test_engine_adapter.py

tests/replay/test_replay_v3_codec.py
tests/replay/test_sampler_projection.py

tests/tuning/test_recipe_validation.py
tests/tuning/test_runtime_watchdog.py

tests/dashboard/test_contract_inspector.py
```

Golden fixtures:

```text
empty board
opening after origin
first-placement turn
second-placement turn with known first move
outside-crop tactical win
forced two-placement cover
large legal table
pair-heavy state
global graph token-heavy state
symmetry-sensitive line state
```

Critical parity tests:

```text
self-play candidate table == sampler candidate table == dashboard candidate table
graph legal table == Rust legal table
D6(history + targets) == D6(rebuilt graph)
pair target mass preserved under D6
dashboard model-cache inputs match training inputs
no full pair enumeration unless PairStrategy explicitly allows it
global_xattn_0 defaults to no pair rows
checkpoint manifest round-trips for every family
autotune rejects incompatible recipe combinations
```

## Migration Plan

Even with implementation cost treated as free, order matters because this system is performance-sensitive.

### Phase 0: Freeze And Observe

Add non-invasive telemetry around current paths:

- worker heartbeat
- policy eval timing
- pair strategy summary
- graph request summary
- runtime sweep no-progress watchdog

This provides before/after evidence and immediately helps diagnose `global_xattn_0` and `graph_hybrid_0` slowdowns.

### Phase 1: Contracts First

Add:

```text
contracts/history.py
contracts/legal.py
contracts/symmetry.py
contracts/targets.py
contracts/candidates.py
contracts/pairs.py
contracts/telemetry.py
```

Redirect graph, sampler, dashboard, tactical oracle, and self-play to shared history/legal/D6 helpers.

Do not redesign models yet.

### Phase 2: Candidate, Pair, And Graph Builders

Add:

```text
CandidateContractBuilder
PairActionTableBuilder
GraphSemanticBuilder
GraphTensorizer
```

Make self-play, sampler, dashboard, and graph training use the same builders.

### Phase 3: Model Registry And Checkpoints

Add:

```text
ModelFamilyRegistry
ModelSpec union
CapabilitySet
CheckpointManifest
CheckpointManager
```

Keep old `build_model_from_config()` as a compatibility wrapper that converts old config to a `ModelSpec`.

### Phase 4: Inference Adapters

Introduce adapters behind the current shared-memory transport:

```text
DenseInferenceAdapter
SparseInferenceAdapter
SparsePairInferenceAdapter
GlobalGraphInferenceAdapter
```

The external process architecture can stay. The request contract changes first; transport internals can evolve later.

### Phase 5: Policy Providers And Pair Strategies

Extract worker branches into:

```text
DensePolicyProvider
RestNetPolicyProvider
GraphHybridPolicyProvider
GlobalGraphPolicyProvider

NoPairStrategy
TwoStageRootOnlyPairStrategy
DiagnosticFullRootPairStrategy
TacticalOnlyPairStrategy
```

Set the default pair strategy to `none` unless a recipe explicitly says otherwise.

### Phase 6: GameRunner And EngineAdapter

Shrink `SelfPlayWorker` to lifecycle and IPC concerns. Move game logic to `GameRunner` and Rust calls to `EngineAdapter`.

Target invariant:

```text
SelfPlayWorker contains no architecture string checks.
```

### Phase 7: Recipe-Based Autotuning

Move family search space out of the Phase 3 script into:

```text
tuning/recipes.py
tuning/family_spaces.py
tuning/runtime_sweep.py
tuning/scoring.py
```

The script becomes an orchestration entrypoint, not the owner of family behavior.

### Phase 8: Replay V3 And Dashboard Inspector

Add `ReplayPositionV3`, migration tools, and `ContractInspector`.

Keep V2 decode compatibility until existing runs are no longer needed.

### Phase 9: Delete Old Paths

Remove:

- direct architecture branches in worker/server/trainer/dashboard/autotune
- duplicate D6 helpers
- manual legal row generation outside `LegalTableProvider`
- pair chunk helpers from worker
- dashboard private imports from sampler internals

## Specific Fixes This Enables

### Global Graph Policy Head Completion

`Docs/SPEC_FIX_MATCH_GLOBAL_GRAPH_MODEL_20260428.md` requires four graph policy heads:

```text
policy_place
policy_pair_first
policy_pair_second
policy_pair_joint
```

The current implementation has code paths for all four names, but they are not all complete as architecture-level contracts.

Required completion status:

| Head | Current Status | Missing Requirement |
|---|---|---|
| `policy_place` | Mostly implemented. `GlobalHexGraphNet` returns logits over `LEGAL` rows and inference/self-play consumes them as global keyed priors. | Promote this to the formal primary graph policy contract everywhere. Dense `0..1088` policy must remain absent from true global graph models except diagnostics. |
| `policy_pair_first` | Partially implemented. The model emits `(B, A)` logits over legal rows and self-play can blend them when pair policy is enabled. | Gate it by `PairStrategy`, validate turn semantics, train it against a first-placement projection only when the position has a meaningful first-placement pair target, and report prior-source telemetry. |
| `policy_pair_second` | Partially implemented. The model can score supplied `(first, second)` row references and returns logits with shape `(B, P)`, often used as conditional second-placement logits. | Define the formal conditional contract: given a selected first placement, expose a masked distribution over legal second placements equivalent to `(B, A_second)`. The implementation may use selected `PAIR_ACTION` rows internally, but the adapter must present and validate the conditional legal table. |
| `policy_pair_joint` | Partially implemented. The model can score provided pair rows and inference can return `(B, P)` logits. | Make this the explicit joint pair-turn prior only for recipes that enable a pair strategy. It must use canonical `PairActionTable` rows, row caps, turn-aware masks, D6-equivariant targets, active MCTS consumption, and telemetry. It must not be confused with the crop-compatible `PairPolicyHead`. |

The redesign must treat these as first-class output contracts, not merely optional tensor keys. A model family should declare which heads it supports, an inference adapter should validate the output shape and legal-row mapping, and a `PairStrategy` should decide whether those heads are consumed by MCTS.

Acceptance requirements:

```text
policy_place returns exactly one logit per legal action row.
policy_pair_first returns exactly one logit per legal first-placement row.
policy_pair_second exposes a conditional legal-second distribution for a known first placement.
policy_pair_joint returns exactly one logit per canonical pair-action row.
PAIR_ACTION rows are built by a canonical PairActionTable.
Full joint pair scoring is opt-in and capped.
Second-placement positions use the legal table after the first placement.
Opening positions have no pair prior and no pair loss.
Pair losses are finite, masked, D6-equivariant, and turn-aware.
MCTS telemetry reports whether pair_first, pair_second, or pair_joint influenced the decision.
The crop-compatible PairPolicyHead is documented as an auxiliary candidate-pair scorer, not the final global graph pair head.
```

### Global X-Attn Freeze Diagnosis

With request-level telemetry and a no-progress watchdog, a freeze can be classified as:

```text
graph build slow
relation build slow
IPC wait deadlock
model forward slow
pair scoring accidentally enabled
worker stalled before inference
server stalled after inference
```

Today, those look too similar.

### Graph Hybrid Slowdown

Graph hybrid performance can be attributed by span:

```text
candidate_build_ms
sparse_request_pack_ms
model_forward_ms
pair_rows_scored
pair_chunk_forward_ms
engine_expand_ms
```

If pair scoring is enabled, the `PairStrategySummary` will show row counts and caps. If not, the slowdown has to be elsewhere.

### Global Graph Versus Graph Hybrid Clarity

The registry makes the distinction explicit:

```text
graph_hybrid_0:
  crop tensor + sparse candidate tokens
  compatible with CNN-style board trunk
  local/crop anchored

global_xattn_0 and friends:
  full global graph input
  legal action rows directly represented
  token/relation schema driven
```

The names stop being loosely similar strings and become different input/output contracts.

### Safer Pair Modeling

Pair-aware experiments become explicit recipes:

```yaml
pair_strategy:
  name: two_stage_root_only
  first_top_k: 16
  max_pair_rows_root: 8192
  max_pair_rows_leaf: 0
```

This keeps pair work measurable and prevents accidental quadratic cost.

### Easier New Architecture Work

To add a new model, implement:

```text
ModelSpec
ModelFamily
InferenceAdapter
TrainAdapter
PolicyProvider
default recipe/search space
checkpoint manifest fields
dashboard inspector hooks
```

No edits should be required in worker, trainer, inference server core, replay sampler core, or autotune scheduler core.

## Non-Negotiable Invariants

```text
SelfPlayWorker contains no architecture string checks.
InferenceServer dispatches by request kind, not architecture string.
No pair scoring happens unless PairStrategy explicitly enables it.
No full A * (A - 1) / 2 pair scoring is allowed without a row cap.
global_xattn_0 default pair strategy is none.
D6 transforms exist in one module.
Legal rows come from LegalTableProvider.
Candidate construction exists in one builder.
Pair action rows exist in one canonical table type.
Graph semantic construction is separate from graph tensorization.
Every model has declared input and output contracts.
Every checkpoint has a manifest.
Every inference response includes telemetry and schema versions.
Every runtime sweep has a no-progress watchdog.
Autotune mutates ModelRecipe, not unrelated raw fields.
Dashboard inspects contracts instead of reconstructing private approximations.
```

## Acceptance Criteria

The redesign is complete when these are true:

1. A new architecture can be added without editing `SelfPlayWorker`.
2. A new architecture can be added without editing the core inference server dispatch loop.
3. `global_xattn_0` can run calibration with pair rows reported as zero by default.
4. Full pair scoring requires an explicit `DiagnosticFullRootPairStrategy`.
5. Dashboard and sampler produce identical candidate tables for a golden position.
6. D6 transforms pass group-action tests across history, dense policy, sparse targets, pair targets, and graph contracts.
7. Every checkpoint can be inspected without loading model weights.
8. Autotune can list recipes and explain why each recipe is valid or rejected.
9. A no-progress runtime sweep emits the last known phase and timing span.
10. Old replay records can be migrated or decoded through a versioned compatibility layer.

## Final Target Shape

The clean flow is:

```text
MoveHistory
-> LegalActionTable
-> TacticalReport
-> PositionContract
-> CandidateTable / PairActionTable / GraphSemanticContract
-> ModelFamily TrainAdapter or InferenceAdapter
-> PolicyProvider
-> PairStrategy
-> EngineAdapter
-> ReplayPositionV3
-> TrainingBatch / Dashboard ContractInspector
```

That structure gives Hexo one source of truth for legal rows, D6, tactical labels, candidates, pairs, graph rows, model contracts, and debug payloads. It makes graph hybrid, true global graph, and future pair-aware models peers instead of special cases. It also makes performance problems explainable: every slow path gets a named owner, a timing span, a contract, and a testable invariant.
