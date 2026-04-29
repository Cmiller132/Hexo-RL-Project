# Modular Hexo Architecture Redesign V2

Date: 2026-04-29

Status: design proposal, breaking refactor

Scope: project structure, Rust/Python rule boundary, model registry, model assembly, inference, self-play, MCTS integration, pair policy, global graph support, replay/data contracts, training, evaluation, autotuning, observability, dashboard/debugging, tests, CI, and dead-code removal.

Supersedes: `Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_20260429.md`

Intent: this is not a compatibility-preserving reshuffle. This is a cohesive project cleanup. Legacy paths, duplicate helpers, deprecated aliases, and stale experiments should be removed unless they are actively needed by the new architecture. Existing runs/checkpoints/replay files can be archived before the cutover, but the main runtime should not carry long-lived compatibility shims.

## Executive Summary

Hexo has the right technical pieces, but the current system is difficult to reason about because ownership is spread across many layers:

- model-family behavior leaks into self-play, inference, replay sampling, dashboard, training, runtime sizing, evaluation, and autotuning
- pair scoring can be enabled by architecture/config side effects instead of an explicit search strategy
- D6, compact history, legal moves, candidates, pair actions, graph rows, and checkpoint cleanup are implemented in multiple places
- Rust is the rules source of truth, but Python has several fallback rule implementations with different semantics
- dashboard and training rebuild model inputs through separate private paths
- CI and developer workflow do not yet prove the proposed architecture invariants

The V2 target is:

```text
Rust rules define legal game state.
Python contracts define versioned data flowing between subsystems.
Model families declare input/output capabilities.
Adapters translate contracts to model/training/inference tensors.
Policy providers and pair strategies decide what search consumes.
Game runners play games without architecture knowledge.
Replay, dashboard, training, and evaluation inspect the same contracts.
Dead code is deleted as each owner is established.
```

The central rule:

```text
No subsystem should infer behavior from an architecture string.
No subsystem should rebuild legal rows, D6 transforms, compact history, candidates, pair rows, graph inputs, or checkpoint cleanup privately.
```

## Non-Goals

- No long-term support for deprecated architecture aliases.
- No indefinite V1/V2 replay dual path inside the main training/runtime stack.
- No compatibility wrapper that keeps old behavior alive after the replacement path is working.
- No duplicate `model` and `models`, `buffer` and `replay`, or old/new graph builders living side by side after the cutover.
- No silent fallback from Rust legal rows to Python legal rows in production.
- No full pair enumeration unless a named pair strategy explicitly requests it and supplies hard caps.

Short-lived migration tools are allowed, but they should live under `tools/migration/` or one-off scripts and should not be imported by runtime code.

## Current Diagnosis

### Worker Owns Too Much

`Python/src/hexorl/selfplay/worker.py` mixes game lifecycle, dense inference, sparse candidates, global graph construction, online pair scoring, tactical oracle calls, MCTS calls, replay records, runtime telemetry, and architecture feature gates.

Problem examples:

```python
self.global_graph_enabled = architecture.startswith("global_")
self.pair_policy_enabled = (
    global_graph_enabled and pair_prior_mix > 0
) or pair_head_present
```

That makes a model like `global_xattn_0` vulnerable to accidental pair work simply because `pair_prior_mix` is nonzero. Pair scoring must be a search strategy, not an implied model side effect.

### Model Construction Is Too Centralized

`Python/src/hexorl/model/network.py` contains dense CNN, RestNet, graph hybrid pieces, sparse heads, pair heads, factory logic, and checkpoint cleanup. `Python/src/hexorl/model/global_graph.py` multiplexes several global graph variants inside one class.

This makes new architecture work expensive because a model change touches training, inference, config validation, runtime sizing, self-play, dashboard, and autotune.

### Config Is A Single Bucket

`Python/src/hexorl/config/schema.py` mixes dense, RestNet, graph hybrid, global graph, sparse policy, pair policy, training defaults, and validation rules. It also repeats global architecture names and makes pair-capable heads imply pair consumption.

The new config should be recipe based:

```text
Recipe = ModelSpec + InputSpec + HeadBundleSpec + SearchSpec + PairStrategySpec + TrainingSpec + RuntimeSpec
```

### Rust/Python Rules Boundary Is Blurry

Rust should be the source of truth for legal game state, encoding, and rule-valid replay. Today, Python has multiple compact history parsers and legal move fallbacks with different ordering/radius/truncation behavior.

The refactor must establish one Python boundary over Rust rules, then delete private Python fallbacks from production paths.

### Pair Policy Is The Main Performance Trap

There are separate full-pair scoring paths for graph and crop/sparse models. Both can reach:

```text
A * (A - 1) / 2
```

This is acceptable only as an explicit diagnostic with caps. The default pair strategy must be `none`, including for `global_xattn_0`.

### Tests And CI Do Not Prove The Target Shape

The existing tests catch important regressions, but the proposed architecture needs CI gates for contracts, Rust/Python parity, inference protocol compatibility, model registry behavior, dashboard build, replay projection, recipe validation, and phase-by-phase deletion.

## North Star Package Layout

The final project should use one coherent layout. The old packages should be removed rather than kept as compatibility facades.

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
    telemetry.py
    validation.py
    debug.py

  engine/
    __init__.py
    rust.py
    legal.py
    history.py
    encoding.py
    parity.py

  models/
    __init__.py
    registry.py
    specs.py
    capabilities.py
    checkpoint.py
    factory.py
    heads/
      policy.py
      value.py
      sparse_policy.py
      pair_policy.py
      regret.py
      tactical.py
    trunks/
      dense_cnn.py
      restnet.py
      graph_hybrid.py
      global_graph.py
    families/
      dense_cnn.py
      restnet.py
      graph_hybrid.py
      global_xattn.py
      global_line_window.py
      global_relation_graph.py

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
      dense.py
      sparse.py
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
    fixtures.py

  train/
    trainer.py
    adapters.py
    losses.py
    schedules.py

  eval/
    arena.py
    players.py
    policy_player.py
    scorecard.py
    league.py

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

Package decisions:

- Use `models/`, not both `model/` and `models/`.
- Use `replay/` for replay records, storage, sampling, and projection. Keep no parallel `buffer/` runtime path.
- Use `engine/` as the only Python-facing Rust rules boundary.
- Keep `contracts/` pure. It should not contain inference transport protocols or search orchestration.
- Keep strategy names out of model-family names. `two_stage` belongs to `PairStrategySpec`; `champion` belongs to a recipe/checkpoint lineage.

## Ownership Map

| Concern | Target Owner |
|---|---|
| Rust game replay and legal rows | `engine/` |
| Compact history codec | `contracts/history.py` plus `engine/history.py` parity |
| D6 transforms | `contracts/symmetry.py`, parity-tested against Rust |
| Legal tables | `contracts/legal.py` and `engine/legal.py` |
| Candidates | `contracts/candidates.py` |
| Pair action rows | `contracts/pairs.py` |
| Graph semantic construction | `graph/semantic_builder.py` |
| Graph tensorization/collation | `graph/tensorize.py`, `graph/collate.py` |
| Model capabilities | `models/capabilities.py` |
| Model construction | `models/registry.py`, family modules |
| Checkpoints | `models/checkpoint.py` |
| Inference request protocol | `inference/protocol.py` |
| Transport lifecycle | `inference/shm_transport.py` |
| Search priors | `search/policy_provider.py` |
| Pair scoring | `search/pair_strategy.py` |
| MCTS Rust calls | `search/engine_adapter.py` |
| Game loop | `selfplay/game_runner.py` |
| Training batch/loss wiring | `train/adapters.py` |
| Evaluation players | `eval/policy_player.py` |
| Autotune recipes | `tuning/recipes.py`, `tuning/family_spaces.py` |
| Dashboard model/debug views | `dashboard/contract_inspector.py` |

## Core Contracts

In this document, a contract is a small, explicit data shape that says:

```text
this is exactly what this subsystem receives
this is exactly what it produces
these fields mean the same thing everywhere
these invariants must hold
```

Contracts are not business logic, model code, or process orchestration. They are shared agreements between parts of the system. For example, a `LegalActionTable` says what legal actions exist, how they are ordered, where they came from, and how to verify that two subsystems are looking at the same legal rows. A `PairActionTable` says how pair moves are represented so crop models, global graph models, MCTS, replay, dashboard, and tests do not each invent a slightly different pair format.

The practical reason to use contracts is debuggability and trust. If self-play, replay sampling, training, dashboard, and evaluation all consume the same `PositionContract`, then a mismatch can be localized to one builder or adapter instead of being hidden in five different reconstruction paths.

Good contracts should be:

```text
plain data, not hidden behavior
typed and versioned
cheap to inspect
hashable or comparable where useful
validated at boundaries
shared by tests and dashboard debug tools
stable enough that logs can reference them
```

Bad contracts would become a new dumping ground. Keep them narrow. They should describe game/data facts, not own inference transport, training loops, dashboard routes, or search strategy.

Contracts must be plain, typed, versioned, and suitable for equality/parity tests. They should avoid hidden side effects and should not import model, inference, training, dashboard, or tuning code.

Hot-path rule:

```text
Contracts can expose cached or zero-copy views. Self-play does not need to allocate full debug dataclasses on every leaf expansion when raw engine bytes already prove the invariant.
```

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

`MoveHistory` replaces all private compact-history parsing in graph, sampler, dashboard, tactical oracle, RGSC, and epoch bootstrap.

Rules:

- The compact byte format has one owner.
- Invalid player/order/duplicate-cell history is rejected at decode time.
- Rust replay parity is tested for golden histories.
- Bootstrap generation uses the same encoder.

### LegalActionTable

```python
@dataclass(frozen=True)
class LegalActionTable:
    rows: np.ndarray
    dense_indices: np.ndarray
    source: str
    radius: int
    occupied_count: int
    table_hash: str
```

Production legal rows come from Rust through `engine/legal.py`.

Rules:

- Python legal fallback is allowed only in unit fixtures that explicitly mark `source="fixture"`.
- Dashboard, sampler, tactical oracle, graph, and self-play may not generate private legal rows.
- Legal row ordering is a contract.
- Any degraded/fallback source is telemetry-visible and test-visible.

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

This is the common root for training projection, graph construction, candidate construction, pair action construction, dashboard inspection, replay validation, and evaluation debugging.

### D6 Symmetry

All Python D6 code lives in `contracts/symmetry.py`.

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

Required parity:

```text
Python D6 == Rust/PyO3 D6 for coordinates, histories, dense tensors, and legal rows.
```

Required invariants:

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
    rows: np.ndarray
    dense_indices: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    mask: np.ndarray
    target: np.ndarray | None
    missing_mass: float
    recall: CandidateRecall
    diagnostics: CandidateDiagnostics
    contract_hash: str
```

Built only by `CandidateContractBuilder`.

Consumers:

```text
self-play
replay sampler
training adapters
graph builder
dashboard
model cache
evaluation debug
```

### PairActionTable

```python
@dataclass(frozen=True)
class PairActionTable:
    rows: np.ndarray
    first_row_indices: np.ndarray
    second_row_indices: np.ndarray
    phase: str
    known_first: tuple[int, int] | None
    generation: str
    total_possible_pairs: int
    selected_pair_rows: int
    table_hash: str
```

This replaces split pair mini-contracts.

Rules:

- Crop pair candidates and global graph `PAIR_ACTION` rows derive from this table.
- `PairCandidateBatch` should be deleted or become a thin tensor projection from `PairActionTable`.
- First-placement unordered pairs and second-placement known-first pairs are separate phases.
- Full pair generation requires a `PairStrategy` with caps.

## Model Family Registry

Model behavior is declared through a registry, not inferred from names.

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

Use discriminated specs:

```python
@dataclass(frozen=True)
class DenseCnnSpec:
    kind: Literal["dense_cnn"]
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
    kind: Literal["graph_hybrid"]
    channels: int
    blocks: int
    sparse_token_budget: int
    candidate_contract: CandidateSpec
    heads: HeadBundleSpec

@dataclass(frozen=True)
class GlobalGraphSpec:
    kind: Literal["global_xattn", "global_line_window", "global_relation_graph"]
    d_model: int
    layers: int
    heads: int
    token_spec: GraphTokenSpec
    relation_spec: GraphRelationSpec
    legal_action_contract: str
    heads: HeadBundleSpec
```

Avoid family names like `global_pair_twostage` and `global_graph_champion`. Those are recipes or strategy/checkpoint labels, not model-family identities.

### Capabilities

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

Capabilities declare what a model can output. They do not decide what MCTS consumes. `PairStrategy` decides consumption.

## Model Assembly

Split model code into:

```text
trunk
heads
family wrapper
train adapter
inference adapter
policy provider
recipe
checkpoint manifest
```

Model core rules:

- Trunks compute features only.
- Heads expose typed output contracts.
- Family wrappers compose trunks and heads.
- Models do not know about self-play, replay storage, dashboard routes, or IPC.
- Avoid generic dicts inside model core. Dicts are acceptable at external boundaries only.

## Checkpoints

Introduce one checkpoint owner:

```python
class CheckpointManager:
    def save(self, bundle: CheckpointBundle, path: Path) -> None: ...
    def load(self, path: Path, *, purpose: LoadPurpose, device: str) -> LoadedCheckpoint: ...
    def inspect(self, path: Path) -> CheckpointManifest: ...
```

Required manifest:

```yaml
checkpoint_schema_version: 1
model_family: global_xattn
model_spec_version: 1
model_spec: ...
input_contract: global_graph_v1
output_contract: global_place_value_v1
action_contract: legal_action_table_v1
graph_schema_version: 1
relation_schema_version: 1
inference_protocol:
  version: 1
  request_kind: global_graph
  max_tokens: 768
  max_legal_rows: 768
  max_pair_rows: 4096
heads:
  - policy_place
  - value
pair_strategy_used: none
created_by:
  git_sha: ...
  command: ...
  config_hash: ...
```

Rules:

- Strict load by default.
- No silent `_orig_mod` stripping outside `CheckpointManager`.
- No duplicate checkpoint cleanup in trainer/server/model code.
- Checkpoints can be inspected without loading model weights.
- Old incompatible checkpoints are archived or migrated by explicit one-off tools, not loaded by production code indefinitely.

## Inference

### Protocol Manifest

Inference has a versioned protocol before adapters are considered complete.

```python
@dataclass(frozen=True)
class InferenceProtocolManifest:
    version: int
    request_kind: str
    input_contract: str
    output_contract: str
    max_batch: int
    max_tokens: int | None
    max_legal_rows: int | None
    max_pair_rows: int | None
    required_heads: tuple[str, ...]
```

The worker and server must handshake on:

```text
protocol version
request kind
schema versions
capacity limits
required output heads
```

Mismatch behavior:

```text
fail fast with a structured error
never wait forever in IPC
emit last request kind and protocol versions
```

### Request Lifecycle

The transport owns shared lifecycle code:

```text
pack
mark ready
wait
timeout
decode
reset
telemetry
```

Adapters own semantic conversion:

```text
DenseInferenceAdapter
SparseInferenceAdapter
GlobalGraphInferenceAdapter
PairScoringInferenceAdapter
```

This avoids preserving duplicated lifecycle code across `submit`, `submit_sparse`, `submit_sparse_pair`, and `submit_graph`.

### Response Telemetry

Every response includes:

```text
request_kind
protocol_version
input_contract
output_contract
model_family
legal_count
candidate_count
pair_rows
token_count
relation_count
pack_ms
queue_wait_ms
forward_ms
decode_ms
warnings
```

## Search And Self-Play

### PolicyProvider

```python
class PolicyProvider(Protocol):
    def evaluate_root(self, context: SearchContext) -> SearchEvaluation: ...
    def evaluate_leaves(self, contexts: list[SearchContext]) -> list[SearchEvaluation]: ...
```

Implementations:

```text
DensePolicyProvider
RestNetPolicyProvider
GraphHybridPolicyProvider
GlobalGraphPolicyProvider
```

### PairStrategy

```python
class PairStrategy(Protocol):
    name: str
    def score_root(self, context: SearchContext, base_eval: SearchEvaluation) -> PairEvaluation: ...
    def score_leaves(self, contexts: list[SearchContext], base_evals: list[SearchEvaluation]) -> list[PairEvaluation]: ...
```

Strategies:

```text
NoPairStrategy
TwoStageRootOnlyPairStrategy
DiagnosticFullRootPairStrategy
TacticalOnlyPairStrategy
```

Required defaults:

```text
default pair strategy = none
global_xattn default pair strategy = none
leaf pair scoring = disabled by default
full pair scoring = diagnostic only, root only, capped
```

Hard invariants:

```text
No pair scoring happens because a model has a pair-capable head.
No pair scoring happens because architecture starts with global_.
Leaf pair scoring requires a separate explicit enable flag and cap.
Full pair rows require a cap and telemetry.
Opening positions have no pair prior and no pair loss.
```

### EngineAdapter

Rust MCTS calls move behind:

```python
class EngineAdapter:
    def expand_root(self, evaluation: SearchEvaluation) -> None: ...
    def apply_pair_priors(self, pair_eval: PairEvaluation) -> None: ...
    def expand_and_backprop(self, evaluations: list[SearchEvaluation]) -> None: ...
```

### GameRunner

`GameRunner` owns game flow. It does not know model families.

```text
GameRunner -> PolicyProvider -> InferenceAdapter -> ModelFamily
GameRunner -> PairStrategy -> PolicyProvider/InferenceAdapter
GameRunner -> EngineAdapter -> Rust MCTS
```

`SelfPlayWorker` should shrink to lifecycle, process management, and IPC wiring.

## Replay And Sampling

This refactor should cut to a new replay contract rather than carry old records indefinitely.

### ReplayPosition

```python
@dataclass(frozen=True)
class ReplayPosition:
    schema_version: int
    identity: PositionIdentity
    history: bytes
    policy_target: PolicyTarget
    opponent_policy_target: PolicyTarget | None
    pair_target: PairPolicyTarget | None
    value_target: ValueTarget
    tactical: TacticalReport
    search_trace: SearchTrace
    contract_trace: ContractTrace
```

Rules:

- New self-play writes only the new replay format.
- Old replay files are archived or migrated by explicit tools outside runtime.
- The sampler consumes `ReplayPosition`, builds `PositionContract`, then calls adapters.
- The sampler does not own private D6, legal fallback, candidate assembly, pair second-placement handling, graph construction, or axis transforms.

### Replay Storage

Use one replay owner:

```text
blob_store:
  exact versioned ReplayPosition records

index_arrays:
  game_id
  move_idx
  player
  value_weight
  policy_weight
  model_family
  flags needed for fast sampling
```

Avoid duplicating every semantic field into a wide ring layout unless profiling proves it is needed.

## Training

Training should be model-family agnostic.

```python
class TrainAdapter(Protocol):
    input_contract: str
    target_contract: str

    def prepare_batch(self, raw: ReplayBatch, device: torch.device) -> ModelTrainBatch: ...
    def compute_losses(self, outputs: ModelOutput, targets: TrainingTargets, loss_plan: LossPlan) -> LossBundle: ...
```

Trainer loop target:

```python
batch = train_adapter.prepare_batch(raw_batch, device)
outputs = model(batch.inputs)
losses = train_adapter.compute_losses(outputs, batch.targets, loss_plan)
losses.total.backward()
```

Rules:

- Trainer does not branch on `GlobalHexGraphNet`.
- Loss wiring belongs to adapters.
- Pair targets include phase/provenance validation.
- `policy_pair_first` cannot silently fall back to unrelated place policy targets.
- `policy_pair_second` validates known-first semantics.
- Joint pair losses validate `PairActionTable` row identity.

## Evaluation

Evaluation must use the same policy-provider path as self-play.

Rules:

- Arena players do not assume dense `model(tensor)["policy"]`.
- Global graph, sparse, and dense models are evaluated through `PolicyProvider`.
- Evaluation can request debug traces from contracts, but it does not rebuild private model inputs.
- Scorecards report model family, recipe, checkpoint manifest, and pair strategy.

## Autotuning

Autotune operates on typed recipes.

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

Ownership:

```text
ModelFamily owns model dimensions, token budgets, head options.
PolicyProvider owns policy source and candidate budgets.
PairStrategy owns pair mode, row caps, root/leaf enablement.
SearchSpec owns sims, c_puct, temperature, root exploration.
TrainingSpec owns lr, batch size, loss weights, augmentation.
RuntimeSpec owns workers, queue sizes, memory mode.
```

The Phase 3 autotune script should become an entrypoint only. Family internals move to `tuning/family_spaces.py`.

## Observability

Observability is a first-class goal of the refactor, not an afterthought. Self-play and autotuning are long-running, multi-process, performance-sensitive workflows. When something slows down, stalls, produces odd training data, or chooses a bad recipe, the system should make it clear:

```text
what is happening
where it is happening
which contract/model/strategy/runtime setting is involved
how long each stage took
what changed recently
what the likely next debugging action is
```

The implementer can choose the exact logging backend, file format, and dashboard presentation. The architecture requirement is that logs and traces must be structured, correlated, and actionable. A plain message like "started sweep" or "worker slow" is not enough. The log should identify the phase, worker/trial/model/recipe, relevant contract hashes or schema versions, timing spans, row/token counts, warnings, and enough context to decide whether the issue is in engine replay, legal rows, graph building, inference IPC, model forward, pair scoring, MCTS expansion, replay writing, or autotune scheduling.

Logging should serve three audiences:

```text
developer debugging an immediate stall or correctness bug
researcher comparing recipes and model families
operator monitoring overnight self-play/autotune runs
```

### Logging Principles

Logs should be:

```text
structured enough to filter and aggregate
human-readable enough to diagnose from a terminal
correlated by run_id, trial_id, game_id, worker_id, trace_id, and checkpoint/recipe identity
emitted at major phase transitions and suspicious no-progress intervals
bounded so high-volume loops do not drown the signal
connected to suggested next actions where possible
```

Every major subsystem should emit both summary logs and trace-level details:

```text
summary logs answer "how is the run doing?"
trace logs answer "why did this specific position/request/trial behave this way?"
```

### Self-Play Logging Priority

Self-play should be debuggable as a pipeline. A position should be traceable from history replay through legal rows, tactical/candidate/pair/graph construction, inference, prior application, MCTS expansion, move selection, and replay record writing.

Self-play logs should make these failure classes distinguishable:

```text
worker is alive but waiting on inference
worker is stuck before inference
Rust engine replay/legal generation is slow or failing
candidate or tactical construction is expensive
graph semantic construction or tensorization is expensive
pair scoring was accidentally enabled or exceeded budget
IPC request was packed but not answered
model forward is slow or non-finite
MCTS expansion/backprop is slow
record writing or dashboard recording is slow
legal rows disagree between engine and model contract
policy priors are missing, masked out, or mapped to wrong rows
```

Self-play should emit periodic heartbeats during games and explicit summaries at game end. Heartbeats should include enough state to know whether progress is being made, but should avoid dumping full tensors or huge legal tables by default.

Useful self-play diagnostics include:

```text
current game/move/phase
positions completed since last heartbeat
last successful inference request
last engine operation
legal/candidate/pair/token/relation counts
active model family and recipe
active policy provider and pair strategy
pair rows possible versus pair rows scored
timing breakdown for the last few positions
warnings and validation failures
suggested next action for common failures
```

For example, if a run stalls while evaluating a global graph model, the logs should help decide whether to inspect graph construction, relation counts, IPC, model forward, pair strategy, or Rust MCTS. The dashboard can present this nicely later, but the raw run logs should already contain the evidence.

### Autotuning Logging Priority

Autotuning should explain its decisions. It should not simply report that a trial started, ended, or scored poorly. It should record enough information to understand why a recipe was selected, rejected, aborted, retried, or promoted.

Autotune logs should make these failure classes distinguishable:

```text
recipe invalid for model family capabilities
runtime candidate invalid for host limits
trial rejected before launch
trial launched but made no self-play progress
trial launched but inference stalled
trial exceeded memory or row/token budgets
trial produced non-finite losses or invalid targets
trial was slower because of graph construction, IPC, model forward, MCTS, or pair scoring
trial score was low because of quality metrics, speed metrics, stability metrics, or budget penalties
scheduler promoted or stopped a trial for a specific reason
```

Autotuning should emit:

```text
recipe identity and diff from baseline
validation results and rejection reasons
host/runtime profile used
trial lifecycle events
resource and throughput summaries
self-play progress summaries
training/evaluation score components
scheduler decisions with reasons
no-progress watchdog reports
recommended next debugging action for common aborts
```

The desired debugging experience is: when an overnight sweep looks frozen or underperforms, the next morning report should say where the time went, which trials were healthy, which failed validation, which stalled, and what subsystem deserves attention first.

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

Required spans:

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

Required events:

```text
selfplay_worker_heartbeat
selfplay_phase_transition
selfplay_no_progress
selfplay_game_summary
policy_eval_timing
pair_strategy_summary
autotune_recipe_validation
autotune_trial_lifecycle
autotune_scheduler_decision
runtime_sweep_no_progress
inference_protocol_mismatch
contract_validation_failure
```

## Dashboard

The dashboard is an inspector, not a second sampler.

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

Views:

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

Dashboard dependency rule:

```text
dashboard may depend on contracts, inspectors, checkpoint manifests, and read-only services.
dashboard may not import sampler-private builders or reconstruct model inputs independently.
```

Migration targets include dashboard model cache and debug payload code that currently rebuild legal/candidate/pair/graph inputs.

## Testing Strategy

Tests stay under `Python/tests` unless the whole package layout is intentionally moved.

Target layout:

```text
Python/tests/contracts/
Python/tests/engine/
Python/tests/models/
Python/tests/inference/
Python/tests/search/
Python/tests/replay/
Python/tests/train/
Python/tests/eval/
Python/tests/tuning/
Python/tests/dashboard/
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

Fixture rules:

- Fixtures have a generator command.
- Fixtures have fixed seeds.
- Rust replay/legal output is the source of truth.
- Snapshot updates are explicit.
- Fixture schema versions are recorded.

Critical tests:

```text
Python/Rust D6 parity
Rust legal table parity
MoveHistory rejects malformed compact history
self-play candidate table == sampler candidate table == dashboard candidate table
graph legal table == Rust legal table
pair target mass preserved under D6
dashboard model inputs match training inputs
no pair scoring unless PairStrategy enables it
no leaf pair scoring unless separately enabled
global_xattn defaults to zero pair rows
checkpoint manifest inspect works without loading weights
autotune rejects incompatible recipes
trainer runs one batch for every registered family
arena can evaluate every registered family through PolicyProvider
inference protocol mismatch fails fast
```

CI matrix:

```text
Rust:
  cargo test --workspace

Python fast:
  pytest Python/tests/contracts Python/tests/engine Python/tests/models

Python integration:
  pytest Python/tests/inference Python/tests/search Python/tests/replay Python/tests/train Python/tests/eval

Dashboard:
  npm run build in Python/dashboard_frontend

Tuning smoke:
  recipe validation dry-run

Optional slow:
  GPU training/inference smoke
  long self-play smoke
```

## Cutover Strategy

This is a breaking refactor, but it should still be controlled.




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

### Phase 0: Freeze, Tag, And Measure

Actions:

- Tag the last pre-refactor commit.
- Archive current important runs/checkpoints/replay data.
- Add non-invasive telemetry for worker heartbeat, self-play phase transitions, policy eval timing, pair summary, graph request summary, runtime sweep no-progress, autotune trial lifecycle, and scheduler decisions.
- Add an immediate guard that `global_xattn` uses `pair_strategy=none`.
- Add a temporary failing test for accidental pair scoring.
- Make logs structured and correlated enough to debug where a run is spending time or where it stalled.

Exit gates:

```text
global_xattn reports zero pair rows by default
pair scoring requires an explicit strategy in tests
baseline self-play/training smoke still runs
self-play and autotune emit actionable progress/no-progress logs
```

### Phase 1: Engine And Contract Foundation

Actions:

- Add `engine/` Rust boundary.
- Add `contracts/history.py`, `legal.py`, `symmetry.py`, `targets.py`, `candidates.py`, `pairs.py`, `telemetry.py`.
- Centralize compact history and D6.
- Replace production Python legal fallbacks with `LegalTableProvider`.
- Add Rust/Python parity tests.

Delete:

```text
private compact-history parsers
private D6 helpers
production Python legal fallbacks
```

Exit gates:

```text
Rust/Python D6 parity passes
Rust legal parity passes
dashboard/sampler/graph use shared history/legal/D6
```

### Phase 2: Candidate, Pair, And Graph Builders

Actions:

- Add `CandidateContractBuilder`.
- Add `PairActionTableBuilder`.
- Split graph semantic builder from tensorizer/collator.
- Make self-play, replay, dashboard, training, and evaluation debug consume shared builders.

Delete:

```text
private candidate construction in sampler/dashboard/worker
parallel crop/global pair mini-contracts
graph batch code that owns semantic construction and tensorization together
```

Exit gates:

```text
self-play candidate table == sampler == dashboard for golden positions
pair tables are phase-aware and D6-equivariant
graph tensorization is pure projection from graph semantic contract
```

### Phase 3: Model Registry, Specs, Training Adapters, Checkpoints

Actions:

- Move from `model/` to `models/`.
- Add `ModelFamilyRegistry`.
- Add discriminated `ModelSpec`.
- Add `TrainAdapter`.
- Add `CheckpointManager`.
- Add checkpoint manifest inspection.
- Convert trainer to adapter path.

Delete:

```text
build_model_from_config switch
GlobalHexGraphNet multi-architecture string switch
trainer isinstance(GlobalHexGraphNet) branch
duplicate checkpoint prefix/state cleanup
deprecated architecture aliases
```

Exit gates:

```text
every registered family builds
every registered family trains one batch
checkpoint manifest round-trips
no trainer architecture branches
```

### Phase 4: Inference Protocol And Adapters

Actions:

- Add `InferenceProtocolManifest`.
- Add request/response contracts.
- Move shared IPC lifecycle into transport.
- Add dense, sparse, global graph, and pair scoring adapters.
- Remove server dispatch by architecture string.

Delete:

```text
client lifecycle duplication across submit methods
server architecture.startswith("global_") dispatch
hidden fixed-cap assumptions not declared by protocol
```

Exit gates:

```text
protocol mismatch fails fast
all adapters round-trip through inference server
responses include schema/protocol telemetry
```

### Phase 5: Policy Providers, Pair Strategies, Engine Adapter

Actions:

- Add policy providers.
- Add pair strategies.
- Add engine adapter.
- Move Rust MCTS calls behind engine adapter.
- Make leaf pair scoring opt-in with separate caps.

Delete:

```text
worker architecture branches
worker pair chunk helpers
pair enablement from pair_prior_mix/head presence
direct MCTS prior wiring in worker
```

Exit gates:

```text
SelfPlayWorker contains no architecture string checks
no pair scoring without PairStrategy
leaf pair scoring disabled by default
full pair scoring diagnostic-only and capped
```

### Phase 6: GameRunner And Self-Play Cleanup

Actions:

- Move game flow to `GameRunner`.
- Keep `SelfPlayWorker` as process/lifecycle shell.
- Move record writing to `record_writer.py`.
- Emit consistent traces.

Delete:

```text
worker-owned game loop details
worker-owned replay record assembly
worker-owned telemetry shape decisions
```

Exit gates:

```text
GameRunner can run dense, graph hybrid, and global graph via same interface
worker code is lifecycle/IPC oriented
```

### Phase 7: Replay Cutover

Actions:

- Introduce new replay codec/storage/projector.
- Make new self-play write only new replay records.
- Make sampler consume only new replay records.
- Move old replay conversion to one-off migration tools, if needed.

Delete:

```text
old ring-buffer semantic duplication
old replay decode path from runtime
old sampler private projection logic
```

Exit gates:

```text
new replay writes/reads round-trip
sampler projection matches contract builders
old replay code is absent from runtime imports
```

### Phase 8: Evaluation, Dashboard, Autotune

Actions:

- Move arena players to `PolicyProvider`.
- Move dashboard to `ContractInspector`.
- Move autotune to typed recipes and family spaces.
- Add recipe validation and dry-run.

Delete:

```text
dense-only eval player assumptions
dashboard private model-input reconstruction
autotune family internals inside scripts
runtime sizing architecture branches
```

Exit gates:

```text
arena evaluates every registered family
dashboard/training inputs match on golden positions
autotune explains valid/rejected recipes
```

### Phase 9: Dead-Code Removal And Import Audit

Actions:

- Run import/code search audit.
- Remove old packages and stale docs references.
- Update getting-started and architecture docs.
- Add import-boundary tests where practical.

Delete:

```text
Python/src/hexorl/model/
Python/src/hexorl/buffer/
old graph batch construction paths
deprecated config aliases
dead scripts superseded by tuning entrypoints
stale dashboard reconstruction helpers
unused model heads or experimental branches
```

Exit gates:

```text
rg finds no old architecture string gates outside registry/spec tests
rg finds no private legal/D6/history parsers outside contracts/engine tests
CI matrix passes
docs describe only the new architecture
```

## Non-Negotiable Invariants

```text
Rust is the production source of legal game state.
Contracts are pure data/value objects.
SelfPlayWorker contains no architecture string checks.
InferenceServer dispatches by request kind/protocol, not architecture string.
Trainer uses TrainAdapter, not model class checks.
Arena uses PolicyProvider, not dense-policy assumptions.
No pair scoring happens unless PairStrategy explicitly enables it.
No leaf pair scoring happens unless separately enabled and capped.
No full A * (A - 1) / 2 pair scoring without diagnostic strategy and cap.
global_xattn default pair strategy is none.
D6 transforms exist in one Python module and are parity-tested against Rust.
Legal rows come from LegalTableProvider.
Candidate construction exists in one builder.
Pair action rows exist in one canonical table type.
Graph semantic construction is separate from graph tensorization.
Every model has declared input/output contracts.
Every checkpoint has a manifest.
Every inference request/response has protocol/schema versions.
Every runtime sweep has a no-progress watchdog.
Self-play logs make stalls, slow phases, pair scoring, inference waits, and contract mismatches diagnosable.
Autotune logs explain recipe validation, trial lifecycle, scheduler decisions, aborts, and likely next debugging action.
Autotune mutates ModelRecipe, not raw config fields.
Dashboard inspects contracts instead of reconstructing private approximations.
Deprecated aliases and stale compatibility shims are deleted.
```

## Final Target Flow

```text
Rust replay/legal source
-> MoveHistory
-> LegalActionTable
-> TacticalReport
-> PositionContract
-> CandidateTable / PairActionTable / GraphSemanticContract
-> ModelFamily TrainAdapter or InferenceAdapter
-> PolicyProvider
-> PairStrategy
-> EngineAdapter
-> ReplayPosition
-> TrainingBatch / Dashboard ContractInspector / Evaluation PolicyPlayer
```

This structure makes Hexo a stronger cohesive project because each concept has one owner, one contract, one test surface, and one deletion path for old code. New architecture work should add a model family, adapters, recipe, and tests. It should not require edits to worker internals, trainer internals, inference server dispatch, replay projection internals, dashboard reconstruction paths, or autotune script internals.
