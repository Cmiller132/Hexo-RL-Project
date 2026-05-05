# Model Architecture Modularization Plan

## Goal

Make model families easier to maintain by moving model-specific decisions out of scattered training, assembly, inference, and self-play code and into explicit architecture definitions.

Today, model behavior is fragmented across:

- model constructors and forward methods
- config normalization
- training loss selection
- replay target construction
- inference server routing
- self-play worker prior mapping
- pair-policy enablement and chunking paths
- graph batch and target builders

The target design is architecture-owned: each architecture declares what it consumes, what it emits, how targets are built, how losses are computed, how outputs are decoded, and which runtime adapter is allowed to consume them.

## Design North Star

The model layer should behave like a small product-line platform.

An architecture is not just a PyTorch module. It is a complete contract bundle:

- what tensors it accepts
- what row tables those tensors represent
- what trunk transforms those inputs
- what heads are attached
- what each head means semantically
- what targets train each head
- what losses are legal for each head
- what inference adapter decodes each output
- what runtime provider may consume each output
- what pair strategy, if any, is allowed to request pair scoring
- what telemetry proves the path was interpreted correctly

The strongest foundation is contract-first, not class-first. Classes are implementation details. Contracts are the durable API between replay, model assembly, training, inference, self-play, dashboard tooling, and Rust MCTS.

The key design rule:

```text
No subsystem should rediscover model behavior from strings, head presence, tensor shapes, or model classes.
```

Every subsystem should ask the architecture spec.



## Success Criteria

- A model architecture can be understood from one registered specification.
- Head names, tensor shapes, semantic masks, target names, loss functions, and inference decoding are declared together.
- Training and inference select behavior through architecture contracts, not architecture-name prefixes or ad hoc `isinstance` checks.
- Runtime consumers never interpret raw model outputs without an adapter selected by the architecture contract.
- Pair heads are output capabilities only; pair scoring remains owned by an explicit pair strategy.
- Dense, sparse, graph hybrid, and global graph families share the same high-level assembly path.
- New architectures can be added by registering specs and modules, not by editing trainer/server/worker switch logic.

## Constraints

- Rust legal rows remain the canonical rules boundary.
- Global graph legal rows must map exactly to canonical legal action rows before MCTS sees priors.
- Pair-action rows must come from canonical pair-row construction and preserve first-placement unordered semantics and second-placement known-first semantics.
- No architecture may enable pair scoring just because pair heads exist.
- No training loss may claim a head is trained unless its target, mask, and semantic phase are present.
- No compatibility facade should be added inside `Python/src/hexorl/` unless a phase doc explicitly permits migration tooling.

## Required Evidence For Completion

- Contract tests for each architecture family.
- Shape and semantic-mask tests for every registered head.
- Training adapter tests proving each head uses the declared target and mask.
- Inference adapter tests proving raw outputs are decoded into canonical runtime rows.
- Search/self-play tests proving pair strategy, not head presence or architecture name, controls pair scoring.
- Import/code-search audit proving prefix checks and scattered head-specific runtime branches are removed.

## Stop Rules

- Stop if an architecture cannot declare its required input contract.
- Stop if a head cannot identify its target, mask, loss, and semantic phase.
- Stop if inference cannot map outputs to canonical legal or pair rows before runtime consumption.
- Stop if pair scoring still depends on model family, head presence, checkpoint metadata, or `pair_prior_mix`.

## High-Level Design Considerations

### Prefer Explicit Contracts Over Clever Discovery

Model code is currently difficult to maintain because behavior is inferred in many places. A head name means one thing in model assembly, another in the trainer, another in inference, and another in self-play.

The modular design should make the implicit explicit:

- architectures declare capabilities
- heads declare output meaning
- targets declare row identity
- losses declare valid masks
- adapters declare decode rules
- providers declare runtime interpretation
- strategies declare optional pair behavior

Discovery should be limited to registry lookup. After lookup, behavior is explicit data.

### Separate Capability From Consumption

A model may expose a head without that head being consumed by search. This is especially important for pair heads.

Examples:

- `policy_pair_first` can be trained as an auxiliary output.
- `policy_pair_joint` can be available for diagnostics.
- `policy_pair_second` can be used in a controlled known-first experiment.
- None of these should alter MCTS unless an explicit `PairStrategy` requests them.

This distinction keeps experimentation safe. New heads can be added without accidentally changing runtime behavior.

### Make Rows First-Class

Most serious bugs in this project are row-identity bugs, not neural-network math bugs.

The design should treat row tables as first-class objects:

- dense board indices
- sparse candidate rows
- global graph legal rows
- opponent legal rows
- pair-action rows
- known-first second-placement rows

Every output that is consumed by runtime should be tied to a row table. Raw logits are not meaningful until paired with their row identity.

### Keep Trunks Reusable And Heads Small

Trunks should produce named representation slots. Heads should consume those slots and declare their output contracts.

This allows new model families to reuse parts:

- a new graph trunk can reuse existing policy and value heads
- a new pair head can attach to an existing global graph trunk
- a new tactical head can train on dense, graph hybrid, or global graph trunks if the required representation slot exists

The architecture spec, not each individual module, decides which combination is valid.

### Keep Adapters Thin But Strict

Adapters should not contain model logic. They should contain boundary logic:

- validate required inputs
- validate schema versions
- call the model
- validate output presence
- validate output shapes
- validate finite masks
- attach row identity
- produce canonical response objects

This makes adapters easy to test and prevents runtime systems from interpreting raw tensors.

### Make Invalid States Unrepresentable When Possible

The design should prefer typed objects and explicit specs over loosely connected dictionaries.

Examples:

- a trainable head without a target should fail spec validation
- a runtime-consumed policy head without a row table should fail spec validation
- a pair-second head without known-first phase semantics should fail spec validation
- a global graph architecture without a graph input contract should fail spec validation

Loose dictionaries may still be used at tensor boundaries, but they should be created from validated specs.

### Optimize For Adding New Parts

Adding a new model type should require registering a spec and implementing only genuinely new pieces.

Adding a new head should require:

- a head module
- a `HeadSpec`
- a target builder if trainable
- a loss entry if the loss is new
- an inference decode rule only if runtime consumes it

Adding a new trunk should require:

- a trunk module
- a `TrunkSpec`
- declared output slots
- architecture specs that consume those slots

Adding a new target should require:

- a target contract
- a target builder
- validation tests
- one or more head specs that reference it

### Avoid Over-Modularization

The goal is not to split every function into tiny files. The goal is to put decisions at the correct ownership boundary.

Good modularization:

- head semantics live with head specs
- row identity lives with row table contracts
- runtime decode lives with inference adapters
- training loss dispatch lives with the loss registry
- architecture composition lives with architecture specs

Bad modularization:

- one-file-per-linear-layer without stronger contracts
- adapters that duplicate model forward logic
- specs that only mirror config without validating anything
- registries that hide behavior instead of making it explicit

### Preserve Fast Experimentation

The structure should make experimentation faster, not slower.

Researchers should be able to create a new architecture by composing existing parts:

```python
register_architecture(
    ArchitectureSpec(
        name="global_graph_tactical_pair_v2",
        input_contract="global_graph_v1",
        trunk="global_relation_graph_trunk",
        heads=("policy_place", "value", "tactical", "policy_pair_first", "policy_pair_joint"),
        target_adapter="global_graph_target_adapter",
        training_adapter="global_graph_training_adapter",
        inference_adapter="global_graph_inference_adapter",
        policy_provider="global_graph_policy_provider",
        pair_capabilities="pair_strategy_required",
    )
)
```

The point is that experiments should compose known-good pieces instead of copying trainer/server/worker branches.

## How The System Works End To End

### 1. Config Resolves To An ArchitectureSpec

Config should not directly drive behavior. Config chooses an architecture id and hyperparameters.

```text
Config
  -> ArchitectureRegistry.resolve(config.model.architecture)
  -> ArchitectureSpec
  -> Spec validation
```

Spec validation checks:

- required heads exist
- requested heads are supported by the trunk
- trainable heads declare targets and losses
- runtime heads declare row tables and inference decode rules
- pair-capable heads do not imply pair consumption
- input schema versions are compatible

### 2. Assembly Builds A ModelBundle

Assembly consumes the spec and config.

```text
ArchitectureSpec
  -> build InputAdapter
  -> build Trunk
  -> build Heads
  -> build Model
  -> return ModelBundle
```

The returned bundle contains both the module and metadata.

```python
ModelBundle(
    model=model,
    architecture=spec,
    input_contract=spec.input_contract,
    head_specs=spec.heads,
)
```

The bundle is what training and inference receive. This avoids model-class checks.

### 3. Replay Builds Canonical Row Tables And Targets

Replay and target adapters produce canonical objects before tensors.

```text
PositionRecord
  -> LegalActionTable
  -> Optional CandidateTable
  -> Optional PairActionTable
  -> TargetBundle
  -> TensorBatch
```

`TargetBundle` should track:

- target name
- row table identity
- schema version
- semantic phase
- mask
- probability mass

This makes it possible to reject mismatched policy, pair, and graph targets before training.

### 4. Training Uses HeadSpec-Driven Loss Dispatch

The trainer should be generic.

```text
TensorBatch
  -> TrainingAdapter.prepare_inputs
  -> model forward
  -> LossRegistry.compute_all(spec.heads, predictions, targets)
```

Loss dispatch should use `HeadSpec`, not broad trainer conditionals.

```python
for head in bundle.architecture.trainable_heads:
    loss = loss_registry.compute(head.loss_name, head, predictions, targets)
```

This keeps model-specific rules out of the trainer.

### 5. Inference Decodes Outputs Into Canonical Responses

Inference adapters convert raw tensors into typed outputs.

```text
InferenceRequest
  -> InferenceAdapter.prepare_inputs
  -> model forward
  -> InferenceAdapter.decode
  -> PolicyOutput / PairOutput / AuxiliaryOutput
```

The server should only pack validated responses. It should not decide what a head means.

### 6. Runtime Consumes Providers And Strategies

Self-play and evaluation consume canonical providers.

```text
SearchContext
  -> PolicyProvider.evaluate
  -> SearchEvaluation
  -> PairStrategy.score_root
  -> PairEvaluation
  -> EngineAdapter.expand_root
```

This is the runtime boundary:

- `PolicyProvider` maps model policy to legal rows.
- `PairStrategy` decides whether pair scoring happens.
- `EngineAdapter` is the only layer that calls Rust MCTS.

### 7. Telemetry Carries Contract Identity

Every runtime evaluation should carry enough metadata to debug a bad move.

Required telemetry:

- architecture id
- architecture spec version
- input contract id
- output contract id
- adapter id
- policy provider id
- pair strategy id
- legal row count
- pair row count possible
- pair row count scored
- row-table hash
- target hash when training
- schema versions
- timing spans
- warnings and validation decisions

This turns model behavior into inspectable evidence.

## Proposed Package Layout

This is the target ownership layout, aligned to the refactor phases. It intentionally avoids putting all modularization under legacy `Python/src/hexorl/model/`.

```text
Python/src/hexorl/models/
  registry.py
  contracts.py
  assembly.py
  bundles.py
  specs/
    dense.py
    restnet.py
    graph_hybrid.py
    global_graph.py
  trunks/
    cnn.py
    residual.py
    graph.py
    relation_graph.py
    cross_attention_graph.py
  heads/
    policy_dense.py
    policy_sparse.py
    policy_place.py
    policy_pair_first.py
    policy_pair_joint.py
    policy_pair_second.py
    value.py
    opponent_policy.py
    tactical.py
    regret.py
    lookahead.py
  losses/
    registry.py
    policy.py
    pair_policy.py
    value.py
    auxiliaries.py
  training/
    training.py
    adapters.py
    loss_plan.py

Python/src/hexorl/contracts/
  row_tables.py
  targets.py
  hashes.py
  schemas.py
  traces.py

Python/src/hexorl/inference/
  manifests.py
  adapters/
    dense.py
    sparse.py
    global_graph.py
    pair_scoring.py

Python/src/hexorl/search/
  policy_provider.py
  pair_strategy.py
  engine_adapter.py

Python/src/hexorl/replay/
  projectors.py
  target_adapters.py
```

This layout separates concerns without making every architecture a completely separate stack. Shared modules stay reusable, but architecture specs decide which modules are assembled and which domain-owned adapters are selected.

The old `Python/src/hexorl/model/` package may exist during migration, but runtime ownership should move toward `Python/src/hexorl/models/` as the authoritative Phase 03 family registry. The old package should become either compatibility-free implementation detail during cutover or be deleted when Phase 09 closes.

## Phase Ownership Map

The modular design spans multiple refactor phases. Each concept must land in the phase-owned module that already governs that boundary.

| Concept | Owner | Phase alignment | Notes |
|---|---|---|---|
| `ArchitectureSpec` / family registry | `models/` | Phase 03 | Source of truth for family capabilities, build facets, heads, loss plan, train adapter, inference capability, policy provider, default recipe, tune space. |
| Trunks and heads | `models/trunks`, `models/heads` | Phase 03 | Implementation modules selected by `ArchitectureSpec`; do not own replay, inference transport, or MCTS behavior. |
| Model assembly / `ModelBundle` | `models/assembly.py` | Phase 03 | Replaces direct `build_model_from_config` branching. |
| Target contracts | `contracts/targets.py` | Phase 01 | Defines target identity, masks, phases, schema versions, hashes. |
| Candidate, pair, and graph row builders | existing action/graph contract modules | Phase 02 | Remain canonical. Adapters wrap these builders first; do not reimplement pair semantics by head. |
| Replay projection | `replay/projectors.py` or buffer-owned projector module | Phase 07 | Decides what replay stores and emits based on architecture target requirements. |
| Training adapters and loss plan | `models/training/` | Phase 03 | Convert target bundles into model inputs and loss calls; preserve AMP/device/weight semantics. |
| Inference manifests and adapters | `inference/` | Phase 04 | Own request kinds, handshake, shared memory, batching, response packing, output validation. |
| Policy providers | `search/` or phase-approved provider module | Phase 05 | Map decoded model outputs to legal-row priors for MCTS. |
| Pair strategies | `search/` or phase-approved strategy module | Phase 05 | Own pair row generation, scoring eligibility, caps, required heads, and telemetry. |
| Engine adapter | `search/engine_adapter.py` or phase-approved engine module | Phase 05/06 | Only boundary allowed to call Rust MCTS expansion APIs. |
| Evaluation/dashboard consumers | `evaluation/`, `dashboard/` | Phase 08 | Consume canonical debug bundles and decoded outputs, not raw model tensors. |

This map is a guardrail against accidentally building a second architecture system beside the V2 refactor plan.

## Core Concepts

### ArchitectureSpec

`ArchitectureSpec` is the source of truth for a model family.

```python
@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    family: str
    input_contract: InputContract
    trunk: TrunkSpec
    heads: tuple[HeadSpec, ...]
    target_adapter: str
    training_adapter: str
    loss_plan: str
    inference_adapter: str
    inference_manifest: str
    policy_provider: str
    pair_capabilities: PairCapabilitySpec
    default_recipe: str
    tune_space: str
    checkpoint_manager: str
    runtime_constraints: RuntimeConstraintSpec
```

Examples:

```text
dense_cnn
restnet
graph_hybrid
global_graph_option1
global_xattn_0
global_pair_twostage_0
global_graph_full_0
global_graph768_champion
```

The architecture spec should replace scattered logic such as:

- `architecture.startswith("global_")`
- model-class checks in trainer setup
- server-side head assumptions
- worker-side architecture branches
- loss routing based only on raw head names

Every family spec should register Phase 03 facets:

- build facet
- train adapter
- inference adapter capability
- inference manifest id
- policy provider id
- loss plan
- checkpoint manager
- default recipe
- tune space
- supported target contracts
- supported output contracts

### InputContract

`InputContract` declares what tensors or row tables the architecture accepts.

```python
@dataclass(frozen=True)
class InputContract:
    name: str
    required_tensors: tuple[str, ...]
    optional_tensors: tuple[str, ...]
    row_tables: tuple[RowTableSpec, ...]
    schema_versions: tuple[SchemaVersionSpec, ...]
    required_hashes: tuple[str, ...]
    mutation_policy: str
```

Dense example:

```text
required_tensors:
- board_tensor

row_tables:
- dense_board_index_to_legal_row
```

Global graph example:

```text
required_tensors:
- token_features
- token_type
- token_qr
- token_mask
- legal_token_indices
- legal_mask
- relation_type
- relation_bias

optional_tensors:
- opp_legal_qr
- opp_legal_mask
- pair_first_indices
- pair_second_indices
- pair_token_indices
- crop_tensor

row_tables:
- legal_qr
- opp_legal_qr
- pair_action_rows
```

### TrunkSpec

`TrunkSpec` describes the representation builder.

```python
@dataclass(frozen=True)
class TrunkSpec:
    name: str
    module_factory: str
    output_slots: tuple[str, ...]
    required_input_contract: str
```

Examples:

```text
cnn_trunk:
- emits board_features
- emits state_embedding

resnet_trunk:
- emits board_features
- emits state_embedding

global_relation_graph_trunk:
- emits token_embeddings
- emits state_embedding
- emits legal_embeddings
- emits pair_row_embeddings when pair rows are supplied
```

### HeadSpec

`HeadSpec` declares head behavior in one place.

```python
@dataclass(frozen=True)
class HeadSpec:
    name: str
    module_factory: str
    input_slot: str
    output_name: str
    target_name: str | None
    mask_name: str | None
    loss_name: str | None
    semantic_phase: str | None
    output_contract: OutputContract
    runtime_consumers: tuple[str, ...]
```

Example head specs:

```text
policy_place:
- input_slot: legal_embeddings
- output_name: policy_place
- target_name: policy_target
- mask_name: legal_mask
- loss_name: graph_policy_ce
- semantic_phase: any_position
- runtime_consumers: policy_provider.global_graph

policy_pair_first:
- input_slot: legal_embeddings
- output_name: policy_pair_first
- target_name: pair_first_policy_target
- mask_name: legal_mask AND first_placement_phase
- loss_name: graph_policy_ce
- semantic_phase: first_placement_only
- runtime_consumers: pair_strategy only

policy_pair_joint:
- input_slot: pair_row_embeddings
- output_name: policy_pair_joint
- target_name: pair_policy_target
- mask_name: pair_row_mask
- loss_name: graph_pair_policy_ce
- semantic_phase: first_placement_only
- runtime_consumers: pair_strategy only

policy_pair_second:
- input_slot: known_first_pair_row_embeddings
- output_name: policy_pair_second
- target_name: pair_second_policy_target
- mask_name: known_first_pair_row_mask
- loss_name: graph_pair_policy_ce
- semantic_phase: second_placement_known_first_only
- runtime_consumers: pair_strategy only
```

This prevents head logic from being duplicated across model forward, trainer, inference server, and worker code.

### OutputContract

`OutputContract` declares shape and semantic identity.

```python
@dataclass(frozen=True)
class OutputContract:
    shape_kind: str
    row_table: str | None
    finite_mask: str | None
    invalid_fill_value: float
    schema_version: str
    row_hash: str | None
    source_label: str
    mutation_policy: str
```

Examples:

```text
policy_place:
- one logit per legal_qr row
- invalid rows are -80.0

policy_pair_first:
- one logit per legal first-placement row
- invalid outside first-placement phase

policy_pair_joint:
- one logit per canonical pair-action row
- unordered first-placement pair identity

policy_pair_second:
- one logit per known-first second-placement row
- first token must be known first placement
- second token must be legal after first placement
```

### RowTableContract

Row identity is core model infrastructure, not optional metadata.

```python
@dataclass(frozen=True)
class RowTableContract:
    name: str
    row_kind: str
    schema_version: int
    rows_hash: str
    source_hash: str
    source_owner: str
    semantic_phase: str | None
    mutation_policy: str
```

Required row kinds:

```text
legal_action_rows
dense_board_rows
sparse_candidate_rows
opponent_legal_rows
first_placement_pair_rows
known_first_second_rows
graph_legal_token_rows
```

Runtime-consumed logits must carry or reference a `RowTableContract`. If the row table is only present on the request side, the inference protocol must prove request rows are immutable and hash-stable through the response.

### TargetContract

Target contracts should live under `contracts/`, not under model implementations.

```python
@dataclass(frozen=True)
class TargetContract:
    name: str
    row_table: str
    semantic_phase: str
    probability_mass_policy: str
    mask_name: str
    weight_name: str | None
    builder_owner: str
    required_negative_tests: tuple[str, ...]
```

Target adapters initially wrap existing canonical builders:

- `build_graph_batch_from_history`
- `graph_batch_with_reference_pair_rows`
- `build_candidate_batch`
- `build_pair_candidate_batch`
- replay policy and value target projection

They must not reimplement first-placement unordered pair semantics, second-placement known-first semantics, opponent legal table semantics, critical candidate overflow handling, or missing-mass behavior until typed contract tests prove exact parity.

## Assembly Flow

Model assembly should use only the registered architecture spec.

```text
Config
  -> ArchitectureRegistry.resolve(config.model.architecture)
  -> ArchitectureSpec
  -> build input adapter
  -> build trunk
  -> build declared heads
  -> attach output contract metadata
  -> return ModelBundle
```

`ModelBundle` should include:

```python
@dataclass(frozen=True)
class ModelBundle:
    model: torch.nn.Module
    architecture: ArchitectureSpec
    head_specs: Mapping[str, HeadSpec]
    input_contract: InputContract
```

The trainer, inference server, dashboard, and self-play worker should receive a `ModelBundle` or architecture metadata rather than rediscovering behavior from strings.

## Training Organization

Training should be adapter-driven.

```text
Replay batch
  -> TargetAdapter builds declared targets
  -> TrainingAdapter builds model inputs
  -> model forward
  -> LossRegistry computes losses from HeadSpec
  -> metrics from HeadSpec metric hooks
```

The trainer should not know that global graph pair heads require:

- `pair_first_policy_target`
- `pair_policy_target`
- `pair_second_policy_target`
- `pair_first_indices`
- `pair_second_indices`
- `legal_mask`
- known-first masking

Instead, the trainer should iterate over registered head specs:

```python
for head in architecture.heads:
    if not head.loss_name:
        continue
    loss = loss_registry.compute(head, predictions, targets)
```

This keeps loss routing out of broad `if head_name == ...` blocks.

## Target Organization

Target builders should be centralized by target contract, not by caller.

```text
targets/dense_policy.py:
- dense board policy target
- dense legal mask

targets/sparse_policy.py:
- candidate-row target
- candidate mask

targets/graph_policy.py:
- legal_qr policy target
- opponent legal target

targets/pair_policy.py:
- pair_first_policy_target
- pair_policy_target
- pair_second_policy_target
- first-placement unordered validation
- second-placement known-first validation
```

Graph batch construction may still build raw row tables, but target semantics should live in target modules that can be tested independently.

## Inference Organization

Inference should decode model outputs through architecture-selected adapters.

```text
Inference request
  -> InferenceAdapter validates input contract
  -> model forward
  -> InferenceAdapter validates output contracts
  -> canonical response object
```

Canonical response types:

```python
@dataclass(frozen=True)
class PolicyOutput:
    legal_rows: LegalActionTable
    logits: np.ndarray
    source: str
    contract_hash: str

@dataclass(frozen=True)
class PairOutput:
    pair_rows: PairActionTable
    logits: np.ndarray
    head_name: str
    semantic_phase: str
    contract_hash: str
```

The inference server should not independently decide which graph heads exist or how to copy them. It should ask the adapter:

```python
response = adapter.decode(model_outputs, request_context)
```

The adapter owns:

- output presence checks
- finite checks
- shape checks
- legal-row alignment metadata
- pair-row count checks
- schema version checks
- response head flags

## Self-Play And Search Organization

Self-play should consume only canonical policy and pair evaluations.

```text
SearchContext
  -> PolicyProvider.evaluate
  -> SearchEvaluation
  -> PairStrategy.score_root or score_leaves
  -> PairEvaluation
  -> EngineAdapter.expand
```

Policy providers:

```text
DensePolicyProvider
RestNetPolicyProvider
GraphHybridPolicyProvider
GlobalGraphPolicyProvider
```

Pair strategies:

```text
none
two_stage_root_only
tactical_only
diagnostic_full_root
```

Hard boundary:

- Policy providers map policy logits to legal rows.
- Pair strategies decide if pair rows are generated or scored.
- Engine adapters are the only code allowed to call Rust MCTS expansion APIs.

This removes direct pair-head handling from self-play worker branches.

## Extension Stories

### Add A New Architecture

A new architecture should be mostly declarative if it reuses existing parts.

Process:

- add or reuse a trunk
- add or reuse heads
- declare an `ArchitectureSpec`
- choose existing target, training, inference, and provider adapters
- add contract tests

No trainer, inference server, or self-play worker branch should be necessary.

### Add A New Head

A new head should be self-contained.

Process:

- implement the head module
- declare `HeadSpec`
- declare output contract
- declare target and mask if trainable
- declare loss if trainable
- declare inference decode only if runtime consumes it
- add shape, mask, loss, and adapter tests

If the head is auxiliary only, runtime adapters should ignore it unless a consumer explicitly asks for it.

### Add A New Target

A target should be defined independently of any one model class.

Process:

- define target contract
- implement target builder
- validate row-table identity
- validate semantic phase
- expose tensor names used by head specs
- add positive and negative tests

This prevents target logic from being duplicated in replay, graph builders, dashboard tools, and trainer code.

### Add A New Runtime Consumer

Runtime consumers should use canonical outputs.

Process:

- define the consumer protocol
- consume `PolicyOutput`, `PairOutput`, or `AuxiliaryOutput`
- reject raw tensor dictionaries
- report contract telemetry

This allows dashboard, evaluation, self-play, and debugging tools to agree on what outputs mean.

### Add A New Pair Strategy

Pair strategies should be explicit and safe by default.

Process:

- declare `PairStrategySpec`
- declare root and leaf eligibility
- declare required heads
- declare row caps
- declare semantic phase support
- use canonical pair-row builders
- return `PairEvaluation`

The strategy must not be activated by architecture name, head presence, or checkpoint metadata.

## Ownership Map

```text
ArchitectureSpec owns:
- which parts are assembled
- which contracts are active
- which adapters are selected

InputContract owns:
- required tensors
- optional tensors
- schema versions
- row table inputs

TrunkSpec owns:
- representation slots emitted by the trunk
- compatibility with input contracts

HeadSpec owns:
- head module
- output name
- target name
- mask name
- loss name
- semantic phase
- output contract

TargetAdapter owns:
- target construction
- target validation
- target tensor naming

TrainingAdapter owns:
- batch-to-model input mapping
- model-call shape
- target bundle handoff

LossRegistry owns:
- loss dispatch
- loss weighting
- missing-target behavior
- metric hooks

InferenceAdapter owns:
- request validation
- output validation
- canonical decoding
- response metadata

PolicyProvider owns:
- row-mapped policy priors
- policy source telemetry

PairStrategy owns:
- whether pair rows are generated
- whether pair heads are scored
- pair row caps
- pair influence telemetry

EngineAdapter owns:
- Rust MCTS expansion calls
- mutation protection
- MCTS input validation
```

## Foundation Quality Bar

The design is only successful if it reduces future local reasoning cost.

A developer adding or debugging a model should be able to answer these questions from the architecture spec and contracts:

- What tensors does this model consume?
- What row tables do those tensors correspond to?
- What heads does this model emit?
- Which heads are trainable?
- Which targets train them?
- Which masks define valid rows?
- Which heads can affect runtime search?
- Which adapter decodes inference outputs?
- Which provider maps priors to MCTS rows?
- Which pair strategy, if any, consumes pair outputs?
- What telemetry proves the path was valid?

If answering any of those requires searching trainer, inference server, worker, replay, and dashboard code, the modularization is incomplete.

## Architecture-Owned Examples

### Dense CNN

```text
architecture: dense_cnn
input_contract: dense_board_v1
trunk: cnn_trunk
heads:
- policy_dense
- value
- optional auxiliaries
target_adapter: dense_target_adapter
training_adapter: dense_training_adapter
inference_adapter: dense_inference_adapter
policy_provider: dense_policy_provider
pair_capabilities: none
```

### Graph Hybrid

```text
architecture: graph_hybrid_0
input_contract: crop_plus_sparse_candidates_v1
trunk: cnn_with_graph_tokens
heads:
- policy_dense
- sparse_policy
- value
- optional pair_policy candidate scorer
target_adapter: sparse_candidate_target_adapter
training_adapter: graph_hybrid_training_adapter
inference_adapter: sparse_candidate_inference_adapter
policy_provider: graph_hybrid_policy_provider
pair_capabilities: candidate_pair_auxiliary_only
```

### Global Graph

```text
architecture: global_graph_option1
input_contract: global_graph_v1
trunk: relation_graph_trunk
heads:
- policy_place
- value
- optional opponent_policy
- optional policy_pair_first
- optional policy_pair_joint
- optional policy_pair_second
- optional tactical
- optional regret
target_adapter: global_graph_target_adapter
training_adapter: global_graph_training_adapter
inference_adapter: global_graph_inference_adapter
policy_provider: global_graph_policy_provider
pair_capabilities: output_heads_only
```

### Global Pair Two-Stage

```text
architecture: global_pair_twostage_0
input_contract: global_graph_v1
trunk: graph_cross_attention_or_relation_trunk
heads:
- policy_place
- value
- policy_pair_first
- policy_pair_joint
- policy_pair_second
target_adapter: global_graph_pair_target_adapter
training_adapter: global_graph_training_adapter
inference_adapter: global_graph_inference_adapter
policy_provider: global_graph_policy_provider
pair_capabilities: pair_strategy_required
```

## Migration Plan

### Step 1: Add Registry And Specs

Create `model/registry.py`, `model/contracts.py`, and architecture specs that describe current behavior without changing runtime behavior.

Completion evidence:

- existing model configs resolve to specs
- every configured head has a `HeadSpec`
- every current architecture has an `InputContract`

### Step 2: Move Model Assembly Behind Specs

Replace direct model-family branching in `build_model_from_config` with registry-based assembly.

Completion evidence:

- dense, graph hybrid, and global graph models build through the same assembly function
- architecture aliases are resolved by registry, not prefix checks

### Step 3: Move Loss Routing To HeadSpec

Replace trainer head-name conditionals with `LossRegistry` and `HeadSpec` declarations.

Completion evidence:

- each trained head declares target, mask, loss, and semantic phase
- skipped heads report missing target or mask explicitly
- pair-second loss only runs on known-first targets

### Step 4: Split Target Builders

Move policy, sparse, graph, pair, opponent, and tactical target construction into `model/targets/`.

Completion evidence:

- target tests no longer need to instantiate full model classes
- pair target tests prove first-placement and second-placement semantics independently

### Step 5: Introduce Inference Adapters

Move output decoding and shared-memory response packing behind architecture-selected inference adapters.

Completion evidence:

- inference server no longer checks graph-specific heads directly
- adapter validates output shapes and row counts
- graph policy and pair outputs are decoded into canonical row-keyed responses

### Step 6: Move Self-Play Consumption To Providers And Strategies

Move root and leaf policy evaluation behind `PolicyProvider`; move pair scoring behind `PairStrategy`.

Completion evidence:

- self-play worker does not directly inspect pair head names
- no pair scoring from head presence, model family, or `pair_prior_mix`
- leaf pair scoring remains disabled unless a validated strategy declares it

### Step 7: Remove Legacy Branches

Delete or quarantine obsolete direct branches after adapters/providers cover all runtime paths.

Completion evidence:

- code search proves no architecture prefix checks remain in runtime selection
- code search proves no direct pair-head MCTS consumption remains outside pair strategies
- tests cover dense, sparse, graph hybrid, and global graph runtime paths

## Acceptance Checklist

- Architecture specs own input contracts.
- Architecture specs own trunk selection.
- Architecture specs own head declarations.
- Head specs own target names.
- Head specs own masks and semantic phases.
- Head specs own loss names.
- Target adapters own target construction.
- Training adapters own model input construction.
- Loss registry owns loss dispatch.
- Inference adapters own raw-output validation and decoding.
- Policy providers own row-mapped priors.
- Pair strategies own pair scoring.
- Engine adapters own Rust MCTS calls.
- No runtime branch uses architecture prefixes as behavior.
- No pair behavior is enabled by head presence.
- No model output reaches MCTS without canonical row mapping.

## Suggested Tests

```text
test_every_architecture_resolves_to_spec
test_every_head_declares_output_contract
test_every_trainable_head_declares_target_mask_and_loss
test_dense_assembly_uses_dense_contract
test_graph_hybrid_assembly_uses_candidate_contract
test_global_graph_assembly_uses_global_graph_contract
test_trainer_loss_dispatch_uses_head_specs
test_pair_second_loss_requires_known_first_phase
test_inference_adapter_rejects_missing_required_head
test_inference_adapter_rejects_output_row_count_mismatch
test_policy_provider_maps_global_graph_logits_to_legal_rows
test_pair_strategy_none_ignores_pair_heads
test_pair_strategy_required_for_pair_scoring
test_worker_does_not_branch_on_architecture_prefix
```

## Long-Term Shape

The end state should feel like this:

```python
spec = ArchitectureRegistry.resolve(cfg.model.architecture)
bundle = assemble_model(spec, cfg)
train_adapter = TrainingAdapterRegistry.resolve(spec.training_adapter)
infer_adapter = InferenceAdapterRegistry.resolve(spec.inference_adapter)
provider = PolicyProviderRegistry.resolve(spec.policy_provider)
pair_strategy = PairStrategyRegistry.resolve(cfg.model.pair_strategy)
```

The architecture spec becomes the place where model logic is defined. Training, inference, dashboard, and self-play become consumers of that definition rather than secondary places where model behavior is rediscovered.
