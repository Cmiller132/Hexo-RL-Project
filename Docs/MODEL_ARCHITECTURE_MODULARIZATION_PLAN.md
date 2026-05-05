# Modular Model Architecture Implementation Plan

## Purpose

This document is the implementation plan for replacing the fragmented model architecture logic with a clean, contract-first system under `Python/src/hexorl/models/`.

The plan is intentionally not a long wrapper migration. Current behavior will be inventoried so we know what exists, but the rewrite will intentionally keep, replace, simplify, or delete each behavior. The target is a clean two-stage cutover with no permanent old/new runtime paths.

## Rechecked Codebase Claims

The plan below was checked against the current source layout and these claims are grounded in codebase searches.

| Claim | Current evidence |
|---|---|
| Model assembly is currently centralized in the legacy `hexorl/model` package. | `Python/src/hexorl/model/network.py` defines `HexNet` and `build_model_from_config`; `Python/src/hexorl/model/global_graph.py` defines `GlobalHexGraphNet`. |
| Global graph family membership is currently embedded in model/config/runtime-adjacent code. | `GlobalHexGraphNet.ARCHITECTURES` exists in `model/global_graph.py`; graph-family names are also repeated in config and buffer code. |
| Training loss routing is currently head-name driven. | `Python/src/hexorl/train/losses.py` defines `compute_losses` with raw branches for `policy`, `sparse_policy`, `pair_policy`, `policy_place`, `policy_pair_first`, `policy_pair_joint`, `policy_pair_second`, and auxiliaries. |
| Trainer behavior still branches on model class and graph tensors. | `Python/src/hexorl/train/trainer.py` imports `GlobalHexGraphNet`, imports `compute_losses`, and has graph-specific model input handling. |
| Graph and pair target construction is split across graph and buffer code. | `Python/src/hexorl/graph/batch.py` builds graph batches, legal rows, pair rows, and graph pair targets; `Python/src/hexorl/buffer/sampler.py` emits aux targets and graph tensors; `Python/src/hexorl/buffer/targets.py` fills replay pair targets. |
| Inference output decode is currently tied to fixed shared-memory arrays and flags. | `Python/src/hexorl/inference/server.py` reads graph request metadata, sanitizes raw dict heads, and fills graph/pair result arrays; `client.py` reconstructs graph outputs; `shm_queue.py` defines fixed arrays including graph pair logits. |
| Self-play pair behavior is partly strategy-gated but still directly consumes raw pair heads. | `Python/src/hexorl/selfplay/worker.py` stores `pair_strategy`, calls graph pair scoring helpers, checks `policy_pair_first`, `policy_pair_joint`, and `policy_pair_second`, and blends/apply pair priors. |
| Config currently owns architecture/head/pair validation that should move into resolved specs where possible. | `Python/src/hexorl/config/schema.py` validates pair strategies, pair heads, architecture/head relationships, and default graph loss weights. |
| Row identity is the core correctness boundary. | Graph legal rows, pair indices, pair targets, legal masks, IPC counts, and Rust legal-row alignment appear across graph batch, sampler, inference, and self-play code. |

## Design Position

This is a clean rewrite, not a compatibility wrapper project.

Inventory exists to make the rewrite intentional. It does not mean preserving every current behavior. Each current behavior must be classified as one of:

```text
keep
replace
simplify
delete
move behind contract
```

The new architecture authority lives under:

```text
Python/src/hexorl/models/
```

The old package remains only as temporary implementation source during the cutover:

```text
Python/src/hexorl/model/
```

After cutover, runtime-facing code must not use `hexorl.model` as an architecture authority. It may import implementation modules only if they have been deliberately retained as low-level components.

## Design Goals

- Define each architecture from one registered spec.
- Make heads, trunks, targets, losses, masks, adapters, providers, and pair capabilities explicit.
- Make row identity first-class everywhere logits or targets are trained or consumed.
- Remove raw head-name loss routing from the trainer.
- Remove model-class and architecture-prefix runtime selection.
- Remove pair scoring from worker branches and move it into executable pair strategies.
- Replace inference head decoding with architecture-selected adapters and protocol objects.
- Keep shared memory as a transport detail, not as the model-output contract.
- Make adding a new architecture mostly declarative when it reuses existing parts.
- End with no permanent old/new dual runtime path.

## Non-Goals

- Do not split every linear layer into a separate file just for aesthetics.
- Do not preserve silent target/loss skips as default behavior.
- Do not add a wrapper facade that keeps current trainer, inference, and self-play behavior intact indefinitely.
- Do not let architecture specs enable pair scoring by head presence.
- Do not make config mutation the source of resolved model behavior.
- Do not rewrite Rust MCTS rules or legal move generation as part of this model refactor.

## Core Architecture

### ArchitectureSpec

`ArchitectureSpec` is the source of truth for a model family.

```python
@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    version: str
    family: str
    input_contract: str
    trunk: str
    heads: tuple[str, ...]
    default_trainable_heads: tuple[str, ...]
    target_adapter: str
    training_adapter: str
    inference_adapter: str
    policy_provider: str
    supported_pair_strategies: tuple[str, ...]
    config_schema: str
    telemetry_contract: str
```

A spec defines what the architecture is allowed to do. It does not directly decide whether pair outputs affect MCTS. That remains the job of `PairStrategySpec`.

### HeadSpec

`HeadSpec` defines one output head.

```python
@dataclass(frozen=True)
class HeadSpec:
    name: str
    module: str
    input_slot: str
    output_name: str
    output_contract: str
    trainable: bool
    target_contract: str | None
    mask_contract: str | None
    loss: str | None
    weight_key: str | None
    semantic_phase: str
    runtime_consumers: tuple[str, ...]
    missing_target_policy: str
```

Allowed `missing_target_policy` values:

```text
error
skip_optional
not_trainable
```

Default for trainable heads is `error`. Current silent skips should be removed unless a head is explicitly optional and non-required for that architecture.

### TrunkSpec

`TrunkSpec` defines reusable representation builders.

```python
@dataclass(frozen=True)
class TrunkSpec:
    name: str
    module: str
    input_contract: str
    output_slots: tuple[str, ...]
    required_features: tuple[str, ...]
```

Example output slots:

```text
board_features
state_embedding
candidate_embeddings
legal_embeddings
pair_embeddings
token_embeddings
```

### RowTableContract

Row tables are in scope for this refactor. They are needed because the highest-risk failures are row/logit mismatches.

```python
@dataclass(frozen=True)
class RowTableContract:
    name: str
    schema_version: int
    semantic_phase: str
    row_count: int
    row_hash: str
    owner: str
```

Required row table families:

```text
dense_board_rows
candidate_action_rows
legal_action_rows
opponent_legal_rows
first_placement_pair_rows
known_first_second_rows
graph_legal_token_rows
```

Every runtime-consumed policy or pair output must reference a row table. Raw logits without row identity are invalid outside the model forward pass.

### TargetContract

Targets are not model implementation details. They are training contracts.

```python
@dataclass(frozen=True)
class TargetContract:
    name: str
    row_table: str
    semantic_phase: str
    tensor_name: str
    mask_name: str
    weight_name: str | None
    probability_mass: str
    builder: str
    negative_tests: tuple[str, ...]
```

Target builders may be rewritten, but they must preserve the rules currently enforced across graph and buffer code.

Important target rules:

```text
legal policy targets map to legal rows
sparse policy targets map to candidate rows
pair first targets map to legal first-placement rows
pair joint targets map to unordered first-placement pair rows
pair second targets map to known-first legal second rows
opponent policy targets map to independent opponent legal rows
invalid duplicate pair rows fail
illegal pair rows fail
stale or mismatched row identity fails
zero target mass is explicit, not silently trained
```

### LossPlan

Loss behavior should be generated from architecture and head specs.

```python
@dataclass(frozen=True)
class LossPlanEntry:
    head: str
    prediction: str
    target: str
    mask: str | None
    loss: str
    weight_key: str | None
    semantic_phase: str
    missing_target_policy: str
```

The trainer should not branch on raw head names. It should ask the resolved `LossPlan`.

### Inference Protocol

The refactor should improve inference, not merely wrap the current shared-memory arrays.

Separate protocol from transport:

```text
InferenceRequest
InferenceResponse
InferenceAdapter
SharedMemoryTransport
```

`InferenceRequest` owns:

```text
request id
architecture id
input contract
requested outputs
schema versions
row table contracts
tensor payload metadata
```

`InferenceResponse` owns:

```text
request id
architecture id
output contracts
policy outputs
pair outputs
value outputs
auxiliary outputs
telemetry
warnings
```

Shared memory remains a high-performance transport. It should pack and unpack protocol fields, but it should not define what heads mean.

### PolicyProvider

Policy providers convert decoded model policy outputs into search evaluations.

Required providers:

```text
DensePolicyProvider
SparseCandidatePolicyProvider
GraphHybridPolicyProvider
GlobalGraphPolicyProvider
```

Each provider returns a row-mapped `SearchEvaluation`.

### PairStrategySpec

Pair strategies are executable runtime logic, not just metadata.

```python
@dataclass(frozen=True)
class PairStrategySpec:
    name: str
    required_heads: tuple[str, ...]
    root_enabled: bool
    leaf_enabled: bool
    phases: tuple[str, ...]
    max_root_rows: int
    max_leaf_rows: int
    max_full_rows: int
    blend_policy: str
    fallback_policy: str
    diagnostic: bool
```

Architecture specs may declare pair output capability. Only pair strategies decide whether pair rows are generated, pair heads are scored, or pair logits influence MCTS.

## Proposed Package Layout

```text
Python/src/hexorl/models/
  __init__.py
  registry.py
  specs.py
  assembly.py
  bundles.py
  validation.py
  configs.py
  specs/
    dense.py
    restnet.py
    graph_hybrid.py
    global_graph.py
  trunks/
    cnn.py
    residual.py
    graph_tokens.py
    relation_graph.py
    cross_attention_graph.py
  heads/
    dense_policy.py
    sparse_policy.py
    graph_policy.py
    pair_first.py
    pair_joint.py
    pair_second.py
    value.py
    opponent_policy.py
    tactical.py
    regret.py
    lookahead.py
  training/
    adapters.py
    loss_plan.py
    losses.py
    metrics.py

Python/src/hexorl/contracts/
  __init__.py
  rows.py
  targets.py
  tensors.py
  hashes.py
  schemas.py
  phases.py
  traces.py

Python/src/hexorl/inference/
  protocol.py
  manifests.py
  transports.py
  adapters/
    dense.py
    sparse.py
    graph_hybrid.py
    global_graph.py
    pair_scoring.py

Python/src/hexorl/search/
  context.py
  policy_provider.py
  pair_strategy.py
  engine_adapter.py

Python/src/hexorl/replay/
  projection.py
  target_builders.py
  training_batch.py
```

## Current Behavior Inventory Required Before Cutover

Stage 1 must produce inventories. These are not compatibility promises. They are decision tables.

### Architecture Inventory

For each current architecture:

```text
architecture id
legacy class or constructor
required input tensors
optional input tensors
supported heads
default heads
loss defaults
inference adapter needed
policy provider needed
pair capabilities
current runtime consumers
keep/replace/delete decision
```

### Head And Loss Inventory

For each current head:

```text
head name
prediction key
current target keys
current mask keys
current weight keys
current loss function
current skip behavior
current fallback behavior
semantic phase
runtime consumers
new decision: keep, replace, simplify, delete
new HeadSpec
```

Fallback target aliases should be removed unless they are explicitly justified.

### Target Inventory

For each target:

```text
target name
source record fields
row table
mask
semantic phase
normalization rule
invalid input behavior
negative tests
new TargetContract
```

### Inference Inventory

For each inference path:

```text
request kind
input tensors
row tables
shared-memory fields
raw model output keys
response fields
head flags
runtime consumers
new protocol object
new adapter
```

### Runtime Inventory

For each runtime consumption path:

```text
consumer
required policy outputs
required pair outputs
row table expectations
MCTS API used
telemetry emitted
new provider or strategy
old branch to delete
```

## Why A Rewrite Is Hard But Manageable

The rewrite is not hard because PyTorch modules are hard. It is hard because semantics are distributed.

The critical risks are:

```text
wrong logits matched to legal rows
wrong pair phase used for pair rows
silent loss skip hides missing targets
shared-memory response flags lose output meaning
pair head availability changes MCTS behavior
config mutation creates behavior outside architecture specs
```

The controls are:

```text
row contracts
target contracts
strict loss plans
inference protocol validation
executable pair strategies
vertical cutover
old branch deletion
```

## Two-Stage Implementation Plan

This rewrite has exactly two stages.

Stage 1 does not install permanent wrappers. It locks design, inventory, and test evidence.

Stage 2 performs the clean implementation cutover and deletes old scattered logic.

## Stage 1: Design Lock, Inventory, And Golden Tests

### Goal

Create the exact implementation blueprint and proof harness needed to rewrite cleanly without preserving accidental complexity.

### Success Criteria

- `hexorl/models/` is selected as the new architecture authority.
- Current architecture, head/loss, target, inference, and runtime behavior is inventoried.
- Each inventory row has a keep/replace/simplify/delete/move decision.
- Row table and target contracts are designed before model code is moved.
- Golden tests capture rules that must survive the rewrite.
- Silent loss skips and fallback aliases are classified for removal or explicit retention.
- Shared-memory constraints are represented as transport constraints, not output semantics.
- Pair strategies are specified as executable runtime plans.

### Constraints

- Do not add permanent runtime wrappers.
- Do not introduce a second runtime path that remains after Stage 2.
- Do not physically split trunks and heads until contracts and specs are locked.
- Do not claim a behavior is preserved unless a golden test covers it.
- Do not use architecture name prefixes as a future behavior mechanism.

### Required Evidence

- `Docs/refactor/artifacts/model_architecture/architecture_inventory.md`
- `Docs/refactor/artifacts/model_architecture/head_loss_inventory.md`
- `Docs/refactor/artifacts/model_architecture/target_inventory.md`
- `Docs/refactor/artifacts/model_architecture/inference_inventory.md`
- `Docs/refactor/artifacts/model_architecture/runtime_inventory.md`
- Contract draft files or design notes for row tables, targets, inference protocol, and pair strategies.
- Golden test list with exact existing tests to keep and new tests to add.

### Stop Rules

- Stop if a current trained head cannot be mapped to a target, mask, loss, and phase.
- Stop if a runtime-consumed output cannot be tied to a row table.
- Stop if shared-memory transport cannot carry required contract identity without a schema change.
- Stop if pair strategy behavior cannot be separated from architecture capability.

### Stage 1 Work Items

1. Create model architecture inventories under `Docs/refactor/artifacts/model_architecture/`.
2. Define semantic phases in `contracts/phases.py` design notes.
3. Define row contracts for dense, candidate, legal, opponent legal, pair joint, and known-first rows.
4. Define target contracts for dense policy, sparse policy, graph policy, pair first, pair joint, pair second, opponent policy, value, tactical, regret, and lookahead.
5. Define architecture specs for current dense CNN, RestNet, graph hybrid, sparse policy, global graph, global x-attn, global pair two-stage, global full graph, and champion graph families.
6. Define head specs for all current heads that remain supported.
7. Define loss plan entries for all trainable heads and mark silent skip behavior for removal unless explicitly optional.
8. Define inference protocol fields and shared-memory transport mapping.
9. Define `PairStrategySpec` entries for `none`, `two_stage_root_only`, `tactical_only`, and `diagnostic_full_root`.
10. Decide which current behavior is deleted instead of migrated.

## Stage 2: Clean Cutover Implementation

### Goal

Implement the new modular architecture system and delete old scattered model behavior in one clean cutover.

### Success Criteria

- `hexorl/models/` is the architecture authority.
- `build_model_from_config` delegates to `hexorl.models.assembly` or is replaced by it.
- Model construction uses registered `ArchitectureSpec` and returns `ModelBundle`.
- Trainer uses `TrainingAdapter` and `LossPlan`, not raw head-name loss routing.
- Target construction uses `TargetContract` and row table contracts.
- Inference uses protocol/adapters and treats shared memory as transport.
- Self-play and evaluation use `PolicyProvider`, `PairStrategy`, and `EngineAdapter` boundaries.
- Config no longer owns architecture default expansion or graph-family capability rules.
- Old architecture-name lists are removed from config, buffer, and model implementation where they are behavior authority.
- Old direct pair-head MCTS consumption is removed from self-play worker.
- Old graph-specific inference head decode branches are removed from server/client hot logic and replaced by adapters.

### Constraints

- No permanent compatibility facade.
- No old/new runtime behavior remains active together after cutover.
- No pair scoring from head presence.
- No trainable head silently skips missing target or mask.
- No model output reaches MCTS without row contract validation.
- Shared-memory performance constraints must be preserved or measured.

### Required Evidence

- Unit tests for architecture registry, head specs, loss plans, target contracts, row contracts, inference adapters, policy providers, pair strategies, and engine adapter inputs.
- Integration tests for dense, sparse, graph hybrid, and global graph training batches.
- Inference adapter round-trip tests for dense and global graph requests.
- Self-play/provider tests proving pair strategy controls pair behavior.
- Code search proving removed behavior branches are gone.
- Performance smoke evidence for inference shared-memory transport if protocol packing changes.

### Stop Rules

- Stop if dense, graph hybrid, or global graph cannot train through the new trainer adapter.
- Stop if inference cannot map policy outputs to row contracts.
- Stop if self-play still directly checks pair output head names outside pair strategies.
- Stop if config still mutates resolved loss/head behavior in place of architecture specs.
- Stop if any old direct branch remains and has no deletion ticket or explicit quarantine record.

## Stage 2 Implementation Work Breakdown

### 2.1 Contracts

Create `hexorl/contracts/`.

Required modules:

```text
rows.py
targets.py
tensors.py
hashes.py
schemas.py
phases.py
traces.py
```

Implement:

```text
RowTableContract
TargetContract
TensorContract
ContractTrace
stable row hash helpers
semantic phase constants
schema version constants
```

Acceptance criteria:

```text
row contracts can hash legal_qr and pair rows
target contracts reject mismatched row hashes
tensor contracts validate shape, dtype, device-neutral metadata
semantic phases cover first-placement, second-placement known-first, any-position, auxiliary-only
```

### 2.2 Model Specs And Assembly

Create `hexorl/models/`.

Required modules:

```text
registry.py
specs.py
assembly.py
bundles.py
validation.py
configs.py
```

Implement:

```text
ArchitectureSpec
HeadSpec
TrunkSpec
OutputContract
ModelBundle
ArchitectureRegistry
build_model_bundle
validate_architecture_request
```

Initial assembly can instantiate retained implementation modules, but architecture authority must come from `hexorl/models/`.

Acceptance criteria:

```text
all current architecture ids resolve through registry
unsupported heads fail at spec resolution
architecture defaults are resolved without mutating Config in scattered validators
build_model_bundle returns model plus spec metadata
dense and global graph build through the same public assembly API
```

### 2.3 Heads And Trunks

Create reusable modules only where they reduce real coupling.

Minimum cutover:

```text
existing HexNet and GlobalHexGraphNet behavior may be ported or retained as trunk modules
new head modules should be split when head semantics need separate specs or masks
```

Preferred end state:

```text
trunks own representations
heads own small output projections
HeadSpec owns semantics
ArchitectureSpec owns composition
```

Acceptance criteria:

```text
model forward emits outputs declared by architecture spec
optional heads are requested through specs, not scattered string checks
pair head masks are phase-aware
output contracts validate shape and row count
```

### 2.4 Targets And Replay Projection

Create `hexorl/replay/` modules for projection and training batch conversion.

Required modules:

```text
projection.py
target_builders.py
training_batch.py
```

Implement clean target builders using contracts.

Important: this is allowed to rewrite target construction, but it must be test-driven against Stage 1 golden rules. Do not preserve old code shape just because it exists.

Acceptance criteria:

```text
replay positions project to canonical row tables first
targets reference row table contracts
pair target builders enforce unordered first-placement and known-first second-placement semantics
illegal target rows fail before tensors reach trainer
global graph training does not accidentally consume dense policy fields
```

### 2.5 Training Adapter And Loss Plan

Create `hexorl/models/training/`.

Required modules:

```text
adapters.py
loss_plan.py
losses.py
metrics.py
```

Implement:

```text
TrainingBatch
TrainingAdapter
LossPlan
LossPlanValidator
LossRegistry
MetricRegistry
```

The trainer flow becomes:

```text
raw replay batch
-> TrainingAdapter.prepare_batch
-> model inputs
-> model forward
-> LossPlanValidator.validate
-> LossRegistry.compute
-> metrics
```

Acceptance criteria:

```text
trainer has no broad raw head-name loss switch
trainable heads fail loudly when required target or mask is absent
optional non-trainable heads are skipped explicitly
loss weights come from resolved loss plan
pair-second loss only runs in known-first phase
```

### 2.6 Inference Protocol And Adapters

Create protocol and adapters under `hexorl/inference/`.

Required modules:

```text
protocol.py
transports.py
adapters/dense.py
adapters/sparse.py
adapters/graph_hybrid.py
adapters/global_graph.py
adapters/pair_scoring.py
```

Implement:

```text
InferenceRequest
InferenceResponse
PolicyOutput
PairOutput
ValueOutput
AuxiliaryOutput
InferenceAdapter
SharedMemoryTransport
```

Shared memory should remain optimized but become transport-only.

Acceptance criteria:

```text
server asks adapter to prepare inputs and decode outputs
adapter validates output presence, shape, finite values, row count, schema version, and head flags
client receives decoded response metadata rather than inferring semantics from arrays alone
existing shm arrays are either preserved with contract metadata or replaced with measured equivalent transport
```

### 2.7 Policy Providers, Pair Strategies, Engine Adapter

Create `hexorl/search/`.

Required modules:

```text
context.py
policy_provider.py
pair_strategy.py
engine_adapter.py
```

Implement:

```text
SearchContext
SearchEvaluation
PairEvaluation
PolicyProvider
PairStrategy
PairStrategySpec
EngineAdapter
```

Acceptance criteria:

```text
self-play worker delegates policy evaluation to provider
pair strategy owns all pair row generation, scoring, caps, phases, and blend behavior
engine adapter owns Rust MCTS calls
worker no longer directly checks pair head names
leaf pair scoring is disabled unless strategy explicitly enables it and tests prove validity
```

### 2.8 Config Resolution

Move architecture behavior resolution out of config mutation.

Config should keep:

```text
syntax validation
range validation
type validation
obvious local invariants
```

Architecture registry should own:

```text
architecture membership
supported heads
default heads
default loss plan
supported input contracts
supported pair strategies
adapter/provider selection
```

Acceptance criteria:

```text
Config no longer duplicates graph architecture membership lists
resolved model/loss behavior comes from registry
invalid head/architecture combinations fail during spec resolution with clear errors
```

### 2.9 Legacy Deletion

Delete or quarantine old behavior after new path owns the boundary.

Delete targets:

```text
raw head-name loss switch in trainer/losses
architecture behavior lists outside registry
self-play direct pair-head consumption
inference server direct graph head decode logic
config mutation of graph head loss defaults
buffer duplicated graph architecture constants
```

Acceptance criteria:

```text
code search confirms no forbidden runtime authority remains
deleted behavior has replacement tests
any quarantine has an owner, reason, and removal date
```

## Required Test Plan

### Contract Tests

```text
test_row_table_contract_hashes_legal_rows
test_row_table_contract_rejects_mismatched_pair_rows
test_target_contract_requires_matching_row_hash
test_semantic_phase_first_vs_second_pair_rows
```

### Architecture Tests

```text
test_every_current_architecture_resolves_to_spec
test_every_supported_head_declares_output_contract
test_every_trainable_head_declares_target_mask_loss_phase
test_invalid_head_for_architecture_fails_spec_resolution
test_pair_capable_architecture_does_not_enable_pair_strategy
```

### Assembly Tests

```text
test_dense_bundle_builds_from_registry
test_graph_hybrid_bundle_builds_from_registry
test_global_graph_bundle_builds_from_registry
test_build_model_from_config_delegates_to_models_assembly
```

### Target Tests

```text
test_dense_policy_target_maps_to_dense_rows
test_sparse_policy_target_maps_to_candidate_rows
test_graph_policy_target_maps_to_legal_rows
test_pair_joint_target_unordered_first_placement
test_pair_second_target_known_first_only
test_pair_targets_reject_duplicates_and_illegal_rows
test_global_graph_training_does_not_consume_dense_policy_target
```

### Loss Tests

```text
test_loss_plan_fails_missing_required_target
test_loss_plan_fails_missing_required_mask
test_optional_head_skip_is_explicit
test_pair_second_loss_requires_known_first_phase
test_loss_weights_resolve_from_architecture_loss_plan
```

### Inference Tests

```text
test_dense_inference_adapter_decodes_policy_value
test_global_graph_inference_adapter_decodes_policy_place
test_global_graph_adapter_rejects_legal_row_count_mismatch
test_pair_output_requires_pair_row_contract
test_shared_memory_transport_preserves_contract_metadata
```

### Runtime Tests

```text
test_policy_provider_maps_dense_policy_to_legal_rows
test_global_graph_provider_maps_logits_to_rust_legal_rows
test_pair_strategy_none_scores_zero_pairs
test_pair_head_presence_does_not_enable_pair_scoring
test_pair_strategy_declares_required_heads_and_caps
test_worker_does_not_directly_consume_pair_head_names
test_engine_adapter_rejects_unmapped_policy_output
```

### Audit Commands

```text
rg -n "architecture\.startswith|startswith\(\"global_|GlobalHexGraphNet\.ARCHITECTURES|GLOBAL_GRAPH_ARCHITECTURES" Python/src/hexorl
rg -n "policy_pair_first|policy_pair_joint|policy_pair_second" Python/src/hexorl/selfplay Python/src/hexorl/inference
rg -n "if head_name ==|elif head_name ==" Python/src/hexorl/train Python/src/hexorl/models
rg -n "pair_prior_mix" Python/src/hexorl/selfplay Python/src/hexorl/search Python/src/hexorl/inference
```

Expected audit outcome: any remaining hits are declarations, specs, tests, or pair strategy code, not scattered runtime behavior authority.

## Implementation Assignments

Each implementation assignment must be framed like this.

```text
Goal
Success criteria
Constraints
Required evidence
Stop rules
```

Example assignment:

```text
Goal
Implement row and target contracts for legal, candidate, graph legal, pair joint, and known-first pair rows.

Success criteria
All target builders produce targets that reference row contracts. Mismatched row hashes fail before training or inference consumption.

Constraints
Do not change Rust legal move generation. Do not allow raw logits or targets to be consumed without row identity.

Required evidence
Contract tests, target negative tests, and examples showing row hashes in training and inference traces.

Stop rules
Stop if any runtime-consumed output cannot identify its row table.
```

## Final Acceptance Checklist

- `hexorl/models/` owns architecture specs, assembly, model bundles, head specs, and loss plans.
- `hexorl/contracts/` owns row, tensor, target, schema, phase, and trace contracts.
- `hexorl/replay/` owns replay projection and training batch conversion.
- `hexorl/inference/` owns protocol, adapters, and transport mapping.
- `hexorl/search/` owns policy providers, pair strategies, and engine adapter boundaries.
- `hexorl/model/` is no longer runtime architecture authority.
- Config validation no longer mutates or derives architecture behavior that belongs to specs.
- Trainer has no broad raw head-name loss switch.
- Inference server does not interpret graph pair heads directly.
- Self-play worker does not directly consume pair output head names.
- Pair behavior is impossible without explicit pair strategy.
- Every runtime-consumed output carries row identity.
- Every trainable head has explicit target, mask, loss, weight, and semantic phase.
- Missing trainable target or mask fails loudly.
- Old scattered branches are deleted or quarantined with explicit owner and removal evidence.

## Ready-To-Implement Decision

This plan is ready to implement only after Stage 1 inventories are complete and reviewed.

The design itself is ready as the target direction because it matches verified current seams:

```text
legacy model assembly -> models registry and assembly
raw loss switch -> loss plan and registry
split target construction -> target contracts and replay projection
shm head flags -> inference protocol plus shm transport
worker pair branches -> pair strategies and engine adapter
config architecture behavior -> registry resolution
row/logit mismatch risk -> row table contracts
```

The implementation should not begin by wrapping the current system. It should begin by locking the inventories and golden tests, then cut over cleanly in Stage 2.
