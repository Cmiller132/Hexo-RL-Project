# Modular Model Architecture Implementation Plan

## Purpose

This document is the implementation plan for replacing the fragmented model architecture logic with a clean, contract-first research system under `Python/src/hexorl/models/`.

The plan is intentionally not a legacy-support migration. Current behavior will be inventoried so we know what exists, but the rewrite will intentionally keep, replace, simplify, or delete each behavior. Legacy checkpoint/API compatibility is out of scope. Old and dead code should be removed, not preserved.

The target is a clean rewrite in four bounded stages, with temporary transition adapters allowed only when they directly support cutover evidence and are deleted before final acceptance.

## Implementation Status

As of the 2026-05-06 cleanup pass, this plan has been implemented. The current
runtime authority is `Python/src/hexorl/models/`, boundary contracts are owned
by `Python/src/hexorl/contracts/`, replay-to-training batch conversion is owned
by `Python/src/hexorl/replay/`, and the old `Python/src/hexorl/model/` package
has been deleted. The deprecated `graph` architecture id is a hard error;
configs must use `graph_hybrid_0` explicitly.

The remainder of this document preserves the design intent and the original
implementation plan. The codebase-claim table below describes the pre-cutover
state that motivated the rewrite.

## Original Codebase Claims

The plan below was checked against the source layout before implementation, and
these claims were grounded in codebase searches at that time.

| Claim | Pre-cutover evidence |
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

The project is an experimental RL research environment whose primary operators are the maintainer and orchestration agents. The design should optimize for research iteration, throughput, stability, readability, and easy deletion of failed experiments. It should not optimize for end-user configuration ergonomics or legacy compatibility.

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

During the cutover, the old package could remain only as temporary
implementation source:

```text
Python/src/hexorl/model/
```

After cutover, `hexorl/model/` should be deleted or its retained
implementation pieces should be moved into the cohesive `hexorl/models/`
structure. Runtime-facing code must not use `hexorl.model` as an architecture
authority. The implemented code follows the deletion path.

## Design Goals

- Define each architecture from one registered spec.
- Make heads, targets, losses, masks, adapters, providers, and pair capabilities explicit at semantic boundaries.
- Make row identity first-class everywhere logits or targets are trained or consumed.
- Require every self-play architecture to resolve a search policy capability and search value capability.
- Forbid config overrides that disable heads required for self-play, including the value head used by MCTS.
- Let architecture specs own default heads and supported optional heads; config may only apply explicit enable/disable overrides.
- Support dynamic head families for parameterized heads such as `lookahead_*`, but expand them to concrete heads during spec resolution.
- Allow cohesive model-family implementations and inline heads when that is clearer than splitting tiny modules.
- Copy readable recipes first; extract reusable components only after repetition or coupling is real.
- Remove raw head-name loss routing from the trainer.
- Remove model-class and architecture-prefix runtime selection.
- Remove pair scoring from worker branches and move it into executable pair strategies.
- Replace inference head decoding with architecture-selected adapters and protocol objects.
- Keep shared memory as a transport detail, not as the model-output contract.
- Make adding a new architecture mostly declarative when it reuses existing parts.
- End with no permanent old/new dual runtime path.

## Non-Goals

- Do not split every linear layer into a separate file just for aesthetics.
- Do not build a general architecture framework beyond the needs of this RL system.
- Do not add contracts for ordinary internal neural-network plumbing unless that tensor crosses a replay, training, inference, or search boundary.
- Do not preserve silent target/loss skips as default behavior.
- Do not add a wrapper facade that keeps current trainer, inference, and self-play behavior intact indefinitely.
- Do not let architecture specs enable pair scoring by head presence.
- Do not make config mutation the source of resolved model behavior.
- Do not preserve checkpoint compatibility or old runtime APIs.
- Do not rewrite Rust MCTS rules or legal move generation as part of this model refactor.

## Core Architecture

### Modularity Rule

Split semantics before splitting math.

Contracts are required at boundaries where wrong assumptions can silently corrupt training or search:

```text
replay projection
row tables
target tensors
model outputs
loss plans
inference requests/responses
search policy/value inputs
pair strategies
```

Contracts are not required for ordinary internal tensors, attention blocks, MLPs, or tiny projections unless those tensors cross one of these boundaries. A model implementation may be a cohesive family class if that is the clearest way to run experiments.

### Implementation Guidance

This plan should guide a strong implementation agent without forcing a predetermined class layout. The invariants below are required; exact dataclass fields, helper names, module splits, and adapter internals are implementation decisions.

Use the smallest structure that satisfies the invariants, makes the tests clear, and leaves future architectures easy to add. Add explicit contracts only at boundaries where a wrong assumption can silently corrupt training, inference, or search.

### Architecture Authority

Registered architecture metadata is the source of truth for a model family. It should answer these questions:

```text
what architecture id is being built
what model family or recipe owns construction
what input contracts are required
what outputs are default, optional, diagnostic, or forbidden
what outputs are required for self-play
what target, training, inference, and search adapters are used
what pair capabilities exist, if any
what config overrides are valid
```

A spec defines what an architecture is allowed to do. It must not decide whether pair outputs affect MCTS; that remains the job of the pair strategy.

All self-play architectures must resolve a search policy capability and a value capability. Config may enable or disable supported optional outputs, but it may not disable outputs required for self-play. Disabling the value output or search policy output is a hard configuration error.

### Output And Head Semantics

Every output returned by `model.forward` must be declared somewhere discoverable by assembly, training, inference, and search. The implementation agent may choose the exact representation, but the resolved metadata must make these distinctions clear:

```text
trainable output
runtime-consumed output
diagnostic output
internal debug output
required output
optional output
conditional output
non-trainable output
```

Diagnostic outputs are allowed, but they cannot affect losses, inference, search, or pair strategies unless promoted to a trainable or runtime-consumed output.

Parameterized output families such as `lookahead_*` should be represented as a family during spec resolution and concrete outputs afterward. The current default examples to preserve are `lookahead_4`, `lookahead_12`, and `lookahead_36`; the implementation may choose how to represent the expansion.

### Recipes And Components

Architecture recipes should be readable Python assembly code. They may build cohesive family classes, compose reusable components, or mix both approaches.

Reusable components are optional. Add them when they remove real duplication, clarify experimentation, or isolate a cross-boundary semantic concern. Do not split every attention block, MLP, or projection just because a contract exists somewhere nearby.

### Row Identity

Row identity is the main correctness boundary.

Policy and pair outputs consumed outside `model.forward` must carry row identity. A row count alone is not enough. Same-count row tables with different order, payload, or backing token mapping must be rejected before loss computation or MCTS consumption.

The implementation should support the row families needed by current dense, sparse/candidate, graph, opponent, pair-first, pair-joint, pair-second, and graph-token workflows. Exact object names are less important than making row identity stable, hashable, testable, and visible at training/inference/search boundaries.

### Output Contracts

Model outputs need enough metadata to prevent consumers from guessing semantics by head name, tensor shape, or shared-memory flag.

Policy and pair outputs must identify their row identity and mask semantics. Value outputs must identify decoder, range, and perspective. Auxiliary outputs must be declared and either consumed by an explicit contract or treated as diagnostic.

This removes the current assumptions that value decoding can be hard-coded by model class or that pair tensors can be interpreted from head flags alone.

### Target Contracts

Targets are training contracts, not model internals. The implementation should make target builders strict enough to catch:

```text
stale or mismatched row identity
illegal target rows
duplicate pair rows
unexpected zero target mass
missing trainable target, mask, weight, or phase
pair-second targets outside known-first phase
configured lookahead heads without configured horizon targets
```

Current behavior must be inventoried, then intentionally kept, replaced, simplified, or deleted. Do not preserve silent skips or synthetic fallback targets unless Stage 1 explicitly classifies them as intentional non-trainable behavior and adds tests for that choice.

Stage 1 must resolve one pair-target semantic decision before implementation: if a pair target is unordered, `policy_pair_first` either trains on a marginal over both cells or becomes diagnostic/non-trainable. If first-position order is intended, the target must be represented as ordered and not described as unordered joint-pair data.

### Loss Planning

The trainer should not route losses through a broad raw head-name switch. It should ask a resolved loss plan, loss registry, or equivalent mechanism that knows:

```text
which prediction is being trained
which target, mask, weight, and phase are required
what loss applies
what behavior is allowed when optional data is absent
```

Trainable outputs default to hard errors when required training data is missing.

### Inference Protocol

Inference should improve the current shared-memory flow, not merely wrap it.

Separate model-output semantics from transport mechanics. The protocol should carry enough information for clients, providers, and search to validate requested outputs, row identity, value decoding, pair phase, and schema compatibility. Shared memory should remain optimized, but it should pack and unpack protocol facts rather than define what heads mean.

### Policy Providers And Pair Strategies

Policy providers convert decoded model outputs into search evaluations. Pair strategies own pair-specific runtime behavior.

Pair-capable architecture specs may declare that pair outputs exist. Only an explicit pair strategy may decide whether pair rows are generated, pair outputs are requested, pair logits are scored, or pair priors influence MCTS.

Runtime search behavior should depend on resolved capabilities and output contracts, not raw head-name presence.

## Suggested Package Shape

This layout is a starting point, not an implementation mandate. Keep the ownership boundaries, but let the implementation agent adjust exact file names and module splits if a simpler shape satisfies the same tests and deletion evidence.

```text
Python/src/hexorl/models/
  __init__.py
  registry.py
  specs.py
  assembly.py
  bundles.py
  validation.py
  recipes/
    dense.py
    restnet.py
    graph_hybrid.py
    global_graph.py
  families/
    dense_cnn.py
    restnet.py
    graph_hybrid.py
    global_graph.py
  components/
    conv_blocks.py
    attention_blocks.py
    graph_blocks.py
    heads.py
    pair_heads.py
  training/
    adapters.py
    loss_plan.py
    losses.py
    metrics.py

Python/src/hexorl/contracts/
  __init__.py
  rows.py
  targets.py
  outputs.py
  hashes.py
  phases.py

Python/src/hexorl/inference/
  protocol.py
  transport_shm.py
  adapters/
    dense.py
    sparse.py
    graph_hybrid.py
    global_graph.py
    pair_outputs.py

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

The intended ownership is stable: `models/` owns model assembly and architecture metadata, `contracts/` owns boundary semantics, `replay/` owns target projection, `inference/` owns protocol and transport mapping, and `search/` owns providers, pair strategies, and engine boundaries.

## Current Behavior Inventory Required Before Cutover

Stage 1 must produce inventories. These are not compatibility promises. They are decision tables.

### Architecture Inventory

The inventory must name the exact architecture ids that exist today, including:

```text
cnn
restnet
graph_hybrid_0
global_graph_option1
global_xattn_0
global_line_window_0
global_pair_twostage_0
global_graph_full_0
global_hybrid_action_0
global_graph768_champion
graph
```

`graph` is a deprecated alias to classify, not a future architecture id. Sparse policy is a policy/head mode to classify under heads and targets, not a standalone architecture family unless Stage 1 explicitly finds otherwise.

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
current required input contracts
semantic phase
runtime consumers
runtime output contracts
new decision: keep, replace, simplify, delete
new resolved head/output metadata
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
duplicate behavior
zero-mass behavior
required weight and phase behavior
negative tests
new resolved target metadata
```

Stage 1 must explicitly classify lookahead target fallback behavior. Synthetic fallback from missing lookahead targets to value targets should be deleted unless the inventory proves a non-trainable diagnostic use and tests lock that down.

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
row-table identity carried or lost
value decoding behavior
runtime consumers
new protocol responsibility
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
engine alignment assumptions
value range and perspective assumptions
telemetry emitted
new provider or strategy
old branch to delete
```

### Test Trust Audit

For each existing model, graph, replay, training, inference, and self-play test:

```text
test path
behavior claimed
actual code boundary covered
classification: golden, rewrite, delete
known blind spots
new contract test replacing or extending it
```

The current codebase is a rough WIP, so existing tests are not automatically acceptance evidence. They become evidence only after this audit classifies them as trusted or replaces them with clearer contract tests.

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

## Improvement Proof

This refactor is only an improvement if it removes real failure modes and makes experiments easier to add.

Required proof:

```text
deletion proof: scattered architecture lists, raw loss switches, direct pair-head worker checks, and graph inference decode branches are gone
bug-class proof: row mismatch, phase mismatch, missing target, disabled self-play value, and pair-head-without-strategy failures become hard errors
experiment proof: adding a new model touches a predictable recipe/spec/adapter/test surface instead of trainer, inference server, config validators, buffer sampler, and worker branches
throughput proof: training/inference hot paths are benchmarked or smoke-profiled when protocol packing or batch preparation changes
```

Contracts should be evaluated by this standard. If a proposed contract does not prevent a real silent failure, simplify model addition, or support deletion of scattered runtime authority, it should not be added.

## Four-Stage Implementation Plan

This rewrite has exactly four stages.

Four stages keeps the rewrite clean without turning it into a long compatibility migration. Each stage must close a real boundary and delete or quarantine obsolete behavior before moving on.

Temporary transition adapters are allowed only inside an active stage and only when they directly support evidence for the next deletion. They are not an accepted end state.

## Stage 1: Inventory, Test Trust Audit, And Contract Design

### Goal

Create the exact implementation blueprint and proof harness needed to rewrite cleanly without preserving accidental complexity or assuming the current tests are already correct.

### Success Criteria

- `hexorl/models/` is selected as the new architecture authority.
- Current architecture, head/loss, target, inference, and runtime behavior is inventoried.
- Each inventory row has a keep/replace/simplify/delete/move decision.
- Existing tests are classified as `golden`, `rewrite`, or `delete`.
- Row table, output, target, and inference contracts are designed before model code is moved.
- Golden tests capture rules that must survive the rewrite, including tests newly added for bug classes the current code can miss.
- Silent loss skips and fallback aliases are classified for removal or explicit retention.
- Shared-memory constraints are represented as transport constraints, not output semantics.
- Pair strategies are specified as executable runtime plans.
- Pair target ordering, duplicate target rows, zero target mass, missing weights/phases, and lookahead fallback behavior are explicitly decided.
- Config override behavior is locked: architecture specs own defaults, config may enable/disable supported optional heads, and self-play required outputs cannot be disabled.
- Dynamic head-family behavior is locked for `lookahead_*` and any future parameterized head families.

### Constraints

- Do not add permanent runtime wrappers.
- Do not introduce a second runtime path that remains after the stage that created it.
- Do not create importable runtime modules in Stage 1. Contract work in this stage is design artifacts, examples, and pseudocode only.
- Do not physically split model internals until contracts and specs are locked.
- Do not claim a behavior is preserved unless a trusted golden test or newly added replacement test covers it.
- Do not use architecture name prefixes as a future behavior mechanism.
- Do not add contracts for internal neural tensors that do not cross replay, training, inference, or search boundaries.

### Required Evidence

- `Docs/artifacts/model_architecture/architecture_inventory.md`
- `Docs/artifacts/model_architecture/head_loss_inventory.md`
- `Docs/artifacts/model_architecture/target_inventory.md`
- `Docs/artifacts/model_architecture/inference_inventory.md`
- `Docs/artifacts/model_architecture/runtime_inventory.md`
- `Docs/artifacts/model_architecture/test_trust_audit.md`
- Baseline command report capturing current import/test health, known failures, and which failures are expected WIP state.
- Contract draft files or design notes for row tables, outputs, targets, inference protocol, and pair strategies.
- Design examples showing row-table definitions, row-table instances, output contracts, value decoding contracts, and pair output specs.
- Golden test list with exact existing tests to keep, existing tests to rewrite, existing tests to delete, and new tests to add.

### Stop Rules

- Stop if a current trained head cannot be mapped to a target, mask, loss, and phase.
- Stop if a runtime-consumed output cannot be tied to a row table.
- Stop if shared-memory transport cannot carry required contract identity without a schema change.
- Stop if pair strategy behavior cannot be separated from architecture capability.
- Stop if policy/value self-play capability cannot be represented without hard-coded model-class checks.

### Stage 1 Work Items

1. Create model architecture inventories under `Docs/artifacts/model_architecture/`.
2. Capture baseline import/test command results so later stages can distinguish regressions from existing WIP failures.
3. Define semantic phases in Stage 1 design notes.
4. Define row-table definitions and row-table instances for dense, candidate, legal, opponent legal, pair joint, known-first, and graph token rows.
5. Define output contracts for policy, pair, value, and auxiliary outputs, including value decoder/range/perspective.
6. Define target contracts for dense policy, sparse policy, graph policy, pair first, pair joint, pair second, opponent policy, value, tactical, regret, and lookahead.
7. Define architecture metadata for `cnn`, `restnet`, `graph_hybrid_0`, `global_graph_option1`, `global_xattn_0`, `global_line_window_0`, `global_pair_twostage_0`, `global_graph_full_0`, `global_hybrid_action_0`, `global_graph768_champion`, and the deprecated `graph` alias decision.
8. Define head/output metadata for all current heads that remain supported.
9. Define resolved `lookahead_*` family expansion from configured horizons.
10. Define search policy/value capabilities for every self-play architecture.
11. Define loss plan entries for all trainable heads and mark silent skip behavior for removal unless explicitly optional.
12. Define inference protocol fields and shared-memory transport mapping, including row hashes and value decoding.
13. Define pair strategy behavior for `none`, current diagnostic behavior, and planned pair strategy variants.
14. Decide which current behavior is deleted instead of migrated.
15. Classify existing tests as `golden`, `rewrite`, or `delete`.

## Stage 2: Architecture Authority And Model Assembly Cutover

### Goal

Implement `hexorl/models/` as the single architecture authority and make model assembly flow through registered specs, recipes, and bundles.

### Success Criteria

- `hexorl/models/` is the architecture authority.
- `build_model_from_config` delegates to `hexorl.models.assembly` or is replaced by it.
- Model construction uses registered architecture metadata and returns the model with enough resolved metadata for training, inference, and search.
- Architecture specs own default heads, supported optional heads, head families, self-play required outputs, and adapter selections.
- Config can enable or disable supported optional heads, but cannot disable required self-play outputs.
- `lookahead_*` and future dynamic head families are expanded to concrete heads during architecture resolution.
- All current self-play architectures resolve a search policy capability and search value capability.
- All current architecture ids listed in Stage 1 resolve through the registry or are explicitly deleted as unsupported aliases.
- Output contracts, row-table definitions, row-table instances, and value decoder contracts are available to assembly outputs.
- Old architecture-name lists are removed from config, buffer, and model implementation where they are behavior authority.
- Existing PyTorch implementations may be temporarily retained only as implementation modules called by new recipes.

### Constraints

- No permanent compatibility facade.
- No checkpoint compatibility work.
- No config mutation that derives architecture behavior outside spec resolution.
- Do not split PyTorch internals unless the split makes experimentation clearer or removes real coupling.
- If `hexorl/model/` is retained temporarily, only `hexorl.models.recipes` may import it, and Stage 2 must produce a quarantine record with the owner and Stage 4 deletion gate.

### Required Evidence

- Unit tests for architecture registry, spec resolution, head-family expansion, config enable/disable overrides, and self-play required-output protection.
- Assembly tests for dense, RestNet, graph hybrid, and global graph bundles.
- Code search proving architecture membership is no longer duplicated as runtime authority outside the registry.
- Import audit proving retained `hexorl/model/` implementation is not runtime authority and is imported only from approved recipes.

### Stop Rules

- Stop if any current supported architecture cannot resolve through `hexorl/models/` or a deprecated alias lacks an explicit deletion decision.
- Stop if config still mutates resolved loss/head behavior in place of architecture specs.
- Stop if a self-play config can disable its policy or value capability.
- Stop if retained `hexorl/model/` code leaks into trainer, inference, self-play, eval, config, buffer, or dashboard runtime paths.

## Stage 3: Training And Replay Cutover

### Goal

Move replay projection, target construction, training adapters, and loss computation behind contracts.

### Success Criteria

- Trainer uses an adapter/loss-plan boundary, not raw head-name loss routing.
- Target construction uses explicit target metadata and row-table instances.
- Trainable heads fail loudly when required targets, masks, weights, or phases are missing.
- Silent loss skips and fallback aliases are deleted unless explicitly represented as optional non-trainable behavior.
- Dense, sparse, graph hybrid, and global graph batches train through the same public trainer flow.
- The prepared training batch carries per-sample or per-head semantic phase data needed by phase-sensitive losses.
- Pair-second loss is gated by explicit known-first phase metadata, not inferred from zero target mass.
- Lookahead trainable heads require exact configured horizon targets and do not fall back to value targets.
- Global graph training cannot accidentally consume dense policy fields.
- Architecture-specific target namespaces prevent unrelated dense, sparse, graph, and pair fields from being consumed accidentally.

### Constraints

- No trainable head silently skips missing target or mask.
- No trainable head silently skips missing weight or phase when its contract requires one.
- No model output reaches loss computation without output contract validation.
- No synthetic fallback target should be kept unless Stage 1 explicitly classifies it as required.

### Required Evidence

- Unit tests for row contracts, target contracts, loss plans, training adapters, and missing-target failures.
- Integration tests for dense, sparse, graph hybrid, and global graph training batches.
- Code search proving the broad raw head-name loss switch is gone from trainer/runtime code.
- Negative tests for missing required target, mask, weight, phase, duplicate rows, zero-mass policy, and pair-second wrong phase.
- Performance smoke/profile evidence for replay projection, batch preparation, and loss computation if those hot paths materially change.

### Stop Rules

- Stop if dense, sparse, graph hybrid, or global graph cannot train through the new trainer adapter.
- Stop if a runtime-consumed or trainable output cannot identify its row table.
- Stop if pair-second loss can run outside the known-first phase.
- Stop if a global graph batch can consume dense policy targets unless that is explicitly declared by its architecture spec.

## Stage 4: Inference, Search, Pair Strategies, And Legacy Deletion

### Goal

Move inference and search runtime behavior to protocol/adapters/providers/strategies, then delete old scattered runtime authority.

### Success Criteria

- Inference uses protocol/adapters and treats shared memory as transport.
- Self-play and evaluation use provider, pair-strategy, and engine-adapter boundaries.
- Pair behavior is impossible without an explicit pair strategy.
- Pair strategy owns pair row generation, scoring caps, phases, blending, and fallback behavior.
- Inference responses carry output contracts and row-table instances so same-count row reorderings cannot pass validation.
- Value decoding is owned by value output contracts and supports binned and scalar values.
- `EngineAdapter` validates root/leaf phase, batch generation, legal-row order/hash, dense offset mapping, Rust legal bytes alignment, value range/perspective, and pair phase.
- Old direct pair-head MCTS consumption is removed from self-play worker.
- Old graph-specific inference head decode branches are removed from server/client hot logic and replaced by adapters.
- `hexorl/model/` is deleted or fully moved into `hexorl/models/`.

### Constraints

- No pair scoring from head presence.
- No model output reaches MCTS without row contract validation.
- Shared-memory throughput and latency must be preserved or measured before protocol packing changes are accepted.
- No old/new runtime behavior remains active together after final cutover.
- Quarantine is allowed only for non-runtime migration/test artifacts with an owner and removal gate.

### Required Evidence

- Inference adapter round-trip tests for dense and global graph requests.
- Self-play/provider tests proving pair strategy controls pair behavior.
- Engine adapter tests proving unmapped policy/value outputs are rejected.
- Eval provider tests or a documented eval quarantine showing no old model-class API remains active.
- Negative tests for same-count reordered row tables, wrong batch generation, wrong legal order, dense out-of-window mapping, wrong pair phase, and invalid value decoder/range.
- Performance smoke evidence for inference shared-memory transport if protocol packing changes.
- Code search proving removed behavior branches are gone.
- Import audit proving `hexorl/model/`, `HexNet`, `GlobalHexGraphNet`, `from_config`, and `load_model_state` are not runtime authority in trainer, inference, self-play, eval, config, buffer, or dashboard paths.

### Stop Rules

- Stop if inference cannot map policy outputs to row contracts.
- Stop if self-play still directly checks pair output head names outside pair strategies.
- Stop if any old direct runtime branch remains. Non-runtime quarantine records do not allow old runtime behavior to stay active.

## Stage Work Breakdown

These work breakdown sections are decomposition notes. Every executable assignment still needs the full `Goal`, `Success criteria`, `Constraints`, `Required evidence`, and `Stop rules` packet from the parent stage before implementation starts.

### 1.1 Contracts And Test Trust Audit

Produce Stage 1 design artifacts for future boundary contracts. Do not create importable runtime modules in Stage 1.

Design outputs should cover:

```text
row identity and stable hashing
target validity rules
output semantics
semantic phases
value decoding
test trust audit artifact
```

Acceptance criteria:

```text
row contracts can hash legal_qr and pair rows
target contracts reject mismatched row hashes
output contracts validate shape, dtype, row-table instance identity, semantic phase, finite policy, and value decoder metadata
semantic phases cover first-placement, second-placement known-first, any-position, auxiliary-only
existing tests are classified as golden, rewrite, or delete before they are used as acceptance evidence
```

### 2.1 Model Specs And Assembly

Create the new model authority under `hexorl/models/`. The implementation agent may choose exact modules and class names, but the boundary must support:

```text
registered architecture metadata
model assembly from config
resolved default and optional outputs
dynamic head-family expansion
self-play required-output validation
adapter/provider selection
bundle metadata returned with the model
clear errors for invalid architecture/head combinations
```

Initial assembly can instantiate retained implementation modules, but architecture authority must come from `hexorl/models/`.

Acceptance criteria:

```text
all current architecture ids resolve through registry
unsupported heads fail at spec resolution
config can enable or disable only supported optional heads
self-play required policy/value outputs cannot be disabled
lookahead head families expand to concrete heads from configured horizons
architecture defaults are resolved without mutating Config in scattered validators
model assembly returns model plus resolved metadata
dense and global graph build through the same public assembly API
retained hexorl/model implementation is imported only from approved recipes during cutover
```

### 2.2 Model Families And Components

Create reusable modules only where they reduce real coupling. A family implementation may keep inline heads when that is the clearest way to express an experiment.

Minimum cutover:

```text
existing HexNet and GlobalHexGraphNet behavior may be ported or temporarily retained as implementation modules
current implementations should move under hexorl/models/families or be deleted
new component modules should be split when reuse is real or semantics need separate specs or masks
```

Preferred end state:

```text
families own cohesive PyTorch model implementations
components own shared blocks only when reuse is real
resolved architecture metadata owns semantics
recipes or assembly code own composition
```

Acceptance criteria:

```text
model forward emits outputs declared by architecture spec
optional heads are requested through specs, not scattered string checks
diagnostic outputs are declared and cannot affect search or loss unless promoted
pair head masks are phase-aware
output contracts validate shape, row-table instance identity, finite policy values, and value decoder metadata
```

### 2.3 Config Resolution

Move architecture behavior resolution out of config mutation.

Config should keep:

```text
syntax validation
range validation
type validation
obvious local invariants
```

Resolved architecture metadata should own:

```text
architecture membership
supported heads
default heads
supported optional heads
required self-play outputs
head family expansion
default loss plan
supported input contracts
supported pair strategies
adapter/provider selection
```

Config uses spec-owned defaults plus explicit overrides:

```text
architecture spec default heads
-> config enable_head overrides
-> config disable_head overrides
-> head family expansion
-> self-play capability validation
-> resolved model bundle or equivalent metadata package
```

Config may enable or disable supported optional heads. It may not invent new architecture behavior, request unsupported heads, or disable outputs required by the self-play capability. Disabling the value output or search policy output for a self-play architecture is a hard error.

Acceptance criteria:

```text
Config no longer duplicates graph architecture membership lists
resolved model/loss behavior comes from registry
invalid head/architecture combinations fail during spec resolution with clear errors
invalid self-play capability overrides fail during spec resolution with clear errors
```

### 3.1 Targets And Replay Projection

Create the replay projection and training-batch boundary in the location that best fits the codebase. The exact module split is flexible, but target construction must become contract-driven and phase-aware.

Important: this is allowed to rewrite target construction, but it must be test-driven against Stage 1 golden rules. Do not preserve old code shape just because it exists.

Acceptance criteria:

```text
replay positions project to canonical row tables first
targets reference row-table instances
pair target builders enforce Stage 1 pair-ordering decision and known-first second-placement semantics
illegal target rows fail before tensors reach trainer
duplicate rows and zero-mass targets follow explicit contract policy
configured lookahead trainable heads require configured horizon targets
global graph training does not accidentally consume dense policy fields
```

### 3.2 Training Adapter And Loss Plan

Create the training boundary needed to remove raw head-name loss routing. Exact class names are flexible, but the implementation should cover:

```text
batch preparation
model input conversion
loss planning
loss-plan validation
loss computation
metric reporting
```

The trainer flow should be equivalent to:

```text
raw replay batch
-> prepare batch
-> model inputs
-> model forward
-> validate loss plan
-> compute losses
-> metrics
```

Acceptance criteria:

```text
trainer has no broad raw head-name loss switch
trainable heads fail loudly when required target, mask, weight, or phase is absent
optional non-trainable heads are skipped explicitly
loss weights come from resolved loss plan
pair-second loss only runs in explicit known-first phase metadata
```

### 4.1 Inference Protocol And Adapters

Create the inference protocol and adapter boundary. Keep shared memory fast, but make it a transport for semantic metadata rather than the source of output meaning.

The implementation should provide the equivalent responsibilities for:

```text
request metadata
response metadata
policy output decoding
pair output decoding
value output decoding
auxiliary output handling
model-specific input/output adapters
shared-memory transport mapping
```

Acceptance criteria:

```text
server asks adapter to prepare inputs and decode outputs
adapter validates output presence, shape, finite values, row-table instance identity, schema version, output contract, and value decoder metadata
client receives decoded response metadata rather than inferring semantics from arrays alone
client and server reject same-count row tables when row hash, payload, or token mapping differs
existing shm arrays are either preserved with contract metadata or replaced with measured equivalent transport
```

### 4.2 Policy Providers, Pair Strategies, Engine Adapter

Create the search boundary that separates decoded model outputs from Rust MCTS calls and pair-runtime behavior.

The implementation should provide the equivalent responsibilities for:

```text
search context
search evaluation
pair evaluation
policy provider
pair strategy
pair strategy metadata
engine adapter
```

Acceptance criteria:

```text
self-play worker delegates policy evaluation to provider
pair strategy owns all pair output requests, row generation, scoring, caps, phases, and blend behavior
pair strategies declare required output contracts instead of raw required head names
engine adapter owns Rust MCTS calls and validates generation, legal-row order/hash, dense offsets, value range/perspective, and pair phase
worker no longer directly checks pair head names
leaf pair scoring is disabled unless strategy explicitly enables it and tests prove validity
```

### 4.3 Legacy Deletion

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
runtime quarantine is forbidden; non-runtime migration/test quarantine has an owner, reason, and removal date
```

## Required Test Plan

The exact test names and file layout are implementation decisions. The required coverage is not optional.

### Contract Coverage

Required coverage:

```text
stable row identity for legal, graph, and pair rows
row-hash mismatch rejection
same-count reordered-row rejection
policy and pair outputs require row identity
value outputs require decoder, range, and perspective
semantic phase separation for first-placement and known-first pair phases
```

### Architecture And Assembly Coverage

Required coverage:

```text
all supported current architecture ids resolve through the new authority
deprecated aliases are deleted or explicitly mapped during transition
unsupported architecture/head combinations fail clearly
config may enable/disable only supported optional outputs
config cannot disable self-play policy or value outputs
lookahead family expansion preserves configured horizons
diagnostic outputs are declared and cannot affect runtime behavior accidentally
dense, graph hybrid, and global graph bundles build through the same public assembly path
```

### Target And Loss Coverage

```text
dense, sparse/candidate, graph, opponent, pair-first, pair-joint, pair-second, value, tactical, regret, and lookahead targets map to the intended row identity
pair ordering semantics are explicitly tested after the Stage 1 decision
illegal rows and duplicate rows fail before training
zero-mass behavior is explicit and tested
missing required target, mask, weight, or phase fails for trainable outputs
optional non-trainable skips are explicit
pair-second loss requires explicit known-first phase metadata
pair-second phase is not inferred from zero target mass
global graph batches cannot accidentally consume dense policy targets
```

### Inference Coverage

```text
dense and global graph inference round-trip through adapters
policy outputs carry row identity through client/server boundaries
same-count row hash mismatches are rejected
pair outputs require pair row identity and phase
binned and scalar value outputs decode through declared contracts
shared-memory transport preserves required semantic metadata or is replaced with measured equivalent transport
```

### Runtime Coverage

```text
policy providers map dense and graph outputs to legal Rust rows
self-play requires policy and value capabilities
pair head presence alone does not enable pair scoring
pair strategy owns pair output requests, caps, phases, and blending
worker no longer consumes pair head names directly
engine adapter rejects unmapped policy/value outputs, wrong generation, wrong legal order, invalid dense offsets, and wrong pair phase
evaluation uses the same provider boundary or is explicitly quarantined from runtime acceptance
```

### Test Trust Audit

Existing model, graph, replay, training, inference, and self-play tests must be classified as `golden`, `rewrite`, or `delete` before they are used as acceptance evidence.

### Audit Commands

```text
rg -n "architecture\.startswith|startswith\(\"global_|GlobalHexGraphNet\.ARCHITECTURES|GLOBAL_GRAPH_ARCHITECTURES" Python/src/hexorl
rg -n "policy_pair_first|policy_pair_joint|policy_pair_second" Python/src/hexorl/selfplay Python/src/hexorl/inference Python/src/hexorl/eval Python/src/hexorl/dashboard
rg -n "if head_name ==|elif head_name ==" Python/src/hexorl/train Python/src/hexorl/models
rg -n "pair_prior_mix" Python/src/hexorl/selfplay Python/src/hexorl/search Python/src/hexorl/inference
rg -n "from hexorl\.model|hexorl\.model|HexNet|GlobalHexGraphNet|from_config|load_model_state" Python/src/hexorl
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
All target builders produce targets that reference row-table instances. Mismatched row hashes fail before training or inference consumption.

Constraints
Do not change Rust legal move generation. Do not allow raw logits or targets to be consumed without row identity.

Required evidence
Contract tests, target negative tests, and examples showing row hashes in training and inference traces.

Stop rules
Stop if any runtime-consumed output cannot identify its row table.
```

## Final Acceptance Checklist

- `hexorl/models/` owns architecture specs, assembly, model bundles, head specs, and loss plans.
- `hexorl/models/` owns spec-defined default heads, optional head overrides, self-play required outputs, and dynamic head-family expansion.
- `hexorl/contracts/` owns row-table definitions, row-table instances, target, output, hash, and phase contracts.
- `hexorl/replay/` owns replay projection and training batch conversion.
- `hexorl/inference/` owns protocol, adapters, and transport mapping.
- `hexorl/search/` owns policy providers, pair strategies, and engine adapter boundaries.
- `hexorl/model/` is deleted or fully moved into the cohesive `hexorl/models/` structure.
- Config validation no longer mutates or derives architecture behavior that belongs to specs.
- Config can enable/disable supported optional heads but cannot disable the self-play search policy or value capability.
- Dynamic heads such as `lookahead_*` resolve from head families to concrete heads before training/inference.
- Diagnostic outputs are allowed only when declared and cannot affect loss/search unless promoted.
- Trainer has no broad raw head-name loss switch.
- Inference server does not interpret graph pair heads directly.
- Self-play worker does not directly consume pair output head names.
- Pair behavior is impossible without explicit pair strategy.
- Every runtime-consumed policy or pair output carries row-table instance identity.
- Every runtime-consumed value output carries decoder, range, and perspective identity.
- Every trainable head has explicit target, mask, loss, weight, and semantic phase.
- Missing trainable target, mask, weight, or phase fails loudly when required by contract.
- Old scattered runtime branches are deleted. Only non-runtime migration/test artifacts may be quarantined with explicit owner and removal evidence.
- No legacy checkpoint/API compatibility path remains.

## Ready-To-Implement Decision

This plan is ready to start Stage 1 now.

It is not ready to start Stage 2-4 runtime cutover until Stage 1 inventories, baseline command report, test trust audit, and contract decisions are complete and reviewed.

The design itself is ready as the target direction because it matches verified current seams:

```text
legacy model assembly -> models registry and assembly
raw loss switch -> loss plan and registry
split target construction -> target contracts and replay projection
shm arrays and head flags -> inference protocol plus shm transport
worker pair branches -> pair strategies and engine adapter
config architecture behavior -> registry resolution
row/logit mismatch risk -> row-table instances and output contracts
config head mutation -> spec-owned defaults plus explicit overrides
dynamic lookahead heads -> resolved head families
self-play assumptions -> search policy/value capabilities
```

The implementation should not begin by wrapping the current system. It should begin by locking inventories, classifying tests, and writing boundary contracts. Each later stage should make one boundary authoritative, prove it with tests/audits/performance evidence, and delete the old path before moving on.
