# Temporary Explainer: Modular Model Architecture Refactor

This document explains the model architecture refactor in plain language.

The detailed implementation plan lives here:

```text
Docs/refactor/MODEL_ARCHITECTURE_MODULARIZATION_PLAN.md
```

This explainer is temporary. It is meant to make the scope and intent easy to understand before implementation begins.

## The Short Version

The current model system works, but too much model behavior is scattered across unrelated files.

For example, to understand one model head, we may need to inspect:

- model construction
- model forward logic
- config validation
- replay target construction
- training loss routing
- inference server output decoding
- self-play MCTS integration
- dashboard/debug tooling

That makes model changes risky. A new head or architecture can accidentally train with the wrong target, decode with the wrong row order, or influence MCTS just because a head exists.

The refactor creates a new architecture system under:

```text
Python/src/hexorl/models/
```

Each model architecture will have one explicit spec that says:

- what inputs it consumes
- what trunk it uses
- what heads it has
- what each head means
- what targets train each head
- what losses apply
- what inference adapter decodes outputs
- what policy provider maps outputs to legal moves
- what pair strategy, if any, may use pair heads

The new system makes model behavior explicit instead of rediscovered in many places.

## The Problem Today

Today, model behavior is fragmented.

### Model Assembly

Model construction currently lives mostly in:

```text
Python/src/hexorl/model/network.py
Python/src/hexorl/model/global_graph.py
```

These files know a lot about architectures, heads, and graph model variants.

### Training

Training currently has head-specific loss logic in:

```text
Python/src/hexorl/train/losses.py
Python/src/hexorl/train/trainer.py
```

The trainer and loss code decide what target each head uses. Some behavior is implicit, such as silently skipping a loss when a target is missing.

### Replay And Targets

Replay and target construction are split across:

```text
Python/src/hexorl/buffer/sampler.py
Python/src/hexorl/buffer/targets.py
Python/src/hexorl/graph/batch.py
```

These files build dense targets, sparse targets, graph legal rows, pair targets, and graph tensors.

### Inference

Inference currently decodes outputs using direct checks for output keys and shared-memory arrays:

```text
Python/src/hexorl/inference/server.py
Python/src/hexorl/inference/client.py
Python/src/hexorl/inference/shm_queue.py
```

This means the inference server has model-head knowledge that should belong to architecture adapters.

### Self-Play And Pair Heads

Self-play currently has direct logic for pair heads in:

```text
Python/src/hexorl/selfplay/worker.py
```

Pair behavior is mostly gated by `pair_strategy`, which is good, but the worker still directly checks and consumes specific pair-head outputs.

## Why This Is Risky

The dangerous bugs are usually not normal PyTorch bugs.

The dangerous bugs are identity bugs.

Examples:

```text
These logits were for graph legal rows, but MCTS interpreted them as Rust legal rows.
```

```text
This pair target was for first-placement unordered pairs, but the model treated it like known-first second-placement pairs.
```

```text
This head silently did not train because its target was missing.
```

```text
This pair head influenced MCTS because it existed, not because a strategy explicitly requested it.
```

The refactor is designed to make those bugs much harder to create.

## The Big Idea

The new system is contract-first.

Instead of asking:

```text
What class is this model?
What strings are in cfg.model.heads?
Does the output dict contain policy_pair_joint?
Does architecture start with global_?
```

the new code asks:

```text
What does the architecture spec say?
What input contract is active?
What output contract does this head declare?
What row table do these logits belong to?
What target contract trains this head?
What pair strategy is active?
```

The architecture spec becomes the central source of truth.

## Important New Concepts

## ArchitectureSpec

An `ArchitectureSpec` describes a model family.

It answers:

- what architecture this is
- what inputs it accepts
- what trunk it uses
- what heads it exposes
- which heads are trainable
- which training adapter to use
- which inference adapter to use
- which policy provider to use
- which pair strategies are supported

Example:

```text
architecture: global_graph_option1
input: global_graph_v1
trunk: relation_graph_trunk
heads:
- policy_place
- value
- tactical
- policy_pair_first
- policy_pair_joint
training adapter: global_graph_training
inference adapter: global_graph_inference
policy provider: global_graph_provider
```

This means a developer can understand the model from one place.

## HeadSpec

A `HeadSpec` describes one model output.

It answers:

- what the head is called
- what tensor it outputs
- what target trains it
- what mask defines valid rows
- what loss it uses
- what phase it applies to
- whether it can be consumed by runtime

Example:

```text
head: policy_pair_second
target: pair_second_policy_target
mask: known_first_pair_row_mask
loss: pair_policy_cross_entropy
phase: second_placement_known_first_only
runtime consumer: pair_strategy only
```

This is better than hardcoding pair-second behavior inside trainer, inference, and worker code.

## Row Tables

A row table is the ordered list that tells us what each tensor row means.

Example:

```text
row 0 = move at (0, 0)
row 1 = move at (1, 0)
row 2 = move at (0, 1)
```

If the model outputs:

```text
logits = [0.2, -1.4, 3.1]
```

then the row table tells us:

```text
0.2 belongs to (0, 0)
-1.4 belongs to (1, 0)
3.1 belongs to (0, 1)
```

This matters because graph legal rows, Rust legal rows, sparse candidate rows, and pair rows can all have different ordering.

The new system makes row tables explicit so logits cannot be consumed without knowing what moves they refer to.

## TargetContract

A `TargetContract` describes a training target.

It answers:

- what rows the target belongs to
- what phase it applies to
- what mask is valid
- how probability mass is handled
- what invalid inputs should fail

Example:

```text
target: pair_joint_policy_target
row table: first_placement_pair_rows
phase: first_placement_unordered
invalid rows: duplicate pair, illegal pair, wrong phase
```

This makes target construction safer and easier to test.

## LossPlan

A `LossPlan` tells the trainer exactly which losses to compute.

Instead of the trainer saying:

```python
if head_name == "policy_pair_second":
    ...
```

the trainer asks the plan:

```text
For this architecture, train these heads with these targets, masks, and weights.
```

Important behavior change:

```text
If a head is declared trainable, missing target or mask is an error.
```

The new system should not silently skip required losses.

## Inference Protocol

Inference will be split into two concepts:

```text
protocol
transport
```

The protocol says what request and response mean.

The transport says how bytes move around.

Shared memory can remain the fast transport, but it should not define model semantics.

New flow:

```text
InferenceRequest
  -> adapter validates inputs
  -> model forward
  -> adapter validates outputs
  -> InferenceResponse
  -> shared-memory transport packs response
```

This keeps inference fast while making output meaning explicit.

## PolicyProvider

A `PolicyProvider` maps model policy outputs to legal MCTS rows.

Examples:

```text
DensePolicyProvider
SparseCandidatePolicyProvider
GlobalGraphPolicyProvider
```

The provider is responsible for turning model output into a row-mapped search evaluation.

## PairStrategy

A `PairStrategy` decides whether pair heads affect search.

This is very important.

A model may expose pair heads. That does not mean MCTS should use them.

Architecture capability:

```text
this model can output policy_pair_joint
```

Pair strategy decision:

```text
use policy_pair_joint for root pair priors with this cap and blend rule
```

This separation makes pair experiments safer.

## What Will Change

## Behavior That Gets Stricter

### Missing Targets Will Fail More Often

Current behavior sometimes silently skips a loss if a target is missing.

New behavior:

```text
If the architecture says a head is trainable, its target and mask must exist.
```

This may expose bugs that were previously hidden.

### Pair Heads Will Not Affect Search Unless A Strategy Says So

Current behavior is partly strategy-gated already, but the worker still directly consumes pair head outputs.

New behavior:

```text
pair heads are model capabilities only
pair strategies decide runtime use
```

### Logits Need Row Identity

Runtime-consumed logits must carry or reference a row table.

New behavior:

```text
raw logits cannot go directly into MCTS
```

They must be decoded through a provider or adapter that proves what rows they belong to.

### Config Will Do Less Architecture Work

Config should still validate simple things like types and ranges.

But architecture-specific behavior should move to the model registry.

New behavior:

```text
config chooses architecture
registry resolves architecture behavior
```

## Behavior That Should Stay The Same

The goal is not to change game rules or model intent.

These should stay semantically equivalent:

- dense policy still maps to legal board moves
- sparse policy still maps to candidate moves
- global graph policy still maps to legal graph rows
- value heads still train/evaluate value
- pair first/joint/second heads still mean the same phases
- Rust remains the canonical legal move boundary
- shared memory remains available as a fast inference transport

What changes is where the behavior is defined and how strictly it is validated.

## Behavior That May Be Removed Or Simplified

The rewrite should inventory old behavior and decide what survives.

Likely candidates for removal or simplification:

- silent loss skips
- implicit target fallbacks
- duplicated architecture lists
- config mutation of graph loss defaults
- direct pair-head checks in self-play worker
- inference server hardcoding graph pair output semantics
- old/new parallel runtime paths

## New Folder Structure

## `hexorl/models/`

Owns model architecture definitions.

Contains:

- architecture registry
- architecture specs
- head specs
- trunk specs
- model assembly
- model bundles
- training loss plans

This is the new source of truth for model families.

## `hexorl/contracts/`

Owns shared contracts.

Contains:

- row table contracts
- target contracts
- tensor contracts
- schema versions
- semantic phases
- hashes
- traces

This keeps row identity and target identity independent of model implementation.

## `hexorl/replay/`

Owns replay projection into training-ready objects.

Contains:

- replay-to-row-table projection
- target builders
- training batch construction

Replay should produce contract-aware targets, not loose arrays with unclear identity.

## `hexorl/inference/`

Owns inference protocol, adapters, and transport.

Contains:

- inference request/response types
- dense inference adapter
- sparse inference adapter
- global graph inference adapter
- pair scoring adapter
- shared-memory transport mapping

## `hexorl/search/`

Owns search-facing model consumption.

Contains:

- search context
- policy providers
- pair strategies
- engine adapter

Self-play should consume these interfaces instead of raw model outputs.

## How Training Will Work After The Refactor

Old shape:

```text
trainer receives batch
trainer branches based on model type
model forward
compute_losses branches on head names
some heads silently skip if targets missing
```

New shape:

```text
replay projects position into row tables and targets
training adapter prepares model inputs
model forward emits declared heads
loss plan validates required targets and masks
loss registry computes declared losses
metrics are reported with contract identity
```

Why better:

- trainer becomes generic
- missing targets fail loudly
- each head has one declared target and mask
- new heads can be added without editing a huge loss switch

## How Inference Will Work After The Refactor

Old shape:

```text
server sees output dict
server checks for known head names
server fills fixed shared-memory arrays
client reconstructs arrays based on flags
worker interprets arrays
```

New shape:

```text
request declares architecture, input contract, requested outputs, row tables
adapter validates request
model forward
adapter validates outputs and row counts
response carries typed policy/pair/value outputs
transport packs response efficiently
runtime consumes decoded outputs through providers/strategies
```

Why better:

- output meaning is explicit
- shared memory is still fast but no longer owns semantics
- graph and pair outputs are validated before runtime use

## How Self-Play Will Work After The Refactor

Old shape:

```text
worker builds inputs
worker calls inference
worker maps priors
worker directly checks pair heads
worker directly applies pair priors
```

New shape:

```text
worker builds SearchContext
PolicyProvider evaluates policy
PairStrategy optionally evaluates pair outputs
EngineAdapter applies validated MCTS inputs
worker records telemetry
```

Why better:

- self-play stops knowing model-head internals
- pair behavior is explicit and testable
- MCTS only receives row-mapped validated inputs

## Why This Is Better

## Easier To Add New Models

A new architecture can reuse existing trunks, heads, targets, losses, and adapters.

Instead of editing trainer, inference, worker, config, and replay code, we register a new spec.

## Easier To Add New Heads

A new head declares:

- output name
- input slot
- target
- mask
- loss
- semantic phase
- runtime consumer, if any

No more hunting through multiple systems to wire it up.

## Easier To Debug

Every output can say:

```text
I came from this architecture
I used this input contract
I correspond to this row table
I was decoded by this adapter
I was consumed by this provider or strategy
```

That makes bad move debugging much easier.

## Safer Pair Experiments

Pair heads can exist without affecting MCTS.

Pair strategies explicitly decide how pair heads are used.

This prevents accidental search behavior changes.

## Fewer Hidden Training Bugs

Missing required targets become errors.

Implicit fallbacks are removed or made explicit.

Loss behavior becomes declarative.

## Better Runtime Boundaries

Training, inference, replay, and self-play stop duplicating model knowledge.

Each subsystem owns its actual job:

```text
models define model structure
contracts define identity
replay builds targets
training computes losses
inference decodes outputs
search consumes row-mapped evaluations
engine adapter talks to Rust MCTS
```

## Implementation Shape

The implementation plan has two stages.

## Stage 1: Inventory And Design Lock

This stage does not add permanent wrappers.

It creates:

- architecture inventory
- head/loss inventory
- target inventory
- inference inventory
- runtime inventory
- row contract design
- target contract design
- inference protocol design
- pair strategy design
- golden tests

The purpose is to decide what old behavior is kept, replaced, simplified, deleted, or moved behind contracts.

## Stage 2: Clean Cutover

This stage implements the new system and deletes old scattered logic.

It creates:

- `hexorl/models/`
- `hexorl/contracts/`
- `hexorl/replay/`
- inference protocol/adapters
- `hexorl/search/`
- model registry
- loss plans
- target contracts
- policy providers
- pair strategies

Then it removes old branches from trainer, inference, self-play, config, and buffer code.

## What Success Looks Like

A developer should be able to answer these questions from the architecture spec and contracts:

- What tensors does this model consume?
- What row tables do those tensors correspond to?
- What heads does this model emit?
- Which heads are trainable?
- Which targets train them?
- Which masks define valid rows?
- Which heads can affect runtime search?
- Which adapter decodes inference outputs?
- Which provider maps priors to MCTS rows?
- Which pair strategy consumes pair outputs?
- What telemetry proves the path was valid?

If answering those questions still requires searching trainer, inference server, self-play worker, replay sampler, config, and dashboard code, the refactor is not done.

## Final Mental Model

The old system is like this:

```text
model behavior is spread across many files
```

The new system should be like this:

```text
architecture spec defines behavior
contracts prove identity
adapters move data across boundaries
providers and strategies consume outputs safely
```

That is the core of the refactor.
