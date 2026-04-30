# Phase 03 - Model Registry, TrainAdapter, and CheckpointManager

## Purpose
Make model behavior explicit and capability-driven. Phase 03 replaces architecture-string inference with a `ModelFamilyRegistry`, moves training wiring behind `TrainAdapter`, and gives checkpoints one strict owner through `CheckpointManager`.

This phase is a breaking runtime cutover. Do not preserve `hexorl/model` through a runtime compatibility shim. Any migration support for old checkpoints, config names, or import paths must live outside runtime code, preferably under `tools/migration/`, and must not be imported by `hexorl` production modules.

## Source Of Truth
Use `Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_V2_20260429.md` as the source of truth for package ownership, model-family interfaces, checkpoint rules, deletion targets, tests, artifacts, and exit gates.

## Target Runtime Structure
Create and cut runtime code over to `Python/src/hexorl/models/`:

```text
Python/src/hexorl/models/
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
```

Rules:

- Use `models/`, not both `model/` and `models/`.
- Runtime imports from `hexorl/model` are forbidden after this phase.
- No `hexorl/model` compatibility facade, alias module, or shim may remain in runtime.
- Migration tools may read old files or old checkpoint layouts, but they must be command-line/offline tools outside runtime.

## Required Model Family Registration
Every registered model family must expose a complete descriptor. The registry is the semantic authority for family identity, capabilities, specs, and component lookup, but it should not become a switchboard full of family-specific branches.

Prefer facet-based registration:

```python
class ModelFamilyDescriptor(Protocol):
    name: str
    aliases: set[str]
    capabilities: CapabilitySet
    spec_schema: type[ModelSpec]
    components: FamilyComponents
```

Required facets:

```text
ModelBuilder
TrainAdapterFactory
InferenceAdapterFactory
PolicyProviderFactory
LossPlanProvider
RecipeProvider
TuneSpaceProvider
CheckpointManifestProvider
```

Registration requirements:

- The descriptor must provide every required facet; no family may be registered with a partial placeholder.
- Model building constructs only the model core: trunk, heads, and family wrapper.
- The train-adapter facet owns batch projection, target validation, loss input assembly, output validation, and training performance assumptions.
- The inference-adapter facet owns inference tensor packing, output decoding, shape validation, legal-row mapping, and manifest declaration.
- The policy-provider facet returns the search-facing policy interface. Search and arena code consume policy providers, not model classes.
- The loss-plan facet declares finite, masked, turn-aware losses for every supported output target.
- The recipe facet produces a complete valid recipe for the family on a host profile.
- The tune-space facet returns only valid family-specific recipe mutations.
- The checkpoint-manifest facet writes enough information to inspect and load the family without architecture-string recovery.
- A fake-family extension test must prove a new family can be registered by adding a descriptor/facets, not by editing trainer, inference, search, dashboard, or self-play internals.

Aliases may exist only as registry-recognized migration names. They must not create non-registry behavior branches, and they must not keep deprecated architecture names alive outside spec validation and offline migration tooling.

## Model Specs And Capabilities
Add discriminated `ModelSpec` types in `models/specs.py` for:

```text
dense_cnn
restnet
graph_hybrid
global_xattn
global_line_window
global_relation_graph
```

Required capability names:

```text
DENSE_PLACE_POLICY
SPARSE_PLACE_POLICY
GLOBAL_PLACE_POLICY
PAIR_FIRST_POLICY
PAIR_SECOND_POLICY
JOINT_PAIR_POLICY
REGRET_HEAD
GLOBAL_GRAPH_INPUT
CROP_INPUT
```

Rules:

- Specs are selected by `kind`, never by architecture-name heuristics.
- Capabilities declare what a model can output. They do not decide what MCTS consumes.
- Pair consumption remains owned by `PairStrategy`, not by model family, head presence, checkpoint metadata, or `pair_prior_mix`.
- Family names must stay architecture identities. Recipe/checkpoint labels such as `champion`, `two_stage`, or strategy names do not become model families.

## TrainAdapter Requirements
Add `TrainAdapter` in `Python/src/hexorl/train/adapters.py` or the phase-approved train package location, then require every family to return one through the registry.

The trainer must:

- Resolve the family from `ModelSpec`.
- Ask the family for `TrainAdapter`.
- Ask the family for `LossPlan`.
- Run the same trainer path for every registered family.
- Contain no `isinstance(GlobalHexGraphNet)`, `architecture.startswith(...)`, family-name branches, or output-key heuristics.

The adapter must:

- Project shared contracts into model inputs.
- Validate required input contracts and schema versions.
- Validate place-policy targets against legal rows.
- Validate pair targets against `PairActionTable`.
- Validate `policy_pair_first`, `policy_pair_second`, and `policy_pair_joint` only on positions where the turn semantics make the target meaningful.
- Reject opening positions with pair prior/loss enabled.
- Reject second-placement pair targets unless the known-first placement and post-first legal table match.
- Preserve pair target mass under D6-transformed batches.
- Produce finite, masked loss tensors.
- Return typed training outputs rather than generic model-core dictionaries.

Training verification rule:
- Do not assume existing training batches or targets are correct.
- Every adapter must support a debug/probe path that shows replay record -> contracts -> model input tensors -> targets -> masks -> model outputs -> loss inputs.
- Target validation must check semantic row identity, not just tensor shape.
- Tensors produced from contracts must either own their memory or be guarded against mutation of the source contract after validation.
- D6-augmented batches must prove target mass, legal-row identity, pair-row identity, and known-first semantics are preserved.
- Negative tests must corrupt legal rows, target rows, pair targets, masks, graph links, tensor shapes, schema versions, and non-finite values, and each failure must identify whether the owner is replay projection, train adapter, model output validation, or loss planning.
- Performance evidence must show vectorized batch projection, device-transfer behavior, and train-step throughput for each registered family or an approved representative family set. CUDA paths should use pinned transfers, AMP, channels-last layout, or compilation only when valid for the family and recorded in the manifest.

## Inference Adapter Manifest And Declaration
Every family must declare its inference protocol through `inference_manifest(spec)`.

The manifest/declaration must include:

```text
protocol_version
request_kind
model_family
model_spec_version
input_contract
output_contract
action_contract
graph_schema_version, when applicable
relation_schema_version, when applicable
max_tokens, when applicable
max_legal_rows
max_pair_rows
required_heads
optional_heads
capabilities
```

Validation rules:

- Manifest values must be written into checkpoints.
- Inference adapters must fail fast on protocol or contract mismatch.
- `policy_place` returns exactly one logit per legal action row.
- `policy_pair_first` returns exactly one logit per legal first-placement row.
- `policy_pair_second` exposes a conditional legal-second distribution for a known first placement.
- `policy_pair_joint` returns exactly one logit per canonical `PairActionTable` row.
- Full joint pair scoring remains opt-in and capped by explicit pair strategy.

## CheckpointManager
Add one checkpoint owner in `models/checkpoint.py`:

```python
class CheckpointManager:
    def save(self, bundle: CheckpointBundle, path: Path) -> None: ...
    def load(self, path: Path, *, purpose: LoadPurpose, device: str) -> LoadedCheckpoint: ...
    def inspect(self, path: Path) -> CheckpointManifest: ...
```

Required manifest fields:

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

Strict checkpoint rules:

- Strict load by default.
- `inspect(path)` must parse and validate the manifest without loading model weights.
- Load must reject missing, unknown, incompatible, or stale manifest fields.
- Load must reject model-family/spec mismatches.
- Load must reject inference-protocol mismatches for inference purpose.
- Load must reject action/input/output contract mismatches for train/inference/eval purposes.
- No silent `_orig_mod` stripping, key-prefix cleanup, shape skipping, partial-state loading, or old-name remapping in runtime.
- Any old checkpoint conversion belongs to offline migration tooling and must emit a new strict manifest.

## Implementation Outcomes
- Runtime uses one model registry/spec/capability system.
- Model families register complete descriptors and facets.
- Trainer construction flows through registry-owned train adapter and loss plan lookup.
- Checkpoint save/load/inspect flows through `CheckpointManager`.
- Pair targets, conditional second-placement targets, legal-row alignment, caps, masks, and finite losses are strictly validated.
- Runtime dependency on old `Python/src/hexorl/model/` is removed.
- Extension-proof tests demonstrate new-family registration without editing unrelated runtime internals.

## Delete
Delete or remove from runtime imports:

```text
build_model_from_config switch
GlobalHexGraphNet multi-architecture string switch
trainer isinstance(GlobalHexGraphNet) branch
duplicate checkpoint prefix/state cleanup
deprecated architecture aliases outside registry/migration tests
Python/src/hexorl/model/ runtime imports
hexorl/model compatibility shim or facade
```

## Exact Tests
Add or update tests under `Python/tests/models/` and `Python/tests/train/`:

```text
test_registry_lists_all_required_families
test_every_registered_family_validates_default_recipe
test_every_registered_family_builds_model
test_every_registered_family_builds_train_adapter
test_every_registered_family_builds_inference_adapter_manifest
test_every_registered_family_builds_policy_provider
test_every_registered_family_declares_loss_plan
test_every_registered_family_declares_tune_space
test_fake_family_registers_without_runtime_internal_edits
test_trainer_runs_one_batch_for_every_registered_family
test_trainer_contains_no_architecture_or_model_class_branches
test_pair_target_validation_rejects_opening_pair_loss
test_pair_target_validation_rejects_missing_known_first
test_pair_target_validation_rejects_stale_post_first_legal_table
test_pair_target_mass_preserved_under_d6
test_train_adapter_debug_bundle_reconstructs_replay_to_loss_inputs
test_train_adapter_rejects_mutated_contract_after_projection
test_train_adapter_rejects_stale_legal_row_identity
test_train_adapter_rejects_corrupt_masks_or_nonfinite_targets
test_train_adapter_projection_and_device_transfer_profile_is_recorded
test_model_output_validation_rejects_wrong_rows_shapes_and_nonfinite_values
test_checkpoint_manifest_round_trips
test_checkpoint_inspect_does_not_load_weights
test_checkpoint_load_rejects_missing_manifest
test_checkpoint_load_rejects_unknown_or_stale_manifest_fields
test_checkpoint_load_rejects_model_family_mismatch
test_checkpoint_load_rejects_inference_protocol_mismatch
test_checkpoint_load_does_not_silently_strip_orig_mod_or_prefixes
test_no_runtime_imports_from_hexorl_model
test_no_model_architecture_string_gates_outside_registry_spec_tests
```

Also update any broader smoke tests needed so each registered family can run one training batch through the same trainer path.

## Import And Deletion Audits
Phase completion requires command-output artifacts for:

```text
rg "hexorl\\.model|from hexorl import model|Python/src/hexorl/model" Python/src Python/tests
rg "architecture\\.startswith|architecture ==|isinstance\\(.*GlobalHexGraphNet|build_model_from_config" Python/src/hexorl
rg "_orig_mod|strip.*prefix|state_dict.*cleanup|strict=False" Python/src/hexorl
rg "pair_prior_mix|pair_head_present" Python/src/hexorl/models Python/src/hexorl/train
```

Allowed matches must be documented inline in the phase artifact and restricted to registry/spec tests or offline migration tooling. Runtime matches fail the phase.

## Required Artifacts
Produce these artifacts before marking the phase complete:

```text
registered model family list with capabilities
default recipe validation output for every family
one-batch trainer smoke output for every family
facet/descriptor extension example for a fake family
training throughput and device-transfer profile for each registered or approved representative family
checkpoint manifest round-trip output
checkpoint inspect-without-weights proof
pair target validation proof for opening, first-placement, second-placement, and D6 cases
single-position training debug bundle showing replay record, contracts, tensors, targets, masks, outputs, losses, hashes, and trace ids
mutation/corruption test output for target and tensor projection boundaries
import/deletion audit output
```

## Hard Exit Gates
Phase 03 is complete only when all gates below pass:

```text
every registered family builds
every registered family exposes complete descriptor/facet registration for model, train adapter, inference adapter manifest/declaration, policy provider, loss plan, default recipe, tune space, and checkpoint manifest
fake-family registration proves the registry is extensible without runtime switch edits
every registered family trains one batch through the same trainer path
trainer has no architecture branches, model-class branches, or output-key behavior inference
checkpoint manifest save/load/inspect round-trips
checkpoint inspect works without loading weights
strict checkpoint load rejects malformed, stale, incompatible, or prefix-cleanup-dependent checkpoints
pair target validation is phase-aware, turn-aware, legal-row-aware, D6-aware, masked, and finite
training debug bundle localizes failures to replay projection, train adapter, model output validation, or loss planning
target/tensor projection paths are mutation-safe and corruption-tested
runtime contains no `hexorl/model` compatibility shim
runtime imports do not reference `hexorl/model`
old model switches, checkpoint cleanup, and deprecated aliases are absent from runtime
all exact tests pass
all import/deletion audits are attached as artifacts
```
