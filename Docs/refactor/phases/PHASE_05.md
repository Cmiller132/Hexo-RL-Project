# Phase 05 - PolicyProvider, PairStrategy, EngineAdapter

## Source Of Truth

This phase implements the Phase 5 search boundary from `Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_V2_20260429.md`.

The goal is to make search consumption explicit:

```text
GameRunner -> PolicyProvider -> InferenceAdapter -> ModelFamily
GameRunner -> PairStrategy -> PolicyProvider/InferenceAdapter
GameRunner -> EngineAdapter -> Rust MCTS
```

No subsystem may infer search behavior from an architecture string, a config side effect, or the presence of a model head.

## Purpose

Move policy priors, pair-action scoring, and Rust MCTS calls behind three explicit interfaces:

- `PolicyProvider` owns policy evaluation and row-mapped priors.
- `PairStrategy` owns pair scoring mode, pair row generation, root/leaf/full caps, and telemetry.
- `EngineAdapter` is the only Python caller of Rust MCTS.

After this phase, `SelfPlayWorker` and `GameRunner` should not wire model outputs directly into MCTS. They should request a `SearchEvaluation`, optionally pass it through a named `PairStrategy`, and hand the result to `EngineAdapter`.

## Target Modules

- `Python/src/hexorl/search/context.py`
- `Python/src/hexorl/search/policy_provider.py`
- `Python/src/hexorl/search/pair_strategy.py`
- `Python/src/hexorl/search/priors.py`
- `Python/src/hexorl/search/expansion.py`
- `Python/src/hexorl/search/mcts_runner.py`
- `Python/src/hexorl/search/engine_adapter.py`
- Existing worker/game-runner call sites only as needed to route through these modules.

Do not add compatibility shims that keep the old direct worker-to-MCTS or worker-owned pair scoring paths alive.

## Required Contracts

### SearchContext

`SearchContext` is the search-facing view of a position. It must carry enough contract identity for policy, pair, and engine layers to prove they agree on the same rows:

```text
position identity/history hash
phase
legal action table
candidate table, if applicable
pair action table request state, if applicable
graph contract/tensor handle, if applicable
model family/spec identity
recipe/search/pair strategy identity
trace id
```

`SearchContext` must not rebuild legal rows, candidates, graph rows, pair rows, compact history, or D6 transforms.

### SearchEvaluation

`SearchEvaluation` is the only policy object accepted by `EngineAdapter`.

Required fields:

```text
context identity and trace id
value estimate
legal row ids
legal dense indices
row-mapped priors
prior source per row
policy provider name
model family/spec identity
inference protocol/schema identity
warnings
timings
```

Acceptance rule: every prior is mapped to a `LegalActionTable` row before MCTS sees it. Dense policy tensors, sparse candidate logits, and global graph `LEGAL` logits must all become row-mapped priors with source traceability.

Reject or hard-fail on:

```text
missing legal rows
duplicate legal rows
prior length != legal row count after mapping
non-finite value or prior
negative prior mass after masking
all-zero prior mass without an explicit fallback reason
model output rows that cannot be traced to legal rows
```

Verification rule: `SearchEvaluation` must prove that model behavior is being interpreted correctly. It must preserve enough raw-output metadata, row ids, masks, prior source labels, and trace ids to determine whether a bad move came from the model, adapter decode, policy mapping, legal table, pair strategy, or MCTS.

### PairEvaluation

`PairEvaluation` is the only pair-prior object accepted by `EngineAdapter`.

Required fields:

```text
strategy name
phase
root/leaf scope
pair action table identity
pair rows
row-mapped pair priors
pair prior source per row
known_first, when phase requires it
total_possible_pairs
selected_pair_rows
scored_pair_rows
caps applied
warnings
timings
```

`PairEvaluation` may be empty. Empty is the required output for `NoPairStrategy`.

## PolicyProvider Requirements

Implement and register providers through model-family capabilities, not architecture-prefix checks:

```python
class PolicyProvider(Protocol):
    def evaluate_root(self, context: SearchContext) -> SearchEvaluation: ...
    def evaluate_leaves(self, contexts: list[SearchContext]) -> list[SearchEvaluation]: ...
```

Required implementations:

```text
DensePolicyProvider
RestNetPolicyProvider
GraphHybridPolicyProvider
GlobalGraphPolicyProvider
```

Acceptance by family:

- `dense`: consumes dense place policy/value outputs and maps dense indices to `LegalActionTable` rows.
- `restnet`: follows the same row mapping as dense while preserving RestNet model/spec identity in telemetry.
- `graph_hybrid`: consumes sparse/candidate policy outputs only through canonical candidate/legal row contracts; no private candidate reconstruction.
- `global_graph`: consumes `policy_place` logits over canonical graph `LEGAL` rows and maps them to the same `LegalActionTable` rows MCTS uses.

Provider hard rules:

- No provider checks `architecture.startswith(...)`.
- No provider enables pair scoring.
- No provider interprets `pair_prior_mix`, `pair_head_present`, head existence, or model-family name as permission to score pairs.
- Every provider returns `SearchEvaluation` with row-mapped priors and prior-source telemetry.
- Evaluation uses the same `PolicyProvider` interface as self-play.

## PairStrategy Requirements

Pair scoring is owned only by `PairStrategy`.

```python
class PairStrategy(Protocol):
    name: str
    def score_root(self, context: SearchContext, base_eval: SearchEvaluation) -> PairEvaluation: ...
    def score_leaves(self, contexts: list[SearchContext], base_evals: list[SearchEvaluation]) -> list[PairEvaluation]: ...
```

`PairStrategySpec` must explicitly declare:

```text
name
enabled heads/sources
root enabled
leaf enabled
max root pair rows
max leaf pair rows
max full pair rows
chunk size
phase eligibility
known-first requirements
diagnostic flag
telemetry level
```

Required strategies:

- `none`: emits zero pair rows, performs zero pair generation calls, performs zero pair scoring calls.
- `two_stage_root_only`: scores only the named root pair path and obeys root caps.
- `tactical_only`: may consume tactical pair hints only through canonical pair-row mapping and explicit caps.
- `diagnostic_full_root`: the only strategy allowed to enumerate full `A * (A - 1) / 2` pair rows; it is root-only, diagnostic-only, and must hard-fail without `max_full_pair_rows`.

Default rules:

```text
default pair strategy = none
global_xattn default pair strategy = none
global_graph default pair strategy = none unless recipe explicitly overrides it
leaf pair scoring = disabled by default
full pair scoring = diagnostic only, root only, capped
```

Default `none` and default `global_xattn` must report:

```text
total_possible_pairs may be observed
selected_pair_rows = 0
scored_pair_rows = 0
pair chunks = 0
pair model calls = 0
MCTS pair influence = none
```

Hard forbidden behavior:

- No implicit pair scoring because a model exposes `policy_pair_first`, `policy_pair_second`, `policy_pair_joint`, or crop-compatible `PairPolicyHead`.
- No implicit pair scoring because `pair_prior_mix > 0`.
- No implicit pair scoring because a checkpoint/config says a pair head exists.
- No implicit pair scoring because architecture/model family starts with `global_`.
- No leaf pair scoring unless `PairStrategySpec.leaf_enabled` is true and `max_leaf_pair_rows` is set.
- No full pair scoring unless `name == "diagnostic_full_root"`, `diagnostic == true`, `root_enabled == true`, `leaf_enabled == false`, and `max_full_pair_rows` is set.

## Global Graph Pair Head Contracts

Global graph pair heads are first-class output contracts, but `PairStrategy` decides whether MCTS consumes them.

Required contracts:

```text
policy_place returns exactly one logit per legal action row.
policy_pair_first returns exactly one logit per legal first-placement row.
policy_pair_second exposes a conditional legal-second distribution for a known first placement.
policy_pair_joint returns exactly one logit per canonical PairActionTable row.
PAIR_ACTION rows are built only by canonical PairActionTable.
```

Known-first semantics:

- First-placement unordered pairs and second-placement known-first pairs are separate phases.
- `policy_pair_second` must validate the selected first placement and expose legal second-placement rows after that first placement.
- Second-placement positions use the legal table after the known first placement.
- Pair rows must preserve unordered identity for first-placement pairs and ordered known-first semantics for second-placement pairs.
- Joint pair losses and priors validate `PairActionTable` row identity.

Opening positions:

```text
opening positions have no pair prior
opening positions have no pair loss
opening positions produce zero MCTS pair influence
```

The crop-compatible `PairPolicyHead` is an auxiliary candidate-pair scorer. It is not the final global graph pair-head contract and cannot authorize MCTS pair consumption.

## EngineAdapter Requirements

Rust MCTS calls move behind one module:

```python
class EngineAdapter:
    def expand_root(self, evaluation: SearchEvaluation) -> None: ...
    def apply_pair_priors(self, pair_eval: PairEvaluation) -> None: ...
    def expand_and_backprop(self, evaluations: list[SearchEvaluation]) -> None: ...
```

`EngineAdapter` is the only Python caller of Rust MCTS APIs.

Hard rules:

- Worker, game runner, policy provider, pair strategy, inference adapters, training, evaluation, dashboard, and replay code do not call Rust MCTS directly.
- `EngineAdapter` accepts only `SearchEvaluation` and `PairEvaluation`, never raw model logits.
- `EngineAdapter` validates legal row identity before expansion.
- `EngineAdapter` validates pair row identity before pair-prior application.
- MCTS telemetry reports whether `pair_first`, `pair_second`, `pair_joint`, tactical pairs, or no pairs influenced the decision.
- Pair-prior application is a no-op for empty `PairEvaluation`, and the trace must show that no pair influence occurred.

## Detailed Policy And MCTS Verification
This phase must verify the model-to-search boundary as if subtle mapping bugs already exist.

Required verification:
- For each model family, compare raw model outputs, decoded adapter outputs, `SearchEvaluation` priors, and `EngineAdapter` inputs for the same golden positions.
- Verify that every prior accepted by MCTS maps to exactly one `LegalActionTable` row with matching row id, dense index, coordinate, phase, schema version, source hash, and trace id.
- Verify that masked rows cannot receive prior mass and legal rows cannot disappear silently.
- Verify that non-finite logits, non-finite values, all-zero prior mass, stale legal hashes, stale pair hashes, duplicate rows, and wrong protocol versions fail before MCTS.
- Verify that MCTS cannot mutate the `SearchEvaluation`, `PairEvaluation`, legal table, pair table, or policy-provider response.
- Verify that search traces report raw prior source, normalized prior, MCTS visit count, selected move, value estimate, pair influence, and fallback reason when a fallback is explicitly allowed.
- Add a single-position policy/search debug bundle containing contracts, raw model outputs, decoded outputs, priors, pair evaluation, MCTS input, MCTS output, selected move, hashes, and timings.

## Parallel Subagent Work

- S1: Implement `SearchContext`, `SearchEvaluation`, and row-mapped prior validation in `search/context.py` and `search/priors.py`.
- S2: Implement `PolicyProvider` registry and dense/restnet/graph_hybrid/global_graph providers.
- S3: Implement `PairStrategySpec`, validation, `NoPairStrategy`, capped root strategies, tactical-only strategy, and diagnostic full root strategy.
- S4: Implement `EngineAdapter` and move all Rust MCTS calls behind it.
- S5: Add tests, import audits, artifacts, and hard exit gates listed below.
- Orchestrator: verify worker-owned architecture branches, worker pair chunk helpers, direct MCTS prior wiring, and pair enablement from config/head presence are removed.

## Mandatory Tests

Create or update tests under `Python/tests/search/` unless an existing integration test is the better owner.

Policy provider tests:

```text
test_dense_policy_provider_returns_row_mapped_priors
test_restnet_policy_provider_returns_row_mapped_priors
test_graph_hybrid_policy_provider_uses_candidate_legal_rows
test_global_graph_policy_provider_maps_legal_logits_to_legal_rows
test_search_evaluation_rejects_prior_length_mismatch
test_search_evaluation_rejects_unmapped_model_rows
test_policy_source_traceability_records_provider_family_protocol
```

Pair strategy tests:

```text
test_pair_strategy_none_generates_zero_pair_rows
test_pair_strategy_none_scores_zero_pairs
test_global_xattn_default_pair_strategy_none_zero_rows
test_global_graph_default_pair_strategy_none_zero_rows
test_pair_head_presence_does_not_enable_pair_scoring
test_pair_prior_mix_does_not_enable_pair_scoring
test_architecture_prefix_does_not_enable_pair_scoring
test_leaf_pair_scoring_requires_explicit_enable_and_cap
test_full_pair_strategy_requires_diagnostic_root_only_and_cap
test_capped_pair_strategy_enforces_root_cap
test_capped_pair_strategy_enforces_leaf_cap
test_diagnostic_full_root_strategy_never_scores_leaves
```

Global graph pair-head contract tests:

```text
test_policy_place_one_logit_per_legal_row
test_policy_pair_first_one_logit_per_legal_first_row
test_policy_pair_second_requires_known_first
test_policy_pair_second_uses_post_first_legal_table
test_policy_pair_joint_one_logit_per_pair_action_row
test_pair_action_rows_from_canonical_pair_action_table
test_opening_position_has_no_pair_prior_or_pair_loss
test_pair_prior_telemetry_reports_pair_head_influence
```

Engine adapter tests:

```text
test_engine_adapter_is_only_rust_mcts_caller
test_engine_adapter_rejects_raw_logits
test_engine_adapter_validates_legal_row_identity
test_engine_adapter_validates_pair_row_identity
test_engine_adapter_empty_pair_eval_is_noop
test_engine_adapter_rejects_mutated_search_evaluation
test_engine_adapter_rejects_stale_hashes_duplicate_rows_and_nonfinite_priors
test_policy_search_debug_bundle_localizes_mapping_or_mcts_failure
test_mcts_integration_consumes_policy_provider_outputs
test_mcts_integration_consumes_pair_strategy_outputs_when_enabled
```

Worker/game-runner regression tests:

```text
test_selfplay_worker_contains_no_architecture_string_checks
test_worker_does_not_call_rust_mcts_directly
test_worker_does_not_score_pair_chunks_directly
test_evaluation_uses_policy_provider_path
```

## Import And Code Audits

Run these audits and save the command/output summaries as phase artifacts:

```text
rg -n "architecture\\.startswith|startswith\\(\"global_|global_graph_enabled|global_xattn|pair_prior_mix|pair_head_present" Python/src/hexorl
rg -n "score_pair|pair_chunk|pair.*forward|policy_pair_first|policy_pair_second|policy_pair_joint|PairPolicyHead" Python/src/hexorl/selfplay Python/src/hexorl/search Python/src/hexorl/inference
rg -n "mcts|MCTS|expand_root|expand_and_backprop|apply_pair_priors" Python/src/hexorl
rg -n "from hexorl\\.engine|import hexorl\\.engine" Python/src/hexorl/search Python/src/hexorl/selfplay Python/src/hexorl/eval
```

Expected audit outcomes:

- Architecture string checks remain only in model registry/spec tests or migration-only code, not runtime search/self-play paths.
- Pair scoring names in runtime code are owned by `search/pair_strategy.py` and inference adapters only.
- Rust MCTS API imports/calls are owned by `search/engine_adapter.py` only.
- Evaluation reaches models through `PolicyProvider`, not dense-only direct model calls.

## Required Artifacts

Produce these artifacts before marking the phase complete:

- `PolicyProvider` API and provider registration docs or docstring.
- `SearchEvaluation` validation tests proving row-mapped priors.
- `PairStrategySpec` schema and validation tests for root/leaf/full caps.
- Default recipe/config evidence showing `none` for global graph families, including `global_xattn`.
- MCTS trace sample showing policy provider, pair strategy, legal row count, pair rows possible, selected pair rows, scored pair rows, and pair influence.
- Single-position policy/search debug bundle showing raw outputs, decoded outputs, row-mapped priors, pair evaluation, MCTS inputs/outputs, hashes, trace ids, and selected move.
- Mutation/corruption verification proof for policy outputs, legal rows, pair rows, priors, and MCTS inputs.
- Import audit output summary proving `EngineAdapter` is the only Rust MCTS caller.
- Import audit output summary proving no worker architecture string checks remain.
- Import audit output summary proving no pair enablement from heads/config/architecture remains.

## Delete

Delete or fully disconnect:

```text
worker architecture branches
worker pair chunk helpers
pair enablement from pair_prior_mix
pair enablement from pair head presence
pair enablement from architecture prefix
direct MCTS prior wiring in worker
direct Rust MCTS calls outside EngineAdapter
dense-only evaluation policy assumptions
```

## Hard Exit Gates

The phase is not complete until all gates pass:

```text
SelfPlayWorker contains no architecture string checks.
SearchEvaluation priors are row-mapped for dense, restnet, graph_hybrid, and global_graph.
PolicyProvider acceptance tests pass for dense/restnet/graph_hybrid/global_graph.
EngineAdapter is the only Python caller of Rust MCTS.
No pair scoring happens without PairStrategy.
No pair scoring happens from head presence, pair_prior_mix, config side effects, or architecture prefix.
PairStrategySpec requires explicit root/leaf/full caps for any scoring mode that can generate rows.
Default none emits zero pair rows and zero pair model calls.
global_xattn default emits zero pair rows and zero pair model calls.
global_graph default emits zero pair rows and zero pair model calls unless recipe explicitly overrides it.
Leaf pair scoring is disabled by default and cannot run without leaf cap.
Full pair scoring is diagnostic-only, root-only, capped, and cannot run at leaves.
Global graph pair heads satisfy legal-row, PairActionTable, and known-first contracts.
Opening positions have no pair prior and no pair loss.
MCTS telemetry reports policy provider, pair strategy, pair rows possible, pair rows scored, and pair influence.
Policy/search debug bundle can identify whether a bad decision came from raw model output, inference decode, policy mapping, pair strategy, legal rows, or MCTS.
MCTS cannot mutate policy, pair, legal, or contract payloads after validation.
Import audits show no direct Rust MCTS calls outside EngineAdapter.
Import audits show no worker-owned pair chunk/scoring path.
Import audits show no runtime architecture string gates outside registry/spec validation.
Mandatory tests pass in the search/integration suite.
```
