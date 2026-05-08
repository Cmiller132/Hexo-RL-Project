# V1 Pair Model Verification Spec

## Purpose

This document verifies the current `global_pair_biaffine_0` /
`sampled_joint_pair_v1` implementation against:

- `Docs/HEXORL_V1_ARCHITECTURE_PROPOSAL.md`
- `Docs/FINAL_HEXORL_IMPLEMENTATION_PLAN.md`
- the current Python and Rust implementation pulled into this workspace

It is not a replacement for the V1 architecture proposal. It is a gap-aware
implementation spec: what exists, what is partial, what behavior that creates,
and what the implementation should do before V1 is claimed complete.

## Verification Method

Verification used originally:

- local source inspection with `rg` and focused file reads,
- two read-only subagent verification passes:
  - model/training/target verification,
  - selector/runtime/candidate verification,
- attempted focused test execution.

Focused tests could not be executed in this shell:

```text
python -m pytest Python\tests\test_v1_pair_candidate_selector.py Python\tests\test_v1_pair_biaffine_model.py Python\tests\test_v1_pair_targets.py Python\tests\test_v1_pair_eval_guardrails.py -q
exit 1: No module named pytest

.\.venv\Scripts\python.exe -m pytest ...
exit 1: .venv python not found in this checkout

cargo test -p hexgame-core v1_pair_search -- --nocapture
exit 1: cargo not found on PATH
```

May 8, 2026 re-verification after the completion patches used executable
Python tests and source audits in this workspace. Rust verification remains
blocked locally because `cargo`, `rustc`, and a built `_engine` extension are
not available on PATH.

Commands that passed locally:

```powershell
$env:PYTHONPATH='Python/src'
python -m pytest Python\tests\test_v1_pair_contract.py Python\tests\test_v1_pair_candidate_selector.py Python\tests\test_v1_pair_targets.py Python\tests\test_v1_pair_training_losses.py Python\tests\test_v1_pair_biaffine_model.py Python\tests\test_v1_pair_ci_audit_gates.py Python\tests\test_v1_selfplay_worker_runtime.py::test_v1_proposal_matrix_reader_accepts_legal_by_legal_outputs Python\tests\test_v1_selfplay_worker_runtime.py::test_sampled_joint_pair_v1_worker_uses_pair_native_runtime -q
```

Observed:

```text
34 passed
```

```powershell
python -m pytest Python\tests\test_training_data_pipeline.py::test_v1_pair_search_metadata_roundtrips_through_compact_record_and_ring Python\tests\test_training_data_pipeline.py::test_v1_pair_search_metadata_compact_blob_rejects_legacy_json Python\tests\test_training_data_pipeline.py::test_v1_pair_search_metadata_schema_two_requires_tactical_payload Python\tests\test_training_data_pipeline.py::test_replay_memory_estimate_accounts_for_compressed_v1_metadata Python\tests\test_training_data_pipeline.py::test_v1_support_type_is_explicit_and_validated Python\tests\test_training_data_pipeline.py::test_v1_unsampled_legal_pairs_are_not_implicit_negatives Python\tests\test_training_data_pipeline.py::test_v1_metadata_rejects_legacy_pair_policy_target_mixing Python\tests\test_training_data_pipeline.py::test_process_game_record_keeps_v1_metadata_out_of_legacy_pair_completeness Python\tests\test_training_data_pipeline.py::test_prepare_dense_training_batch_masks_legacy_pair_weight_for_v1_schema_marker -q
```

Observed:

```text
9 passed
```

`Python\tests\test_v1_pair_search_ffi.py -q` is skipped locally because
`_engine` is unavailable.

Additional local verification after the unstaged Response1 review fixes:

```powershell
$env:PYTHONPATH='Python/src'
python -m pytest Python\tests\test_v1_selfplay_worker_runtime.py::test_sampled_joint_pair_v1_worker_uses_pair_native_runtime -q
```

Observed:

```text
1 passed
```

```powershell
python -m compileall -q Python/src/hexorl/selfplay/worker.py Python/tests/test_v1_selfplay_worker_runtime.py
```

Observed:

```text
passed with no output
```

```powershell
git diff --check -- Python/src/hexorl/selfplay/worker.py Python/tests/test_v1_selfplay_worker_runtime.py crates/hexgame-core/src/v1_pair_search.rs Docs/V1_PAIR_MCTS_SEARCH.md Docs/V1_PAIR_MODEL_VERIFICATION_SPEC.md Docs/V1_PAIR_MCTS_RESPONSE1_REVIEW.md
```

Observed:

```text
passed with no output
```

Commands still blocked locally:

```powershell
cargo test -p hexgame-core v1_pair_search
```

Observed:

```text
cargo: The term 'cargo' is not recognized
```

```powershell
rustfmt crates/hexgame-core/src/v1_pair_search.rs
```

Observed:

```text
rustfmt: The term 'rustfmt' is not recognized
```

## Current Unstaged Change Snapshot

This document reflects the current dirty worktree, including the V1 completion
patches and the follow-up fixes from `Docs/Response1.md`.

Primary source changes to verify:

- `crates/hexgame-core/src/v1_pair_search.rs`
  - recursive pair-action search protocol,
  - terminal-safe pair application order,
  - raw-model-logit search priors independent of proposal correction weights,
  - revealed-row prior renormalization for interior PUCT/widening,
  - completed-Q-first final root selection.
- `crates/hexgame-py/src/engine.rs`
  - PyO3 exposure for V1 root init, expansion requests, expansion completion,
    selected action, applied action, telemetry, legal-row tables, pair rows, and
    tactical payloads.
- `Python/src/hexorl/selfplay/worker.py`
  - `sampled_joint_pair_v1` root proposal/scoring,
  - recursive Rust expansion loop,
  - request-batch proposal and admitted-pair scoring through
    `submit_graph_many(...)`,
  - replay metadata construction from Rust telemetry.
- `Python/src/hexorl/models/families/global_graph.py`
  - V1 legal projection outputs and final pair heads.
- `Python/src/hexorl/search/pair_candidate_selector_v1.py`
  - learned proposal admission, direct retrieval, tactical protection,
    canary/eviction telemetry.
- `Python/src/hexorl/v1_pair_contract.py`
  - canonical schema-2 V1 pair features and tactical replay contract.
- `Python/src/hexorl/selfplay/records.py`,
  `Python/src/hexorl/train/v1_pair_targets.py`, and
  `Python/src/hexorl/train/loss_plan.py`
  - compact V1 replay, completed-Q targets, tactical targets from payload, and
    ranking-head loss contract.
- `Python/src/hexorl/search/legacy_pair_projection.py` and
  `Python/src/hexorl/search/pair_strategy.py`
  - legacy pair-to-single projection quarantine and normal V1 strategy gates.

Unstaged documentation/evidence files:

- `Docs/V1_PAIR_MCTS_SEARCH.md`
- `Docs/V1_PAIR_MCTS_RESPONSE1_REVIEW.md`
- `Docs/V1_PAIR_MODEL_COMPLETION_VERIFICATION.md`
- `Docs/V1_PAIR_MODEL_VERIFICATION_SPEC.md`

Known unrelated/user-authored change to preserve:

- `Docs/Notes.md`

## Current Implementation Summary

The implementation has a real V1 spine:

- `global_pair_biaffine_0` is registered as the V1 pair model.
- `sampled_joint_pair_v1` is an explicit pair strategy.
- V1 config validation requires:
  - `model.architecture = "global_pair_biaffine_0"`,
  - explicit V1 heads,
  - `selfplay.legal_row_mode = "full_rust_legal"`,
  - `selfplay.tactical_mode = "proposal_and_label"`,
  - `selfplay.constrain_threats = false`.
- Rust owns V1 legal-row identity, canonical pair rows, tactical payloads,
  root pair admission, Gumbel admission, and selected-action apply.
- Python owns candidate selection, source metadata, graph batching, V1 replay
  metadata, V1 target construction, and V1 loss contracts.
- The graph model implements bounded unordered pair scoring without
  materializing all pair actions as attention tokens.

May 8 cutover status: the source-level V1 runtime, replay, model output, target,
selector, and Python self-play gaps identified below have been patched. The
remaining unclaimed work is external evidence: Rust/PyO3 execution in a
Rust-capable environment, search trace artifacts, performance artifacts,
candidate/direct-retrieval recall artifacts, D6 consistency artifacts, and
equal-wall-clock arena scorecards.

## May 8, 2026 Cutover Patch Summary

Implemented source changes:

- Added `hexorl.v1_pair_contract` as the canonical feature/tactical contract,
  schema version `2`, with required 12-field pair features and schema-2 tactical
  replay payloads.
- Updated graph batching, replay records, compact metadata, sampler paths, and
  V1 target construction to consume the canonical contract and reject stale V1
  metadata.
- Added runtime-only legal projection outputs for `global_pair_biaffine_0`:
  `legal_proposal_embeddings`, `legal_completion_query`, and
  `legal_completion_key`.
- Reused the learned V1 pair projection parameters for those legal projection
  outputs; no new training targets were introduced.
- Updated loss planning to ignore those runtime-only tensors while keeping
  `pair_proposal_score`, `pair_completion_logits`, and `pair_joint_logits`
  as normal trainable heads.
- Replaced live heuristic legal embeddings with learned proposal projection and
  completion projection signals in `sampled_joint_pair_v1` admission.
- Preserved final bounded pair scoring through `pair_joint_logits` and required
  graph `value` for V1 inference.
- Added Rust `run_search_step`, `complete_expansion`, and `select_root_action`
  protocol methods and PyO3 bindings.
- Added recursive Rust V1 search nodes with stable node keys, node visit/value
  statistics, legal row identity, tactical payload, terminal value, and cached
  pair reservoirs.
- Updated `run_search_step` to spend the simulation budget by descending from
  the root through expanded interior nodes with PUCT and progressive widening
  until a terminal or unexpanded leaf is reached.
- Updated `complete_expansion` to attach exactly one reservoir to the expanded
  node, widen from cache, and back up the supplied current-player value through
  the pending simulation path.
- Updated `select_root_action` to reject normal full-turn roots while expansion
  requests are pending or the recursive simulation budget is incomplete.
- Updated Python V1 self-play to loop over recursive Rust expansion requests
  and score interior nodes with the same proposal/scoring path as the root.
- Updated Python V1 expansion completion to batch all proposal graphs returned
  by one `run_search_step(...)` call, batch all admitted-pair scoring graphs,
  and then complete each expansion back into Rust.
- Added terminal-safe unordered pair application: if exactly one cell in a pair
  is an immediate current-player win, Rust applies that cell before filler.
- Search priors now use raw model logits; proposal correction weights are
  validated and retained for replay/training but are not subtracted from PUCT
  prior logits.
- Interior PUCT and widening now compute prior mass from raw logits normalized
  over the currently revealed rows.
- Final root pair selection now orders by completed-Q first, then visit count,
  prior, and deterministic pair-key tie break.
- Removed Rust's dormant shallow nonterminal `evaluate_pair(...)=0.0` fallback.
- Stored interior expanded-node, reservoir-build, widening, neural-call, and
  explicit zero-refill telemetry in replay metadata.
- Moved `pair_logits_to_action_logits` into
  `hexorl.search.legacy_pair_projection` and removed it from `pair_strategy.py`.
- Quarantined `root_pair_mcts` and `full_pair_mcts` behind
  `build_legacy_pair_baseline_strategy`; normal config/autotune surfaces now
  expose `none` and `sampled_joint_pair_v1`.

Source audits:

```powershell
rg -n "simulate_candidate\(|evaluate_pair\(|_v1_legal_embedding_features|v1_pair_proposal_score_target|pair_proposal_score_target" Python/src Python/tests crates/hexgame-core/src
```

Observed:

```text
no matches
```

```powershell
rg -n "pair_logits_to_action_logits|root_pair_mcts|full_pair_mcts" Python/src/hexorl/search Python/src/hexorl/selfplay Python/src/hexorl/config Python/src/hexorl/autotune Python/src/hexorl/tuning
```

Observed remaining references are limited to the legacy projection module,
legacy non-V1 worker branch, and quarantined legacy constants in
`pair_strategy.py`.

## Verified Behaviors That Match The Spec

### Explicit V1 Strategy Gate

Current behavior:

- V1 behavior is activated by `sampled_joint_pair_v1`, not by merely requesting
  V1-looking output heads.
- `build_pair_strategy` declares required V1 contracts:
  - `cell_marginal_logits`,
  - `pair_completion_logits`,
  - `pair_proposal_score`,
  - `pair_joint_logits`,
  - `terminal_tactical_v1`.

Evidence:

- `Python/src/hexorl/search/pair_strategy.py`
- `Python/src/hexorl/config/schema.py`
- `Python/tests/test_v1_selfplay_worker_runtime.py`

Behavioral effect:

- This prevents accidental V1 runtime behavior from head names alone.
- It is a likely improvement over looser head-detected activation.

### Full Legal Rows Are Guarded For V1

Current behavior:

- Config and runtime reject threat-constrained V1 self-play.
- The V1 root path explicitly refuses `constrain_threats`.
- Graph legal rows are validated against Rust legal rows and required pair
  cells.

Evidence:

- `Python/src/hexorl/config/schema.py`
- `Python/src/hexorl/selfplay/worker.py`

Behavioral effect:

- Tactical facts cannot silently censor the semantic legal row table in the V1
  path.
- This matches the V1 contract.

### Pair Joint Model Is Bounded And Symmetric

Current behavior:

- `global_pair_biaffine_0` creates:
  - per-legal-cell `cell_marginal_logits`,
  - low-rank left/right pair projections,
  - symmetric biaffine proposal score,
  - symmetric pair MLP interaction,
  - final `pair_joint_logits`.
- Pair rows reference legal-token indices. They are not added as all-pair graph
  tokens.

The implemented joint form is:

```text
joint(i, j) =
  cell_marginal(i)
  + cell_marginal(j)
  + symmetric_biaffine(i, j)
  + pair_feature_score(i, j)
  + symmetric_interaction(i, j)
```

Evidence:

- `Python/src/hexorl/models/families/global_graph.py`
- `Python/tests/test_v1_pair_biaffine_model.py`

Behavioral effect:

- The model can represent both independent cell quality and pair-specific
  interaction.
- Order-invariance is enforced for unordered pair logits.

### Single-Action Exceptions Are Narrow

Current behavior:

- Rust hardcodes:
  - opening single,
  - true one-placement terminal single,
  - terminal no-action.
- Normal two-placement tactical situations still require admitted pair
  candidates and search.

Evidence:

- `crates/hexgame-core/src/v1_pair_search.rs`
- `Python/src/hexorl/selfplay/worker.py`

Behavioral effect:

- This matches the intended exception model: hardcode the rare structural
  exceptions, not normal tactical two-placement decisions.
- This is a likely improvement over asking the model to learn forced structural
  exceptions.

### V1 Replay Is Separated From Legacy Pair Targets

Current behavior:

- V1 metadata is stored in `V1SearchPairMetadata`.
- V1 records are not treated as complete legacy `pair_policy_target_v2`.
- The sampler masks legacy pair-head training when V1 metadata is present.

Evidence:

- `Python/src/hexorl/selfplay/records.py`
- `Python/src/hexorl/buffer/targets.py`
- `Python/src/hexorl/buffer/sampler.py`
- `Python/tests/test_training_data_pipeline.py`

Behavioral effect:

- V1 sampled support is not accidentally interpreted as exhaustive pair labels.
- This is a likely improvement over compatibility bridges that would blur
  sampled, admitted, explicit-negative, and unsampled rows.

## Historical Gaps And Current Status

The sections below are retained as the original verification trail. Items
marked as gaps in the first pass have source-level fixes in the May 8 patch
summary unless explicitly called out as external evidence.

### 1. Live Candidate Admission Is Not Model-Led

Status: fixed at source level; requires recall/performance artifacts for final
acceptance.

Spec expectation:

At each expanded full-turn node, V1 should run model inference to produce
legal-cell embeddings and proposal heads, then build the bounded candidate
reservoir using those model-informed signals:

```text
legal-cell embeddings
cell_marginal_logits
pair_completion_logits
pair_proposal_score
candidate selector
bounded reservoir
pair_joint_logits
```

Current behavior:

- The V1 root path runs `select_pair_candidates_v1(...)` before graph
  inference.
- The selector receives:
  - Rust legal table,
  - tactical payload,
  - handcrafted `legal_cell_embeddings`,
  - selector config.
- It does not receive live `cell_marginal_logits`,
  `pair_completion_logits`, or `pair_proposal_score`.
- Graph inference then scores only the already-admitted rows with
  `pair_joint_logits`.

Evidence:

- `Python/src/hexorl/selfplay/worker.py`
- `Python/src/hexorl/search/pair_candidate_selector_v1.py`

Behavioral effect:

- `cell_marginal_logits` and `pair_completion_logits` are trainable heads, but
  they do not currently shape live root candidate support.
- If the selector misses an important pair, the model cannot recover it through
  `pair_joint_logits`, because that head only scores admitted rows.
- The current system behaves more like:

```text
handcrafted/tactical selector -> model reranks admitted pair rows
```

It should behave more like:

```text
model-informed selector -> bounded joint pair scorer -> pair search
```

Required correction:

- Add a proposal pass that obtains legal-cell embeddings and proposal outputs
  before final candidate admission.
- Feed learned proposal signals into:
  - `direct_pair_retrieval`,
  - `anchor_conditioned_completion`,
  - `cell_marginal_cross`,
  - optional rich rerank.
- Keep tactical and diagnostic sources active, but make the live reservoir
  model-informed.

Acceptance evidence:

- Runtime trace showing nonzero final admitted rows from
  `anchor_conditioned_completion` and `cell_marginal_cross` using live model
  outputs.
- Tests proving `cell_marginal_logits` and `pair_completion_logits` affect
  final admitted support without becoming final move-choice authority.

### 2. Direct Retrieval Uses Handcrafted Coordinates, Not Learned Embeddings

Status: verified gap.

Spec expectation:

`direct_pair_retrieval` starts as blockwise-exact retrieval over a cheap learned
symmetric score:

```text
u_i = W_u h_i
v_i = W_v h_i
s_prop(i, j) = 0.5 * (u_i dot v_j + u_j dot v_i) + cheap_pair_features(i, j)
```

Current behavior:

- Live runtime calls `_v1_legal_embedding_features(...)`.
- Those "embeddings" are deterministic coordinate/id features:
  - q,
  - r,
  - q + r,
  - absolute q/r,
  - distance from origin,
  - row-id parity,
  - constant bias.
- `direct_pair_retrieval_v1(...)` scores those features exactly.

Evidence:

- `Python/src/hexorl/selfplay/worker.py`
- `Python/src/hexorl/search/pair_scorer_v1.py`

Behavioral effect:

- Retrieval quality cannot improve through training.
- It can learn only through downstream `pair_joint_logits` reranking after
  admission.
- Candidate recall remains bounded by handcrafted geometry plus tactical and
  diversity sources.

Required correction:

- Expose learned legal embeddings or learned retrieval projections from the
  graph model.
- Use those learned embeddings for direct retrieval while preserving exact
  blockwise top-k semantics.
- Keep deterministic coordinate features as auxiliary cheap features, not the
  main embedding.

Acceptance evidence:

- Direct retrieval recall improves over coordinate-only and unary baselines.
- Chunking, row-order, deduplication, and D6 stability tests still pass.

### 3. Protected Tactical Candidates Can Be Evicted By Global Budget

Status: verified gap.

Spec expectation:

`terminal_exact_v1` is non-optional, protected from eviction, and never forces
normal two-placement play.

Current behavior:

- Tactical proposals are admitted first and marked protected.
- Final selection globally truncates protected candidates to
  `candidate_budget`.
- A test explicitly expects tactical protected candidates to be evicted when
  there are more tactical rows than budget.

Evidence:

- `Python/src/hexorl/search/pair_candidate_selector_v1.py`
- `Python/tests/test_v1_pair_candidate_selector.py`

Behavioral effect:

- Exact hot completion or cover pairs can be absent from search if tactical
  support exceeds budget.
- This violates the "protected from eviction" contract.
- Search may fail to see the exact tactical candidate it is supposed to judge
  naturally.

Required correction:

- Protected tactical rows must either:
  - expand the effective candidate budget for that root, or
  - occupy a separate mandatory support band outside ordinary source quotas.
- If a deliberate cap is still needed for safety, overflow must be explicit:
  - fail the root,
  - emit a hard diagnostic blocker,
  - or use a documented deterministic tactical overflow policy that is accepted
    by the spec.

Acceptance evidence:

- Tests where tactical protected count exceeds ordinary budget and all
  tactical rows survive.
- Telemetry distinguishing ordinary budget from protected tactical overflow.

### 4. Source Quota Telemetry Is Pre-Final-Truncation

Status: verified gap.

Spec expectation:

Source quotas and telemetry should explain final candidate support.

Current behavior:

- `admitted_by_source` is updated during source admission into intermediate
  state.
- The final global budget truncation happens after that.
- Final candidates may not include all rows counted as admitted by source.

Evidence:

- `Python/src/hexorl/search/pair_candidate_selector_v1.py`

Behavioral effect:

- Telemetry can overstate final source representation.
- Candidate source mix can look healthier than actual search support.
- This weakens debugging for support collapse and canary survival.

Required correction:

- Emit separate telemetry fields:
  - proposed by source,
  - pre-budget admitted by source,
  - final admitted by source,
  - evicted by source and reason.

Acceptance evidence:

- Unit tests where source rows are admitted pre-budget then evicted, with final
  telemetry reflecting the eviction.

### 5. Blind Canaries Are Diagnostic But Not Guaranteed Observable

Status: nuanced gap.

Spec expectation:

Blind canaries are a tiny deterministic diagnostic source for support-collapse
probing.

Current behavior:

- Canaries are deterministic and marked `training_forbidden`.
- Rust excludes `training_forbidden` rows from selectable root admission.
- They can still be removed by final candidate-budget truncation.

Evidence:

- `Python/src/hexorl/search/pair_candidate_selector_v1.py`
- `crates/hexgame-core/src/v1_pair_search.rs`
- `Python/tests/test_v1_pair_candidate_selector.py`

Behavioral effect:

- Canary semantics are good when rows survive.
- If canaries are budget-evicted, the diagnostic can silently disappear unless
  telemetry is inspected carefully.

Required correction:

- Either guarantee the configured canary quota in final support or explicitly
  report final canary loss.
- Keep canaries `training_forbidden` and non-selectable.

Acceptance evidence:

- Final support telemetry includes `final_canary_count`.
- Tests cover canary survival or explicit canary eviction accounting.

### 6. Runtime And Training Pair Features Are Mismatched

Status: verified gap.

Spec expectation:

Pair features should be stable across training and inference. Source
contribution features are intended as ablation or diagnostics, not default
scorer inputs.

Current runtime behavior:

- `_v1_pair_features_from_candidates(...)` supplies pair features during
  self-play inference.
- Runtime features include:
  - distance,
  - same line,
  - origin axis,
  - tactical protected,
  - forced exploration,
  - terminal exact,
  - terminal equivalent,
  - source count,
  - selector score,
  - rich rerank score,
  - admission generation,
  - root/interior flag.

Current training behavior:

- The sampler attaches V1 pair rows without supplying the same pair features.
- The model falls back to internally computed features, with tactical/source
  slots zeroed or simplified.

Evidence:

- `Python/src/hexorl/selfplay/worker.py`
- `Python/src/hexorl/buffer/sampler.py`
- `Python/src/hexorl/models/families/global_graph.py`

Behavioral effect:

- The pair scorer sees different input distributions in training and
  self-play.
- The model can learn dependencies on runtime-only source/procedure features
  that are missing during training.
- `pair_joint_logits` may be miscalibrated in either direction.

Required correction:

- Define one canonical V1 pair-feature schema.
- Store required pair features in replay or deterministically reconstruct the
  same features in training.
- Remove source/procedure features from the default scorer unless explicitly
  enabled as an ablation.

Acceptance evidence:

- Training and inference pair-feature tensors match for the same metadata row.
- Schema/version tests fail if feature order or meaning changes.

### 7. Required Pair Features Are Incomplete

Status: verified gap.

Spec expectation:

Required V1 pair features include:

- axial distance,
- same-axis indicator,
- same-line indicator,
- same-window indicator,
- terminal exact win flag,
- terminal-equivalent win flag,
- terminal exact cover flag,
- covers-all-opponent-win-requirements flag,
- impossible-to-cover state flag,
- legal-row phase flags.

Current behavior:

- Runtime has some geometry and tactical flags, but lacks distinct:
  - terminal exact cover,
  - covers-all-opponent-win-requirements,
  - impossible-to-cover,
  - full phase flags.
- Runtime includes source/procedure fields not intended as default model
  inputs.
- The model fallback path conflates or simplifies some geometry, including
  setting `same_axis = same_line`.

Evidence:

- `Python/src/hexorl/selfplay/worker.py`
- `Python/src/hexorl/models/families/global_graph.py`
- `Docs/FINAL_HEXORL_IMPLEMENTATION_PLAN.md`

Behavioral effect:

- The pair scorer cannot directly distinguish a winning completion from an
  exact cover or impossible cover state except through collapsed flags.
- Tactical target supervision loses specificity.
- Source labels can become shortcuts for tactical relevance.

Required correction:

- Replace the current feature schema with the required feature set.
- Keep source/procedure features only in an ablation feature group.
- Validate schema width, version, and semantic field names at graph boundary.

Acceptance evidence:

- Tests showing each required pair feature is populated from a known tactical
  fixture.
- Tests showing source/procedure features are absent from default scorer input.

### 8. `terminal_tactical_v1` Replay And Target Fidelity Are Incomplete

Status: verified gap.

Spec expectation:

Replay should preserve enough tactical data to train exact V1 tactical labels:

- status,
- winning single cells,
- hot completion pairs,
- terminal equivalent pairs,
- opponent win requirements,
- hot cover pairs,
- impossible-to-cover flag.

Current behavior:

- Rust exposes a full `TerminalTacticalSetV1`.
- Replay stores candidate/search metadata, not the full tactical payload.
- `v1_terminal_tactical_target` is synthesized from candidate masks rather than
  the full Rust tactical payload.

Evidence:

- `crates/hexgame-core/src/v1.rs`
- `Python/src/hexorl/selfplay/records.py`
- `Python/src/hexorl/train/v1_pair_targets.py`

Behavioral effect:

- Tactical labels are lossy.
- Impossible-to-cover, opponent requirement count, and exact cover semantics
  are collapsed or absent.
- If a tactical row is not admitted, its tactical fact may be missing from the
  training target.

Required correction:

- Store the Rust tactical payload or a faithful compact encoding in V1 replay.
- Build `terminal_tactical_v1` targets from that payload, not only from
  candidate masks.
- Keep candidate-source flags for support/debug, but do not treat them as the
  tactical label source of truth.

Acceptance evidence:

- Tests where tactical payload contains impossible-to-cover or cover facts even
  when no candidate row survives, and the target still records the tactical
  fact.

### 9. `pair_proposal_score` Target Contract

Status: fixed in source.

Spec expectation:

`pair_proposal_score` should train proposal/retrieval quality or ranking in a
way that matches its target contract.

Current behavior after the May 8 target cleanup:

- `pair_proposal_score` is the retrieval/ranking head for sampled V1 pair rows.
- Its loss contract points to `v1_pair_ranking_target`.
- The unused `v1_pair_proposal_score_target` arrays and tests were removed.

Evidence:

- `Python/src/hexorl/train/v1_pair_targets.py`
- `Python/src/hexorl/train/loss_plan.py`

Behavioral effect:

- The runtime can use `pair_proposal_score` as a learned proposal/retrieval
  signal without carrying an unused target tensor.
- Loss naming and target production now agree that this is a ranking contract.

Required correction:

- No source correction remains for this contract.

Acceptance evidence:

- `Python/tests/test_v1_pair_training_losses.py`
- `Python/tests/test_v1_pair_targets.py`
- Source audit:

```powershell
rg -n "v1_pair_proposal_score_target|pair_proposal_score_target" Python/src Python/tests
```

Observed:

```text
no matches
```

### 10. Pair Search Is Recursive Pair-Native MCTS, But Root Gumbel SH Is Not Complete

Status: source fixed for recursive pair-action search; root Gumbel sequential
halving remains an unclaimed gap.

Spec expectation:

`sampled_joint_pair_v1` should own:

- pair prior construction,
- root Gumbel admission,
- proposal-aware PUCT,
- cached progressive widening,
- pair expansion and backup,
- search telemetry,
- candidate-aware targets.

Current behavior after the May 8 recursive-search patch:

- Rust root search admits pair candidates and runs recursive pair-action PUCT
  with deterministic Gumbel admission/tie metadata.
- `run_search_step(...)` descends from the root through expanded interior pair
  nodes until it reaches a terminal/structural leaf or an unexpanded full-turn
  node.
- `complete_expansion(...)` attaches exactly one scored reservoir to the
  expanded node, widens it from cache, and backs up the supplied neural value.
- Python V1 self-play loops on `run_search_step(...)` and uses the same
  learned selector plus bounded final pair scoring path for root and interior
  nodes.
- The stale shallow nonterminal `0.0` fallback has been removed.
- This is still not a complete `gumbel_sequential_halving_v1` root policy
  improvement operator. Root admission and replay expose Gumbel values and
  allocation metadata, but simulation allocation is recursive PUCT and final
  selection is completed-Q-first over visited admitted candidates.

Evidence:

- `crates/hexgame-core/src/v1_pair_search.rs`
- `Python/src/hexorl/selfplay/worker.py`

Behavioral effect:

- Non-terminal pair selection now receives neural value backup from future
  pair-turn states.
- The implementation should be described as recursive pair-action PUCT with
  Gumbel admission metadata, not Gumbel sequential halving.
- Final root policy targets are completed-Q candidate posteriors, so they avoid
  raw visit-count targets over sampled support, but they are not a formal
  Gumbel improved-policy target.

Required correction:

- No source correction remains for recursive pair-action search.
- If the architecture keeps `gumbel_sequential_halving_v1` as a hard
  acceptance criterion, add a real root sequential-halving schedule and
  corresponding improved-policy target, or amend the architecture to accept
  completed-Q recursive PUCT as the V1 root operator.
- Produce Rust/PyO3 search traces in a Rust-capable environment.

Acceptance evidence:

- Search traces showing interior full-turn node expansion and backup.
- Telemetry showing neural calls per expanded full-turn node and reservoir
  lifecycle.
- Tests where deeper non-terminal value changes selected root pairs.

### 11. Interior Reservoir And Progressive Widening Are Mainline, With Performance Evidence Still Open

Status: source fixed for mainline search lifecycle and request-batch provider
submission; performance evidence remains open.

Spec expectation:

Interior lifecycle is a hard runtime contract:

```text
one neural evaluation
one candidate-reservoir build
one bounded scoring pass per expanded full-turn node
widening reveals cached rows
```

Current behavior after the May 8 recursive-search patch:

- Rust exposes `run_search_step(...)`, `complete_expansion(...)`, and
  `select_root_action(...)` as the mainline V1 protocol.
- Interior reservoirs are built from Python expansion completions and attached
  to recursive search nodes.
- Progressive widening reveals rows from cached reservoirs without rescoring.
- PUCT priors for interior selection are renormalized over the currently
  revealed rows.
- Python batches proposal graphs for all expansion requests returned by one
  `run_search_step(...)`, then batches all admitted-pair scoring graphs before
  completing the expansions back into Rust. Test fakes without
  `submit_graph_many(...)` use a fallback to `submit_graph(...)`.

Evidence:

- `crates/hexgame-core/src/v1_pair_search.rs`
- `crates/hexgame-py/src/engine.rs`
- `Python/src/hexorl/selfplay/worker.py`

Behavioral effect:

- Progressive widening is active in the recursive V1 search lifecycle.
- The remaining risk is empirical throughput: the source now uses batched graph
  submission, but the final wall-clock gates still need to be measured in the
  real inference environment.

Required correction:

- No source correction remains for request-batch graph submission.
- Record and gate encoder-forward count, pair-scorer-forward count, proposal
  latency, pair-scoring latency, pair-scores/sec, queue/backpressure, and GPU
  utilization in performance artifacts.

Acceptance evidence:

- Runtime tests or traces proving interior cache/widen calls occur during
  self-play, not only unit tests.

### 12. Reservoir Refill Event Details Are Not Preserved

Status: verified gap.

Spec expectation:

Replay must store `reservoir_refill_events`.

Current behavior:

- `V1SearchPairMetadata` supports `reservoir_refill_events`.
- Runtime metadata sets `reservoir_refill_events=()`.
- Only refill counts appear in surprise metrics.

Evidence:

- `Python/src/hexorl/selfplay/records.py`
- `Python/src/hexorl/selfplay/worker.py`

Behavioral effect:

- Replay cannot audit why a reservoir refilled, how many rows were requested,
  or which rows were added.
- This limits debugging of support collapse and widening behavior.

Required correction:

- Populate `reservoir_refill_events` from Rust telemetry when refills occur.
- If refills are not implemented, keep the field empty but document that V1
  mainline currently has no refill behavior.

Acceptance evidence:

- Test with a forced refill event and replay round-trip preserving details.

### 13. Evaluation And Cutover Are Schema-Gated, Not Completed

Status: verified gap.

Spec expectation:

V1 final acceptance requires:

- fair sequential DAG neural baseline,
- equal-wall-clock arena evidence,
- candidate recall artifacts,
- tactical suite accuracy,
- direct retrieval recall proof,
- D6 identity/policy/value consistency,
- performance profile,
- deletion/import proof for obsolete paths.

Current behavior:

- `v1_pair_scorecard.py` defines required baselines, metrics, and
  equal-wall-clock fields.
- Tests validate schema-only mode and reject unbacked strength claims.
- There is no evidence in the inspected paths that the equal-wall-clock
  comparison or final acceptance artifacts have been produced.

Evidence:

- `Python/src/hexorl/eval/v1_pair_scorecard.py`
- `Python/tests/test_v1_pair_eval_guardrails.py`

Behavioral effect:

- The project correctly prevents unbacked V1 strength claims.
- V1 cannot yet be called accepted or cut over.

Required correction:

- Generate real scorecard payloads with evidence artifact paths.
- Run fair equal-wall-clock comparisons against required baselines.
- Attach candidate recall, tactical suite, D6, and performance artifacts.

Acceptance evidence:

- Non-schema-only V1 scorecard passing all hard gates.

### 14. Legacy Pair Paths Still Exist

Status: fixed for flagship V1 runtime; legacy authority is quarantined for
explicit offline/eval baselines.

Spec expectation:

Old paths may exist during development as baselines, but V1 final move choice
must not use pair-to-single projection, and obsolete paths must be removed or
quarantined before cutover.

Current behavior after the May 8 patch:

- `root_pair_mcts` and `full_pair_mcts` are rejected by normal
  `build_pair_strategy(...)`.
- Explicit baseline tooling must call `build_legacy_pair_baseline_strategy(...)`
  to construct those old modes.
- `pair_logits_to_action_logits` lives in
  `hexorl.search.legacy_pair_projection`, not in `pair_strategy.py`.
- The `sampled_joint_pair_v1` segment of `SelfPlayWorker` does not contain the
  legacy projection or prior-application tokens.

Evidence:

- `Python/src/hexorl/search/pair_strategy.py`
- `Python/src/hexorl/config/schema.py`
- `Python/tests/test_v1_pair_ci_audit_gates.py`
- `Python/tests/test_v1_selfplay_worker_runtime.py`

Behavioral effect:

- This is acceptable before cutover because legacy modes are protected
  baselines.
- It remains a blocker for final V1 cutover unless quarantined or removed.

Required correction:

- No source correction remains for flagship runtime quarantine.
- Final acceptance still needs an archived cutover audit artifact produced in
  the target CI/Rust-capable environment.

Acceptance evidence:

- Cutover audit artifact showing no V1 path calls pair-to-single projection.

## Historical Behavioral Difference Summary

This table reflects the first-pass pre-patch review. Use the May 8 patch
summary and checklist statuses above for current source status.

| Area | Pre-Patch Behavior | Spec Behavior | Effect |
|---|---|---|---|
| Candidate admission | Built before graph inference | Model-informed before final reservoir | Model cannot rescue missed pairs |
| Direct retrieval | Coordinate/id heuristic | Learned embedding retrieval | Retrieval recall cannot learn |
| Tactical protection | Protected unless over budget | Never evicted by ordinary budget | Exact tactical rows can be absent |
| Source telemetry | Pre-final truncation | Final support auditable | Source mix can be misleading |
| Pair features | Runtime source/procedure features; training fallback features | Canonical required feature schema | Train/infer shift |
| Tactical targets | Candidate-mask-derived | Rust tactical-payload-derived | Lossy tactical supervision |
| Proposal score loss | Ranking target | Explicit chosen proposal/ranking contract | Built target may be unused |
| Search | Root Gumbel + shallow terminal eval | Recursive pair-action PUCT with interior expansion/widening | Source fixed; Rust/PyO3 evidence and Gumbel-SH decision remain |
| Widening | Rust API exists | Integrated mainline lifecycle | Source fixed; performance contract unproven |
| Evaluation | Schema gate only | Evidence-backed hard pass | No strength/cutover claim |

## Likely Improvements Over The Written Spec

These deviations or implementation choices appear beneficial and should be
preserved unless later evidence says otherwise.

### Explicit Activation By Strategy

The implementation requires `sampled_joint_pair_v1` and rejects accidental V1
activation by heads alone. This is safer than head-name inference.

### Narrow Hardcoded Exceptions

Opening and one-placement terminal phases are hardcoded in Rust. Normal
two-placement tactical states are still searched. This keeps the model focused
on real pair decisions and removes needless complexity for structural
exceptions.

### Separate V1 Replay Schema

Separating V1 metadata from legacy pair targets avoids treating sampled support
as exhaustive supervision. This should remain a non-negotiable boundary.

### Deterministic Auxiliary Row Pools

Structured diversity and blind canaries use deterministic bounded pools for
large legal tables. This is a practical runtime improvement if final telemetry
clearly reports survival and eviction.

### Schema-Only Scorecard Gate

The scorecard guardrails correctly allow schema validation without pretending
strength evidence exists. This is an honest and useful intermediate state.

## Required Completion Checklist

Before claiming the V1 pair model is fully accepted, the implementation should
close or produce evidence for these items:

1. Source fixed: model-informed candidate admission using learned legal
   projection and proposal heads.
2. Source fixed: learned direct retrieval replaces live coordinate-only
   retrieval.
3. Source fixed: protected tactical candidates survive ordinary budget; overflow
   is represented explicitly.
4. Source fixed: candidate-source telemetry is split into proposed,
   pre-budget, final, and evicted counts.
5. Source fixed: blind-canary survival/loss is explicitly counted.
6. Source fixed: one canonical V1 pair-feature schema is shared by training and
   runtime.
7. Source fixed: default pair features exclude source/procedure fields.
8. Source fixed: Rust tactical payload is stored in schema-2 V1 replay.
9. Source fixed: `terminal_tactical_v1` targets come from tactical payload.
10. Source fixed: `pair_proposal_score` trains through the ranking target; unused
    target arrays were removed.
11. Source fixed, Rust verification pending: recursive interior pair MCTS
    descent, PUCT selection, widening, and path backup protocol.
12. Source fixed, Rust verification pending: mainline V1 self-play uses the
    recursive expansion protocol for normal full-turn nodes.
13. Source fixed for zero-refill default; non-default refill serialization still
    needs Rust-capable replay evidence.
14. Evidence pending: candidate recall, tactical suite, D6, direct retrieval,
    and performance artifacts.
15. Evidence pending: fair equal-wall-clock comparisons against required
    baselines.
16. Source fixed locally; final CI artifact pending for obsolete path audit.

## Remaining Verification Order

The source-level behavior gaps have been patched. Remaining work should produce
evidence in this order:

1. **Rust search tests**
   Run:

```powershell
cargo test -p hexgame-core v1_pair_search -- --nocapture
```

   Required verification:

   - Rust compiles with the new recursive search structs and helpers.
   - Nonterminal neural backup changes root choice.
   - `num_simulations` counts completed recursive simulations, not just root
     candidate evaluation.
   - PUCT descends beyond depth 1 after a child reservoir is expanded.
   - One reservoir is built per expanded full-turn node.
   - Repeated visits widen from cache without rescoring.
   - Interior PUCT/widening uses priors renormalized over revealed rows.
   - Terminal-after-first-placement and terminal-after-second-placement paths
     short-circuit and back up correctly.
   - Terminal-safe pair ordering applies an immediate winning cell before a
     filler cell.
   - Completion rejects stale, duplicate, illegal, non-finite, or wrong-node
     payloads.
   - Proposal correction weights are validated and preserved but do not alter
     search priors.
   - Final root selection is completed-Q-first among visited admitted rows.
   - No `simulate_candidate` or shallow `evaluate_pair(...)=0.0` fallback
     remains.

   Also run:

```powershell
rustfmt --check crates/hexgame-core/src/v1_pair_search.rs crates/hexgame-py/src/engine.rs
cargo test -p hexgame-py v1_pair -- --nocapture
```

   The exact PyO3 crate test selector may need adjustment if the crate does not
   expose Rust-side tests with `v1_pair` in the name.

2. **PyO3 lifecycle tests**
   Build `_engine`, then run:

```powershell
$env:PYTHONPATH='Python/src'
python -m pytest Python\tests\test_v1_pair_search_ffi.py -q
```

   Required verification:

   - `init_root_v1()` returns schema/version/hash/state identity fields.
   - `run_search_step(max_expansions)` returns expansion requests containing:
     `node_key`, compact move history bytes, legal row table, tactical payload,
     parent visits, node visit count, root generation, phase, and legal row
     table hash.
   - Python can complete a root expansion, then receive and complete a deeper
     interior expansion.
   - `complete_expansion(...)` rejects stale legal rows and root-level pair rows
     for an interior node.
   - `select_root_action()` rejects selection while requests are pending or the
     simulation budget is incomplete.
   - `apply_selected_action(...)` validates generation, legal-table hash, and
     pair key before mutating the Rust game.
   - Replay telemetry exposes root visits/Q/completed-Q, interior expanded
     node count, reservoir build count, widening events, neural-call count, and
     explicit zero refill events by default.

3. **Search trace artifact**
   Capture root and interior expansion, reservoir cache, widening, backup, and
   selected action identity in one replayable trace.

   Required artifact contents:

   - root legal row table hash plus turn-start/current state hashes,
   - root admitted candidate rows with priors, Gumbel values, correction modes,
     visits, Q, completed-Q, allocation, and selected pair key,
   - at least one expansion request at depth 1 and one deeper expansion request,
   - one reservoir build per expanded node,
   - widening events showing reveal from cached rows,
   - model value at each expanded node with current-player perspective,
   - backup path from expanded node to root,
   - final selected action and applied action identity.

   Suggested command, after `_engine` is built:

```powershell
$env:PYTHONPATH='Python/src'
python scripts\run_v1_selfplay_coherence_smoke.py --target-states 8 --mcts-simulations 4 --max-game-moves 8 --pair-budget 32
```

   If the script arguments differ, keep the artifact requirement above and use
   the repository's current smoke entrypoint.

4. **Recall and D6 artifacts**
   Produce candidate recall, direct retrieval recall, tactical fixture, and D6
   identity/policy/value consistency reports.

   Required verification:

   - learned direct retrieval recall changes when legal proposal embeddings
     change,
   - tactical protected rows survive ordinary budget,
   - blind canaries are non-selectable/training-forbidden and counted,
   - Rust legal row identity matches graph LEGAL row identity for D6 fixtures,
   - train/infer V1 pair features are byte/field-order equivalent for the same
     candidate row,
   - unsampled legal pairs remain unlabelled rather than implicit negatives.

5. **Performance profile**
   Measure candidate generation p95, pair scores/sec, inference latency, neural
   calls per expanded full-turn node, queue/backpressure, and GPU utilization.

   Required verification:

   - request-batch proposal graphs use `submit_graph_many(...)`,
   - admitted-pair scoring graphs use `submit_graph_many(...)`,
   - no per-pair neural calls occur during widening,
   - widening reveals from cached reservoir rows,
   - telemetry separates proposal forward time, candidate build time, final
     scoring forward time, proposal batch size, scoring batch size, pair
     scores/sec, and queue/backpressure,
   - equal-wall-clock throughput is reported against required baselines.

6. **Acceptance scorecard**
   Run equal-wall-clock arena comparisons and attach non-schema-only scorecard
   evidence.

   Required verification:

   - schema-only scorecard mode is not used for final acceptance,
   - baseline list includes required non-pair and legacy/offline pair baselines,
   - scorecard points to concrete recall, D6, tactical, trace, and performance
     artifacts,
   - results are equal-wall-clock, not equal-simulation-only.

7. **Code-search and quarantine audit**
   Run:

```powershell
rg -n "simulate_candidate\(|evaluate_pair\(|_v1_legal_embedding_features|v1_pair_proposal_score_target|pair_proposal_score_target" Python/src Python/tests crates/hexgame-core/src
rg -n "pair_logits_to_action_logits|root_pair_mcts|full_pair_mcts" Python/src/hexorl/search Python/src/hexorl/selfplay Python/src/hexorl/config Python/src/hexorl/autotune Python/src/hexorl/tuning
```

   Required verification:

   - first audit has no matches,
   - second audit has only explicitly quarantined legacy/offline references and
     no `sampled_joint_pair_v1` runtime calls to pair-to-single projection.

## Verification Notes For Future Reviewers

When rechecking this document, prioritize these audits:

```powershell
rg -n "select_pair_candidates_v1|cell_marginal_logits|pair_completion_logits|pair_proposal_score|pair_joint_logits" Python/src/hexorl/selfplay Python/src/hexorl/search
rg -n "_v1_pair_features_from_candidates|pair_features|V1_PAIR_FEATURE_DIM" Python/src/hexorl
rg -n "terminal_tactical_v1|TerminalTacticalSetV1|impossible_to_cover|hot_cover_pairs|opponent_win_requirements" Python/src crates/hexgame-core/src
rg -n "cache_interior_reservoir|widen_interior_reservoir|run_root_search|simulate_candidate|evaluate_pair" crates/hexgame-core/src Python/src
rg -n "pair_logits_to_action_logits|root_pair_mcts|full_pair_mcts|sampled_joint_pair_v1" Python/src Python/tests
```

The most important success signal is not that these terms exist. It is that
the `sampled_joint_pair_v1` runtime consumes the V1 contracts in the same order
and with the same semantics described by the spec.
