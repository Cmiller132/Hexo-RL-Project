# V1 Pair MCTS Search

## Purpose

This document describes how `sampled_joint_pair_v1` searches with pair actions and how that search integrates with `global_pair_biaffine_0`.

The V1 pair path is intentionally different from the legacy pair-prior modes. It does not project pair logits into single-cell action logits. Python supplies model inference and candidate reservoirs. Rust owns legal-row identity, pair application, tree traversal, PUCT, progressive widening, backup, and final action choice.

Important naming boundary: the current Rust root operator is recursive pair-action PUCT with deterministic Gumbel admission and tie metadata. It is not a complete Gumbel AlphaZero / Gumbel MuZero sequential-halving policy-improvement operator. Replay stores Gumbel values, admission/allocation metadata, visits, Q, and completed-Q, but final selection is currently completed-Q-first over visited admitted candidates.

## Main Components

### Python Runtime Provider

`SelfPlayWorker` is the Python owner of model calls. In the V1 path it acts as an expansion provider for Rust:

- Builds graph batches from Rust legal rows and compact move history.
- Runs the proposal pass without pair rows.
- Selects bounded candidate pairs using learned and tactical signals.
- Runs the final pair-scoring pass only for admitted pairs.
- Sends scored pairs and model value back to Rust.
- Records Rust telemetry into replay metadata.

The main provider methods are:

- `_v1_build_and_score_root(...)`
- `_v1_complete_expansion_request(...)`
- `_play_one_game_v1_pair(...)`

Despite the name, `_v1_build_and_score_root(...)` is used for both root and interior full-turn nodes. The `root_or_interior` argument controls metadata and selector labeling.

### Model Outputs

For `global_pair_biaffine_0`, V1 search consumes these outputs:

- `value`: required for V1 inference. There is no silent `0.0` fallback.
- `cell_marginal_logits`: learned single LEGAL-row quality used during proposal admission.
- `legal_proposal_embeddings`: runtime-only learned LEGAL-row embeddings for direct pair retrieval.
- `legal_completion_query` and `legal_completion_key`: runtime-only learned LEGAL-row projections for anchor-completion scoring.
- `pair_completion_logits`: pair-row completion head, consumed when present in legal-by-legal form.
- `pair_proposal_score`: pair-row ranking/proposal head.
- `pair_joint_logits`: final score for admitted candidate pairs.
- `terminal_tactical_v1`: tactical supervision head during training, with replay targets built from Rust tactical payload.

The runtime-only legal projection tensors reuse learned V1 pair projection parameters. They are not separate trainable heads and the loss planner ignores them as non-head outputs.

### Rust Search Engine

Rust search is implemented by `V1PairSearchEngine`.

Important state:

- `root_identity`: root generation, root legal row table, and root tactical payload.
- `root_candidates`: admitted root pair reservoir with priors, visits, Q values, Gumbel values, flags, and target-support metadata.
- `search_nodes`: recursive interior tree nodes keyed by stable `node_key`.
- `pending_expansions`: unexpanded nonterminal leaves waiting for Python model output.

Each `V1SearchNode` stores:

- game state at that node,
- legal row table and hash,
- turn-start and current Rust state hashes inside the legal row table,
- tactical payload,
- parent edge and incoming pair,
- current player,
- visit count and total value,
- terminal value when known,
- optional cached pair reservoir.

Each cached pair reservoir stores:

- canonical `PairRowV1` rows,
- priors,
- edge visits,
- edge total values,
- child node keys,
- reservoir telemetry including revealed count, build count, scoring count, widening count, and refill count.

## Full Search Lifecycle

### 1. Rust Initializes The Root

At the start of a move, Python creates a Rust `PyV1PairSearchEngine` from the current game, simulation count, PUCT config, seed, and V1 pair budget.

Python then calls:

```text
init_root_v1()
```

Rust returns:

- root generation token,
- phase,
- legal row table,
- terminal tactical payload,
- legal pair count.

The root phase decides the path:

- `opening_single`: Rust selects the center/origin single exception.
- `one_placement`: Rust selects a terminal single if available.
- `terminal`: no action.
- `normal_two_placement`: full V1 pair search.

Only `normal_two_placement` uses recursive pair MCTS.

The `one_placement` phase is reserved for structural single-placement roots. A normal post-opening turn whose turn-start state has two placements remaining is `normal_two_placement`, including turns with immediate one-cell wins inside the pair-action candidate set.

### 2. Python Runs Root Proposal Inference

For a normal full-turn root, Python calls `_v1_build_and_score_root(...)`.

The first model pass is a proposal pass:

```text
build_graph_batch_from_history(..., include_pair_rows=False)
client.submit_graph(proposal_graph)
```

This graph contains the Rust LEGAL rows but no pair rows. The model encodes the graph once and returns LEGAL-row outputs:

- `cell_marginal_logits`
- `legal_proposal_embeddings`
- `legal_completion_query`
- `legal_completion_key`
- optionally legal-by-legal proposal/completion matrices
- `value`

Python validates that `cell_marginal_logits` covers the Rust LEGAL rows and that V1 is not using threat-filtered legal rows.

### 3. Python Selects Root Candidate Pairs

Python calls `select_pair_candidates_v1(...)` with:

- Rust legal row table,
- Rust tactical payload,
- learned legal proposal embeddings,
- learned completion matrix,
- pair proposal matrix when available,
- cell marginal logits,
- selector config,
- admission generation,
- `root_or_interior="root"`.

The selector combines multiple candidate sources:

- direct retrieval from learned legal embeddings,
- anchor-conditioned completion,
- cell-marginal cross,
- structured diversity,
- terminal/tactical protected pairs,
- blind canaries.

Important selector semantics:

- Tactical-protected candidates are not evicted by ordinary candidate budget.
- Blind canaries are diagnostic/training-forbidden where appropriate.
- Unsampled legal pairs are not treated as negatives.
- Telemetry separates proposed, pre-budget, admitted, evicted, protected, and canary counts.

The result is a bounded candidate reservoir, not the final move.

### 4. Python Runs Final Root Pair Scoring

Python converts admitted candidates into pair QR rows and canonical V1 pair features:

```text
v1_pair_features_for_candidates(...)
graph_batch_with_admitted_pair_rows(...)
client.submit_graph(graph_batch)
```

This second graph pass contains only admitted pair rows. The model returns:

- `pair_joint_logits` for the admitted pairs,
- `value` for the current node.

Python validates that inference metadata did not reorder or replace the LEGAL rows.

### 5. Python Admits Root Pairs Into Rust

Python sends the final root pair scores to Rust:

```text
admit_root_pairs(pair_qr, pair_logits, correction_weights, correction_modes, root_generation)
```

Rust validates:

- root generation,
- normal full-turn phase,
- metadata lengths,
- finite logits and weights,
- canonical pair rows against the Rust root legal row table,
- duplicate and illegal pair rows.

Rust then stores root candidates with:

- model logit,
- correction weight and mode for replay/training correction,
- model prior logit used for search,
- softmax prior,
- deterministic Gumbel value,
- visit and value stats initialized to zero,
- tactical and support flags.

If no root candidates are explicitly admitted by search, Rust performs deterministic Gumbel admission over the supplied candidate reservoir, respecting forced/tactical candidates and excluding training-forbidden rows. This admission step is not the same as a full root sequential-halving policy-improvement operator.

### 6. Rust Starts Recursive Simulations

Python enters the expansion loop:

```text
while True:
    requests = v1_engine.run_search_step(max_expansions)
    if not requests:
        break
    for request in requests:
        complete that expansion with model output
```

`run_search_step(max_expansions)` repeatedly starts simulations until:

- the configured simulation budget is reached,
- the batch has `max_expansions` unexpanded leaves,
- or all currently selectable paths are blocked by pending expansion.

Rust keeps pending requests out of further selection so the same leaf is not requested twice before Python completes it.

### 7. Rust Selects A Root Edge With PUCT

Each simulation starts at the root.

Rust computes a root PUCT score for admitted, selectable root candidates:

```text
score = Q(root_edge)
      + c_puct * prior(root_edge) * sqrt(max(1, parent_visits))
        / (1 + edge_visits)
      + tiny_gumbel_tiebreak
```

The selected root pair is atomically applied to a cloned game. Pair rows are canonical unordered rows, but physical application uses a terminal-safe deterministic order:

1. If exactly one cell in the pair is an immediate current-player win, apply that winning cell first.
2. Otherwise apply the canonical first placement.
3. If terminal, back up terminal value immediately.
4. Apply second placement.
5. If terminal, back up terminal value immediately.
6. Otherwise create or reuse the child search node.

The child node key is stable and derived from parent node key plus pair key. The root parent key is `0`.

### 8. Rust Descends Through Interior Nodes

At each interior node, Rust checks:

- If the node has terminal value, back it up immediately.
- If the phase is not `normal_two_placement`, evaluate the structural single-action leaf or error if it is not structurally resolvable.
- If the node has no reservoir, create an expansion request for Python.
- If the node has a reservoir, widen and select an interior edge with PUCT.

Interior PUCT uses the node-local edge stats and renormalizes priors over the currently revealed subset:

```text
score = Q(interior_edge)
      + c_puct * prior_revealed(interior_edge) * sqrt(max(1, node_visits))
        / (1 + edge_visits)
```

The selected interior pair is applied atomically in the same way as root pairs. If nonterminal, Rust creates or reuses the child node and continues descent.

This is the recursive part of V1 pair MCTS: once a child has been expanded and has a reservoir, later simulations can select from that reservoir, apply another pair, and request deeper expansions.

### 9. Progressive Widening

Every expanded full-turn node has a cached reservoir of scored candidate pairs. Rust does not ask Python to rescore that reservoir.

Before selecting an interior edge, Rust computes:

```text
revealed_limit =
  ceil(c_pw * max(1, node_visits) ** alpha_pw)
```

The limit is clamped to `[1, reservoir_size]`.

If the new limit is larger than the current revealed count, Rust reveals more cached pair rows and increments widening telemetry. PUCT selection only considers revealed rows.

This keeps the branching factor bounded while allowing more candidates to become selectable as visit count grows. Rust stores raw prior logits for the reservoir and computes the PUCT prior from a softmax over the revealed rows, so hidden rows do not suppress early exploration mass.

### 10. Rust Requests Neural Expansion

When a simulation reaches a nonterminal full-turn node with no reservoir, Rust stores a `V1PendingExpansion` containing:

- full path from root through selected edges,
- node path for visit updates,
- root player,
- node game state,
- legal row table hash.

Rust returns `V1ExpansionRequest` to Python with:

- `node_key`,
- compact move history bytes,
- legal row table,
- tactical payload,
- parent visits,
- node visit count,
- root generation,
- legal row table hash,
- phase.

The request is the only point where Python gets involved in tree expansion.

The legal row table contains both `turn_start_state_hash` and `current_state_hash`, so Python can build from compact history while still seeing the Rust state identity that produced the rows.

### 11. Python Completes The Expansion

For every request, Python calls `_v1_complete_expansion_request(...)`.

The provider batches expansion work at the request-batch level: it builds proposal graphs for all requests returned by one `run_search_step(...)`, submits them together, builds admitted-pair scoring graphs for all resulting candidate sets, submits those together, and then completes each expansion back into Rust. Test fakes that only expose `submit_graph(...)` use a compatibility fallback, but the normal inference client path uses `submit_graph_many(...)`.

It builds a node-root wrapper from the request:

```text
{
  "legal_row_table": request["legal_row_table"],
  "terminal_tactical": request["terminal_tactical"],
}
```

Then it calls `_v1_build_and_score_root(...)` with:

```text
root_or_interior="interior"
move_history=request["move_history_bytes"]
```

This means interior nodes use the same model and candidate pipeline as the root:

1. proposal graph pass,
2. learned candidate admission,
3. final bounded pair scoring pass,
4. required value extraction.

Python sends the result back to Rust:

```text
complete_expansion(
    node_key,
    node_value,
    pair_qr,
    pair_logits,
    correction_weights,
    correction_modes,
)
```

### 12. Rust Completes The Node

`complete_expansion(...)` validates:

- node exists,
- expansion is pending,
- node does not already have a reservoir,
- legal row table hash still matches,
- value is finite,
- candidate metadata lengths match,
- candidate pairs are legal canonical rows for that node,
- candidate rows are not duplicates.

Rust then:

1. Builds one reservoir for that node.
2. Stores rows, priors, visits, total values, child keys, and telemetry.
3. Marks the node expanded by attaching the reservoir.
4. Performs initial widening from cache.
5. Removes the pending expansion entry.
6. Converts the model value to root-player perspective.
7. Backs up through the stored pending simulation path.

The value convention is:

- Python/model returns value from the current player perspective at the expanded node.
- Rust converts it to root-player perspective.
- During backup, Rust updates each edge/node in the appropriate local perspective.

### 13. Backup

Backup updates both root and interior statistics.

For the root edge:

- increment root candidate visit count,
- increment allocation,
- add root-perspective value to total value,
- recompute root `simulation_count` as sum of root allocations.

For each interior edge on the path:

- increment edge visit count,
- add value in that node current-player perspective to edge total value.

For each node on the path:

- increment node visit count,
- add value in that node current-player perspective to node total value.

Completed simulations, not expansion requests, drive `simulation_count`.

### 14. Search Completion And Final Action

After Python sees `run_search_step(...)` return no requests, it calls:

```text
select_root_action()
```

For normal full-turn roots, Rust rejects final selection if:

- any expansion is still pending,
- the recursive simulation budget is incomplete,
- no admitted root pair has a backed-up visit.

When selection is legal, Rust completes root Q values and chooses the final pair among admitted, visited, selectable root candidates. The ordering prioritizes:

1. completed Q,
2. visit count,
3. prior,
4. deterministic pair-key tie break.

Rust returns the selected pair with:

- pair row,
- root generation,
- root legal row table hash.

Python then calls:

```text
apply_selected_action(root_generation, legal_row_table_hash, pair_key)
```

Rust validates the root token, current legal row table hash, and pair key, then atomically applies the selected pair to the authoritative Rust game.

Python mirrors the applied placements into its tracking move history for replay.

## Replay And Telemetry

After action application, Python reads Rust replay telemetry and stores it in `V1SearchPairMetadata`.

Important metadata includes:

- candidate pairs,
- selected pair,
- root visit counts,
- root allocations,
- Q values,
- completed Q values,
- improved policy target,
- forced/tactical/support flags,
- root Gumbel/admission order,
- reservoir build count,
- scoring pass count,
- neural calls per expanded full-turn node,
- reservoir refill events,
- interior expanded full-turn node count,
- interior reservoir build count,
- interior scoring pass count.

Selector and model-provider metrics are merged into `search_surprise_metrics`, including:

- model eval count,
- bounded scoring pass count,
- proposal forward time,
- candidate build time,
- final graph forward time,
- selector source/eviction/canary/protected counts,
- learned projection usage,
- graph legal row count,
- required pair cell count,
- interior candidate count,
- interior revealed count,
- interior widening events.

Reservoir refill is disabled by default. The default replay representation must explicitly report zero refill events.

## Training Integration

V1 replay is not treated as exhaustive pair supervision.

Training target construction preserves these rules:

- Admitted sampled pairs can receive policy/ranking/Q targets.
- Unsampled legal pairs are not implicit negatives.
- Tactical labels are built from the Rust tactical payload, not from candidate masks.
- `pair_proposal_score` trains through `v1_pair_ranking_target`.
- Runtime-only legal projection outputs are ignored by the loss planner.
- Canonical V1 pair features are shared by runtime graph batching and training.

This keeps train/infer feature semantics aligned and prevents sampled support from being mistaken for an exhaustive pair-label table.

## Failure Modes

V1 pair search fails loudly for contract violations:

- missing `value` output,
- threat-filtered LEGAL rows,
- proposal logits shorter than Rust legal rows,
- inference metadata LEGAL row identity changes,
- selector admits no candidates,
- stale root generation,
- stale or mismatched legal row table hash,
- duplicate or illegal pair rows,
- non-finite logits, weights, or values,
- selecting while expansions are pending,
- selecting before recursive simulation budget completes,
- applying a pair with a mismatched pair key.

These failures are intentional. V1 is a clean schema/runtime break and does not include compatibility facades for stale pair replay or stale runtime schemas.

## Current Verification Boundary

The source implements recursive V1 pair MCTS and Python model integration. Local Python tests cover the provider loop with fake recursive expansion, replay metadata paths, V1 training contracts, and V1 audit gates.

This environment does not have `cargo`, `rustc`, or a built `_engine`, so Rust compile/tests and PyO3 lifecycle tests still need to run in a Rust-capable environment:

```powershell
cargo test -p hexgame-core v1_pair_search
$env:PYTHONPATH='Python/src'
python -m pytest Python\tests\test_v1_pair_search_ffi.py -q
```

Final V1 acceptance also still requires search trace artifacts, performance profiles, candidate/direct retrieval recall artifacts, D6 consistency artifacts, and equal-wall-clock arena scorecards.
