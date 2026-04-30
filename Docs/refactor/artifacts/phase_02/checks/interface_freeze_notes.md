# Phase 02 Interface Freeze Notes

- Created: `2026-04-30T00:29:18-04:00`
- Git SHA: `549c14d8f4837b98257a95f4fd5b4496e26b76b1`
- Branch: `codex/phase-02-builder-convergence`

## Frozen Public Owners

`CandidateContractBuilder` in `Python/src/hexorl/contracts/candidates.py` owns candidate row identity, dense board indices, storage mask, target projection, feature blocks, diagnostics, and candidate table hash.

`PairActionTableBuilder` in `Python/src/hexorl/contracts/pairs.py` owns pair row identity, first/second candidate references, first-placement unordered semantics, second-placement known-first semantics, generation mode, caps, selected/possible counts, target projection, and table hash.

`GraphSemanticBuilder` in `Python/src/hexorl/graph/semantic_builder.py` owns graph token identity, token semantic rows, relation identity, legal links, candidate links, pair links, tactical labels, debug metadata, and graph semantic hash.

`GraphTensorizer` in `Python/src/hexorl/graph/tensorize.py` owns pure tensor projection from graph, legal, candidate, and pair contracts. It may validate shape/schema/hash compatibility but must not regenerate semantic rows.

`collate_graph_batches` in `Python/src/hexorl/graph/collate.py` owns only padding and batching of already tensorized graph batches.

## Candidate Interface

Required stable fields:

- `CandidateTable.rows`: canonical `(q, r)` rows.
- `CandidateTable.dense_indices`: dense board indices for rows.
- `CandidateTable.mask`: storage-width mask for active rows.
- `CandidateTable.target`: normalized target over active rows.
- `CandidateTable.features`: registered feature-block projection.
- `CandidateTable.diagnostics`: missing mass, recall/discovery metrics, critical overflow details, source/hash/schema.
- `CandidateTable.table_hash`: deterministic hash over semantic rows, dense indices, mask, targets, features, diagnostics-defining metadata, source, and schema.

Extension freeze:

- Feature blocks contribute through registered callables invoked by the builder.
- Extensions cannot reorder rows, change dense identity, bypass mask validation, or override hash inputs.

## Pair Interface

Required stable fields:

- `PairActionTable.rows`: canonical `(first_q, first_r, second_q, second_r)` rows.
- `PairActionTable.first_candidate_rows` and `second_candidate_rows`: references into the candidate table when available.
- `PairActionTable.phase`: `first_placement`, `second_placement_known_first`, or `empty`.
- `PairActionTable.known_first`: required for second-placement known-first rows.
- `PairActionTable.generation_mode`: no-pairs, selected, capped-fill, or full-capped.
- `PairActionTable.possible_pair_count` and `selected_pair_count`.
- `PairActionTable.mask`, `target`, `missing_mass`, and `table_hash`.

Cap freeze:

- Full `A * (A - 1) / 2` generation is forbidden unless a Phase 02 explicit capped pair strategy requests it and supplies a cap at least as large as the selected row count.
- Pair selection/caps may choose rows; only `PairActionTableBuilder` defines row identity and validation.

## Graph Interface

`GraphSemanticContract` is the semantic object. It contains immutable arrays or tuples for token types, token coordinates, legal-token references, candidate-token references, pair-token references, relation types, relation bias semantics, target semantics, tactical labels, schema versions, and debug metadata.

`GraphBatch` is the tensor projection object. It contains NumPy arrays used by model/training/inference code and carries source graph/candidate/pair/legal hashes for mutation and stale-schema tests.

## Consumer Cutover

The following runtime paths must import shared builders, not old private constructors:

- `Python/src/hexorl/selfplay/worker.py`
- `Python/src/hexorl/buffer/sampler.py`
- `Python/src/hexorl/dashboard/app.py`
- `Python/src/hexorl/dashboard/model_cache.py`
- `Python/src/hexorl/dashboard/replay.py`
- tests and fixtures that build model inputs

## Deleted Or Demoted Interfaces

- `PairCandidateBatch` cannot remain a semantic owner.
- `hexorl.action_contract.candidates` cannot remain the production owner of candidate or pair semantics. It may be deleted or limited to import-free projection aliases only if no runtime path imports it.
- `hexorl.graph.batch` cannot own both graph semantics and tensorization after implementation. Runtime consumers must import the split modules.

