# Phase 02 Builder Contract Examples

## Candidate Builder

`CandidateContractBuilder.build(...)` is the only production owner for candidate rows, dense indices, masks, targets, features, diagnostics, and `CandidateTable.table_hash`.

Feature extensions are registered with the builder as feature-block callables. A feature block receives `CandidateFeatureContext` and may append feature columns only; it cannot reorder rows, mutate masks, change dense identity, or bypass hash inputs.

## Pair Builder

`PairActionTableBuilder.build(...)` is the only production owner for pair row identity.

- First placement rows are canonical unordered `(first_q, first_r, second_q, second_r)` rows.
- Second placement rows require `known_first` and preserve ordered known-first semantics.
- `PairStrategy` declares generation mode and hard caps for Phase 02 row construction. Final search strategy and scoring remain Phase 05-owned.
- `PairActionTable.pair_indices` is a tensor projection from canonical first/second candidate row references.

## Graph Builder

`GraphSemanticBuilder.build(...)` owns graph token identity, legal-token links, pair-token links, relation identities, tactical labels, and debug metadata.

`GraphTensorizer.tensorize(...)` copies a `GraphSemanticContract` into model-facing arrays and validates shape/schema compatibility. It does not parse history, rebuild legal rows, generate candidates, generate pair rows, scan tactics, or apply D6 transforms.

`collate_graph_batches(...)` pads and batches tensorized `GraphBatch` objects only.

