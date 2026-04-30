# Phase 02 Deletion Manifest

| Deleted or demoted path | Result | Evidence |
|---|---|---|
| `Python/src/hexorl/action_contract/candidates.py` | Deleted. Candidate and pair semantics moved to `contracts/candidates.py` and `contracts/pairs.py`. | `import_audits/runtime_private_builder_audit.txt` |
| `PairCandidateBatch` production semantic owner | Deleted. Pair tensor indices are projected from `PairActionTable.pair_indices`. | `Python/tests/contracts/test_phase02_builders.py` |
| Private runtime imports of `hexorl.action_contract.candidates` | Removed from runtime consumers. | `import_audits/runtime_private_builder_audit.txt` |
| Monolithic graph semantic/tensor/collate ownership in `graph/batch.py` | Demoted to compatibility exports only; semantic builder, tensorizer, and collator live in split modules. | `import_audits/runtime_graph_batch_audit.txt` |
| `graph_batch_with_reference_pair_rows` | Removed from runtime. Replaced by `graph_batch_with_pair_table`, a projection from `PairActionTable`. | `Python/tests/test_global_graph_contract.py`, `Python/tests/contracts/test_phase02_builders.py` |

No old runtime candidate builder, pair mini-contract, or graph semantic/tensor monolith remains imported by production consumers.

