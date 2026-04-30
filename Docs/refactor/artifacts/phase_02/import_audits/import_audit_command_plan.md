# Phase 02 Import Audit Command Plan

Use `git grep` because local `rg.exe` access has previously failed.

```powershell
git grep -n "CandidateBatch\|PairCandidateBatch\|build_candidate_batch\|build_pair_candidate_batch" -- Python/src Python/tests
git grep -n "build_graph_batch_from_history\|graph_batch_with_reference_pair_rows\|collate_graph_batches" -- Python/src Python/tests
git grep -n "legal_moves_for_stones\|parse_history\|_target_for_pairs\|PAIR_ACTION" -- Python/src/hexorl Python/tests
git grep -n "source=\"fallback\"\|fallback source" -- Python/src/hexorl Python/tests
```

Closing expectation: runtime consumers import `CandidateContractBuilder`, `PairActionTableBuilder`, `GraphSemanticBuilder`, `GraphTensorizer`, and `collate_graph_batches` from the approved modules. Any remaining old names must be test-only compatibility coverage or absent.

