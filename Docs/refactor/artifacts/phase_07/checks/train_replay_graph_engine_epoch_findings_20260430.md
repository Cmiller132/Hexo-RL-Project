# Train/Replay/Graph/Engine/Epoch Findings Closure - 2026-04-30

## Scope

Closed follow-up findings for the runtime replay-to-train boundary, graph pair semantics, Rust legal-byte ownership, and epoch bootstrap legal-row ownership.

## Closed V2 Rows

- V2-012: Legal rows used by production `engine/legal.py` now require the Rust/PyO3 `encode_board_and_legal` byte protocol.
- V2-021/V2-025: Graph pair projection and train graph target validation now validate phase-aware `PairActionTable` row semantics, known-first second-placement rows, legal token references, and canonical unordered first-placement rows.
- V2-071/V2-072: Training adapters consume only `ProjectedReplayBatch` from `replay/projector.py`; raw tuple handoff and legacy tuple projection were removed.

## Runtime Consumers Changed

- `train/adapters.py`: accepts only `ProjectedReplayBatch`, rejects raw tuples, and validates graph pair targets semantically against canonical pair-table metadata.
- `replay/projector.py`: removed legacy tuple export and carries canonical graph pair-table metadata into projected training batches.
- `graph/tensorize.py` and `graph/collate.py`: preserve canonical pair rows, phase, known-first metadata, and masks through graph projection/collation.
- `graph/semantic_builder.py`: freezes graph semantic ndarray fields and removed the production reference-pair helper.
- `engine/legal.py`: rejects objects without the centralized Rust legal-byte protocol.
- `epoch/pipeline.py`: removed Python bootstrap legal generation fallback; Rust/bootstrap failures raise a structured runtime error.

## Legacy Paths Deleted Or Quarantined

- Deleted `ProjectedReplayBatch.as_legacy_tuple`.
- Deleted production `graph_batch_with_reference_pair_rows`.
- Deleted `_make_fallback_bootstrap_game`, `_fallback_bootstrap_legal_moves`, and unused dense bootstrap helper that depended on Python legal fallback rows.
- Rejected arbitrary `game.legal_moves()` objects in `LegalTableProvider.from_game`.

## Verification

- `PYTHONPATH=Python/src .venv/bin/python -m py_compile Python/src/hexorl/train/adapters.py Python/src/hexorl/replay/projector.py Python/src/hexorl/graph/semantic_builder.py Python/src/hexorl/graph/tensorize.py Python/src/hexorl/graph/collate.py Python/src/hexorl/engine/legal.py Python/src/hexorl/epoch/pipeline.py Python/tests/train/test_phase03_train_adapter_checkpoint.py Python/tests/contracts/test_phase02_builders.py Python/tests/engine/test_phase01_engine_contract_parity.py Python/tests/replay/test_phase07_import_audit.py Python/tests/search/test_global_graph_pair_contracts.py Python/tests/test_global_graph_contract.py` -> exit 0.
- `PYTHONPATH=Python/src .venv/bin/python -m pytest Python/tests/train/test_phase03_train_adapter_checkpoint.py -q` -> exit 0, 22 passed.
- `PYTHONPATH=Python/src .venv/bin/python -m pytest Python/tests/contracts/test_phase02_builders.py -q` -> exit 0, 8 passed.
- `PYTHONPATH=Python/src .venv/bin/python -m pytest Python/tests/engine/test_phase01_engine_contract_parity.py -q` -> exit 0, 5 passed.
- `PYTHONPATH=Python/src .venv/bin/python -m pytest Python/tests/replay/test_phase07_import_audit.py -q` -> exit 0, 4 passed.
- `PYTHONPATH=Python/src .venv/bin/python -m pytest Python/tests/replay/test_phase07_codec_storage_projector.py -q` -> exit 0, 8 passed.
- `PYTHONPATH=Python/src .venv/bin/python -m pytest Python/tests/search/test_global_graph_pair_contracts.py -q` -> exit 0, 8 passed.
- `PYTHONPATH=Python/src .venv/bin/python -m pytest Python/tests/test_global_graph_contract.py -q` -> exit 0, 29 passed.
- `PYTHONPATH=Python/src .venv/bin/python -m pytest Python/tests/test_production_smoke.py -q` -> exit 0, 1 passed.
- `rg -n "as_legacy_tuple|len\\(batch\\) == [45]|raw tuple|legacy tuple" Python/src/hexorl/train Python/src/hexorl/replay Python/src/hexorl/epoch Python/src/hexorl/graph Python/src/hexorl/engine` -> exit 1, no matches.
- `rg -n "graph_batch_with_reference_pair_rows" Python/src/hexorl Python/tests` -> exit 1, no matches.
- `rg -n "_fallback_bootstrap_legal_moves|_make_fallback_bootstrap_game|fallback_bootstrap|Python legal fallback|game\\.legal_moves\\(\\)" Python/src/hexorl/train Python/src/hexorl/replay Python/src/hexorl/graph Python/src/hexorl/engine/legal.py Python/src/hexorl/epoch Python/tests/replay/test_phase07_import_audit.py` -> exit 1, no matches.

## Artifacts And Performance

- This closure report is the artifact for the finding fix.
- No hot-path performance benchmark was run; changes are validation/deletion focused. Existing replay projection throughput fields remain intact.

## Known Blockers

- None. Rust `_engine` was available in `.venv` and all focused verification ran against it.

No skipped, deferred, flaky, quarantined, or manual-only requirement is claimed complete by this report.
