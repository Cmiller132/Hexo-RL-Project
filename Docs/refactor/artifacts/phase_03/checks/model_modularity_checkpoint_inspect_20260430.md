# Phase 03 Model Modularity And Checkpoint Inspect Evidence - 2026-04-30

## Scope
- Closed corrective evidence for V2-031 model family descriptor/facet registration quality.
- Closed corrective evidence for V2-034 checkpoint inspect metadata path.
- Write scope was limited to `Python/src/hexorl/models/**`, `Python/tests/models/**`, `Python/tests/train/test_phase03_train_adapter_checkpoint.py`, and this Phase 03 artifact.

## Implementation Evidence
- Built-in descriptors are constructed in family modules:
  - `hexorl.models.families.dense_cnn`
  - `hexorl.models.families.restnet`
  - `hexorl.models.families.graph_hybrid`
  - `hexorl.models.families.global_xattn`
  - `hexorl.models.families.global_line_window`
  - `hexorl.models.families.global_relation_graph`
- `hexorl.models.factory` now registers descriptors from `hexorl.models.families.builtin_descriptors()` and routes public APIs through descriptor facets.
- Head modules expose named head component sets consumed by family descriptors.
- Trunk modules expose model builders consumed by family descriptors.
- `CheckpointManager.save()` writes a lightweight `checkpoint_manifest.json` member inside the PyTorch zip archive.
- `CheckpointManager.inspect()` reads the JSON manifest member directly with `zipfile` and does not call `torch.load()`.

## Commands
- `PYTHONPATH=Python/src python3 -m pytest -q Python/tests/models/test_phase03_model_registry.py Python/tests/train/test_phase03_train_adapter_checkpoint.py::test_checkpoint_manifest_round_trips Python/tests/train/test_phase03_train_adapter_checkpoint.py::test_checkpoint_inspect_does_not_load_weights Python/tests/train/test_phase03_train_adapter_checkpoint.py::test_checkpoint_load_rejects_missing_manifest Python/tests/train/test_phase03_train_adapter_checkpoint.py::test_checkpoint_load_rejects_unknown_or_stale_manifest_fields Python/tests/train/test_phase03_train_adapter_checkpoint.py::test_checkpoint_load_rejects_model_family_mismatch Python/tests/train/test_phase03_train_adapter_checkpoint.py::test_checkpoint_load_rejects_inference_protocol_mismatch Python/tests/train/test_phase03_train_adapter_checkpoint.py::test_checkpoint_load_does_not_silently_strip_orig_mod_or_prefixes Python/tests/train/test_phase03_train_adapter_checkpoint.py::test_no_runtime_imports_from_hexorl_model Python/tests/train/test_phase03_train_adapter_checkpoint.py::test_no_model_architecture_string_gates_outside_registry_spec_tests`
  - Exit status: 0
  - Result: `20 passed in 1.67s`
- `rg -n "_builtin_descriptors|_build_crop_model|_build_global_model|\[spec\.kind\]|if spec\.kind|elif spec\.kind|architecture\.startswith|build_model_from_config" Python/src/hexorl/models Python/tests/models Python/tests/train/test_phase03_train_adapter_checkpoint.py`
  - Exit status: 0
  - Result: no model-runtime switchboard hits; only test guardrail strings matched.
- `rg -n "def inspect\(|torch\.load\([^\n]*weights_only=False|checkpoint_manifest\.json" Python/src/hexorl/models/checkpoint.py Python/tests/train/test_phase03_train_adapter_checkpoint.py`
  - Exit status: 0
  - Result: `inspect()` and manifest JSON path found; `weights_only=False` remains only in full `load()` and mutation tests, not in `inspect()`.

## Broader Test Blocker
- Attempted full focused file run:
  - `PYTHONPATH=Python/src python3 -m pytest -q Python/tests/models/test_phase03_model_registry.py Python/tests/train/test_phase03_train_adapter_checkpoint.py`
  - Exit status: 1
  - Result: 26 passed, 7 failed.
- Blocker was outside this corrective scope:
  - Rust `_engine` extension is unavailable, blocking graph fixture construction.

## Integrated Verification Update
- Reran the same area through the repository `.venv`, where `_engine` is available:
  - `PYTHONPATH=Python/src ./.venv/bin/python -m pytest -q Python/tests/models/test_phase03_model_registry.py Python/tests/train/test_phase03_train_adapter_checkpoint.py Python/tests/contracts/test_phase02_builders.py Python/tests/engine/test_phase01_engine_contract_parity.py Python/tests/replay/test_phase07_import_audit.py Python/tests/replay/test_phase07_codec_storage_projector.py Python/tests/search/test_global_graph_pair_contracts.py Python/tests/test_global_graph_contract.py Python/tests/test_production_smoke.py`
  - Exit status: 0
  - Result: `96 passed in 214.51s`.
- The earlier `python3` blocker is an environment/runtime-extension blocker only; the integrated `.venv` verification closed the graph-backed coverage path.

## Completion Statement
No skipped, deferred, flaky, or manual-only check is being claimed complete for this corrective packet.
