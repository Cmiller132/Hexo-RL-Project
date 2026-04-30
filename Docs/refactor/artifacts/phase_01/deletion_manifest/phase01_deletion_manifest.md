# Phase 01 Deletion And Quarantine Manifest

## Removed Or Replaced Runtime Ownership

| Legacy owner | Replacement | Evidence |
|---|---|---|
| Runtime direct `_engine` imports in dashboard, eval, self-play, epoch, tactical, graph | `hexorl.engine.rust`, `hexorl.engine.history`, `hexorl.engine.legal`, `hexorl.engine.encoding` | `import_audits/direct_engine_import_audit.txt`, engine boundary tests |
| Sampler private D6 helpers `_hex_transform`, `_py_apply_d6_symmetry`, `_transform_*` | `hexorl.contracts.symmetry` | `import_audits/private_helper_audit.txt`, `test_training_data_pipeline.py` |
| Sampler Python tensor/legal fallback decode path | Rust-backed `encode_tensor_for_history`; missing Rust now fails hard | `Python/src/hexorl/buffer/sampler.py`, focused pytest |
| Graph private D6/history/legal ownership | `MoveHistory`, `turn_state_after`, `contracts.symmetry`, `LegalTableProvider` | `Python/src/hexorl/graph/batch.py`, graph contract tests |
| RGSC private compact-history codec | `MoveHistory` and shared encoder | `Python/src/hexorl/selfplay/rgsc.py`, RGSC tests |
| Dashboard replay private compact-history codec and Python legal fallback | `MoveHistory`, `engine.history`, `engine.legal` | `Python/src/hexorl/dashboard/replay.py`, dashboard replay tests |
| Runtime legal-byte decoders using `np.frombuffer(legal_bytes)` | `decode_legal_bytes` | `import_audits/protocol_decode_audit.txt` |
| Buffer target private compact-history parsing | `MoveHistory` and `legal_rows_from_history` | `Python/src/hexorl/buffer/targets.py`, training pipeline tests |

## Remaining Allowed Paths

- `Python/src/hexorl/selfplay/records.py` still owns replay-record storage byte layout. This is not the Rust FFI legal/history protocol and is owned by later replay phases.
- Tactical Python scanner remains an explicit diagnostic implementation behind `allow_fixture_scan=True`; production tactical history/game paths require Rust `tactical_oracle`.
- Dashboard fixture generation may use random legal choices for fixture variety, but legal rows are sourced from the Rust game object.
