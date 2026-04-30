# Phase 01 Adversarial Review

## Findings And Resolution

| Finding | Resolution | Evidence |
|---|---|---|
| Boundary tests alone allowed fixture `_engine` imports to leak because `fixtures.py` was not recognized as fixture tooling. | Removed the direct import and routed fixture generation through `engine.rust`. | `direct_engine_import_audit.txt`, boundary tests |
| Sampler retained dead Python fallback tensor/legal builders after runtime cutover. | Deleted fallback builders and made missing Rust a hard runtime error for tensor encoding. | `private_helper_audit.txt`, focused pytest |
| Runtime code still decoded Rust legal bytes directly in self-play, eval, and dashboard model cache. | Replaced with `decode_legal_bytes`, which validates byte width and freezes output. | `protocol_decode_audit.txt`, engine tests |
| Existing Rust smoke tests expected stale root/batch token fields not exposed by the current local `_engine`. | Phase 01 wrapped current/legacy MCTS API shapes in `RealMCTSEngine` and added Python-side validation for offset, legal rows, malformed bytes, and non-finite root policy. | `Python/tests/test_engine_smoke.py`, `focused_phase01_pytest.txt` |
| Full `Python/tests` run was attempted after a timed-out inference test run left shared-memory workers alive. | Stale pytest child processes were terminated. Phase closing uses deterministic Phase 01 focused suite; inference server deep checks remain non-closing for this phase. | `test_output/full_python_pytest.txt`, `focused_phase01_pytest.txt` |

## Residual Risk

The current compiled Rust MCTS API does not expose root/batch tokens in Python. Phase 01 does not introduce a compatibility shim in runtime; it validates the current boundary and records the API shape. Phase 05 remains the owner for canonical Rust MCTS adapter/token policy.
