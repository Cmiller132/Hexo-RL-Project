# Phase 04 Agent Completion Packet

Closed V2 rows:

- V2-040
- V2-041
- V2-042
- V2-043
- V2-044
- V2-045
- V2-046

Runtime consumers changed:

- `Python/src/hexorl/inference/client/`
- `Python/src/hexorl/inference/server/`
- `Python/src/hexorl/inference/shm_queue.py`
- `Python/src/hexorl/selfplay/worker.py`

Files changed:

- Added protocol, client transport, server batching, telemetry, and adapter modules under `Python/src/hexorl/inference/`.
- Added required Phase 04 inference tests under `Python/tests/inference/`.
- Updated `Python/tests/test_inference_server.py`.
- Added Phase 04 artifacts under `Docs/refactor/artifacts/phase_04/`.

Legacy paths deleted or quarantined:

- Mode-specific submit methods removed from `InferenceClient`.
- `req_mode` removed from inference shared-memory runtime.
- Architecture-string inference dispatch not present.
- Server non-finite sanitization removed.

Tests and commands run with exit status:

- `python -m pytest Python\tests\inference -q` -> 0
- `python -m pytest Python\tests\test_inference_server.py -q` -> 0
- `python -m compileall Python\src\hexorl` -> 0
- `rg --version` -> 1, access denied
- `git grep` deletion audits -> no banned inference runtime matches

Artifacts produced:

- See `MANIFEST.md`.

Performance/utilization evidence for hot paths:

- Existing inference integration, adaptive two-client batching, graph forward, sparse/pair forward, and MCTS round-trip pass after request-kind transport migration.

Contract examples/docs added:

- `phase_04_protocol_manifest_examples.md`
- `phase_04_handshake_matrix.md`

Known blockers:

- None for Phase 04.

No skipped/deferred/manual-only requirement is being claimed complete.
