# Phase 07 Artifact Manifest

Git base before Phase 07 implementation: `88227077ab124871e8f4954482ec356e243d1ec4`.

Goal
Cut runtime replay over to canonical `hexorl.replay` records only.

Closed rows
- V2-070: self-play writer emits `ReplayGameRecord` with schema version, Rust history/legal source markers, history hash, legal-table hash, and reconstructed legal-table hash.
- V2-071: runtime sampler reads `ReplayStorage` only and produces training input through `replay/projector.py`.
- V2-072: self-play, replay, train, and epoch runtime have no `hexorl.buffer` imports.
- V2-073: codec roundtrip, corruption handling, projection, Rust legal replay, and data-quality tests pass.
- V2-074: trace-to-record-to-projector identities are preserved and transient MCTS root/batch tokens are rejected as replay semantics.
- V2-075: storage/sampler/projector expose bounded capacity, prefetch, throughput, queue/backpressure, and memory evidence.

Runtime path
`selfplay/record_writer.py -> replay/codec.py -> replay/storage.py -> replay/sampler.py -> replay/projector.py -> train/adapters.py`.

Artifact index
- `checks/phase_07_acceptance_checklist.md`
- `checks/interface_freeze_notes.md`
- `fixtures_or_references/fixture_artifact_plan.md`
- `commands/ci_routing_plan.md`
- `commands/command_transcripts.md`
- `import_audits/phase_07_import_audit.md`
- `deletion_manifest/deletion_manifest.md`
- `telemetry_samples/phase_07_debug_bundle_sample.json`
- `performance/phase_07_replay_throughput_profile.json`
- `contract_examples/replay_contract_examples.md`
- `adversarial_review.md`
- `agent_completion_packet.md`
- `evidence_reconciliation.md`
- `exit_gate_report.md`
