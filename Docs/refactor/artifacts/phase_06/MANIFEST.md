# Phase 06 Artifact Manifest

Phase: 06 - GameRunner + SelfPlayWorker Cleanup

Rows closed: V2-060, V2-061, V2-062, V2-063, V2-064, V2-065

Implementation SHA at capture time: recorded by `git rev-parse HEAD` in command transcripts.

Primary runtime changes:

- `Python/src/hexorl/selfplay/game_runner.py`
- `Python/src/hexorl/selfplay/worker.py`
- `Python/src/hexorl/selfplay/record_writer.py`
- `Python/src/hexorl/selfplay/telemetry.py`
- `Python/src/hexorl/selfplay/orchestrator.py`
- `Python/src/hexorl/search/policy_provider.py`

Required evidence files:

- `checks/phase_06_acceptance_checklist.md`
- `checks/interface_freeze_notes.md`
- `fixtures_or_references/fixture_artifact_plan.md`
- `commands/ci_routing_plan.md`
- `commands/command_transcripts.md`
- `import_audits/phase_06_import_audit.md`
- `deletion_manifest/deletion_manifest.md`
- `telemetry_samples/phase_06_telemetry_samples.json`
- `telemetry_samples/phase_06_debug_bundle_sample.json`
- `performance/phase_06_selfplay_smoke_profile.json`
- `contract_examples/game_runner_contract_examples.md`
- `adversarial_review.md`
- `agent_completion_packet.md`
- `evidence_reconciliation.md`
- `exit_gate_report.md`

No intentional runtime compatibility shim was added.
