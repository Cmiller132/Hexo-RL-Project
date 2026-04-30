# Adversarial Review

Findings checked
- Missing `HXR7` replay magic fails in `replay.codec`.
- Unknown schema version fails in `replay.codec`.
- Truncated payload fails in `replay.codec`.
- Stale legal-table hash fails before storage/projector consumption.
- Non-finite policy targets fail before training.
- Sampler rejects non-`ReplayStorage` inputs.
- Projector preserves source record hash across D6 projection.
- Runtime import audit found no `hexorl.buffer` imports in self-play, replay, train, or epoch.

Resolution
- All adversarial checks are covered by `Python/tests/replay/test_phase07_codec_storage_projector.py` and `Python/tests/replay/test_phase07_import_audit.py`.

Residual risk
- Legacy buffer files remain present for older tests and dashboard/evaluation Phase 08 work, but Phase 07 runtime scopes do not import them.
