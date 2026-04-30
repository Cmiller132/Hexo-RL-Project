# Evidence Reconciliation

V2-070
- Implementation: `replay/codec.py`, `selfplay/record_writer.py`
- Tests: `test_replay_codec_roundtrip_preserves_semantic_identities`, `test_replay_record_rejects_stale_legal_hash_and_bad_target`, self-play record writer tests.

V2-071
- Implementation: `replay/storage.py`, `replay/sampler.py`, `replay/projector.py`, `train/adapters.py`
- Tests: `test_storage_sampler_reads_only_new_replay_records`, `test_sample_to_loss_uses_projected_replay_batch`, production smoke.

V2-072
- Implementation: runtime imports changed in self-play, epoch, train.
- Tests/audit: `test_phase07_runtime_has_no_buffer_imports`; git grep import audit.

V2-073
- Implementation: replay codec validation and projector ownership.
- Tests: roundtrip, corruption, stale legal hash, non-finite targets.

V2-074
- Implementation: `ReplayPositionRecord.record_hash`, `ReplayGameRecord.game_hash`, projector `projection_id`.
- Tests: D6 mutation-safety projector test and sample-to-loss identity path.

V2-075
- Implementation: bounded `ReplayStorage`, `ReplayDataset`, batched `ReplayProjector`.
- Evidence: throughput/memory profile JSON and storage stats in tests.
