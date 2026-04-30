# Deletion Manifest

Runtime imports removed
- `hexorl.buffer.ring` from `Python/src/hexorl/selfplay/orchestrator.py`
- `hexorl.buffer.ring` from `Python/src/hexorl/epoch/pipeline.py`
- `hexorl.buffer.sampler` from `Python/src/hexorl/epoch/pipeline.py`
- `hexorl.buffer.targets` from `Python/src/hexorl/epoch/pipeline.py`
- `hexorl.buffer.regret_buffer` from `Python/src/hexorl/selfplay/rgsc.py`

Runtime behavior removed
- self-play no longer queues `GameRecord`/`PositionRecord` into a ring buffer
- epoch runtime no longer constructs `RingBuffer`
- epoch runtime no longer constructs old `ReplayDataset`
- train runtime consumes projector-owned `ProjectedReplayBatch`
- magic-less legacy `GameRecord.from_compact_bytes` payloads are rejected

Quarantine note
Old `Python/src/hexorl/buffer/` modules remain not imported by Phase 07 runtime scopes because dashboard/evaluation cleanup is Phase 08. The Phase 07 gate is enforced by runtime import audit.
