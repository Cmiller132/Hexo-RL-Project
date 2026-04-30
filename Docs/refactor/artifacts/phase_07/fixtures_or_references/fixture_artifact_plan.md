# Fixture And Artifact Plan

Golden fixture source
- `Python/src/hexorl/replay/fixtures.py`

Golden fixture contents
- deterministic self-play-shaped `GameRecord`
- canonical `ReplayGameRecord`
- encoded `HXR7` byte payload
- corruption helpers for bad magic, bad schema version, and truncation

Verification use
- codec roundtrip identity preservation
- stale legal-hash rejection
- non-finite target rejection
- sampler/projector sample generation
- sample-to-loss smoke through `TrainAdapter`

Fixed seed
- Fixture generation is deterministic by construction and does not rely on global RNG state.
