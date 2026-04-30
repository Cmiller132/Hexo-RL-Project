# Interface Freeze Notes

Frozen public runtime interfaces
- `ReplayPositionRecord`
- `ReplayGameRecord`
- `encode_replay_game`
- `decode_replay_game`
- `replay_game_from_selfplay`
- `ReplayStorage.append_game`
- `ReplayStorage.sample_records`
- `ReplayDataset`
- `ReplayProjector.project`
- `ProjectedReplayBatch`

Frozen boundaries
- Self-play writer accepts transient `GameRecord` only as source input and writes canonical `ReplayGameRecord`.
- Runtime readers reject unknown replay schema versions and missing `HXR7` magic.
- Projector is the only runtime sample-to-train projection path.
- Train adapter accepts `ProjectedReplayBatch` with `source == "replay/projector.py"`.

Out of scope
- Dashboard/evaluation inspection cleanup remains Phase 08.
