# Phase 06 Deletion Manifest

Deleted from `SelfPlayWorker`:

- Game loop and terminal-state handling.
- Replay record assembly.
- Direct output queue write loop.
- Candidate construction.
- Graph batch construction.
- Pair strategy configuration and summaries.
- MCTS root/leaf evaluation calls.
- Direct `process_game_record` use.
- Inference failure uniform-policy fallback path.
- Engine availability fallback naming covered by the banned `HAS_ENGINE` audit.

Moved or centralized:

- Game execution: `selfplay/game_runner.py`
- Replay validation/write boundary: `selfplay/record_writer.py`
- Telemetry schemas and debug bundles: `selfplay/telemetry.py`
- Pair strategy checks: `search/pair_strategy.py` and runner dependency injection

No quarantined runtime compatibility path remains for Phase 06-owned worker logic.
