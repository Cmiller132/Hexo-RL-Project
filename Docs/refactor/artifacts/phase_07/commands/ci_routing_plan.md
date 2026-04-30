# CI Routing Plan

Local
- `python -m pytest Python\tests\replay\test_phase07_codec_storage_projector.py Python\tests\replay\test_phase07_import_audit.py -q`
- `python -m pytest Python\tests\selfplay Python\tests\replay Python\tests\search\test_pair_strategy_selfplay_integration.py Python\tests\test_production_smoke.py -q`
- `python -m compileall Python\src\hexorl\replay Python\src\hexorl\selfplay Python\src\hexorl\epoch Python\src\hexorl\train`

PR required
- Replay codec/storage/projector tests.
- Runtime banned import audit.
- Self-play record writer tests.
- Production sample-to-loss smoke.

Deep/scheduled
- Full training data corruption suite.
- Longer replay throughput comparison on target host profile.
- GPU train-starvation profile where CUDA is available.

Promotion rule
Phase 07 closes only when local and PR-required checks pass and performance evidence is attached.
