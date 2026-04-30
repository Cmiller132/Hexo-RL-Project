# CI Routing Plan

Local:

- `python -m pytest Python\tests\selfplay Python\tests\search\test_pair_strategy_selfplay_integration.py -q`
- `python -m pytest Python\tests\inference Python\tests\replay -q`
- `python -m compileall Python\src\hexorl`
- Phase 06 import audits in `import_audits/phase_06_import_audit.md`

PR required:

- Focused self-play, search, inference, replay tests listed above.
- Worker lifecycle-only audit.
- Banned fallback/import audit.

Deep:

- Full self-play shaped smoke with real inference server and Rust extension.
- Longer queue/backpressure sweep under HostProfile budgets.

Scheduled:

- Self-play throughput comparison against stable runners.
- Long no-progress diagnostic run.

Final:

- Phase 09 final import graph and performance comparison.
