# Phase 06 Adversarial Review

Findings and resolution:

- Finding: Worker source could still hide architecture gates or search wiring after the move.
  Resolution: Added worker source tests and import audits proving no architecture, candidate, pair, graph, replay, MCTS, direct submit, or old processing terms remain.

- Finding: Policy providers could silently emit uniform outputs when inference is unavailable.
  Resolution: Removed no-client uniform fallback paths from `search/policy_provider.py`; providers now raise structured `ContractValidationError`.

- Finding: Old replay target processing could remain reachable through self-play.
  Resolution: Removed direct `process_game_record` use from self-play/search runtime; record validation and queue backpressure now live in `selfplay/record_writer.py`.

- Finding: Record writer queue saturation could hang indefinitely.
  Resolution: `QueueSelfPlayRecordWriter` has bounded retry budget, emits `selfplay_backpressure`, and returns structured failure evidence.

- Finding: Debug bundles could be narrative-only.
  Resolution: `SelfPlayDebugBundle` validates required sections and owner subsystem names; mutation guard tests assert owner-specific failure.

No unresolved adversarial findings remain for Phase 06 rows.
