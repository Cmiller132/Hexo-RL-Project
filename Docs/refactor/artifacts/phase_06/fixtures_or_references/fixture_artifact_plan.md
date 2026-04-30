# Fixture And Artifact Plan

Deterministic fixtures:

- `Python/tests/selfplay/conftest.py` defines fake policy provider, fake engine adapter, fake graph batch, in-memory telemetry, and in-memory writer.
- Golden runner tests assert legal rows, policy priors, MCTS selected move, and replay payload identity for a one-position game.
- Negative tests corrupt prior lengths and non-finite values before search consumption.

Artifacts:

- `telemetry_samples/phase_06_telemetry_samples.json`
- `telemetry_samples/phase_06_debug_bundle_sample.json`
- `performance/phase_06_selfplay_smoke_profile.json`

No old runtime behavior is used as the oracle.
