# Evaluation Fairness

## Purpose

If models can own custom search, preprocessing, and replay interpretation,
evaluation must make comparisons honest. The current project already has
self-play, benchmarks, profiles, and scorecards; the split should make fairness
metadata explicit.

## Proposal

Evaluation records should capture:

- model package and checkpoint identity;
- opponent pool and scenario/opening set;
- equal wall-clock or equal decision-budget policy;
- hardware and shared resource profile;
- inference precision, batch policy, and worker counts;
- search settings, temperature, resignation, and stochastic policy settings;
- replay/schema versions and scorecard version.

Custom model behavior is allowed, but scorecards should make it visible.

## Simplification Guardrails

Avoid one universal fairness rule. Some comparisons need equal wall-clock;
others need equal simulations or fixed inference batches. The important part is
that the chosen rule is recorded and enforced consistently.

Keep evaluation fixtures small and repeatable before expanding to large arenas.
