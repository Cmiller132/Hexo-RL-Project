# Testing And Contract Tests

## Purpose

The package split should reduce test confusion, not multiply it. Each package
should own its internal behavior, while cross-package tests should prove that
the contracts compose correctly.

## Proposal

Keep ownership clear:

- `hexo-engine`: rules, legality, state identity, tactical payloads, FFI, and
  malformed input rejection;
- `hexo-runner`: player contract, game loop, budget handling, replay/event
  emission, and controlled failures;
- `hexo-utils`: schemas, batching helpers, queues, replay utilities,
  resource profiles, and adapter framework behavior;
- `hexo-model-*`: encoding, targets, losses, inference adapters, checkpoints,
  search behavior, and model diagnostics.

Add a small number of contract tests around real package boundaries: runner to
engine, runner to player, model adapter to replay utilities, and evaluation to
scorecard output.

## Simplification Guardrails

Do not duplicate every engine rule test through every model. Models should test
their use of engine contracts, not revalidate the whole game.

Prefer small fixtures with clear failure messages over broad end-to-end tests
that hide which contract broke.
