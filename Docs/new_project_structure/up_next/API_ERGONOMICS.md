# API Ergonomics

## Purpose

The new structure should make the correct path easy. If adding a model requires
too much ceremony, model authors will bypass the boundaries and recreate hidden
coupling.

## Proposal

A new model package should have a short, predictable path:

1. Define the model architecture and config.
2. Define state-to-input construction from engine/runner payloads.
3. Implement inference decoding into the player contract.
4. Implement training adapters for examples, batches, losses, and checkpoints.
5. Register player and training factories.
6. Run shared smoke tests, self-play, training, and evaluation tools.

The public interfaces should favor simple method contracts and plain data
objects. Advanced models can add custom search or diagnostics behind the same
runner-facing player boundary.

## Simplification Guardrails

Keep the first successful model path small. Avoid deep inheritance trees,
mandatory plugin frameworks, or required dashboards.

A model that only needs policy/value inference should not pay the complexity
cost of V1-style search, graph batching, or custom replay sidecars.
