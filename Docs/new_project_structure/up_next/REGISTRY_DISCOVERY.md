# Registry And Discovery

## Purpose

The runner and training tools need to find model players, training adapters,
checkpoint loaders, and evaluation hooks without hardcoding every architecture.
The current project already has registries; the split should keep discovery
explicit and lightweight.

## Proposal

Use small typed registries for:

- player factories;
- model package descriptors;
- training adapter factories;
- checkpoint loaders;
- evaluation and diagnostic hooks.

Each registration should include a stable name, version or capability metadata,
required dependencies, and the callable needed by the runner or training tool.

## Simplification Guardrails

Avoid plugin auto-magic at first. Local Python registration or simple entry
points are enough if errors are clear when a requested model is unavailable.

The registry should not become a configuration system. It should answer "what
can I instantiate?" while run configs answer "what should I run?"
