# Phase 03 CI Routing Plan

## Tier 0: Local

- Python compile for changed model, train, checkpoint, inference, dashboard, eval, epoch, and tests.
- Focused model registry/spec/checkpoint tests.
- Focused train adapter and pair target validation tests.
- Import/deletion audits for old model package, architecture gates, checkpoint cleanup, and pair consumption bypasses.

## Tier 1: PR Required

- Every registered family builds.
- Every registered family exposes complete descriptor facets.
- Fake-family registration test.
- Trainer no-branch audit.
- Checkpoint manifest round-trip and strict rejection tests.
- One representative one-batch training smoke for dense/crop and global graph families.

## Tier 2: Deep

- One-batch training smoke for every registered family.
- Training debug bundle generation.
- Mutation/corruption tests for adapter projection and output validation.
- Performance timing for adapter projection, device transfer, and train-step throughput.

No Phase 03 close invariant may rely on manual-only, skipped, xfailed, or flaky-only checks.
