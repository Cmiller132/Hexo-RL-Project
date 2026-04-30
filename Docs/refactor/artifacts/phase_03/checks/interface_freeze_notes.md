# Phase 03 Interface Freeze Notes

- Created: `2026-04-30T01:01:35-04:00`
- Git SHA: `5638e8bc6b20b2dc27821602d4fa1f5adac9b4f8`
- Branch: `codex/phase-03-model-registry-specs`

## Frozen Runtime Package

The runtime package is `Python/src/hexorl/models/`. `Python/src/hexorl/model/` is not a runtime package after Phase 03.

Required public modules:

- `models/capabilities.py`
- `models/specs.py`
- `models/registry.py`
- `models/factory.py`
- `models/checkpoint.py`
- `models/families/*`

## Model Spec Freeze

`ModelSpec` objects are discriminated by `kind`:

- `dense_cnn`
- `restnet`
- `graph_hybrid`
- `global_xattn`
- `global_line_window`
- `global_relation_graph`

Deprecated architecture names may be parsed only by registry/spec migration helpers, not by runtime model, trainer, inference, search, dashboard, or self-play branches.

## Descriptor Facet Freeze

Every `ModelFamilyDescriptor` must provide:

- model builder
- train adapter factory
- inference adapter manifest factory
- policy provider factory
- loss plan provider
- default recipe provider
- tune-space provider
- checkpoint manifest provider

The registry validates descriptor completeness at registration time.

## Train Adapter Freeze

`TrainAdapter` owns:

- raw replay batch projection into model inputs
- target validation
- output validation
- loss input assembly
- finite/mask checks
- pair target phase validation
- training debug bundle

The trainer asks the registry for the adapter and loss plan. It must not branch on model class, architecture name, or head/output-key heuristics.

## Checkpoint Freeze

`CheckpointManager` owns:

- strict save
- strict load
- inspect without loading weights
- manifest validation
- family/spec/protocol/contract compatibility checks

Runtime checkpoint loading must reject missing/stale/unknown manifests, incompatible model family/spec, protocol mismatch, and prefix-cleanup-dependent state dicts.

## Import Freeze

Forbidden in runtime after implementation:

- `hexorl.model`
- `Python/src/hexorl/model`
- `build_model_from_config`
- `architecture.startswith`
- `isinstance(...GlobalHexGraphNet...)`
- `_orig_mod` stripping outside approved offline migration tooling
- `strict=False` model state loading in runtime
