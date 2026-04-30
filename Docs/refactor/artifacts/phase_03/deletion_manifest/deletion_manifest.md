# Phase 03 Deletion Manifest

Deleted or removed from runtime:

- `Python/src/hexorl/model/` runtime package path.
- `hexorl.models.network.build_model_from_config`.
- `hexorl.models.network.from_config`.
- `hexorl.models.network.load_model_state`.
- `hexorl.models.network.strip_compiled_prefix`.
- Trainer `isinstance(GlobalHexGraphNet)` training branch.
- Inference server `architecture.startswith("global_")` mode switch.
- Epoch/replay graph-policy `architecture.startswith("global_")` gates.
- Runtime `_orig_mod` state-dict cleanup.

Retained implementation cores:

- `HexNet` remains the dense/crop model core behind registry descriptors.
- `GlobalHexGraphNet` remains the global graph model core, but its old multi-architecture constructor switch was replaced with explicit `family_kind` profiles consumed through the registry.

No singular `Python/src/hexorl/model/` compatibility shim or facade remains.
