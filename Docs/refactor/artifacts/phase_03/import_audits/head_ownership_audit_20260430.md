# Phase 03 Crop Model Ownership Audit - 2026-04-30

## Scope

This audit covers the corrective model-organization pass for
`Python/src/hexorl/models`. The previous `hexorl.models.network` mega-module
has been removed from runtime source. Ownership is now:

- crop model wrapper: `hexorl.models.crop_network.HexNet`;
- shared crop constants: `hexorl.models.constants`;
- dense crop blocks: `hexorl.models.trunks.dense_cnn`;
- RestNet attention block: `hexorl.models.trunks.restnet`;
- graph-hybrid sparse encoder: `hexorl.models.trunks.graph_hybrid`;
- neural output heads: `hexorl.models.heads.*`;
- value-bin projection utilities: `hexorl.models.heads.value`.

## Ownership Checks

```text
PYTHONPATH=Python/src python3 -m py_compile Python/src/hexorl/models/crop_network.py Python/src/hexorl/models/constants.py Python/src/hexorl/models/heads/*.py Python/src/hexorl/models/trunks/*.py Python/tests/models/test_phase03_model_registry.py
exit=0
```

```text
PYTHONPATH=Python/src ./.venv/bin/python -m pytest -q Python/tests/models/test_phase03_model_registry.py
exit=0
12 passed in 1.26s
```

```text
PYTHONPATH=Python/src ./.venv/bin/python -m pytest -q Python/tests/test_config_and_guardrails.py
exit=0
31 passed in 373.72s (0:06:13)
```

## Import And Deletion Proof

```text
test_crop_components_are_not_owned_by_legacy_network_module
asserts:
- Python/src/hexorl/models/network.py does not exist;
- Python/src/hexorl/models/heads/*.py do not import hexorl.models.network;
- Python/src/hexorl/models/trunks/*.py do not import hexorl.models.network;
- Python/src/hexorl/**/*.py does not import hexorl.models.network.
```

```text
rg -n "from hexorl\.models\.network|import hexorl\.models\.network|hexorl\.models\.network" Python/src scripts tools -S
exit=1
no matches
```

```text
rg -n "from hexorl\.models\.network|import hexorl\.models\.network|from \.\.network|from \.network" Python/src/hexorl/models/heads Python/src/hexorl/models/trunks -S
exit=1
no matches
```

```text
rg -n "^class .*Head\b|^class HexConv2d\b|^class GatedResBlock\b|^class SpatialTransformerBlock\b|^class SparseHexGraphHybrid0Encoder\b" Python/src/hexorl/models -S
exit=0
all matches are in focused heads/trunks modules, not a network mega-module
```

## Regression Coverage

`Python/tests/models/test_phase03_model_registry.py` now asserts:

- exported head classes resolve from focused `hexorl.models.heads.*` modules;
- exported crop trunk classes resolve from focused `hexorl.models.trunks.*`
  modules;
- the legacy `network.py` module is absent;
- focused head/trunk modules and production runtime code do not import
  `hexorl.models.network`.

No skipped, deferred, quarantined, flaky, or manual-only requirement is claimed
complete by this audit.
