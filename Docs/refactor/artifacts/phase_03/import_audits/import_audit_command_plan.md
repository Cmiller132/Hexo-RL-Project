# Phase 03 Import Audit Command Plan

Use `git grep` because local `rg.exe` access has failed with access denied.

```powershell
git grep -n "hexorl\\.model\\|from hexorl import model\\|Python/src/hexorl/model" -- Python/src Python/tests
git grep -n "architecture\\.startswith\\|architecture ==\\|isinstance(.*GlobalHexGraphNet\\|build_model_from_config" -- Python/src/hexorl
git grep -n "_orig_mod\\|strip.*prefix\\|state_dict.*cleanup\\|strict=False" -- Python/src/hexorl
git grep -n "pair_prior_mix\\|pair_head_present" -- Python/src/hexorl/models Python/src/hexorl/train
```

Closing expectation: runtime hits are absent. Any allowed test-only or offline-migration hits must be documented inline in the final audit artifacts.
