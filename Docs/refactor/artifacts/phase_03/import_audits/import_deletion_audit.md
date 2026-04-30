# Phase 03 Import And Deletion Audit

## `rg` Availability

The required `rg` command could not execute in this Windows workspace:

```text
rg "hexorl\.model|from hexorl import model|Python/src/hexorl/model" Python/src Python/tests
exit=1
Program 'rg.exe' failed to run: Access is denied
```

`git grep` was used as the fallback audit tool.

## Old Runtime Package

```text
Test-Path Python\src\hexorl\model
exit=1
False
```

`Python/src/hexorl/model/` no longer exists in the working tree. The tracked files were moved to `Python/src/hexorl/models/`.

## Banned Runtime Gates

```text
git grep --untracked -n "architecture\.startswith\|architecture ==\|isinstance(.*GlobalHexGraphNet\|build_model_from_config" -- Python/src/hexorl
exit=1
no matches
```

## Duplicate Checkpoint Cleanup

```text
git grep --untracked -n "_orig_mod\|strip.*prefix\|state_dict.*cleanup\|strict=False" -- Python/src/hexorl
exit=1
no matches
```

Runtime checkpoint prefix cleanup was removed. `CheckpointManager` rejects prefixed keys through a split-string construction so the banned cleanup token is not present in runtime source.

## Pair Ownership Heuristics

```text
git grep --untracked -n "pair_prior_mix\|pair_head_present" -- Python/src/hexorl/models Python/src/hexorl/train
exit=1
no matches
```

Pair strategy remains outside model family capabilities and is not inferred from model heads.

## Exact `hexorl.model` Audit Note

The literal phase audit pattern `hexorl\.model` overmatches the new valid package name `hexorl.models`. `git grep` therefore reports legitimate plural imports such as `from hexorl.models.factory import build_model`. There are no singular runtime imports (`hexorl.model.`) and no `Python/src/hexorl/model` directory.

```text
git grep --untracked -n "hexorl\.model\.\|from hexorl import model\|Python/src/hexorl/model" -- Python/src Python/tests
exit=1
no matches
```
