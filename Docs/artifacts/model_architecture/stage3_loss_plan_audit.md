# Stage 3 Loss Plan And Training Boundary Audit

Captured on 2026-05-06 in `D:\Hexo\Hexo-RL-Project`.

## Raw Loss Switch Audit

Command:

```powershell
Select-String -Path Python\src\hexorl\train\losses.py,Python\src\hexorl\train\trainer.py `
  -Pattern 'elif head_name','for head_name, pred','if head_name ==','head_name in',`
           'targets\.get\("policy_target", targets\.get','targets\.get\("opp_policy", targets\.get',`
           'build_loss_plan\(tuple\(predictions\.keys\(\)\)'
```

Result:

```text
<no matches>
```

The broad head-name switch in `compute_losses` is gone. Loss routing now goes
through `LossPlan` entries built from resolved output contracts.

## Training Boundary Audit

Command:

```powershell
Get-ChildItem -Path Python\src\hexorl -Recurse -File -Include *.py |
  Select-String -Pattern 'compute_losses\(','build_loss_plan\(','prepare_dense_training_batch','prepare_global_graph_training_batch'
```

Result:

```text
Python/src/hexorl/replay/training_batch.py:22: def prepare_dense_training_batch(
Python/src/hexorl/replay/training_batch.py:55: def prepare_global_graph_training_batch(
Python/src/hexorl/train/losses.py:352: def compute_losses(
Python/src/hexorl/train/loss_plan.py:83: def build_loss_plan(
Python/src/hexorl/train/trainer.py:22: prepare_dense_training_batch,
Python/src/hexorl/train/trainer.py:23: prepare_global_graph_training_batch,
Python/src/hexorl/train/trainer.py:91: self._loss_plan = build_loss_plan(self._resolved_spec, self._loss_weights)
Python/src/hexorl/train/trainer.py:227: prepared = prepare_dense_training_batch(
Python/src/hexorl/train/trainer.py:253: total_loss, per_head = compute_losses(
Python/src/hexorl/train/trainer.py:421: prepared = prepare_global_graph_training_batch(
Python/src/hexorl/train/trainer.py:440: total_loss, per_head = compute_losses(
```

Trainer constructs one architecture-resolved loss plan and both dense and
global graph paths call `compute_losses` only with adapter-prepared row tables.
There is no runtime fallback that builds a loss plan from observed prediction
names.

## Explicit Loss Plan Boundary Audit

Command:

```powershell
Select-String -Path Python\src\hexorl\train\losses.py `
  -Pattern 'compute_losses requires an explicit loss_plan','build_loss_plan\(tuple\(predictions\.keys\(\)\)'
```

Result:

```text
Python/src/hexorl/train/losses.py:375: raise LossContractError("compute_losses requires an explicit loss_plan")
```

Direct loss calls must provide a validated `LossPlan`; tests now cover the
missing-plan failure explicitly.

## Resolved Output Coverage Audit

Command:

```powershell
$env:PYTHONPATH='Python/src'; @'
# Resolve every registered architecture/static-output combination and verify
# each trainable resolved output has a loss-plan entry.
'@ | python -
```

Result:

```text
all valid resolved trainable outputs have loss-plan entries
```

The audit covers the registered architecture IDs and all static output sets
accepted by the architecture resolver. Invalid architecture/output
combinations are rejected by resolver contracts before loss-plan construction.

## Lookahead And Pair-Second Audit

Command:

```powershell
Select-String -Path Python\src\hexorl\buffer\ring.py,Python\src\hexorl\buffer\sampler.py,Python\src\hexorl\train\*.py `
  -Pattern 'lookahead\[idx, k:\] = self\._values','lookahead_arrays\[.*=\s*values','pair_second.*target_mass','pair_second.*sum','policy_pair_second.*pair_policy_target'
```

Result:

```text
<no matches>
```

Compact replay storage and replay sampling no longer fall back from missing
lookahead horizons to value targets, and pair-second routing is no longer
inferred from target mass or the joint pair target.

## Runtime Legacy Import Audit

Command:

```powershell
Get-ChildItem -Path Python\src\hexorl -Recurse -File -Include *.py |
  Select-String -Pattern 'from hexorl\.model[\s\.]','import hexorl\.model[\s\.]','\bGlobalHexGraphNet\b','\bHexNet\b' |
  Where-Object { $_.Path -notmatch '\\model\\' -and $_.Path -notmatch '\\models\\recipes\\' }
```

Result:

```text
<no matches>
```

Runtime consumers no longer import the legacy model modules directly outside
the model implementation and recipe compatibility scope.

## Contract Failure Coverage

`Python/tests/test_model_architecture_stage3.py` covers:

- Missing required target.
- Missing required mask.
- Missing required weight.
- Missing required phase.
- Missing explicit loss plan.
- Auxiliary target namespace overwrite in the training adapter.
- Duplicate active row-table rows.
- Zero-mass active policy target.
- Pair-second positive targets outside known-first phase.
- Exact lookahead target requirement.
- Compact replay rows preserve the actual number of lookahead targets and make
  the sampler raise when a configured horizon is absent after slot overwrite.
- Global graph rejection of dense policy target fields.
- Global graph adapter phase metadata and graph policy namespace generation.
- Crop pair replay rows with empty pair target mass are masked out of pair-head
  training instead of reaching the loss plan with active weight.

## Notes

`Select-String` was used for this audit because `rg` was not executable in the
current Windows shell session. The searched source set is limited to the
training/runtime files relevant to Stage 3 loss routing.
