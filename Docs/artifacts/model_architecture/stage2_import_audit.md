# Stage 2 Import And Authority Audit

Captured on 2026-05-06 in `D:\Hexo\Hexo-RL-Project`.

## Direct Legacy Import Audit

Command:

```powershell
Get-ChildItem -Path Python\src\hexorl -Recurse -File -Include *.py |
  Where-Object { $_.FullName -notmatch '\\model\\' -and $_.FullName -notmatch '\\models\\recipes\\' } |
  Select-String -Pattern 'from hexorl\.model[\s\.]','import hexorl\.model[\s\.]','\bGlobalHexGraphNet\b','\bHexNet\b'
```

Result:

```text
<no matches>
```

## Approved Legacy Recipe Imports

Command:

```powershell
Get-ChildItem -Path Python\src\hexorl\models\recipes -Recurse -File -Include *.py |
  Select-String -Pattern 'from hexorl\.model[\s\.]','import hexorl\.model[\s\.]'
```

Result:

```text
<no matches>
```

The Stage 2 legacy recipe quarantine is closed by Stage 4. Family
implementations now live under `Python/src/hexorl/models/families/`.

## Duplicated Global Authority Audit

Command:

```powershell
Get-ChildItem -Path Python\src\hexorl,scripts -Recurse -File -Include *.py |
  Select-String -Pattern 'startswith\("global_','startswith\(''global_','GLOBAL_GRAPH_ARCHITECTURES'
```

Result:

```text
<no matches>
```

## Runtime And Orchestration Legacy Import Audit

Command:

```powershell
Get-ChildItem -Path Python\src\hexorl,scripts -Recurse -File -Include *.py |
  Where-Object { $_.FullName -notmatch '\\model\\' -and $_.FullName -notmatch '\\models\\recipes\\' } |
  Select-String -Pattern 'from hexorl\.model[\s\.]','import hexorl\.model[\s\.]','\bGlobalHexGraphNet\b','\bHexNet\b'
```

Result:

```text
<no matches>
```

## Notes

`Python/src/hexorl/model/` no longer exists. Its implementation contents moved
under `Python/src/hexorl/models/families/`, and the final Stage 4 runtime audit
found no `hexorl.model` imports.

`scripts/run_phase3_48h_autotune.py` retains an explicit
`GLOBAL_GRAPH_SCOUT_FAMILIES` experiment scope for the four pre-champion
families, but validates that scope against `hexorl.models.registry` at import
time and delegates global-graph membership checks to the registry.
