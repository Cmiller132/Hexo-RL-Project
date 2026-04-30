# Phase 07 Import Audit

Command
```text
git grep -n "hexorl\.buffer\|from hexorl.buffer\|import hexorl.buffer" -- Python/src/hexorl/selfplay Python/src/hexorl/replay Python/src/hexorl/train Python/src/hexorl/epoch
```

Exit code: `1`

Result: no matches.

Additional audit
```text
Select-String -Path Python\src\hexorl\selfplay\records.py -Pattern 'magic-less|legacy records|hexorl.buffer|root_generation|batch_generation'
```

Result
```text
Python\src\hexorl\selfplay\records.py:315: raise ValueError("Legacy magic-less GameRecord payloads are not accepted by runtime")
```
