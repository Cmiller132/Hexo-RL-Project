# Phase 08 Contract Examples

```python
from hexorl.dashboard.contract_inspector import ContractInspector

payload = ContractInspector().inspect("model-input", history=history_bytes)
assert payload["dashboard_training_parity"]
assert payload["facts"]["legal_table_hash"]
```

```python
from hexorl.tuning import family_space, default_runtime_spec, HostProfile, dry_run_validate_recipe

recipe = family_space("global_xattn").default_recipe(seed=17)
runtime = default_runtime_spec(HostProfile.local())
assert all(row["ok"] for row in dry_run_validate_recipe(recipe, runtime, HostProfile.local()))
```
