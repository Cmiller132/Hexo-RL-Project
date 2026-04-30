# Evidence Reconciliation

| Row | Implementation proof | Test proof | Deletion/import proof | Telemetry/perf proof |
|---|---|---|---|---|
| V2-080 | `eval/players.py`, `eval/arena.py` | `test_phase08_eval_policy_provider.py` | eval audit | eval trace sample |
| V2-081 | `dashboard/contract_inspector.py`, route cutover | `test_phase08_contract_inspector.py` | dashboard audit | dashboard parity report |
| V2-082 | inspector facts payloads | dashboard required-view tests | dashboard audit | debug samples |
| V2-083 | `tuning/recipes.py`, `family_spaces.py` | typed autotune tests | deletion manifest | dry-run report |
| V2-084 | `scheduler.py`, `validation.py`, `runtime_sweep.py`, `reporting.py` | typed autotune tests | tuning audit | watchdog report |
| V2-085 | debug-bundle/mismatch inspectors and poor-learning report | dashboard/tuning tests | dashboard/tuning audits | debug samples |
| V2-086 | `RuntimeSpec`, scoring, utilization report | runtime sweep tests | old runtime sizing deletion | runtime utilization JSON |
