# CI Routing Plan

| Check | Tier | Command | Promotion rule |
|---|---|---|---|
| Phase 08 focused tests | local/pr_required | `python -m pytest Python\tests\eval\test_phase08_eval_policy_provider.py Python\tests\dashboard\test_phase08_contract_inspector.py Python\tests\tuning\test_phase08_typed_autotune.py -q` | Required for rows `V2-080` through `V2-086`. |
| Existing dashboard smoke | pr_required | `python -m pytest Python\tests\test_dashboard_foundation.py Python\tests\test_dashboard_replay_debug.py -q` | Required because dashboard routes changed. |
| Import/deletion audits | pr_required | `git grep` commands in `command_transcripts.md` | Must return no banned runtime matches. |
| Runtime utilization artifact | local/deep | generated `performance/runtime_utilization_sweep.json` | Deep CI can replace local proxy with runner-normalized utilization. |
