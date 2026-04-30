# Phase 08 Deletion Manifest

Deleted:
- `Python/src/hexorl/tuning/asha.py`
- `Python/src/hexorl/tuning/bohb.py`
- `Python/src/hexorl/tuning/pb2.py`
- `Python/tests/test_phase3_autotune.py`
- `scripts/run_phase3_48h_autotune.py`
- `scripts/launch_phase3_48h_autotune.sh`
- `scripts/launch_wsl_phase3_services.ps1`

Cleaned runtime paths:
- Eval dense-only direct model policy path removed from `arena.py`.
- Dashboard private candidate/pair/graph/D6 reconstruction removed from app routes and model-cache route helpers.
- Tuning package exports only typed Phase 08 recipe/runtime/scheduler/reporting interfaces.
