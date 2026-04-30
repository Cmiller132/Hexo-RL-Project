# Phase 08 Import Audit

- Eval audit: no direct `hexorl.models.network` imports, architecture equality dispatch, or architecture prefix dispatch in `eval/`.
- Dashboard audit: no `CandidateContractBuilder`, `PairActionTableBuilder`, `build_graph_batch_from_history`, or D6 transform imports outside `dashboard/contract_inspector.py`.
- Tuning audit: no runtime imports of deleted `hexorl.tuning.asha`, `bohb`, `pb2`, `ASHARungTable`, `BOHBSampler`, `PB2Scheduler`, or `run_phase3_48h_autotune` under `Python`, `scripts`, or `.github`.
- Deprecated runtime sizing script path was deleted rather than kept as a runtime entrypoint.
