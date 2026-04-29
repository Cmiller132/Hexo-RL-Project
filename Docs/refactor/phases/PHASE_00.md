# Phase 00 — Program Setup and Baseline Freeze

## Purpose
Establish an immutable baseline before any breaking refactor work. This phase creates the evidence set that all later phases compare against.

## Inputs from V2 Spec
- Worker currently mixes lifecycle, inference, candidates, graph, pair, MCTS, telemetry responsibilities.
- Model construction centralized in `model/network.py` and `model/global_graph.py`.
- Config schema overload and architecture-string coupling.
- Rust/Python rule boundary ambiguity and pair policy performance risks.

## In-Scope Repository Context
- Python hotspots: `selfplay/worker.py`, `model/network.py`, `model/global_graph.py`, `config/schema.py`.
- Existing package families: `inference/`, `selfplay/`, `graph/`, `train/`, `eval/`, `dashboard/`, `tuning/`.
- Rust rule source: `crates/hexgame-core/`, `crates/hexgame-py/`.

## Required Deliverables
1. Baseline behavior report (functional + perf + CI timings).
2. Architecture-string dependency inventory (where behavior is inferred from names).
3. Legacy fallback inventory (Python legal/history/candidate/pair fallback paths).
4. Refactor artifact directory conventions under `Docs/refactor/artifacts/phase_00/`.

## Parallel Subagent Split
- S1 Contracts/Schema: enumerate all shared data shapes currently implicit.
- S2 Engine/Runtime: map Rust vs Python ownership boundaries.
- S3 Models/Search: inventory model family checks and pair-policy coupling points.
- S4 Data/Train/Eval: baseline replay->sampler->trainer path and known invariants.
- S5 Quality/Obs/Docs: establish metrics dashboard and evidence template.

## Mandatory Checks
- `cargo test --workspace`
- `pytest -q Python/tests`
- `python -m hexorl.cli --help`
- Baseline perf commands used by team (selfplay/inference/training smoke benches).

## Exit Criteria
- Baseline reports archived and signed by orchestrator.
- All critical coupling/fallback points are cataloged with owning phase mapping.
