from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "Python" / "src"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from hexorl.config import Config  # noqa: E402
from hexorl.runtime import detect_host  # noqa: E402
from hexorl.selfplay.worker import SelfPlayWorker  # noqa: E402


REQUIRED_TIMING_SPANS = (
    "history_parse_ms",
    "engine_replay_ms",
    "legal_table_ms",
    "tactical_oracle_ms",
    "candidate_build_ms",
    "pair_table_build_ms",
    "graph_token_build_ms",
    "graph_relation_build_ms",
    "graph_tensorize_ms",
    "ipc_pack_ms",
    "ipc_wait_ms",
    "queue_wait_ms",
    "collate_ms",
    "model_forward_ms",
    "scatter_ms",
    "decode_ms",
    "pair_chunk_count",
    "pair_chunk_forward_ms",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Phase 00 baseline artifacts.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "Docs" / "refactor" / "artifacts" / "phase_00",
    )
    parser.add_argument(
        "--watchdog-smoke",
        action="store_true",
        help="Run only the controlled no-progress smoke and exit with code 2.",
    )
    parser.add_argument("--watchdog-threshold-s", type=float, default=0.01)
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    for name in ("config_hashes", "logs", "traces", "watchdog", "performance", "baseline"):
        (artifact_root / name).mkdir(parents=True, exist_ok=True)

    git_sha = _git("rev-parse", "HEAD")
    created_at = _utc_timestamp()
    config_hashes = _config_hashes(REPO_ROOT / "Configs")
    host = detect_host()

    if args.watchdog_smoke:
        event = _controlled_watchdog_event(
            git_sha=git_sha,
            threshold_s=float(args.watchdog_threshold_s),
        )
        _write_json(artifact_root / "watchdog" / "no_progress_smoke_event.json", event)
        print(json.dumps(event, sort_keys=True))
        return 2

    cfg = Config.model_validate(
        {
            "model": {
                "architecture": "global_xattn_0",
                "channels": 16,
                "attention_heads": 4,
                "graph_layers": 1,
                "heads": ["value", "policy_place", "policy_pair_first", "policy_pair_joint"],
                "pair_prior_mix": 0.75,
            },
            "inference": {"fp16": False},
            "selfplay": {"mcts_simulations": 1, "max_game_moves": 1},
        }
    )
    t0 = time.perf_counter()
    worker = SelfPlayWorker(0, cfg, record_queue=None, num_workers=1, max_batch_size=1)
    pair_guard_init_ms = (time.perf_counter() - t0) * 1000.0
    pair_summary = worker.pair_strategy_summary(pair_rows_possible=128, pair_rows_scored=0)

    common = {
        "git_sha": git_sha,
        "created_at": created_at,
        "sample_source": "scripts/phase00_capture_baseline.py",
    }
    events = [
        {
            **common,
            "event": "selfplay_worker_heartbeat",
            "worker_id": 0,
            "game_id": 0,
            "move_index": 0,
            "phase": "phase00_sample",
            "positions_completed_since_last_heartbeat": 0,
            "last_successful_inference_request": None,
            "last_engine_operation": "worker_init",
            "pair_strategy": pair_summary["pair_strategy"],
            "pair_rows_scored": 0,
            "suggested_next_action": "inspect inference queue if no request follows heartbeat",
        },
        {
            **common,
            "event": "selfplay_phase_transition",
            "worker_id": 0,
            "from_phase": "worker_init",
            "to_phase": "root_inference",
            "elapsed_ms": round(pair_guard_init_ms, 3),
        },
        {
            **common,
            "event": "policy_eval_timing",
            "model_family": cfg.model.architecture,
            "request_kind": "global_graph",
            "batch_size": 1,
            "queue_wait_ms": 0.0,
            "model_forward_ms": 0.0,
            "decode_ms": 0.0,
            "warning": "sample does not execute model forward",
        },
        {
            **common,
            "event": "pair_strategy_summary",
            **pair_summary,
        },
        {
            **common,
            "event": "graph_request_summary",
            "model_family": cfg.model.architecture,
            "graph_token_count": 0,
            "graph_relation_count": 0,
            "legal_count": 0,
            "pair_rows_total": 128,
            "pair_rows_scored": 0,
            "pair_strategy": pair_summary["pair_strategy"],
        },
        {
            **common,
            "event": "autotune_recipe_validation",
            "recipe_id": "phase00-global-xattn-default",
            "valid": True,
            "pair_strategy": cfg.model.pair_strategy,
            "validation_notes": ["pair scoring disabled by default"],
        },
        {
            **common,
            "event": "autotune_trial_lifecycle",
            "trial_id": "phase00-sample",
            "stage": "created",
            "status": "diagnostic_sample",
        },
        {
            **common,
            "event": "autotune_scheduler_decision",
            "trial_id": "phase00-sample",
            "decision": "hold",
            "reason": "baseline artifact sample only",
        },
        {
            **common,
            "event": "runtime_sweep_no_progress",
            "trial_id": "phase00-controlled-stall",
            "elapsed_s": 0.01,
            "last_successful_phase": "runtime_validation",
            "last_inference_request": None,
            "last_engine_operation": "not_started",
            "progress_counters": {"games": 0, "positions": 0, "train_batches": 0},
            "pair_strategy": cfg.model.pair_strategy,
            "pair_rows_scored": 0,
            "suggested_next_action": "inspect runtime scheduler and inference startup",
            "outcome": "failed_predictably",
        },
        {
            **common,
            "event": "inference_protocol_mismatch",
            "protocol": "phase00_sample",
            "expected_version": 1,
            "observed_version": 0,
            "structured_error": "InferenceProtocolMismatch",
            "outcome": "fail_fast_sample",
        },
        {
            **common,
            "event": "contract_validation_failure",
            "contract": "LegalActionTable",
            "failure_owner": "python_contract_validation",
            "reason": "sample_bad_schema_version",
            "outcome": "rejected",
        },
        {
            **common,
            "event": "selfplay_no_progress",
            "worker_id": 0,
            "elapsed_s": 0.01,
            "last_successful_phase": "worker_init",
            "last_successful_inference_request": None,
            "last_engine_operation": "worker_init",
            "progress_counters": {"games": 0, "positions": 0},
            "pair_strategy": cfg.model.pair_strategy,
            "pair_rows_scored": 0,
            "suggested_next_action": "inspect inference startup or worker phase transition",
            "outcome": "aborted_predictably",
        },
        {
            **common,
            "event": "selfplay_game_summary",
            "worker_id": 0,
            "game_id": 0,
            "game_length": 0,
            "terminal_reason": "phase00_sample",
            "pair_strategy": cfg.model.pair_strategy,
            "pair_rows_scored": 0,
        },
    ]

    trace = {
        **common,
        "trace_id": f"phase00-{git_sha[:12]}",
        "history_hash": "empty-board-sample",
        "model_family": cfg.model.architecture,
        "phase": "phase00_baseline",
        "legal_count": 0,
        "candidate_count": 0,
        "pair_rows_total": 128,
        "pair_rows_scored": 0,
        "graph_token_count": 0,
        "graph_relation_count": 0,
        "timings_ms": {name: 0.0 for name in REQUIRED_TIMING_SPANS},
        "warnings": (
            "phase00 sample records required trace shape; full subsystem timings are captured by later hot-path probes",
        ),
    }
    trace["timings_ms"]["pair_table_build_ms"] = round(pair_guard_init_ms, 3)

    performance = {
        **common,
        "command": "python scripts/phase00_capture_baseline.py",
        "config_hashes": config_hashes,
        "host_profile": asdict(host),
        "runner_profile": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "cwd": str(REPO_ROOT),
            "pid": os.getpid(),
        },
        "workload": "Phase 00 baseline artifact capture and pair guard initialization",
        "throughput_metrics": {
            "pair_guard_workers_initialized_per_s": round(1000.0 / max(pair_guard_init_ms, 0.001), 3),
        },
        "latency_metrics": {"pair_guard_init_ms": round(pair_guard_init_ms, 3)},
        "queue_backpressure_metrics": {"controlled_no_progress_events": 2},
        "cpu_gpu_utilization_or_proxy": {
            "cuda_available": host.cuda_available,
            "cuda_name": host.cuda_name,
            "proxy": "pair guard initialization wall time",
        },
        "comparison_baseline": "phase00 initial baseline",
        "accepted_regressions": [],
    }

    _write_json(artifact_root / "config_hashes" / "config_hashes.json", config_hashes)
    _write_json(artifact_root / "performance" / "host_profile.json", asdict(host))
    _write_json(artifact_root / "performance" / "baseline_probe.json", performance)
    _write_json(artifact_root / "traces" / "contract_trace_sample.json", trace)
    _write_json(artifact_root / "watchdog" / "no_progress_event.json", events[-2])
    _write_text(
        artifact_root / "logs" / "structured_events.jsonl",
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
    )
    _write_text(
        artifact_root / "performance" / "baseline_probe.md",
        "\n".join(
            [
                "# Phase 00 Baseline Probe",
                "",
                f"- git_sha: `{git_sha}`",
                f"- host: `{platform.platform()}`",
                f"- cuda_available: `{host.cuda_available}`",
                f"- pair_guard_init_ms: `{pair_guard_init_ms:.3f}`",
                "- inference/self-play/replay/training hot-path benchmarks: recorded by mandatory command transcripts where locally available.",
                "",
            ]
        ),
    )
    _write_text(
        artifact_root / "baseline" / "baseline_freeze.md",
        "\n".join(
            [
                "# Phase 00 Baseline Freeze",
                "",
                f"- git_sha: `{git_sha}`",
                f"- created_at: `{created_at}`",
                f"- branch: `{_git('branch', '--show-current')}`",
                "- baseline_tag: `v2-phase-00-pre-python-foundation`",
                "- rust_phase_2_state: captured by `Docs/refactor/rust_review/PHASE_2_VERIFICATION_REPORT.md`.",
                "- current_runtime_is_trusted_oracle: `false`",
                "",
            ]
        ),
    )
    print(json.dumps({"artifact_root": str(artifact_root), "git_sha": git_sha, "events": len(events)}, sort_keys=True))
    return 0


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _config_hashes(config_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(config_dir.glob("*.toml")):
        data = path.read_bytes()
        rows.append(
            {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return rows


def _controlled_watchdog_event(*, git_sha: str, threshold_s: float) -> dict[str, Any]:
    threshold_s = max(0.0, float(threshold_s))
    start = time.perf_counter()
    time.sleep(threshold_s)
    elapsed_s = time.perf_counter() - start
    return {
        "event": "runtime_sweep_no_progress",
        "git_sha": git_sha,
        "created_at": _utc_timestamp(),
        "sample_source": "scripts/phase00_capture_baseline.py --watchdog-smoke",
        "trial_id": "phase00-controlled-stall",
        "threshold_s": threshold_s,
        "elapsed_s": round(elapsed_s, 6),
        "last_successful_phase": "runtime_validation",
        "last_inference_request": None,
        "last_engine_operation": "not_started",
        "progress_counters": {"games": 0, "positions": 0, "train_batches": 0},
        "pair_strategy": "none",
        "pair_rows_scored": 0,
        "suggested_next_action": "inspect runtime scheduler and inference startup",
        "outcome": "aborted_predictably",
        "exit_code": 2,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
