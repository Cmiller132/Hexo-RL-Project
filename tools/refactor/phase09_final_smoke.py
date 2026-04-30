"""Generate the Phase 09 final end-to-end smoke bundle."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hexorl.config import Config
from hexorl.dashboard.contract_inspector import ContractInspector
from hexorl.epoch.pipeline import _make_bootstrap_game_records
from hexorl.eval.players import greedy_model_player
from hexorl.models.factory import build_model, get_model_registry, train_adapter_for
from hexorl.replay.codec import decode_replay_game, encode_replay_game, replay_game_from_selfplay
from hexorl.replay.fixtures import corrupt_replay_bytes, golden_replay_game
from hexorl.replay.sampler import ReplayDataset
from hexorl.replay.storage import ReplayStorage
from hexorl.tuning.family_spaces import family_space
from hexorl.tuning.recipes import ModelRecipe, RecipeTransform
from hexorl.tuning.reporting import trial_lifecycle_report
from hexorl.tuning.runtime_sweep import HostProfile, default_runtime_spec, simulate_no_progress
from hexorl.tuning.validation import dry_run_validate_recipe


ROOT = Path(__file__).resolve().parents[2]


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, default=_json_default) + "\n" for row in rows), encoding="utf-8")


def _cfg() -> Config:
    cfg = Config()
    cfg.run.seed = 20260430
    cfg.model.architecture = "graph_hybrid_0"
    cfg.model.channels = 4
    cfg.model.blocks = 1
    cfg.model.graph_layers = 1
    cfg.model.attention_heads = 1
    cfg.model.heads = ["policy", "value", "sparse_policy", "pair_policy"]
    cfg.model.sparse_policy = True
    cfg.model.candidate_budget = 8
    cfg.model.pair_strategy = "none"
    cfg.buffer.capacity = 32
    cfg.buffer.lookahead_horizons = []
    cfg.buffer.lookahead_lambdas = []
    cfg.selfplay.max_game_moves = 8
    cfg.selfplay.policy_target_top_k = 4
    cfg.train.batch_size = 2
    cfg.train.batches_per_epoch = 1
    cfg.train.lr_schedule = "constant"
    cfg.train.loss_weights = {"policy": 1.0, "value": 1.0, "sparse_policy": 0.1, "pair_policy": 0.05}
    cfg.runtime.dataloader_workers = 0
    cfg.inference.fp16 = False
    return cfg


def _one_train_step(cfg: Config, storage: ReplayStorage) -> dict[str, object]:
    dataset = ReplayDataset(
        storage,
        batch_size=cfg.train.batch_size,
        use_symmetry=False,
        include_sparse_policy=True,
        include_pair_policy=True,
        candidate_budget=cfg.model.candidate_budget,
    )
    batch = next(iter(DataLoader(dataset, batch_size=None, num_workers=0)))
    model = build_model(cfg, device=torch.device("cpu"), inference=False)
    adapter = train_adapter_for(model, cfg, device=torch.device("cpu"))
    projected = adapter.project_batch(batch)
    outputs = adapter.forward(projected)
    total, parts = adapter.losses(outputs, projected.targets, n_bins=model.n_bins)
    total.backward()
    return {
        "family": "graph_hybrid",
        "batch_source": batch.source,
        "loss_total": float(total.detach().cpu()),
        "loss_parts": {key: float(value.detach().cpu()) for key, value in parts.items() if torch.is_tensor(value)},
        "target_keys": sorted(projected.targets),
    }


def _tuning_probe() -> dict[str, object]:
    host = HostProfile.local()
    runtime = default_runtime_spec(host)
    recipe = ModelRecipe(
        recipe_id="phase09-smoke-graph",
        model_family="graph_hybrid",
        channels=4,
        blocks=1,
        heads=("policy", "value", "sparse_policy"),
        candidate_budget=8,
    )
    valid = tuple(dry_run_validate_recipe(recipe, runtime, host))
    try:
        RecipeTransform.from_raw_config({"model": {"architecture": "raw-old"}})
        raw_rejected = {"ok": True, "message": "unexpected raw config acceptance"}
    except TypeError as exc:
        raw_rejected = {"ok": False, "message": str(exc)}
    stalled = simulate_no_progress(runtime, "selfplay", trace_id="phase09-smoke")
    report = trial_lifecycle_report(
        [
            {
                "trial_id": "phase09-smoke",
                "action": "select",
                "reason_code": "validation_passed",
                "score_components": {"throughput": 1.0, "utilization": 1.0, "stability": 1.0, "stall_penalty": 0.0},
                "trace_ids": ["phase09-final-smoke"],
                "likely_owner": "tuning/scheduler",
            }
        ]
    )
    return {
        "valid_recipe": {"ok": all(item["ok"] for item in valid), "checks": list(valid), "recipe": recipe.to_manifest()},
        "rejected_raw_config": raw_rejected,
        "family_space": family_space("graph_hybrid").to_manifest(),
        "watchdog": stalled,
        "report": report,
    }


def run(output_dir: Path) -> dict[str, object]:
    started = time.monotonic()
    cfg = _cfg()
    games = _make_bootstrap_game_records(cfg, 1)
    storage = ReplayStorage(capacity=cfg.buffer.capacity)
    replay_records = []
    for game in games:
        replay = replay_game_from_selfplay(
            game,
            lookahead_horizons=cfg.buffer.lookahead_horizons,
            lookahead_lambdas=cfg.buffer.lookahead_lambdas,
            config_identity="phase09-final-smoke",
            checkpoint_identity="phase09-smoke",
        )
        storage.append_game(replay)
        replay_records.append(replay)

    encoded = encode_replay_game(replay_records[0])
    decoded = decode_replay_game(encoded)
    train = _one_train_step(cfg, storage)

    model = build_model(cfg, device=torch.device("cpu"), inference=False)
    player = greedy_model_player(model, cfg=cfg, device=torch.device("cpu"))
    eval_move = player([], 1, 0)
    eval_trace = player.telemetry[-1].to_dict()

    inspector = ContractInspector()
    history = games[0].positions[0].move_history
    debug_bundle = inspector.inspect(
        "debug-bundle",
        history=history,
        policy_target=tuple(games[0].positions[0].policy_target_v2),
        pair_policy_target=tuple(games[0].positions[0].pair_policy_target_v2),
        model_output={"heads": ["policy", "value", "sparse_policy"]},
        replay_identity={
            "game_hash": decoded.game_hash,
            "position_hash": decoded.positions[0].record_hash,
        },
        trace={
            "trace_id": "phase09-final-smoke",
            "mcts_token_lifecycle": "engine_adapter_root_batch_tokens",
            "ffi_protocol_source": "hexgame-py protocol",
            "invariant_probe_status": "checked_by_engine_tests",
            "structured_rust_error_owner": "search.engine_adapter",
        },
    )
    mutation_corruption = {
        "bad_replay_magic_rejected": False,
        "bad_schema_rejected": False,
        "stale_hash_rejected": False,
    }
    for kind, key in (("bad_magic", "bad_replay_magic_rejected"), ("bad_version", "bad_schema_rejected")):
        try:
            decode_replay_game(corrupt_replay_bytes(kind))
        except Exception as exc:
            mutation_corruption[key] = True
            mutation_corruption[f"{key}_error"] = str(exc)
    stale_payload = golden_replay_game().to_dict()
    stale_payload["positions"][0]["legal_table_hash"] = "stale"
    try:
        type(golden_replay_game()).from_dict(stale_payload)
    except Exception as exc:
        mutation_corruption["stale_hash_rejected"] = True
        mutation_corruption["stale_hash_error"] = str(exc)

    rust_suspicion = {
        "malformed_ffi_bytes": "covered_by Python/tests/engine/test_phase01_engine_contract_parity.py",
        "stale_mcts_tokens": "covered_by Python/tests/search/test_engine_adapter.py",
        "invalid_policy_lengths": "covered_by Rust MCTSError tests and engine adapter tests",
        "far_coordinate_mismatches": "covered_by legal/candidate contract validation",
        "illegal_pair_rows": "covered_by Python/tests/contracts/test_phase02_builders.py",
        "structured_error_owner": "search.engine_adapter",
    }
    tuning = _tuning_probe()
    telemetry = [
        {"event": "selfplay_game_generation", "games": len(games), "positions": len(games[0].positions)},
        {"event": "canonical_replay_write_read", "game_hash": decoded.game_hash, "positions": len(decoded.positions)},
        {"event": "train_step", **train},
        {"event": "eval_policy_provider", "move": list(eval_move), "trace": eval_trace},
        {"event": "dashboard_contract_inspector", "debug_sections": sorted(debug_bundle)},
        {"event": "autotune_recipe_dry_run", "valid": tuning["valid_recipe"]["ok"], "rejected": not tuning["rejected_raw_config"]["ok"]},
    ]
    summary = {
        "schema_version": 1,
        "ok": all(
            [
                bool(games),
                decoded.game_hash == replay_records[0].game_hash,
                train["loss_total"] >= 0.0,
                eval_move[0] is not None,
                bool(debug_bundle["engine"]),
                tuning["valid_recipe"]["ok"],
                not tuning["rejected_raw_config"]["ok"],
                all(bool(v) for k, v in mutation_corruption.items() if k.endswith("_rejected")),
            ]
        ),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "registered_families": list(get_model_registry().names()),
        "representative_train_family_set": ["graph_hybrid"],
        "elapsed_s": time.monotonic() - started,
        "replay": {
            "game_hash": decoded.game_hash,
            "position_count": len(decoded.positions),
            "storage_stats": storage.stats,
        },
        "train": train,
        "eval": {"move": list(eval_move), "trace": eval_trace},
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "debug_bundle.json", debug_bundle)
    _write_json(output_dir.parent / "verification" / "mutation_corruption_report.json", mutation_corruption)
    _write_json(output_dir.parent / "verification" / "rust_suspicion_report.json", rust_suspicion)
    _write_json(output_dir / "autotune_dry_run.json", tuning)
    _write_jsonl(output_dir.parent / "telemetry_samples" / "phase09_trace_samples.jsonl", telemetry)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Docs/refactor/artifacts/phase_09/final_smoke")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    summary = run(output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
