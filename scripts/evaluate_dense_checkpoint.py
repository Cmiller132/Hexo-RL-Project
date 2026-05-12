"""Evaluate a dense checkpoint with several direct-policy arena modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hexorl.config import Config, load_config
from hexorl.eval.arena import load_checkpoint_model, model_move_fn, run_arena
from hexorl.eval.classical import classical_opponent_fn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved JSON config or TOML config.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--classical-time-ms", type=int, default=50)
    parser.add_argument("--classical-depth", type=int, default=1)
    parser.add_argument("--near-radius", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    checkpoint = Path(args.checkpoint)
    model = load_checkpoint_model(checkpoint, cfg)
    modes = {
        "sampled_t005_p095": dict(temperature=0.05, top_p=0.95),
        "greedy": dict(temperature=1.0e-5, top_p=1.0),
        "sampled_t010_p090": dict(temperature=0.10, top_p=0.90),
    }
    results: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "config": str(args.config),
        "games_per_mode": int(args.games),
        "classical_time_ms": int(args.classical_time_ms),
        "classical_depth": int(args.classical_depth),
        "near_radius": int(args.near_radius),
        "modes": {},
    }
    for mode_idx, (name, policy_kwargs) in enumerate(modes.items()):
        player = model_move_fn(
            model,
            temperature=float(policy_kwargs["temperature"]),
            top_p=float(policy_kwargs["top_p"]),
            near_radius=int(args.near_radius),
            seed=int(args.seed) + 1009 * mode_idx,
        )
        classical = classical_opponent_fn(
            time_ms=int(args.classical_time_ms),
            max_depth=int(args.classical_depth),
            noise_level=0.0,
        )
        stats = run_arena(player, classical, num_games=int(args.games), sims=128)
        results["modes"][name] = {
            "games": stats.total_games,
            "model_wins": stats.wins_a,
            "opponent_wins": stats.wins_b,
            "draws": stats.draws,
            "model_win_rate": stats.win_rate_a,
            "avg_moves": stats.avg_moves,
            "reason_counts": stats.reason_counts,
            "first_game_moves": stats.results[0].move_history[:20] if stats.results else [],
        }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def _load_config(path: Path) -> Config:
    if path.suffix.lower() == ".json":
        return Config.model_validate_json(path.read_text(encoding="utf-8"))
    return load_config(path)


if __name__ == "__main__":
    raise SystemExit(main())
