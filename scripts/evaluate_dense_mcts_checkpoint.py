"""Evaluate a dense checkpoint through Rust MCTS, optionally with root priors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hexorl.config import Config, load_config
from hexorl.eval.arena import load_checkpoint_model, run_arena
from hexorl.eval.classical import classical_opponent_fn
from hexorl.eval.players import model_input_dtype
from hexorl.inference.adapters import decode_dense_outputs
from hexorl.inference.server import decode_binned_value_logits
from hexorl.selfplay.worker import HAS_ENGINE, RealMCTSEngine

try:
    import _engine
except ImportError:  # pragma: no cover - local extension dependent
    _engine = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--classical-time-ms", type=int, default=50)
    parser.add_argument("--classical-depth", type=int, default=1)
    parser.add_argument("--root-prior-mix", type=float, default=0.5)
    parser.add_argument(
        "--root-prior-mixes",
        default="",
        help="Optional comma-separated mix sweep. Overrides --root-prior-mix.",
    )
    parser.add_argument("--include-uniform-rootprior", action="store_true")
    parser.add_argument("--include-policy-value-ablation", action="store_true")
    parser.add_argument("--root-prior-time-ms", type=int, default=10)
    parser.add_argument("--root-prior-depth", type=int, default=3)
    parser.add_argument("--root-prior-near-radius", type=int, default=2)
    parser.add_argument("--near-radius", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not HAS_ENGINE or _engine is None:
        raise RuntimeError("Rust _engine extension is required for MCTS checkpoint eval")

    cfg = _load_config(Path(args.config))
    model = load_checkpoint_model(Path(args.checkpoint), cfg)
    mixes = _parse_mixes(args.root_prior_mixes, float(args.root_prior_mix))
    modes: list[tuple[str, float, bool, bool]] = [("mcts_model_only", 0.0, True, True)]
    for mix in mixes:
        if mix > 0.0:
            modes.append((f"mcts_rootprior_{mix:.2f}", mix, True, True))
            if args.include_policy_value_ablation:
                modes.append((f"model_policy_zero_value_rootprior_{mix:.2f}", mix, True, False))
                modes.append((f"uniform_policy_model_value_rootprior_{mix:.2f}", mix, False, True))
    if args.include_uniform_rootprior:
        for mix in mixes:
            if mix > 0.0:
                modes.append((f"uniform_rootprior_{mix:.2f}", mix, False, False))
    results: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "games_per_mode": int(args.games),
        "sims": int(args.sims),
        "batch_size": int(args.batch_size),
        "classical_time_ms": int(args.classical_time_ms),
        "classical_depth": int(args.classical_depth),
        "modes": {},
    }
    for mode_idx, (mode_name, root_prior_mix, use_model_policy, use_model_value) in enumerate(modes):
        player = DenseMCTSPlayer(
            model,
            cfg,
            sims=int(args.sims),
            batch_size=int(args.batch_size),
            near_radius=int(args.near_radius),
            root_prior_mix=float(root_prior_mix),
            root_prior_time_ms=int(args.root_prior_time_ms),
            root_prior_depth=int(args.root_prior_depth),
            root_prior_near_radius=int(args.root_prior_near_radius),
            seed=int(args.seed) + 1009 * mode_idx,
            use_model_policy=bool(use_model_policy),
            use_model_value=bool(use_model_value),
        )
        classical = classical_opponent_fn(
            time_ms=int(args.classical_time_ms),
            max_depth=int(args.classical_depth),
            noise_level=0.0,
        )
        stats = run_arena(player, classical, num_games=int(args.games), sims=int(args.sims))
        results["modes"][mode_name] = {
            "games": stats.total_games,
            "model_wins": stats.wins_a,
            "opponent_wins": stats.wins_b,
            "draws": stats.draws,
            "model_win_rate": stats.win_rate_a,
            "avg_moves": stats.avg_moves,
            "reason_counts": stats.reason_counts,
            "first_game_moves": stats.results[0].move_history[:20] if stats.results else [],
            "mcts_calls": player.calls,
            "avg_root_children": player.avg_root_children,
            "avg_root_prior_hits": player.avg_root_prior_hits,
            "use_model_policy": bool(use_model_policy),
            "use_model_value": bool(use_model_value),
        }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


class DenseMCTSPlayer:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: Config,
        *,
        sims: int,
        batch_size: int,
        near_radius: int,
        root_prior_mix: float,
        root_prior_time_ms: int,
        root_prior_depth: int,
        root_prior_near_radius: int,
        seed: int,
        use_model_policy: bool,
        use_model_value: bool,
    ):
        self.model = model
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.dtype = model_input_dtype(model)
        self.sims = int(sims)
        self.batch_size = int(batch_size)
        self.near_radius = int(near_radius)
        self.root_prior_mix = max(0.0, min(float(root_prior_mix), 0.95))
        self.root_prior_time_ms = int(root_prior_time_ms)
        self.root_prior_depth = int(root_prior_depth)
        self.root_prior_near_radius = int(root_prior_near_radius)
        self.seed = int(seed)
        self.use_model_policy = bool(use_model_policy)
        self.use_model_value = bool(use_model_value)
        self.calls = 0
        self._root_children_total = 0
        self._root_prior_hits = 0
        self.model.eval()

    @property
    def avg_root_children(self) -> float:
        return self._root_children_total / max(1, self.calls)

    @property
    def avg_root_prior_hits(self) -> float:
        return self._root_prior_hits / max(1, self.calls)

    def __call__(
        self,
        move_history: list[tuple[int, int, int]],
        time_ms_override: int,
        player: int,
    ) -> tuple[int | None, int | None]:
        game = _engine.HexGame()
        for _p, q, r in move_history:
            game.place(int(q), int(r))
        if bool(getattr(game, "is_over", False)):
            return None, None
        engine = RealMCTSEngine(
            game,
            self.sims,
            float(self.cfg.selfplay.c_puct),
            self.near_radius,
            self.seed + self.calls,
            c_puct_init=float(self.cfg.selfplay.c_puct_init),
            constrain_threats=bool(self.cfg.selfplay.constrain_threats),
            subtree_reuse=False,
            max_children=None,
        )
        self.calls += 1
        init = engine.init_root()
        if init is None:
            return None, None
        tensor, offset_q, offset_r, legal_bytes, root_generation = init
        legal = np.frombuffer(legal_bytes, dtype=np.int32).reshape(-1, 2)
        policy, value = self._infer(np.asarray(tensor, dtype=np.float32).reshape(1, 13, 33, 33))
        prior_qr, prior_logits = self._classical_root_prior(game, legal)
        if prior_qr.shape[0] > 0:
            self._root_prior_hits += 1
            engine.expand_root_with_sparse_priors(
                policy,
                float(value[0]),
                int(offset_q),
                int(offset_r),
                legal_bytes,
                root_generation,
                prior_qr,
                prior_logits,
                1,
                self.root_prior_mix,
            )
        else:
            engine.expand_root(
                policy,
                float(value[0]),
                int(offset_q),
                int(offset_r),
                legal_bytes,
                root_generation,
            )
        while not engine.done():
            batch_tensor, count, batch_generation = engine.select_leaves(self.batch_size)
            if int(count) <= 0:
                engine.expand_and_backprop(
                    np.zeros(0, dtype=np.float32),
                    np.zeros(0, dtype=np.float32),
                    batch_generation,
                )
                break
            p, v = self._infer(np.asarray(batch_tensor, dtype=np.float32)[: int(count)])
            engine.expand_and_backprop(p, v, batch_generation)
        moves_q, moves_r, visits, _root_value = engine.get_results()
        self._root_children_total += len(moves_q)
        if not moves_q:
            return None, None
        best = max(range(len(moves_q)), key=lambda idx: (int(visits[idx]), -idx))
        return int(moves_q[best]), int(moves_r[best])

    def _infer(self, tensor_4d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.use_model_policy and not self.use_model_value:
            count = int(tensor_4d.shape[0])
            return np.zeros(count * 1089, dtype=np.float32), np.zeros(count, dtype=np.float32)
        x = torch.from_numpy(tensor_4d.astype(np.float32, copy=False)).to(
            device=self.device,
            dtype=self.dtype,
        )
        with torch.no_grad():
            out = self.model(x)
        decoded = decode_dense_outputs(
            out,
            value_decoder=decode_binned_value_logits,
            sparse_requested=False,
        )
        count = int(tensor_4d.shape[0])
        policy = (
            decoded.policy.reshape(-1)
            if self.use_model_policy
            else np.zeros(count * 1089, dtype=np.float32)
        )
        value = (
            decoded.value.reshape(-1)
            if self.use_model_value
            else np.zeros(count, dtype=np.float32)
        )
        return policy, value

    def _classical_root_prior(
        self,
        game,
        legal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.root_prior_mix <= 0.0 or legal.shape[0] <= 0:
            return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float32)
        try:
            q, r, _score, _depth, _nodes = game.classical_search(
                time_ms=self.root_prior_time_ms,
                max_depth=self.root_prior_depth,
                near_radius=self.root_prior_near_radius,
                noise_level=0.0,
            )
        except Exception:
            return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float32)
        q, r = int(q), int(r)
        legal_set = {(int(lq), int(lr)) for lq, lr in legal.tolist()}
        if (q, r) not in legal_set:
            return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float32)
        return np.asarray([(q, r)], dtype=np.int32), np.asarray([10.0], dtype=np.float32)


def _load_config(path: Path) -> Config:
    if path.suffix.lower() == ".json":
        return Config.model_validate_json(path.read_text(encoding="utf-8"))
    return load_config(path)


def _parse_mixes(raw: str, fallback: float) -> list[float]:
    if not raw.strip():
        return [float(fallback)]
    out: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = max(0.0, min(float(item), 0.95))
        if value not in out:
            out.append(value)
    return out or [float(fallback)]


if __name__ == "__main__":
    raise SystemExit(main())
