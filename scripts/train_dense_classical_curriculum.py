"""Train a tiny dense CNN on shallow-classical labels from persisted positions."""

from __future__ import annotations

import argparse
import base64
import json
import random
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import _engine

    HAS_ENGINE = True
except ImportError:
    _engine = None
    HAS_ENGINE = False

from hexorl.config import load_config
from hexorl.dashboard.replay import encode_tensor_for_history
from hexorl.eval.arena import model_move_fn, run_arena
from hexorl.eval.classical import classical_opponent_fn
from hexorl.models.assembly import build_model_from_config
from hexorl.selfplay.records import BOARD_AREA, action_to_board_index
from hexorl.train.loss_plan import build_loss_plan
from hexorl.train.losses import compute_losses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", help="dashboard.sqlite3 path")
    parser.add_argument("--config", default="Configs/wsl_speed_probe.toml")
    parser.add_argument("--output", default="runs/dense_policy_alignment/classical_curriculum.json")
    parser.add_argument("--max-turn", type=int, default=60)
    parser.add_argument("--sample", type=int, default=256)
    parser.add_argument("--generate-games", type=int, default=0)
    parser.add_argument("--generated-max-turn", type=int, default=160)
    parser.add_argument("--random-opening-plies", type=int, default=8)
    parser.add_argument("--random-opening-prob", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--classical-time-ms", type=int, default=10)
    parser.add_argument("--classical-max-depth", type=int, default=3)
    parser.add_argument("--classical-near-radius", type=int, default=2)
    parser.add_argument("--eval-games", type=int, default=4)
    parser.add_argument("--eval-time-ms", type=int, default=50)
    parser.add_argument("--eval-depth", type=int, default=1)
    parser.add_argument(
        "--target-mode",
        choices=("classical", "mixed_selfplay"),
        default="classical",
        help="Use pure classical labels or blend classical labels into persisted self-play policy targets.",
    )
    parser.add_argument("--classical-mix-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=9501)
    args = parser.parse_args()

    if not HAS_ENGINE:
        raise RuntimeError("_engine is required for classical curriculum labels")
    _seed(args.seed)
    if int(args.generate_games) > 0:
        examples = _generate_examples(args)
    else:
        rows = _load_rows(Path(args.db), max_turn=args.max_turn, limit=args.sample)
        examples = [_example_from_row(row, args) for row in rows]
        examples = [ex for ex in examples if ex is not None]
    if len(examples) < 8:
        raise RuntimeError(f"not enough usable examples: {len(examples)}")
    random.shuffle(examples)
    split = max(1, int(len(examples) * 0.8))
    train = examples[:split]
    val = examples[split:] or examples[: max(1, min(16, len(examples)))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _make_cfg(args)
    model = build_model_from_config(cfg, device=device, inference=False)
    model.train()
    loss_plan = build_loss_plan(("policy", "value"), {"policy": 1.0, "value": 1.0})
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    started = time.monotonic()
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        random.shuffle(train)
        for batch in _batches(train, int(args.batch_size)):
            x, y = _batch_tensors(batch, device)
            targets = {
                "policy": y,
                "policy_weight": torch.ones(y.shape[0], device=device),
                "value": torch.zeros(y.shape[0], device=device),
                "value_weight": torch.zeros(y.shape[0], device=device),
            }
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss, _ = compute_losses(pred, targets, {"policy": 1.0, "value": 1.0}, loss_plan=loss_plan)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        history.append({"epoch": epoch, "train": _score(model, train, device), "val": _score(model, val, device)})
        if history[-1]["train"]["top1_acc"] >= 0.98 and history[-1]["val"]["top1_acc"] >= 0.80:
            break

    arena = {}
    if int(args.eval_games) > 0:
        model.eval()
        model_player = model_move_fn(model, temperature=1e-5, top_p=1.0, seed=int(args.seed))
        classical = classical_opponent_fn(time_ms=int(args.eval_time_ms), max_depth=int(args.eval_depth), noise_level=0.0)
        stats = run_arena(model_player, classical, num_games=int(args.eval_games), sims=128)
        arena = {
            "games": stats.total_games,
            "model_wins": stats.wins_a,
            "opponent_wins": stats.wins_b,
            "draws": stats.draws,
            "model_win_rate": stats.win_rate_a,
            "avg_moves": stats.avg_moves,
            "reason_counts": stats.reason_counts,
        }

    output = {
        "event": "dense_classical_curriculum",
        "db": str(args.db),
        "generated_games": int(args.generate_games),
        "target_mode": str(args.target_mode),
        "classical_mix_alpha": float(args.classical_mix_alpha),
        "usable_examples": len(examples),
        "train_examples": len(train),
        "val_examples": len(val),
        "device": str(device),
        "elapsed_s": time.monotonic() - started,
        "history": history,
        "final": history[-1],
        "arena": arena,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _make_cfg(args: argparse.Namespace):
    cfg = load_config(Path(args.config)).model_copy(deep=True)
    cfg.model.architecture = "cnn"
    cfg.model.channels = 32
    cfg.model.blocks = 4
    cfg.model.heads = ["policy", "value"]
    cfg.model.sparse_policy = False
    cfg.model.attention_positions = []
    cfg.inference.fp16 = False
    cfg.runtime.compile_model = False
    cfg.runtime.compile_inference = False
    cfg.train.loss_weights = {"policy": 1.0, "value": 1.0}
    cfg.train.peak_lr = float(args.lr)
    return cfg


def _load_rows(db: Path, *, max_turn: int, limit: int) -> list[dict[str, Any]]:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows: list[dict[str, Any]] = []
    query = """
        SELECT g.final_history_b64, p.turn_index, p.policy_json
        FROM positions p
        JOIN games g ON g.game_id = p.game_id
        WHERE g.source = 'selfplay' AND p.turn_index <= ?
        ORDER BY p.position_id DESC
        LIMIT ?
    """
    for row in con.execute(query, (int(max_turn), int(limit))):
        item = dict(row)
        item["history"] = _position_history(item["final_history_b64"], int(item["turn_index"]))
        item["policy"] = {int(k): float(v) for k, v in json.loads(item.pop("policy_json") or "{}").items()}
        rows.append(item)
    return rows


def _position_history(final_history_b64: str, turn_index: int) -> bytes:
    raw = base64.b64decode(final_history_b64) if final_history_b64 else b""
    return raw[: max(0, int(turn_index)) * 12]


def _decode_moves(history: bytes) -> list[tuple[int, int, int]]:
    return [struct.unpack_from("<iii", history, off) for off in range(0, len(history), 12)]


def _example_from_row(row: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray] | None:
    tensor, offset_q, offset_r, _legal = encode_tensor_for_history(row["history"])
    game = _engine.HexGame()
    for _player, q, r in _decode_moves(row["history"]):
        game.place(int(q), int(r))
    try:
        q, r, _score, _depth, _nodes = game.classical_search(
            time_ms=int(args.classical_time_ms),
            max_depth=int(args.classical_max_depth),
            near_radius=int(args.classical_near_radius),
            noise_level=0.0,
        )
    except Exception:
        return None
    idx = action_to_board_index(int(q), int(r), int(offset_q), int(offset_r))
    if idx < 0 or idx >= BOARD_AREA:
        return None
    target = np.zeros(BOARD_AREA, dtype=np.float32)
    target[int(idx)] = 1.0
    if str(args.target_mode) == "mixed_selfplay":
        policy = np.zeros(BOARD_AREA, dtype=np.float32)
        for policy_idx, prob in row.get("policy", {}).items():
            if 0 <= int(policy_idx) < BOARD_AREA and float(prob) > 0.0:
                policy[int(policy_idx)] = float(prob)
        policy_mass = float(policy.sum())
        if policy_mass > 0.0:
            policy /= policy_mass
            alpha = min(max(float(args.classical_mix_alpha), 0.0), 1.0)
            target = (1.0 - alpha) * policy + alpha * target
            target_sum = float(target.sum())
            if target_sum > 0.0:
                target /= target_sum
    return tensor.astype(np.float32), target


def _generate_examples(args: argparse.Namespace) -> list[tuple[np.ndarray, np.ndarray]]:
    examples: list[tuple[np.ndarray, np.ndarray]] = []
    for game_idx in range(int(args.generate_games)):
        game = _engine.HexGame()
        history: list[tuple[int, int, int]] = []
        for move_idx in range(int(args.generated_max_turn)):
            if bool(getattr(game, "is_over", False)):
                break
            history_bytes = _pack_history(history)
            tensor, offset_q, offset_r, legal_bytes = encode_tensor_for_history(history_bytes)
            try:
                q, r, _score, _depth, _nodes = game.classical_search(
                    time_ms=int(args.classical_time_ms),
                    max_depth=int(args.classical_max_depth),
                    near_radius=int(args.classical_near_radius),
                    noise_level=0.0,
                )
            except Exception:
                break
            idx = action_to_board_index(int(q), int(r), int(offset_q), int(offset_r))
            if 0 <= idx < BOARD_AREA:
                target = np.zeros(BOARD_AREA, dtype=np.float32)
                target[int(idx)] = 1.0
                examples.append((tensor.astype(np.float32), target))

            play_q, play_r = int(q), int(r)
            if (
                move_idx < int(args.random_opening_plies)
                and random.random() < float(args.random_opening_prob)
                and legal_bytes
            ):
                legal = np.frombuffer(legal_bytes, dtype=np.int32).reshape(-1, 2)
                if len(legal) > 0:
                    chosen = legal[random.randrange(len(legal))]
                    play_q, play_r = int(chosen[0]), int(chosen[1])
            player = int(getattr(game, "current_player", 0))
            try:
                game.place(play_q, play_r)
            except Exception:
                try:
                    game.place(int(q), int(r))
                    play_q, play_r = int(q), int(r)
                except Exception:
                    break
            history.append((player, play_q, play_r))
    return examples


def _pack_history(history: list[tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for player, q, r in history:
        out.extend(struct.pack("<iii", int(player), int(q), int(r)))
    return bytes(out)


def _batches(items: list[tuple[np.ndarray, np.ndarray]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _batch_tensors(batch: list[tuple[np.ndarray, np.ndarray]], device: torch.device):
    x = torch.as_tensor(np.stack([item[0] for item in batch]), device=device)
    y = torch.as_tensor(np.stack([item[1] for item in batch]), device=device)
    return x, y


def _score(model: torch.nn.Module, items: list[tuple[np.ndarray, np.ndarray]], device: torch.device) -> dict[str, float]:
    model.eval()
    total = correct = 0
    target_probs: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in _batches(items, 64):
            x, y = _batch_tensors(batch, device)
            logits = model(x)["policy"]
            probs = torch.softmax(logits, dim=-1)
            target = y.argmax(dim=-1)
            pred = probs.argmax(dim=-1)
            total += int(y.shape[0])
            correct += int((pred == target).sum().detach().cpu())
            target_probs.extend(probs.gather(1, target[:, None]).squeeze(1).detach().cpu().tolist())
            losses.extend((-(y * torch.log_softmax(logits, dim=-1)).sum(dim=1)).detach().cpu().tolist())
    model.train()
    return {
        "top1_acc": float(correct / max(total, 1)),
        "target_prob": float(np.mean(target_probs)) if target_probs else 0.0,
        "loss": float(np.mean(losses)) if losses else 0.0,
    }


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
