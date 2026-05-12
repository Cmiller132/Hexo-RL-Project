"""Dense policy action-index and tiny-batch overfit diagnostics."""

from __future__ import annotations

import argparse
import json
import random
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hexorl.config import load_config
from hexorl.dashboard.replay import encode_tensor_for_history
from hexorl.models.assembly import build_model_from_config
from hexorl.selfplay.records import BOARD_AREA, action_to_board_index, dense_policy_from_v2
from hexorl.train.loss_plan import build_loss_plan
from hexorl.train.losses import compute_losses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Configs/wsl_speed_probe.toml")
    parser.add_argument("--output", default="runs/dense_policy_alignment/latest.json")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=9301)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    samples = _build_samples()
    tensors, policies, alignment = _encode_samples(samples)
    overfit = _run_overfit(args, tensors, policies)
    payload = {
        "event": "dense_policy_alignment",
        "samples": alignment,
        "all_targets_in_window": all(row["target_index"] >= 0 for row in alignment),
        "all_targets_legal": all(row["target_in_legal"] for row in alignment),
        "overfit": overfit,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["all_targets_in_window"] and payload["all_targets_legal"] and overfit["passed"] else 1


def _hist(*moves: tuple[int, int, int]) -> bytes:
    data = bytearray()
    for player, q, r in moves:
        data.extend(struct.pack("<iii", int(player), int(q), int(r)))
    return bytes(data)


def _build_samples() -> list[dict[str, Any]]:
    return [
        {"name": "empty_center", "history": _hist(), "target": (0, 0)},
        {"name": "after_center_east", "history": _hist((0, 0, 0)), "target": (1, 0)},
        {"name": "two_moves_north", "history": _hist((0, 0, 0), (1, 1, 0)), "target": (0, 1)},
        {"name": "three_moves_west", "history": _hist((0, 0, 0), (1, 1, 0), (0, 0, 1)), "target": (-1, 0)},
    ]


def _encode_samples(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    tensors: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    alignment: list[dict[str, Any]] = []
    for sample in samples:
        tensor, offset_q, offset_r, legal_bytes = encode_tensor_for_history(sample["history"])
        q, r = sample["target"]
        policy_dict, outside_mass = dense_policy_from_v2([(q, r, 1.0)], offset_q, offset_r, top_k=1)
        policy = np.zeros(BOARD_AREA, dtype=np.float32)
        for idx, prob in policy_dict.items():
            policy[int(idx)] = float(prob)
        legal = (
            np.frombuffer(legal_bytes, dtype=np.int32).reshape(-1, 2)
            if legal_bytes
            else np.empty((0, 2), dtype=np.int32)
        )
        target_index = action_to_board_index(q, r, offset_q, offset_r)
        top_index = int(policy.argmax()) if float(policy.sum()) > 0.0 else -1
        legal_set = {(int(lq), int(lr)) for lq, lr in legal}
        alignment.append(
            {
                "name": sample["name"],
                "offset_q": int(offset_q),
                "offset_r": int(offset_r),
                "target_q": int(q),
                "target_r": int(r),
                "target_index": int(target_index),
                "dense_top_index": int(top_index),
                "dense_mass": float(policy.sum()),
                "outside_mass": float(outside_mass),
                "target_in_legal": (int(q), int(r)) in legal_set,
                "legal_count": int(len(legal_set)),
            }
        )
        tensors.append(tensor)
        policies.append(policy)
    return np.stack(tensors).astype(np.float32), np.stack(policies).astype(np.float32), alignment


def _run_overfit(args: argparse.Namespace, tensors: np.ndarray, policies: np.ndarray) -> dict[str, Any]:
    cfg = load_config(Path(args.config)).model_copy(deep=True)
    cfg.model.architecture = "cnn"
    cfg.model.channels = 32
    cfg.model.blocks = 4
    cfg.model.heads = ["policy", "value"]
    cfg.model.sparse_policy = False
    cfg.model.attention_positions = []
    cfg.runtime.compile_model = False
    cfg.runtime.compile_inference = False
    cfg.inference.fp16 = False
    cfg.train.loss_weights = {"policy": 1.0, "value": 1.0}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg, device=device, inference=False)
    model.train()
    x = torch.as_tensor(tensors, device=device)
    y = torch.as_tensor(policies, device=device)
    targets = {
        "policy": y,
        "policy_weight": torch.ones(y.shape[0], device=device),
        "value": torch.zeros(y.shape[0], device=device),
        "value_weight": torch.zeros(y.shape[0], device=device),
    }
    loss_plan = build_loss_plan(("policy", "value"), {"policy": 1.0, "value": 1.0})
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.0)
    started = time.monotonic()
    first_loss = None
    final_loss = None
    final_top1 = 0.0
    final_top_prob = 0.0
    for step in range(1, int(args.steps) + 1):
        opt.zero_grad(set_to_none=True)
        predictions = model(x)
        loss, _per_head = compute_losses(
            predictions,
            targets,
            {"policy": 1.0},
            loss_plan=loss_plan,
        )
        if first_loss is None:
            first_loss = float(loss.detach().cpu())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        final_loss = float(loss.detach().cpu())
        if step == int(args.steps) or step % 20 == 0:
            with torch.no_grad():
                probs = torch.softmax(model(x)["policy"], dim=-1)
                target_top = y.argmax(dim=-1)
                pred_top = probs.argmax(dim=-1)
                final_top1 = float((pred_top == target_top).float().mean().detach().cpu())
                final_top_prob = float(probs.gather(1, target_top[:, None]).mean().detach().cpu())
            if final_top1 >= 1.0 and final_top_prob >= 0.95:
                break
    return {
        "device": str(device),
        "steps_requested": int(args.steps),
        "steps_run": int(step),
        "elapsed_s": time.monotonic() - started,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "final_top1_acc": final_top1,
        "final_target_prob": final_top_prob,
        "passed": bool(final_top1 >= 1.0 and final_top_prob >= 0.95),
    }


if __name__ == "__main__":
    raise SystemExit(main())
