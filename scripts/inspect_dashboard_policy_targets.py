"""Inspect persisted dashboard self-play policy targets for legality and overfitability."""

from __future__ import annotations

import argparse
import base64
import json
import math
import random
import sqlite3
import struct
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hexorl.config import Config, load_config
from hexorl.action_contract.tactical_oracle import scan_tactical_oracle_from_history
from hexorl.dashboard.replay import encode_tensor_for_history
from hexorl.models.assembly import build_model_from_config
from hexorl.selfplay.records import BOARD_AREA, BOARD_SIZE, action_to_board_index
from hexorl.train.loss_plan import build_loss_plan
from hexorl.train.losses import compute_losses

try:
    import _engine

    HAS_ENGINE = True
except ImportError:
    _engine = None
    HAS_ENGINE = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", help="dashboard.sqlite3 path")
    parser.add_argument("--config", default="Configs/wsl_speed_probe.toml")
    parser.add_argument("--output", default="runs/dense_policy_alignment/dashboard_target_inspection.json")
    parser.add_argument("--sample", type=int, default=256)
    parser.add_argument("--overfit", type=int, default=64)
    parser.add_argument("--source", default="", help="Optional game source filter, e.g. selfplay or bootstrap.")
    parser.add_argument("--max-turn", type=int, default=-1, help="Only inspect positions at or before this turn index.")
    parser.add_argument("--min-turn", type=int, default=-1, help="Only inspect positions at or after this turn index.")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--classical-time-ms", type=int, default=0)
    parser.add_argument("--classical-max-depth", type=int, default=3)
    parser.add_argument("--classical-near-radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9401)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = _load_rows(
        Path(args.db),
        limit=max(int(args.sample), int(args.overfit)),
        source=str(args.source or ""),
        min_turn=int(args.min_turn),
        max_turn=int(args.max_turn),
    )
    analysis = _analyze_rows(rows[: int(args.sample)], args=args)
    overfit = _overfit_rows(args, rows[: int(args.overfit)])
    payload = {
        "event": "dashboard_policy_target_inspection",
        "db": str(args.db),
        "rows_loaded": len(rows),
        "analysis": analysis,
        "overfit": overfit,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    bad = analysis["overall"]["nonzero_mass_frac"] < 0.95 or analysis["overall"]["top_legal_frac"] < 0.95
    bad = bad or not overfit["passed"]
    return 1 if bad else 0


def _load_rows(db: Path, *, limit: int, source: str = "", min_turn: int = -1, max_turn: int = -1) -> list[dict[str, Any]]:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    filters = []
    params: list[Any] = []
    if source:
        filters.append("g.source = ?")
        params.append(source)
    if min_turn >= 0:
        filters.append("p.turn_index >= ?")
        params.append(int(min_turn))
    if max_turn >= 0:
        filters.append("p.turn_index <= ?")
        params.append(int(max_turn))
    where = "WHERE " + " AND ".join(filters) if filters else ""
    query = """
        SELECT
            g.game_id, g.source, g.epoch, g.outcome, g.move_count, g.final_history_b64,
            p.position_id, p.turn_index, p.player, p.root_value, p.policy_json, p.debug_json
        FROM positions p
        JOIN games g ON g.game_id = p.game_id
        {where}
        ORDER BY
            CASE g.source WHEN 'selfplay' THEN 0 ELSE 1 END,
            p.position_id DESC
        LIMIT ?
    """.format(where=where)
    params.append(int(limit))
    rows: list[dict[str, Any]] = []
    for row in con.execute(query, tuple(params)):
        item = dict(row)
        item["policy"] = {int(k): float(v) for k, v in json.loads(item.pop("policy_json") or "{}").items()}
        item["debug"] = json.loads(item.pop("debug_json") or "{}")
        item["history"] = _position_history(item["final_history_b64"], int(item["turn_index"]))
        rows.append(item)
    return rows


def _position_history(final_history_b64: str, turn_index: int) -> bytes:
    raw = base64.b64decode(final_history_b64) if final_history_b64 else b""
    return raw[: max(0, int(turn_index)) * 12]


def _decode_moves(history: bytes) -> list[tuple[int, int, int]]:
    return [struct.unpack_from("<iii", history, off) for off in range(0, len(history), 12)]


def _flat_to_qr(index: int, offset_q: int, offset_r: int) -> tuple[int, int]:
    gi, gj = divmod(int(index), BOARD_SIZE)
    return int(gi + offset_q), int(gj + offset_r)


def _row_target_stats(row: dict[str, Any], args: argparse.Namespace | None = None) -> dict[str, Any]:
    tensor, offset_q, offset_r, legal_bytes = encode_tensor_for_history(row["history"])
    del tensor
    legal = (
        np.frombuffer(legal_bytes, dtype=np.int32).reshape(-1, 2)
        if legal_bytes
        else np.empty((0, 2), dtype=np.int32)
    )
    legal_set = {(int(q), int(r)) for q, r in legal}
    positive = [(idx, prob) for idx, prob in row["policy"].items() if float(prob) > 0.0]
    mass = float(sum(prob for _idx, prob in positive))
    top_idx, top_prob = max(positive, key=lambda item: item[1]) if positive else (-1, 0.0)
    top_qr = _flat_to_qr(top_idx, offset_q, offset_r) if top_idx >= 0 else None
    norm_probs = [prob / mass for _idx, prob in positive if mass > 0]
    entropy = float(-sum(p * math.log(max(p, 1e-12)) for p in norm_probs)) if norm_probs else 0.0
    v2 = [
        (int(q), int(r), float(prob))
        for q, r, prob in row["debug"].get("policy_target_v2", [])
        if float(prob) > 0.0
    ]
    v2_mass = float(sum(prob for _q, _r, prob in v2))
    v2_top = max(v2, key=lambda item: item[2]) if v2 else None
    v2_top_index = (
        action_to_board_index(v2_top[0], v2_top[1], int(offset_q), int(offset_r))
        if v2_top is not None
        else -1
    )
    oracle_payload = _oracle_stats(row["history"], legal, int(offset_q), int(offset_r), top_qr)
    classical_payload = _classical_stats(
        row["history"],
        row["policy"],
        int(offset_q),
        int(offset_r),
        top_qr,
        args=args,
    )
    return {
        "source": row["source"],
        "epoch": row["epoch"],
        "turn_index": row["turn_index"],
        "player": row["player"],
        "is_full_search": bool(row["debug"].get("is_full_search", False)),
        "policy_weight": float(row["debug"].get("policy_weight", 0.0) or 0.0),
        "root_value": float(row["root_value"]),
        "selected_action_value": row["debug"].get("selected_action_value"),
        "outcome": float(row["debug"].get("outcome", row["outcome"])),
        "legal_count": int(len(legal_set)),
        "mass": mass,
        "top_prob": float(top_prob),
        "entropy": entropy,
        "top_index": int(top_idx),
        "top_qr": top_qr,
        "top_legal": bool(top_qr in legal_set) if top_qr is not None else False,
        "v2_mass": v2_mass,
        "v2_top_index": int(v2_top_index),
        "v2_top_matches_dense": bool(v2_top_index == top_idx) if v2_top_index >= 0 and top_idx >= 0 else False,
        "outside_mass": float(row["debug"].get("target_policy_mass_outside_window", 0.0) or 0.0),
        "missing_mass": float(row["debug"].get("missing_target_policy_mass", 0.0) or 0.0),
        "candidate_recall_top1": float(row["debug"].get("candidate_recall_mcts_top1", 0.0) or 0.0),
        "moves": len(_decode_moves(row["history"])),
        **oracle_payload,
        **classical_payload,
    }


def _classical_stats(
    history: bytes,
    policy: dict[int, float],
    offset_q: int,
    offset_r: int,
    top_qr: tuple[int, int] | None,
    *,
    args: argparse.Namespace | None,
) -> dict[str, Any]:
    if args is None or int(getattr(args, "classical_time_ms", 0)) <= 0:
        return {
            "classical_available": False,
            "classical_top_hit": False,
            "classical_target_mass": 0.0,
            "classical_qr": None,
            "top_classical_hex_distance": 0,
            "classical_error": "disabled",
        }
    if not HAS_ENGINE or _engine is None:
        return {
            "classical_available": False,
            "classical_top_hit": False,
            "classical_target_mass": 0.0,
            "classical_qr": None,
            "top_classical_hex_distance": 0,
            "classical_error": "engine_unavailable",
        }
    try:
        game = _engine.HexGame()
        for _player, q, r in _decode_moves(history):
            game.place(int(q), int(r))
        q, r, _score, _depth, _nodes = game.classical_search(
            time_ms=int(args.classical_time_ms),
            max_depth=int(args.classical_max_depth),
            near_radius=int(args.classical_near_radius),
            noise_level=0.0,
        )
        qr = (int(q), int(r))
        idx = action_to_board_index(qr[0], qr[1], offset_q, offset_r)
        target_mass = float(policy.get(int(idx), 0.0)) if idx >= 0 else 0.0
        top_classical_distance = (
            _hex_distance(top_qr[0] - qr[0], top_qr[1] - qr[1]) if top_qr is not None else 0
        )
        return {
            "classical_available": True,
            "classical_top_hit": bool(top_qr == qr) if top_qr is not None else False,
            "classical_target_mass": target_mass,
            "classical_qr": qr,
            "top_classical_hex_distance": int(top_classical_distance),
            "classical_error": "",
        }
    except Exception as exc:
        return {
            "classical_available": False,
            "classical_top_hit": False,
            "classical_target_mass": 0.0,
            "classical_qr": None,
            "top_classical_hex_distance": 0,
            "classical_error": type(exc).__name__,
        }


def _hex_distance(dq: int, dr: int) -> int:
    return int(max(abs(int(dq)), abs(int(dr)), abs(int(dq) + int(dr))))


def _oracle_stats(
    history: bytes,
    legal: np.ndarray,
    offset_q: int,
    offset_r: int,
    top_qr: tuple[int, int] | None,
) -> dict[str, Any]:
    try:
        oracle = scan_tactical_oracle_from_history(
            history,
            [(int(q), int(r)) for q, r in legal],
            offset_q=offset_q,
            offset_r=offset_r,
        )
    except Exception as exc:
        return {
            "oracle_error": type(exc).__name__,
            "win_now_available": False,
            "win_now_top_hit": False,
            "forced_block_available": False,
            "forced_block_top_hit": False,
            "critical_available": False,
            "critical_top_hit": False,
            "oracle_status": "error",
        }
    win_now = {(int(q), int(r)) for q, r in oracle.win_now_cells}
    forced = {(int(q), int(r)) for q, r in oracle.forced_block_cells}
    open_four = {(int(q), int(r)) for q, r in oracle.open_four_cells}
    open_five = {(int(q), int(r)) for q, r in oracle.open_five_cells}
    cover = {(int(q), int(r)) for q, r in oracle.cover_cells}
    critical = win_now | forced | open_four | open_five | cover
    return {
        "oracle_error": "",
        "oracle_status": str(oracle.status),
        "win_now_available": bool(win_now),
        "win_now_top_hit": bool(top_qr in win_now) if top_qr is not None else False,
        "forced_block_available": bool(forced),
        "forced_block_top_hit": bool(top_qr in forced) if top_qr is not None else False,
        "critical_available": bool(critical),
        "critical_top_hit": bool(top_qr in critical) if top_qr is not None else False,
        "open_four_available": bool(open_four),
        "open_four_top_hit": bool(top_qr in open_four) if top_qr is not None else False,
        "open_five_available": bool(open_five),
        "open_five_top_hit": bool(top_qr in open_five) if top_qr is not None else False,
        "cover_available": bool(cover),
        "cover_top_hit": bool(top_qr in cover) if top_qr is not None else False,
    }


def _analyze_rows(rows: list[dict[str, Any]], *, args: argparse.Namespace | None = None) -> dict[str, Any]:
    stats = [_row_target_stats(row, args=args) for row in rows]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_search_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_turn_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for stat in stats:
        grouped[str(stat["source"])].append(stat)
        by_search_mode["full_search" if bool(stat["is_full_search"]) else "pcr"].append(stat)
        by_turn_bucket[_turn_bucket(int(stat["turn_index"]))].append(stat)
    return {
        "overall": _summarize(stats),
        "by_source": {source: _summarize(items) for source, items in grouped.items()},
        "by_search_mode": {mode: _summarize(items) for mode, items in by_search_mode.items()},
        "by_turn_bucket": {bucket: _summarize(items) for bucket, items in sorted(by_turn_bucket.items())},
        "source_counts": dict(Counter(str(row["source"]) for row in rows)),
        "examples": stats[:12],
    }


def _turn_bucket(turn_index: int) -> str:
    if turn_index <= 24:
        return "000_024"
    if turn_index <= 60:
        return "025_060"
    if turn_index <= 120:
        return "061_120"
    if turn_index <= 240:
        return "121_240"
    return "241_plus"


def _summarize(stats: list[dict[str, Any]]) -> dict[str, Any]:
    if not stats:
        return {"count": 0}
    arr = lambda key: np.array([float(s[key]) for s in stats], dtype=np.float64)
    return {
        "count": len(stats),
        "nonzero_mass_frac": float(np.mean(arr("mass") > 0.0)),
        "top_legal_frac": float(np.mean([bool(s["top_legal"]) for s in stats])),
        "v2_top_matches_dense_frac": float(np.mean([bool(s["v2_top_matches_dense"]) for s in stats if s["v2_top_index"] >= 0])) if any(s["v2_top_index"] >= 0 for s in stats) else 0.0,
        "full_search_frac": float(np.mean([bool(s["is_full_search"]) for s in stats])),
        "mass_mean": float(arr("mass").mean()),
        "top_prob_mean": float(arr("top_prob").mean()),
        "entropy_mean": float(arr("entropy").mean()),
        "outside_mass_mean": float(arr("outside_mass").mean()),
        "missing_mass_mean": float(arr("missing_mass").mean()),
        "candidate_recall_top1_mean": float(arr("candidate_recall_top1").mean()),
        "turn_index_mean": float(arr("turn_index").mean()),
        "legal_count_mean": float(arr("legal_count").mean()),
        "selected_root_abs_delta_mean": float(
            np.mean(
                [
                    abs(float(s["selected_action_value"]) - float(s["root_value"]))
                    for s in stats
                    if s["selected_action_value"] is not None
                ]
            )
        )
        if any(s["selected_action_value"] is not None for s in stats)
        else 0.0,
        "selected_root_negated_abs_delta_mean": float(
            np.mean(
                [
                    abs(-float(s["selected_action_value"]) - float(s["root_value"]))
                    for s in stats
                    if s["selected_action_value"] is not None
                ]
            )
        )
        if any(s["selected_action_value"] is not None for s in stats)
        else 0.0,
        "selected_root_large_delta_frac": float(
            np.mean(
                [
                    abs(float(s["selected_action_value"]) - float(s["root_value"])) > 0.5
                    for s in stats
                    if s["selected_action_value"] is not None
                ]
            )
        )
        if any(s["selected_action_value"] is not None for s in stats)
        else 0.0,
        "win_now_available_frac": _available_frac(stats, "win_now_available"),
        "win_now_top_hit_frac_when_available": _hit_frac(stats, "win_now_available", "win_now_top_hit"),
        "forced_block_available_frac": _available_frac(stats, "forced_block_available"),
        "forced_block_top_hit_frac_when_available": _hit_frac(stats, "forced_block_available", "forced_block_top_hit"),
        "critical_available_frac": _available_frac(stats, "critical_available"),
        "critical_top_hit_frac_when_available": _hit_frac(stats, "critical_available", "critical_top_hit"),
        "open_four_available_frac": _available_frac(stats, "open_four_available"),
        "open_four_top_hit_frac_when_available": _hit_frac(stats, "open_four_available", "open_four_top_hit"),
        "open_five_available_frac": _available_frac(stats, "open_five_available"),
        "open_five_top_hit_frac_when_available": _hit_frac(stats, "open_five_available", "open_five_top_hit"),
        "classical_available_frac": _available_frac(stats, "classical_available"),
        "classical_top_hit_frac_when_available": _hit_frac(stats, "classical_available", "classical_top_hit"),
        "classical_target_mass_mean_when_available": _mean_when_available(
            stats,
            "classical_available",
            "classical_target_mass",
        ),
        "top_classical_hex_distance_mean_when_available": _mean_when_available(
            stats,
            "classical_available",
            "top_classical_hex_distance",
        ),
    }


def _available_frac(stats: list[dict[str, Any]], available_key: str) -> float:
    return float(np.mean([bool(s.get(available_key, False)) for s in stats])) if stats else 0.0


def _hit_frac(stats: list[dict[str, Any]], available_key: str, hit_key: str) -> float:
    available = [s for s in stats if bool(s.get(available_key, False))]
    if not available:
        return 0.0
    return float(np.mean([bool(s.get(hit_key, False)) for s in available]))


def _mean_when_available(stats: list[dict[str, Any]], available_key: str, value_key: str) -> float:
    available = [s for s in stats if bool(s.get(available_key, False))]
    if not available:
        return 0.0
    return float(np.mean([float(s.get(value_key, 0.0) or 0.0) for s in available]))


def _overfit_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable: list[tuple[np.ndarray, np.ndarray]] = []
    for row in rows:
        stat = _row_target_stats(row)
        if stat["mass"] <= 0.0 or not stat["top_legal"]:
            continue
        tensor, _oq, _or, _legal = encode_tensor_for_history(row["history"])
        policy = np.zeros(BOARD_AREA, dtype=np.float32)
        for idx, prob in row["policy"].items():
            if 0 <= int(idx) < BOARD_AREA and float(prob) > 0.0:
                policy[int(idx)] = float(prob)
        if policy.sum() > 0:
            policy /= policy.sum()
            usable.append((tensor.astype(np.float32), policy.astype(np.float32)))
    if not usable:
        return {"passed": False, "reason": "no usable rows"}
    tensors = np.stack([x for x, _y in usable])
    policies = np.stack([y for _x, y in usable])
    target_entropy = float(
        np.mean(
            [
                -float(np.sum(row[row > 0.0] * np.log(np.maximum(row[row > 0.0], 1e-12))))
                for row in policies
            ]
        )
    )
    target_top_mass = float(np.max(policies, axis=1).mean())
    cfg = _load_config(Path(args.config)).model_copy(deep=True)
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
    first_loss = final_loss = None
    final_top1 = final_top_prob = 0.0
    for step in range(1, int(args.steps) + 1):
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss, _per_head = compute_losses(pred, targets, {"policy": 1.0, "value": 1.0}, loss_plan=loss_plan)
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
            if final_top1 >= 0.98 and final_loss is not None and final_loss <= max(target_entropy + 0.25, 0.75):
                break
    return {
        "usable_rows": len(usable),
        "device": str(device),
        "steps_run": int(step),
        "elapsed_s": time.monotonic() - started,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "target_entropy_mean": target_entropy,
        "target_top_mass_mean": target_top_mass,
        "final_top1_acc": final_top1,
        "final_target_prob": final_top_prob,
        "passed": bool(final_top1 >= 0.98 and final_loss is not None and final_loss <= max(target_entropy + 0.25, 0.75)),
    }


def _load_config(path: Path) -> Config:
    if path.suffix.lower() == ".json":
        return Config.model_validate_json(path.read_text(encoding="utf-8"))
    return load_config(path)


if __name__ == "__main__":
    raise SystemExit(main())
