"""Shared dashboard route helpers."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from hexorl.axis_policy.core import AxisPolicyInput
from hexorl.contracts.history import MoveHistory
from hexorl.dashboard.contract_inspector import ContractInspector
from hexorl.dashboard.db import DashboardStore
from hexorl.dashboard.play import session_payload
from hexorl.dashboard.replay import get_replay_position, position_payload
from hexorl.dashboard.services.suite import suite_store_for_run, suite_trial_dirs
from hexorl.selfplay.records import BOARD_SIZE


def game_summary(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload_json", {}) or {}
    return {
        "game_id": row["game_id"],
        "run_id": row["run_id"],
        "trial_id": row.get("trial_id") or row["run_id"],
        "external_game_id": row["external_game_id"],
        "source": row["source"],
        "epoch": row["epoch"],
        "outcome": row["outcome"],
        "move_count": row["move_count"],
        "created_at": row["created_at"],
        "terminal_reason": payload.get("terminal_reason", ""),
        "truncated": bool(payload.get("truncated", False)),
        "positions": payload.get("positions"),
        "payload": payload,
    }


def game_row_for_request(
    store: DashboardStore,
    run_root: Path | None,
    game_id: int,
    run_id: str | None,
) -> dict[str, Any] | None:
    if run_id:
        trial_store = suite_store_for_run(run_root, run_id)
        if trial_store is not None:
            rows = trial_store.rows("SELECT * FROM games WHERE game_id=?", (game_id,))
            if rows:
                rows[0]["source_db"] = str(trial_store.path)
                return rows[0]
        rows = store.rows("SELECT * FROM games WHERE game_id=? AND run_id=?", (game_id, run_id))
        return rows[0] if rows else None
    rows = store.rows("SELECT * FROM games WHERE game_id=?", (game_id,))
    if rows:
        return rows[0]
    if run_root is None:
        return None
    for trial_dir in suite_trial_dirs(run_root):
        db = trial_dir / "dashboard.sqlite3"
        try:
            trial_store = DashboardStore(db)
            rows = trial_store.rows("SELECT * FROM games WHERE game_id=?", (game_id,))
        except Exception:
            continue
        if rows:
            rows[0]["trial_id"] = trial_dir.name
            rows[0]["source_db"] = str(db)
            return rows[0]
    return None


def parse_policy_target_v2(rows: list[Any]) -> list[tuple[int, int, float]]:
    parsed: list[tuple[int, int, float]] = []
    for row in rows or []:
        if isinstance(row, dict):
            q = row.get("q")
            r = row.get("r")
            prob = row.get("prob", row.get("p", row.get("weight", 0.0)))
        else:
            if len(row) != 3:
                raise HTTPException(400, "policy_target_v2 rows must be [q, r, probability]")
            q, r, prob = row
        prob_f = float(prob)
        if prob_f > 0.0:
            parsed.append((int(q), int(r), prob_f))
    return parsed


def parse_pair_policy_target_v2(rows: list[Any]) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    parsed: list[tuple[tuple[int, int], tuple[int, int], float]] = []
    for row in rows or []:
        if isinstance(row, dict):
            first_raw = row.get("first")
            second_raw = row.get("second")
            prob = row.get("prob", row.get("p", row.get("weight", 0.0)))
        else:
            if len(row) != 3:
                raise HTTPException(400, "pair_policy_target_v2 rows must be [[q1, r1], [q2, r2], probability]")
            first_raw, second_raw, prob = row
        first = (
            (int(first_raw["q"]), int(first_raw["r"]))
            if isinstance(first_raw, dict)
            else (int(first_raw[0]), int(first_raw[1]))
        )
        second = (
            (int(second_raw["q"]), int(second_raw["r"]))
            if isinstance(second_raw, dict)
            else (int(second_raw[0]), int(second_raw[1]))
        )
        prob_f = float(prob)
        if prob_f > 0.0:
            parsed.append((first, second, prob_f))
    return parsed


def axis_input_from_request(store: DashboardStore, req: Any) -> AxisPolicyInput:
    if req.position:
        offset_q, offset_r = fit_axis_offsets(
            list(req.position.get("stones", [])),
            list(req.position.get("legal_moves", [])),
            int(req.position.get("offset_q", -16)),
            int(req.position.get("offset_r", -16)),
        )
        return AxisPolicyInput(
            stones=list(req.position.get("stones", [])),
            legal_moves=list(req.position.get("legal_moves", [])),
            current_player=int(req.position.get("current_player", 0)),
            offset_q=offset_q,
            offset_r=offset_r,
            metadata={
                "placements_remaining": int(req.position.get("placements_remaining", 2)),
                **dict(req.position.get("metadata", {})),
            },
        )
    if req.session_id:
        pos = session_payload(store, req.session_id)["position"]
    else:
        history = b""
        if req.game_id is not None:
            rows = store.rows("SELECT final_history_b64 FROM games WHERE game_id=?", (req.game_id,))
            if not rows:
                raise HTTPException(404, f"Game not found: {req.game_id}")
            history = rows[0]["final_history_b64"]
        elif req.history_b64:
            history = base64.b64decode(req.history_b64)
        pos = position_payload(get_replay_position(history, turn_index=req.turn_index))
    offset_q, offset_r = fit_axis_offsets(
        pos["stones"],
        pos["legal_moves"],
        int(pos["encoding"].get("offset_q", -16)),
        int(pos["encoding"].get("offset_r", -16)),
    )
    return AxisPolicyInput(
        stones=pos["stones"],
        legal_moves=pos["legal_moves"],
        current_player=int(pos["current_player"]),
        offset_q=offset_q,
        offset_r=offset_r,
        metadata={
            "source": "dashboard",
            "turn_index": int(pos.get("turn_index", 0)),
            "placements_remaining": int(pos.get("placements_remaining", 2)),
        },
    )


def d6_contract_payload(history: bytes, position: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    axis_input = AxisPolicyInput(
        stones=list(position.get("stones", [])),
        legal_moves=list(position.get("legal_moves", [])),
        current_player=int(position.get("current_player", 0)),
        offset_q=int(position.get("encoding", {}).get("offset_q", -16)),
        offset_r=int(position.get("encoding", {}).get("offset_r", -16)),
        metadata={
            "source": "dashboard_d6_debug",
            "placements_remaining": int(position.get("placements_remaining", 1)),
            "history_b64": base64.b64encode(history).decode("ascii"),
        },
    )
    from hexorl.axis_policy.registry import evaluate_all

    inspector = ContractInspector()
    return {
        "dense_legal_mask": {
            "offset_q": int(position.get("encoding", {}).get("offset_q", -16)),
            "offset_r": int(position.get("encoding", {}).get("offset_r", -16)),
            "legal_indices": list(position.get("encoding", {}).get("legal_mask", []))[:128],
            "legal_count": len(position.get("legal_moves", [])),
        },
        "sparse_candidates": inspector.inspect("candidates", history=history),
        "pair_rows": inspector.inspect("pairs", history=history),
        "axis": {"prototype_count": len(evaluate_all(axis_input, {})), "results": evaluate_all(axis_input, {})[:8]},
        "graph_targets": {
            "legal_count": int(graph["legal_count"]),
            "pair_count": int(graph["pair_count"]),
            "opp_legal_count": int(graph["opp_legal_count"]),
            "token_counts": graph["token_counts"],
            "target_masses": graph.get("target_masses", {}),
        },
    }


def last_history_qr(history: bytes) -> tuple[int, int] | None:
    if len(history) < 12:
        return None
    moves = MoveHistory.decode(history, source="rust").rows
    if not moves:
        return None
    _player, q, r = moves[-1]
    return q, r


def fit_axis_offsets(stones: list[dict[str, Any]], legal_moves: list[dict[str, Any]], offset_q: int, offset_r: int) -> tuple[int, int]:
    primary = [(int(m["q"]), int(m["r"])) for m in legal_moves if "q" in m and "r" in m]
    secondary = [(int(s["q"]), int(s["r"])) for s in stones if "q" in s and "r" in s]
    if _all_inside(primary or secondary, offset_q, offset_r):
        return offset_q, offset_r
    coords = primary or secondary
    return _best_offset_pair(coords, offset_q, offset_r) if coords else (offset_q, offset_r)


def _all_inside(coords: list[tuple[int, int]], offset_q: int, offset_r: int) -> bool:
    return all(offset_q <= q < offset_q + BOARD_SIZE and offset_r <= r < offset_r + BOARD_SIZE for q, r in coords)


def _best_axis_start(values: list[int], current: int) -> int:
    if not values:
        return current
    if max(values) - min(values) < BOARD_SIZE:
        return int(round((min(values) + max(values) - BOARD_SIZE + 1) / 2))
    starts = sorted(set(values + [value - BOARD_SIZE + 1 for value in values]))
    median = sorted(values)[len(values) // 2]
    return max(starts, key=lambda start: (sum(1 for value in values if start <= value < start + BOARD_SIZE), -abs((start + (BOARD_SIZE - 1) / 2) - median)))


def _best_offset_pair(coords: list[tuple[int, int]], current_q: int, current_r: int) -> tuple[int, int]:
    q_values = [q for q, _r in coords]
    r_values = [r for _q, r in coords]
    if max(q_values) - min(q_values) < BOARD_SIZE and max(r_values) - min(r_values) < BOARD_SIZE:
        return _best_axis_start(q_values, current_q), _best_axis_start(r_values, current_r)
    q_starts = sorted(set(q_values + [q - BOARD_SIZE + 1 for q in q_values]))
    r_starts = sorted(set(r_values + [r - BOARD_SIZE + 1 for r in r_values]))
    q_median = sorted(q_values)[len(q_values) // 2]
    r_median = sorted(r_values)[len(r_values) // 2]
    return max(
        ((q_start, r_start) for q_start in q_starts for r_start in r_starts),
        key=lambda start: (
            sum(1 for q, r in coords if start[0] <= q < start[0] + BOARD_SIZE and start[1] <= r < start[1] + BOARD_SIZE),
            -abs((start[0] + (BOARD_SIZE - 1) / 2) - q_median) - abs((start[1] + (BOARD_SIZE - 1) / 2) - r_median),
        ),
    )

