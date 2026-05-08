"""Run telemetry recorder shared by training, eval, and dashboard tooling."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from hexorl.dashboard.checkpoints import index_checkpoint
from hexorl.dashboard.db import DashboardStore
from hexorl.selfplay.records import GameRecord


@dataclass
class RunRecorder:
    """Append-only JSONL plus SQLite recorder.

    The recorder is deliberately independent of the web app.  Training and
    evaluation can emit facts without importing FastAPI or frontend code.
    """

    store: DashboardStore
    run_id: str
    jsonl_path: Path | None = None
    _opened: bool = field(default=False, init=False)
    _write_failures: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.jsonl_path is not None:
            self.jsonl_path = Path(self.jsonl_path)
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.store.upsert_run(self.run_id)
        except Exception as exc:
            self._recording_error("upsert_run", exc)

    @classmethod
    def for_run_dir(cls, run_dir: Path | str, run_id: str | None = None) -> "RunRecorder":
        run_dir = Path(run_dir)
        run_id = run_id or run_dir.name
        return cls(
            DashboardStore(run_dir / "dashboard.sqlite3"),
            run_id,
            run_dir / "events.jsonl",
        )

    def event(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        phase: str | None = None,
        epoch: int | None = None,
        global_step: int | None = None,
    ) -> int:
        payload_dict = dict(payload or {})
        row = {
            "schema_version": 1,
            "time": time.time(),
            "run_id": self.run_id,
            "event_type": event_type,
            "phase": phase,
            "epoch": epoch,
            "global_step": global_step,
            "payload": payload_dict,
        }
        self._append_jsonl(row)
        try:
            return self.store.record_event(
                self.run_id,
                event_type,
                payload_dict,
                phase=phase,
                epoch=epoch,
                global_step=global_step,
            )
        except Exception as exc:
            return self._recording_error("record_event", exc, phase=phase, epoch=epoch, global_step=global_step)

    def metric(
        self,
        metrics: Mapping[str, Any],
        *,
        phase: str,
        epoch: int | None = None,
        global_step: int | None = None,
    ) -> int:
        payload = dict(metrics)
        self._append_jsonl(
            {
                "schema_version": 1,
                "time": time.time(),
                "run_id": self.run_id,
                "event_type": "metric",
                "phase": phase,
                "epoch": epoch,
                "global_step": global_step,
                "payload": payload,
            }
        )
        try:
            return self.store.record_metric(
                self.run_id,
                epoch=epoch,
                global_step=global_step,
                phase=phase,
                metrics=payload,
            )
        except Exception as exc:
            return self._recording_error("record_metric", exc, phase=phase, epoch=epoch, global_step=global_step)

    def checkpoint(
        self,
        path: Path | str,
        payload: Mapping[str, Any] | None = None,
        *,
        epoch: int | None = None,
        global_step: int | None = None,
    ) -> int:
        path = Path(path)
        checkpoint_id: int | None = None
        try:
            checkpoint_id = index_checkpoint(path, self.store, run_id=self.run_id).checkpoint_id
        except Exception as exc:
            payload = {"checkpoint_index_error": str(exc), **dict(payload or {})}

        event_id = self.event(
            "checkpoint",
            {"path": str(path), "checkpoint_id": checkpoint_id, **dict(payload or {})},
            phase="checkpoint",
            epoch=epoch,
            global_step=global_step,
        )
        return checkpoint_id or event_id

    def game(
        self,
        record: GameRecord,
        *,
        source: str,
        epoch: int | None = None,
        checkpoint_id: int | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> int:
        try:
            game_row_id = self.store.insert_game_with_positions(
                run_id=self.run_id,
                game_id=record.game_id,
                source=source,
                final_move_history=record.final_move_history,
                outcome=record.outcome,
                epoch=epoch,
                checkpoint_id=checkpoint_id,
                payload={
                    "game_length": record.game_length,
                    "positions": len(record.positions),
                    "truncated": getattr(record, "truncated", False),
                    "terminal_reason": getattr(record, "terminal_reason", "unknown"),
                    "rgsc_metrics": dict(getattr(record, "rgsc_metrics", {}) or {}),
                    "rgsc_prb_snapshot": list(getattr(record, "rgsc_prb_snapshot", []) or []),
                    **dict(payload or {}),
                },
                positions=[
                    {
                        "turn_index": pos.turn_index,
                        "player": pos.player,
                        "root_value": pos.root_value,
                        "policy_target": pos.policy_target,
                        "debug": {
                            "outcome": pos.outcome,
                            "is_full_search": pos.is_full_search,
                            "lookahead_values": pos.lookahead_values,
                            "selected_action_value": pos.selected_action_value,
                            "regret_rank": pos.regret_rank,
                            "regret_value": pos.regret_value,
                            "regret_weight": pos.regret_weight,
                            "axis_label": pos.axis_label,
                            "moves_left": pos.moves_left,
                            "value_weight": pos.value_weight,
                            "policy_weight": 1.0 if pos.is_full_search else 0.0,
                            "opp_policy_weight": pos.opp_policy_weight,
                            "policy_target_v2": pos.policy_target_v2,
                            "opp_policy_target_v2": pos.opp_policy_target_v2,
                            "opp_policy_legal_v2_count": len(pos.opp_policy_legal_v2 or []),
                            "pair_policy_target_v2": pos.pair_policy_target_v2,
                            "target_policy_mass_outside_window": pos.target_policy_mass_outside_window,
                            "missing_target_policy_mass": pos.missing_target_policy_mass,
                            "candidate_recall_mcts_top1": pos.candidate_recall_mcts_top1,
                            "candidate_recall_mcts_top4": pos.candidate_recall_mcts_top4,
                            "candidate_recall_mcts_top8": pos.candidate_recall_mcts_top8,
                            "candidate_recall_winning_move": pos.candidate_recall_winning_move,
                            "candidate_recall_forced_block": pos.candidate_recall_forced_block,
                            "candidate_recall_two_placement_cover": pos.candidate_recall_two_placement_cover,
                            "candidate_discovery_top1": pos.candidate_discovery_top1,
                            "candidate_discovery_top4": pos.candidate_discovery_top4,
                            "candidate_discovery_top8": pos.candidate_discovery_top8,
                            "candidate_discovery_winning_move": pos.candidate_discovery_winning_move,
                            "candidate_discovery_forced_block": pos.candidate_discovery_forced_block,
                            "candidate_discovery_two_placement_cover": pos.candidate_discovery_two_placement_cover,
                            "candidate_discovery_open_four": pos.candidate_discovery_open_four,
                            "candidate_discovery_open_five": pos.candidate_discovery_open_five,
                            "candidate_critical_count": pos.candidate_critical_count,
                            "candidate_critical_overflow_count": pos.candidate_critical_overflow_count,
                            "candidate_critical_overflow_examples": pos.candidate_critical_overflow_examples,
                            "sparse_prior_stage": pos.sparse_prior_stage,
                            "sparse_prior_root_candidate_count": pos.sparse_prior_root_candidate_count,
                            "sparse_prior_leaf_candidate_count": pos.sparse_prior_leaf_candidate_count,
                            "sparse_prior_root_hit_frac": pos.sparse_prior_root_hit_frac,
                            "sparse_prior_leaf_hit_frac": pos.sparse_prior_leaf_hit_frac,
                            "fallback_prior_use": pos.fallback_prior_use,
                            "fallback_prior_use_on_mcts_top1": pos.fallback_prior_use_on_mcts_top1,
                            "fallback_prior_use_on_mcts_top4": pos.fallback_prior_use_on_mcts_top4,
                            "fallback_prior_use_on_mcts_top8": pos.fallback_prior_use_on_mcts_top8,
                            "sparse_vs_dense_disagreement": pos.sparse_vs_dense_disagreement,
                            "sparse_prior_forward_ms": pos.sparse_prior_forward_ms,
                            "sparse_prior_candidate_build_ms": pos.sparse_prior_candidate_build_ms,
                            "pair_prior_candidate_count": pos.pair_prior_candidate_count,
                            "pair_prior_hit_frac": pos.pair_prior_hit_frac,
                            "pair_fallback_prior_use": pos.pair_fallback_prior_use,
                            "pair_fallback_prior_use_on_mcts_top1": pos.pair_fallback_prior_use_on_mcts_top1,
                            "pair_fallback_prior_use_on_mcts_top4": pos.pair_fallback_prior_use_on_mcts_top4,
                            "pair_fallback_prior_use_on_mcts_top8": pos.pair_fallback_prior_use_on_mcts_top8,
                        },
                    }
                    for pos in record.positions
                ],
            )
        except Exception as exc:
            return self._recording_error("insert_game_with_positions", exc, phase=source, epoch=epoch)
        self.event(
            "game_recorded",
            {"game_id": record.game_id, "game_row_id": game_row_id, "source": source},
            phase=source,
            epoch=epoch,
        )
        return game_row_id

    def _recording_error(
        self,
        operation: str,
        exc: Exception,
        *,
        phase: str | None = None,
        epoch: int | None = None,
        global_step: int | None = None,
    ) -> int:
        self._write_failures += 1
        try:
            self._append_jsonl(
                {
                    "schema_version": 1,
                    "time": time.time(),
                    "run_id": self.run_id,
                    "event_type": "dashboard_write_failed",
                    "phase": phase,
                    "epoch": epoch,
                    "global_step": global_step,
                    "payload": {
                        "operation": operation,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "failure_count": self._write_failures,
                        "sqlite_path": str(self.store.path),
                    },
                }
            )
        except Exception:
            pass
        return -1

    def _append_jsonl(self, row: Mapping[str, Any]) -> None:
        if self.jsonl_path is None:
            return
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
