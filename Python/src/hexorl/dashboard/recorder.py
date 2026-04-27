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

    def __post_init__(self) -> None:
        if self.jsonl_path is not None:
            self.jsonl_path = Path(self.jsonl_path)
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.store.upsert_run(self.run_id)

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
        return self.store.record_event(
            self.run_id,
            event_type,
            payload_dict,
            phase=phase,
            epoch=epoch,
            global_step=global_step,
        )

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
        return self.store.record_metric(
            self.run_id,
            epoch=epoch,
            global_step=global_step,
            phase=phase,
            metrics=payload,
        )

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
                **dict(payload or {}),
            },
            positions=[
                {
                    "turn_index": pos.turn_index,
                    "player": pos.player,
                    "move_history": pos.move_history,
                    "root_value": pos.root_value,
                    "policy_target": pos.policy_target,
                    "debug": {
                        "outcome": pos.outcome,
                        "is_full_search": pos.is_full_search,
                        "lookahead_values": pos.lookahead_values,
                        "regret_rank": pos.regret_rank,
                        "regret_value": pos.regret_value,
                        "axis_label": pos.axis_label,
                        "moves_left": pos.moves_left,
                        "value_weight": pos.value_weight,
                    },
                }
                for pos in record.positions
            ],
        )
        self.event(
            "game_recorded",
            {"game_id": record.game_id, "game_row_id": game_row_id, "source": source},
            phase=source,
            epoch=epoch,
        )
        return game_row_id

    def _append_jsonl(self, row: Mapping[str, Any]) -> None:
        if self.jsonl_path is None:
            return
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
