"""RGSC restart service for self-play search control.

The service owns the prioritized regret buffer used by workers, restores exact
restart histories, refreshes sampled entries with EMA regret, and accepts both
played trajectory states and MCTS tree-node candidates. Tree-node candidates
must arrive with explicit caller-provided rank/regret scores so scout heuristics
cannot masquerade as regret-network scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional

import numpy as np

from hexorl.selfplay.regret_buffer import PrioritizedRegretBuffer
from hexorl.contracts.history import (
    MoveHistory,
    encode_move_history as contract_encode_move_history,
)
from hexorl.contracts.validation import ContractValidationError
from hexorl.selfplay.records import GameRecord, PositionRecord


HISTORY_STRIDE = 12


@dataclass
class RestoreResult:
    ok: bool
    game: object | None = None
    move_history: bytes = b""
    move_count: int = 0
    current_player: int = 0
    placements_remaining: int = 1
    reason: str = ""


@dataclass
class RGSCRestartDecision:
    attempted: bool
    used: bool
    reason: str
    game: object | None = None
    move_history: bytes = b""
    move_count: int = 0
    entry_index: int | None = None
    entry_id: int | None = None
    regret: float = 0.0
    rank_score: float = 0.0


@dataclass(frozen=True)
class RGSCCandidate:
    move_history: bytes
    rank_score: float
    regret: float
    game_id: int = 0
    source: str = "trajectory_ranked_regret"


def decode_move_history(move_history: bytes) -> list[tuple[int, int, int]]:
    """Decode compact `(player, q, r)` little-endian i32 move history."""
    return list(MoveHistory.decode(move_history, source="rust").rows)


def encode_move_history(moves: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...]) -> bytes:
    """Encode `(player, q, r)` triples into compact little-endian i32 history."""
    return contract_encode_move_history(moves)


def _attr_value(obj: object, name: str, default: int | bool | None = None):
    value = getattr(obj, name, default)
    return value() if callable(value) else value


def restore_game_from_history(
    move_history: bytes,
    game_factory: Callable[[], object],
    *,
    max_game_moves: int | None = None,
) -> RestoreResult:
    """Replay a compact history into a fresh game and validate turn phase."""
    try:
        moves = decode_move_history(move_history)
    except (ValueError, ContractValidationError) as exc:
        reason = str(exc)
        if "invalid player order" in reason:
            reason = f"player_mismatch: {reason}"
        return RestoreResult(ok=False, reason=reason)
    if max_game_moves is not None and len(moves) >= max_game_moves:
        return RestoreResult(ok=False, reason="history_at_or_past_move_cap")

    game = game_factory()
    for idx, (player, q, r) in enumerate(moves):
        if _attr_value(game, "is_over", False):
            return RestoreResult(ok=False, reason="history_continues_after_terminal")
        current_player = int(_attr_value(game, "current_player", player))
        if player != current_player:
            return RestoreResult(
                ok=False,
                reason=f"player_mismatch_at_{idx}: expected {current_player}, got {player}",
            )
        try:
            game.place(q, r)
        except Exception as exc:
            return RestoreResult(ok=False, reason=f"illegal_history_at_{idx}: {exc}")

    if _attr_value(game, "is_over", False):
        return RestoreResult(ok=False, reason="terminal_history")
    return RestoreResult(
        ok=True,
        game=game,
        move_history=move_history,
        move_count=len(moves),
        current_player=int(_attr_value(game, "current_player", 0)),
        placements_remaining=int(_attr_value(game, "placements_remaining", 1)),
    )


class RGSCRestartService:
    """Worker-owned PRB restart service with explicit, auditable metrics."""

    def __init__(
        self,
        *,
        beta: float = 0.0,
        capacity: int = 100,
        ema_alpha: float = 0.5,
        sampling_temperature: float = 0.1,
        seed: int = 0,
        enabled: bool = True,
    ):
        self.beta = float(beta)
        self.enabled = bool(enabled) and capacity > 0 and self.beta > 0.0
        self.prb = PrioritizedRegretBuffer(
            capacity=max(1, int(capacity)),
            ema_alpha=float(ema_alpha),
            sampling_temperature=float(sampling_temperature),
        )
        self.rng = np.random.RandomState(int(seed))
        self.restart_attempts = 0
        self.restart_successes = 0
        self.restart_rejections = 0
        self.insertions = 0
        self.refreshes = 0
        self.tree_node_insertions = 0
        self.last_ema_delta = 0.0
        self.last_staleness = 0.0

    def maybe_restart(
        self,
        game_factory: Callable[[], object],
        *,
        max_game_moves: int,
    ) -> RGSCRestartDecision:
        if not self.enabled:
            return RGSCRestartDecision(False, False, "disabled")
        if self.rng.random_sample() >= self.beta:
            return RGSCRestartDecision(False, False, "not_sampled")
        self.restart_attempts += 1
        sampled = self.prb.sample_with_index(self.rng)
        if sampled is None:
            self.restart_rejections += 1
            return RGSCRestartDecision(True, False, "prb_empty")
        entry_index, entry = sampled
        if not entry.move_history:
            self.restart_rejections += 1
            return RGSCRestartDecision(True, False, "empty_history", entry_index=entry_index, entry_id=entry.entry_id)
        restored = restore_game_from_history(
            entry.move_history,
            game_factory,
            max_game_moves=max_game_moves,
        )
        if not restored.ok:
            self.restart_rejections += 1
            return RGSCRestartDecision(
                True,
                False,
                restored.reason,
                entry_index=entry_index,
                entry_id=entry.entry_id,
                regret=float(entry.regret),
                rank_score=float(entry.rank_score),
            )
        self.restart_successes += 1
        return RGSCRestartDecision(
            True,
            True,
            "ok",
            game=restored.game,
            move_history=restored.move_history,
            move_count=restored.move_count,
            entry_index=entry_index,
            entry_id=entry.entry_id,
            regret=float(entry.regret),
            rank_score=float(entry.rank_score),
        )

    def observe_game(
        self,
        record: GameRecord,
        *,
        restart_entry_index: int | None = None,
    ) -> bool:
        """Refresh sampled entries or insert the rank-network selected candidate."""
        if not self.enabled:
            return False
        if restart_entry_index is not None:
            if record.positions:
                restart_pos = record.positions[0]
                if float(getattr(restart_pos, "regret_weight", 0.0)) > 0.0:
                    before = self.prb.get_entries()
                    sampled = before[restart_entry_index] if 0 <= restart_entry_index < len(before) else None
                    self.last_ema_delta = self.prb.update_regret(
                        restart_entry_index,
                        float(restart_pos.regret_value),
                    )
                    if sampled is not None:
                        self.last_staleness = float(max(0, len(record.positions) - 1))
                    self.refreshes += 1
            return False

        candidate = self._selected_ranked_candidate(record)
        if candidate is None:
            return False
        inserted = self.prb.add(
            candidate.move_history,
            regret=float(candidate.regret),
            rank_score=float(candidate.rank_score),
            game_id=int(candidate.game_id),
            source=str(candidate.source),
        )
        if inserted:
            self.insertions += 1
            if str(candidate.source).startswith("mcts_tree_node"):
                self.tree_node_insertions += 1
        return inserted

    def observe_tree_node_candidates(
        self,
        candidates: list[tuple[bytes, float, float]],
        *,
        game_id: int = 0,
        score_source: str = "mcts_tree_node_scored_candidate",
    ) -> int:
        """Insert scored MCTS tree-node candidates into the PRB.

        Each candidate is `(move_history, rank_score, regret_estimate)`. Empty
        histories, non-finite scores, and non-positive estimates are rejected.
        The caller is responsible for producing the scores from the active
        regret rank/value path or another explicitly labeled source.
        """
        if not self.enabled:
            return 0
        inserted = 0
        for move_history, rank_score, regret_estimate in candidates:
            rank = float(rank_score)
            regret = float(regret_estimate)
            if not move_history or not math.isfinite(rank) or not math.isfinite(regret):
                continue
            if regret <= 0.0:
                continue
            if self.prb.add(
                bytes(move_history),
                regret=regret,
                rank_score=rank,
                game_id=int(game_id),
                source=str(score_source),
            ):
                inserted += 1
        self.tree_node_insertions += inserted
        self.insertions += inserted
        return inserted

    @staticmethod
    def _selected_ranked_candidate(record: GameRecord) -> Optional[RGSCCandidate]:
        raw_candidates = getattr(record, "rgsc_ranked_candidates", ()) or ()
        candidates: list[RGSCCandidate] = []
        for raw in raw_candidates:
            if isinstance(raw, RGSCCandidate):
                candidate = raw
            else:
                try:
                    candidate = RGSCCandidate(
                        move_history=bytes(raw["move_history"]),
                        rank_score=float(raw["rank_score"]),
                        regret=float(raw["regret"]),
                        game_id=int(raw.get("game_id", getattr(record, "game_id", 0))),
                        source=str(raw.get("source", "ranked_regret_candidate")),
                    )
                except Exception:
                    continue
            if (
                candidate.move_history
                and math.isfinite(float(candidate.rank_score))
                and math.isfinite(float(candidate.regret))
                and float(candidate.regret) > 0.0
            ):
                candidates.append(candidate)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda candidate: (
                float(candidate.rank_score),
                float(candidate.regret),
                len(candidate.move_history),
            ),
        )

    @property
    def metrics(self) -> dict[str, float]:
        return {
            "rgsc_prb_size": float(len(self.prb)),
            "rgsc_restart_attempts": float(self.restart_attempts),
            "rgsc_restart_successes": float(self.restart_successes),
            "rgsc_restart_rejections": float(self.restart_rejections),
            "rgsc_prb_insertions": float(self.insertions),
            "rgsc_prb_refreshes": float(self.refreshes),
            "rgsc_tree_node_insertions": float(self.tree_node_insertions),
            "rgsc_last_ema_delta": float(self.last_ema_delta),
            "rgsc_last_staleness": float(self.last_staleness),
        }

    def snapshot_entries(self, limit: int = 32) -> list[dict[str, float | int | str]]:
        entries = sorted(
            self.prb.get_entries(),
            key=lambda entry: (float(entry.regret), float(entry.rank_score)),
            reverse=True,
        )
        return [
            {
                "entry_id": int(entry.entry_id),
                "game_id": int(entry.game_id),
                "move_count": int(len(entry.move_history) // HISTORY_STRIDE),
                "ema_regret": float(entry.regret),
                "observed_regret": float(entry.observed_regret),
                "rank_score": float(entry.rank_score),
                "refresh_count": int(entry.refresh_count),
                "inserted_step": int(entry.inserted_step),
                "last_sampled_step": int(entry.last_sampled_step),
                "last_updated_step": int(entry.last_updated_step),
                "source": str(entry.source),
            }
            for entry in entries[: max(0, int(limit))]
        ]
