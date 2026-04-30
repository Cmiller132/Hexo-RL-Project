"""Versioned canonical replay records and byte codec."""

from __future__ import annotations

import json
import math
import struct
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from hexorl.contracts.history import MoveHistory
from hexorl.contracts.identity import stable_digest
from hexorl.contracts.validation import ContractValidationError
from hexorl.engine.legal import LegalTableProvider
from hexorl.selfplay.records import GameRecord, PositionRecord, action_to_board_index


REPLAY_CODEC_MAGIC = b"HXR7"
REPLAY_RECORD_SCHEMA_VERSION = 1
REPLAY_WRITER_VERSION = "phase07.replay.codec.v1"
LEGAL_SOURCE_MARKER = "rust:legal"
HISTORY_SOURCE_MARKER = "rust"


class ReplayCodecError(ContractValidationError):
    """Replay codec/storage/projector boundary error with ownership context."""


@dataclass(frozen=True)
class ReplayPositionRecord:
    schema_version: int
    game_id: int
    position_index: int
    position_id: str
    move_history: bytes
    history_source: str
    history_schema_version: int
    history_hash: str
    compact_history_row_count: int
    legal_rows: tuple[tuple[int, int], ...]
    legal_dense_indices: tuple[int, ...]
    legal_source: str
    legal_schema_version: int
    legal_table_hash: str
    reconstructed_legal_table_hash: str
    ffi_protocol_version: str
    current_player: int
    placements_remaining: int
    policy_target_v2: tuple[tuple[int, int, float], ...]
    policy_target_dense: tuple[tuple[int, float], ...]
    policy_target_global_row_identity: str
    value_target: float
    root_value: float
    player: int
    outcome: float
    is_full_search: bool
    turn_index: int
    selected_action_value: float | None
    lookahead_values: tuple[float, ...] = ()
    opp_policy_target_v2: tuple[tuple[int, int, float], ...] = ()
    opp_policy_dense: tuple[tuple[int, float], ...] = ()
    opp_policy_legal_v2: tuple[tuple[int, int], ...] = ()
    opp_policy_weight: float = 0.0
    pair_policy_target_v2: tuple[tuple[tuple[int, int], tuple[int, int], float], ...] = ()
    pair_policy_complete: bool = False
    pair_known_first: tuple[int, int] | None = None
    pair_completeness: str = "none"
    target_policy_mass_outside_window: float = 0.0
    missing_target_policy_mass: float = 0.0
    candidate_critical_count: int = 0
    candidate_critical_overflow_count: int = 0
    candidate_critical_overflow_examples: tuple[tuple[int, int], ...] = ()
    regret_rank: float = 0.0
    regret_value: float = 0.0
    regret_weight: float = 0.0
    axis_label: int = -1
    moves_left: float = 0.0
    value_weight: float = 1.0
    policy_weight: float = 1.0
    contract_trace: Mapping[str, Any] = field(default_factory=dict)
    record_hash: str = ""

    def __post_init__(self) -> None:
        _validate_position(self)
        digest = _position_hash(self)
        if self.record_hash and self.record_hash != digest:
            raise ReplayCodecError(
                f"replay position hash mismatch for game={self.game_id} index={self.position_index}",
                owner="replay.codec",
            )
        object.__setattr__(self, "record_hash", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "game_id": self.game_id,
            "position_index": self.position_index,
            "position_id": self.position_id,
            "move_history_hex": self.move_history.hex(),
            "history_source": self.history_source,
            "history_schema_version": self.history_schema_version,
            "history_hash": self.history_hash,
            "compact_history_row_count": self.compact_history_row_count,
            "legal_rows": [list(row) for row in self.legal_rows],
            "legal_dense_indices": list(self.legal_dense_indices),
            "legal_source": self.legal_source,
            "legal_schema_version": self.legal_schema_version,
            "legal_table_hash": self.legal_table_hash,
            "reconstructed_legal_table_hash": self.reconstructed_legal_table_hash,
            "ffi_protocol_version": self.ffi_protocol_version,
            "current_player": self.current_player,
            "placements_remaining": self.placements_remaining,
            "policy_target_v2": [list(row) for row in self.policy_target_v2],
            "policy_target_dense": [list(row) for row in self.policy_target_dense],
            "policy_target_global_row_identity": self.policy_target_global_row_identity,
            "value_target": self.value_target,
            "root_value": self.root_value,
            "player": self.player,
            "outcome": self.outcome,
            "is_full_search": self.is_full_search,
            "turn_index": self.turn_index,
            "selected_action_value": self.selected_action_value,
            "lookahead_values": list(self.lookahead_values),
            "opp_policy_target_v2": [list(row) for row in self.opp_policy_target_v2],
            "opp_policy_dense": [list(row) for row in self.opp_policy_dense],
            "opp_policy_legal_v2": [list(row) for row in self.opp_policy_legal_v2],
            "opp_policy_weight": self.opp_policy_weight,
            "pair_policy_target_v2": [
                [list(first), list(second), prob]
                for first, second, prob in self.pair_policy_target_v2
            ],
            "pair_policy_complete": self.pair_policy_complete,
            "pair_known_first": list(self.pair_known_first) if self.pair_known_first is not None else None,
            "pair_completeness": self.pair_completeness,
            "target_policy_mass_outside_window": self.target_policy_mass_outside_window,
            "missing_target_policy_mass": self.missing_target_policy_mass,
            "candidate_critical_count": self.candidate_critical_count,
            "candidate_critical_overflow_count": self.candidate_critical_overflow_count,
            "candidate_critical_overflow_examples": [list(row) for row in self.candidate_critical_overflow_examples],
            "regret_rank": self.regret_rank,
            "regret_value": self.regret_value,
            "regret_weight": self.regret_weight,
            "axis_label": self.axis_label,
            "moves_left": self.moves_left,
            "value_weight": self.value_weight,
            "policy_weight": self.policy_weight,
            "contract_trace": dict(self.contract_trace),
            "record_hash": self.record_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReplayPositionRecord":
        pairs = []
        for first, second, prob in payload.get("pair_policy_target_v2", []):
            pairs.append(((int(first[0]), int(first[1])), (int(second[0]), int(second[1])), float(prob)))
        known = payload.get("pair_known_first")
        return cls(
            schema_version=int(payload["schema_version"]),
            game_id=int(payload["game_id"]),
            position_index=int(payload["position_index"]),
            position_id=str(payload["position_id"]),
            move_history=bytes.fromhex(str(payload["move_history_hex"])),
            history_source=str(payload["history_source"]),
            history_schema_version=int(payload["history_schema_version"]),
            history_hash=str(payload["history_hash"]),
            compact_history_row_count=int(payload["compact_history_row_count"]),
            legal_rows=tuple((int(q), int(r)) for q, r in payload["legal_rows"]),
            legal_dense_indices=tuple(int(x) for x in payload["legal_dense_indices"]),
            legal_source=str(payload["legal_source"]),
            legal_schema_version=int(payload["legal_schema_version"]),
            legal_table_hash=str(payload["legal_table_hash"]),
            reconstructed_legal_table_hash=str(payload["reconstructed_legal_table_hash"]),
            ffi_protocol_version=str(payload["ffi_protocol_version"]),
            current_player=int(payload["current_player"]),
            placements_remaining=int(payload["placements_remaining"]),
            policy_target_v2=tuple((int(q), int(r), float(p)) for q, r, p in payload["policy_target_v2"]),
            policy_target_dense=tuple((int(idx), float(p)) for idx, p in payload["policy_target_dense"]),
            policy_target_global_row_identity=str(payload["policy_target_global_row_identity"]),
            value_target=float(payload["value_target"]),
            root_value=float(payload["root_value"]),
            player=int(payload["player"]),
            outcome=float(payload["outcome"]),
            is_full_search=bool(payload["is_full_search"]),
            turn_index=int(payload["turn_index"]),
            selected_action_value=None if payload.get("selected_action_value") is None else float(payload["selected_action_value"]),
            lookahead_values=tuple(float(x) for x in payload.get("lookahead_values", [])),
            opp_policy_target_v2=tuple((int(q), int(r), float(p)) for q, r, p in payload.get("opp_policy_target_v2", [])),
            opp_policy_dense=tuple((int(idx), float(p)) for idx, p in payload.get("opp_policy_dense", [])),
            opp_policy_legal_v2=tuple((int(q), int(r)) for q, r in payload.get("opp_policy_legal_v2", [])),
            opp_policy_weight=float(payload.get("opp_policy_weight", 0.0)),
            pair_policy_target_v2=tuple(pairs),
            pair_policy_complete=bool(payload.get("pair_policy_complete", False)),
            pair_known_first=None if known is None else (int(known[0]), int(known[1])),
            pair_completeness=str(payload.get("pair_completeness", "none")),
            target_policy_mass_outside_window=float(payload.get("target_policy_mass_outside_window", 0.0)),
            missing_target_policy_mass=float(payload.get("missing_target_policy_mass", 0.0)),
            candidate_critical_count=int(payload.get("candidate_critical_count", 0)),
            candidate_critical_overflow_count=int(payload.get("candidate_critical_overflow_count", 0)),
            candidate_critical_overflow_examples=tuple(
                (int(q), int(r)) for q, r in payload.get("candidate_critical_overflow_examples", [])
            ),
            regret_rank=float(payload.get("regret_rank", 0.0)),
            regret_value=float(payload.get("regret_value", 0.0)),
            regret_weight=float(payload.get("regret_weight", 0.0)),
            axis_label=int(payload.get("axis_label", -1)),
            moves_left=float(payload.get("moves_left", 0.0)),
            value_weight=float(payload.get("value_weight", 1.0)),
            policy_weight=float(payload.get("policy_weight", 1.0)),
            contract_trace=dict(payload.get("contract_trace", {})),
            record_hash=str(payload.get("record_hash", "")),
        )


@dataclass(frozen=True)
class ReplayGameRecord:
    schema_version: int
    game_id: int
    outcome: float
    game_length: int
    final_move_history: bytes
    final_history_hash: str
    terminal_reason: str
    truncated: bool
    writer_version: str
    config_identity: str
    checkpoint_identity: str
    positions: tuple[ReplayPositionRecord, ...]
    game_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_RECORD_SCHEMA_VERSION:
            raise ReplayCodecError(f"unsupported replay schema version {self.schema_version}", owner="replay.codec")
        if self.game_length != len(self.positions):
            raise ReplayCodecError("game_length does not match replay positions", owner="replay.codec")
        final = MoveHistory.decode(self.final_move_history, source=HISTORY_SOURCE_MARKER)
        if final.history_hash != self.final_history_hash:
            raise ReplayCodecError("final history hash mismatch", owner="replay.codec")
        for idx, pos in enumerate(self.positions):
            if pos.game_id != self.game_id or pos.position_index != idx:
                raise ReplayCodecError("replay game position identity mismatch", owner="replay.codec")
        digest = _game_hash(self)
        if self.game_hash and self.game_hash != digest:
            raise ReplayCodecError("replay game hash mismatch", owner="replay.codec")
        object.__setattr__(self, "game_hash", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "game_id": self.game_id,
            "outcome": self.outcome,
            "game_length": self.game_length,
            "final_move_history_hex": self.final_move_history.hex(),
            "final_history_hash": self.final_history_hash,
            "terminal_reason": self.terminal_reason,
            "truncated": self.truncated,
            "writer_version": self.writer_version,
            "config_identity": self.config_identity,
            "checkpoint_identity": self.checkpoint_identity,
            "positions": [pos.to_dict() for pos in self.positions],
            "game_hash": self.game_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReplayGameRecord":
        return cls(
            schema_version=int(payload["schema_version"]),
            game_id=int(payload["game_id"]),
            outcome=float(payload["outcome"]),
            game_length=int(payload["game_length"]),
            final_move_history=bytes.fromhex(str(payload["final_move_history_hex"])),
            final_history_hash=str(payload["final_history_hash"]),
            terminal_reason=str(payload["terminal_reason"]),
            truncated=bool(payload["truncated"]),
            writer_version=str(payload["writer_version"]),
            config_identity=str(payload["config_identity"]),
            checkpoint_identity=str(payload["checkpoint_identity"]),
            positions=tuple(ReplayPositionRecord.from_dict(item) for item in payload["positions"]),
            game_hash=str(payload.get("game_hash", "")),
        )

    def to_game_record(self) -> GameRecord:
        positions = []
        for rec in self.positions:
            positions.append(
                PositionRecord(
                    move_history=bytes(rec.move_history),
                    policy_target=dict(rec.policy_target_dense),
                    root_value=float(rec.root_value),
                    player=int(rec.player),
                    outcome=float(rec.outcome),
                    game_id=int(rec.game_id),
                    is_full_search=bool(rec.is_full_search),
                    turn_index=int(rec.turn_index),
                    selected_action_value=rec.selected_action_value,
                    lookahead_values=list(rec.lookahead_values),
                    opp_policy_target=dict(rec.opp_policy_dense),
                    opp_policy_weight=float(rec.opp_policy_weight),
                    policy_target_v2=list(rec.policy_target_v2),
                    opp_policy_target_v2=list(rec.opp_policy_target_v2),
                    opp_policy_legal_v2=list(rec.opp_policy_legal_v2),
                    pair_policy_target_v2=list(rec.pair_policy_target_v2),
                    pair_policy_complete=bool(rec.pair_policy_complete),
                    target_policy_mass_outside_window=float(rec.target_policy_mass_outside_window),
                    missing_target_policy_mass=float(rec.missing_target_policy_mass),
                    candidate_critical_count=int(rec.candidate_critical_count),
                    candidate_critical_overflow_count=int(rec.candidate_critical_overflow_count),
                    candidate_critical_overflow_examples=tuple(rec.candidate_critical_overflow_examples),
                    regret_rank=float(rec.regret_rank),
                    regret_value=float(rec.regret_value),
                    regret_weight=float(rec.regret_weight),
                    axis_label=int(rec.axis_label),
                    moves_left=float(rec.moves_left),
                    value_weight=float(rec.value_weight),
                )
            )
            setattr(positions[-1], "policy_weight", float(rec.policy_weight))
        return GameRecord(
            positions=positions,
            outcome=float(self.outcome),
            game_id=int(self.game_id),
            game_length=int(self.game_length),
            final_move_history=bytes(self.final_move_history),
            truncated=bool(self.truncated),
            terminal_reason=str(self.terminal_reason),
        )


def replay_game_from_selfplay(
    record: GameRecord,
    *,
    lookahead_horizons: Sequence[int] = (),
    lookahead_lambdas: Sequence[float] = (),
    config_identity: str = "",
    checkpoint_identity: str = "",
    ffi_protocol_version: str = "rust-pyo3:v1",
) -> ReplayGameRecord:
    if not isinstance(record, GameRecord):
        raise ReplayCodecError("self-play writer accepts only GameRecord as source input", owner="replay.codec")
    source = _prepare_game_record(record, lookahead_horizons=lookahead_horizons, lookahead_lambdas=lookahead_lambdas)
    positions = tuple(
        _position_from_selfplay(
            pos,
            schema_version=REPLAY_RECORD_SCHEMA_VERSION,
            config_identity=config_identity,
            checkpoint_identity=checkpoint_identity,
            ffi_protocol_version=ffi_protocol_version,
        )
        for pos in source.positions
    )
    final = MoveHistory.decode(bytes(source.final_move_history), source=HISTORY_SOURCE_MARKER)
    return ReplayGameRecord(
        schema_version=REPLAY_RECORD_SCHEMA_VERSION,
        game_id=int(source.game_id),
        outcome=float(source.outcome),
        game_length=len(positions),
        final_move_history=bytes(source.final_move_history),
        final_history_hash=final.history_hash,
        terminal_reason=str(source.terminal_reason),
        truncated=bool(source.truncated),
        writer_version=REPLAY_WRITER_VERSION,
        config_identity=str(config_identity),
        checkpoint_identity=str(checkpoint_identity),
        positions=positions,
    )


def encode_replay_game(record: ReplayGameRecord) -> bytes:
    if not isinstance(record, ReplayGameRecord):
        raise ReplayCodecError("encode_replay_game requires ReplayGameRecord", owner="replay.codec")
    payload = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return REPLAY_CODEC_MAGIC + struct.pack("<HI", REPLAY_RECORD_SCHEMA_VERSION, len(payload)) + payload


def decode_replay_game(data: bytes, *, allow_migration: bool = False) -> ReplayGameRecord:
    if not data.startswith(REPLAY_CODEC_MAGIC):
        raise ReplayCodecError("malformed replay header: missing HXR7 magic", owner="replay.codec")
    offset = len(REPLAY_CODEC_MAGIC)
    try:
        schema_version, length = struct.unpack_from("<HI", data, offset)
    except struct.error as exc:
        raise ReplayCodecError("malformed replay header: truncated schema/length", owner="replay.codec") from exc
    offset += struct.calcsize("<HI")
    if schema_version != REPLAY_RECORD_SCHEMA_VERSION and not allow_migration:
        raise ReplayCodecError(f"unknown replay schema version {schema_version}", owner="replay.codec")
    payload = data[offset:]
    if len(payload) != int(length):
        raise ReplayCodecError("truncated replay payload length", owner="replay.codec")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise ReplayCodecError("malformed replay json payload", owner="replay.codec") from exc
    return ReplayGameRecord.from_dict(decoded)


def replay_feature_flags(heads, **_kwargs) -> dict[str, bool]:
    head_set = set(heads or [])
    return {
        "include_sparse_policy": bool("sparse_policy" in head_set or "pair_policy" in head_set),
        "include_pair_policy": bool("pair_policy" in head_set or {"policy_pair_first", "policy_pair_second", "policy_pair_joint"} & head_set),
        "include_graph_policy": bool({"policy_place", "policy_pair_first", "policy_pair_second", "policy_pair_joint"} & head_set),
    }


def compute_value_targets(positions: Sequence[PositionRecord], outcome: float) -> None:
    for pos in positions:
        pos.outcome = float(outcome)


def compute_ema_lookahead(
    positions: Sequence[PositionRecord],
    horizon: int,
    lambda_: float,
) -> np.ndarray:
    n = len(positions)
    if n == 0:
        return np.array([], dtype=np.float32)
    boundaries = _hexo_turn_start_indices(positions)
    mcts_values = np.array([float(pos.root_value) for pos in positions], dtype=np.float32)
    result = np.copy(mcts_values)
    for i in range(n - 1, -1, -1):
        target_bi = bisect_right(boundaries, i) + int(horizon) - 1
        if target_bi < len(boundaries):
            j = boundaries[target_bi]
            future = _value_from_source_perspective(result[j], positions[j].player, positions[i].player)
            result[i] = (1.0 - float(lambda_)) * mcts_values[i] + float(lambda_) * future
        else:
            result[i] = mcts_values[i]
    return result


def _prepare_game_record(
    record: GameRecord,
    *,
    lookahead_horizons: Sequence[int],
    lookahead_lambdas: Sequence[float],
) -> GameRecord:
    game = replace(record, positions=[replace(pos) for pos in record.positions])
    compute_value_targets(game.positions, float(game.outcome))
    for horizon, lambda_ in zip(lookahead_horizons, lookahead_lambdas):
        values = compute_ema_lookahead(game.positions, int(horizon), float(lambda_))
        for idx, pos in enumerate(game.positions):
            current = list(getattr(pos, "lookahead_values", []))
            current.append(float(values[idx]))
            pos.lookahead_values = current
    _assign_auxiliary_targets(game)
    return game


def _position_from_selfplay(
    pos: PositionRecord,
    *,
    schema_version: int,
    config_identity: str,
    checkpoint_identity: str,
    ffi_protocol_version: str,
) -> ReplayPositionRecord:
    history = MoveHistory.decode(bytes(pos.move_history), source=HISTORY_SOURCE_MARKER)
    legal = LegalTableProvider(near_radius=8, constrain_threats=False).from_history(history)
    policy_v2 = tuple((int(q), int(r), float(prob)) for q, r, prob in getattr(pos, "policy_target_v2", []))
    policy_dense = tuple(sorted((int(k), float(v)) for k, v in getattr(pos, "policy_target", {}).items() if float(v) > 0.0))
    pair_target = tuple(
        ((int(first[0]), int(first[1])), (int(second[0]), int(second[1])), float(prob))
        for first, second, prob in getattr(pos, "pair_policy_target_v2", [])
    )
    known_first = _last_move_qr(bytes(pos.move_history)) if legal.placements_remaining == 1 else None
    return ReplayPositionRecord(
        schema_version=schema_version,
        game_id=int(pos.game_id),
        position_index=int(pos.turn_index),
        position_id=stable_digest(("ReplayPosition", int(pos.game_id), int(pos.turn_index), history.history_hash)),
        move_history=bytes(pos.move_history),
        history_source=history.source,
        history_schema_version=history.schema_version,
        history_hash=history.history_hash,
        compact_history_row_count=history.move_count,
        legal_rows=tuple((int(q), int(r)) for q, r in legal.rows.tolist()),
        legal_dense_indices=tuple(int(x) for x in legal.dense_indices.tolist()),
        legal_source=legal.source,
        legal_schema_version=legal.schema_version,
        legal_table_hash=legal.table_hash,
        reconstructed_legal_table_hash=LegalTableProvider(near_radius=8, constrain_threats=False).from_history(history).table_hash,
        ffi_protocol_version=ffi_protocol_version,
        current_player=legal.current_player,
        placements_remaining=legal.placements_remaining,
        policy_target_v2=policy_v2,
        policy_target_dense=policy_dense,
        policy_target_global_row_identity=stable_digest(("PolicyTargetV2", history.history_hash, policy_v2)),
        value_target=float(pos.to_value_target()),
        root_value=float(pos.root_value),
        player=int(pos.player),
        outcome=float(pos.outcome if pos.outcome is not None else 0.0),
        is_full_search=bool(pos.is_full_search),
        turn_index=int(pos.turn_index),
        selected_action_value=None if pos.selected_action_value is None else float(pos.selected_action_value),
        lookahead_values=tuple(float(v) for v in getattr(pos, "lookahead_values", [])),
        opp_policy_target_v2=tuple((int(q), int(r), float(prob)) for q, r, prob in getattr(pos, "opp_policy_target_v2", [])),
        opp_policy_dense=tuple(sorted((int(k), float(v)) for k, v in getattr(pos, "opp_policy_target", {}).items() if float(v) > 0.0)),
        opp_policy_legal_v2=tuple((int(q), int(r)) for q, r in getattr(pos, "opp_policy_legal_v2", [])),
        opp_policy_weight=float(getattr(pos, "opp_policy_weight", 0.0)),
        pair_policy_target_v2=pair_target,
        pair_policy_complete=bool(getattr(pos, "pair_policy_complete", False)),
        pair_known_first=known_first,
        pair_completeness="complete" if bool(getattr(pos, "pair_policy_complete", False)) else ("partial" if pair_target else "none"),
        target_policy_mass_outside_window=float(getattr(pos, "target_policy_mass_outside_window", 0.0)),
        missing_target_policy_mass=float(getattr(pos, "missing_target_policy_mass", 0.0)),
        candidate_critical_count=int(getattr(pos, "candidate_critical_count", 0)),
        candidate_critical_overflow_count=int(getattr(pos, "candidate_critical_overflow_count", 0)),
        candidate_critical_overflow_examples=tuple(
            (int(q), int(r)) for q, r in getattr(pos, "candidate_critical_overflow_examples", ())
        ),
        regret_rank=float(getattr(pos, "regret_rank", 0.0)),
        regret_value=float(getattr(pos, "regret_value", 0.0)),
        regret_weight=float(getattr(pos, "regret_weight", 0.0)),
        axis_label=int(getattr(pos, "axis_label", -1)),
        moves_left=float(getattr(pos, "moves_left", 0.0)),
        value_weight=float(getattr(pos, "value_weight", 1.0)),
        policy_weight=float(getattr(pos, "policy_weight", 1.0 if getattr(pos, "is_full_search", True) else 0.0)),
        contract_trace={
            "history_hash": history.history_hash,
            "legal_table_hash": legal.table_hash,
            "config_identity": str(config_identity),
            "checkpoint_identity": str(checkpoint_identity),
        },
    )


def _validate_position(pos: ReplayPositionRecord) -> None:
    if pos.schema_version != REPLAY_RECORD_SCHEMA_VERSION:
        raise ReplayCodecError(f"unsupported replay position schema version {pos.schema_version}", owner="replay.codec")
    history = MoveHistory.decode(pos.move_history, source=pos.history_source)
    if history.history_hash != pos.history_hash:
        raise ReplayCodecError("history hash mismatch", owner="replay.codec")
    if history.move_count != pos.compact_history_row_count:
        raise ReplayCodecError("compact history row count mismatch", owner="replay.codec")
    legal = LegalTableProvider(near_radius=8, constrain_threats=False).from_history(history)
    if legal.table_hash != pos.reconstructed_legal_table_hash or legal.table_hash != pos.legal_table_hash:
        raise ReplayCodecError("stale legal hash for replay record", owner="replay.codec")
    if tuple((int(q), int(r)) for q, r in legal.rows.tolist()) != tuple(pos.legal_rows):
        raise ReplayCodecError("legal rows changed or reordered for replay record", owner="replay.codec")
    legal_set = set(pos.legal_rows)
    _validate_probability_target(pos.policy_target_v2, legal_set, "policy_target_v2")
    if pos.opp_policy_target_v2 and not pos.opp_policy_legal_v2:
        raise ReplayCodecError("opponent policy target requires opponent legal rows", owner="replay.codec")
    _validate_probability_target(pos.opp_policy_target_v2, set(pos.opp_policy_legal_v2), "opp_policy_target_v2", allow_empty_legal=True)
    _validate_pair_targets(pos)
    for field_name in (
        "value_target",
        "root_value",
        "outcome",
        "target_policy_mass_outside_window",
        "missing_target_policy_mass",
        "regret_rank",
        "regret_value",
        "regret_weight",
        "moves_left",
        "value_weight",
        "policy_weight",
    ):
        if not math.isfinite(float(getattr(pos, field_name))):
            raise ReplayCodecError(f"non-finite replay field: {field_name}", owner="replay.codec")
    if any(not math.isfinite(float(v)) for v in pos.lookahead_values):
        raise ReplayCodecError("non-finite lookahead target", owner="replay.codec")
    forbidden = {"root_generation", "batch_generation"}
    if forbidden & set(pos.contract_trace.keys()):
        raise ReplayCodecError("transient MCTS token stored as replay semantics", owner="replay.codec")


def _validate_probability_target(
    rows: Sequence[tuple[int, int, float]],
    legal_set: set[tuple[int, int]],
    field: str,
    *,
    allow_empty_legal: bool = False,
) -> None:
    total = 0.0
    seen: set[tuple[int, int]] = set()
    for q, r, prob in rows:
        qr = (int(q), int(r))
        if qr in seen:
            raise ReplayCodecError(f"{field} contains duplicate row {qr}", owner="replay.codec")
        seen.add(qr)
        if legal_set or not allow_empty_legal:
            if qr not in legal_set:
                raise ReplayCodecError(f"{field} contains illegal row {qr}", owner="replay.codec")
        if not math.isfinite(float(prob)) or float(prob) < 0.0:
            raise ReplayCodecError(f"{field} contains invalid target probability", owner="replay.codec")
        total += float(prob)
    if rows and not (0.999 <= total <= 1.001):
        raise ReplayCodecError(f"{field} target mass must be 1.0, got {total:.6f}", owner="replay.codec")


def _validate_pair_targets(pos: ReplayPositionRecord) -> None:
    if not pos.pair_policy_target_v2:
        return
    legal_set = set(pos.legal_rows)
    total = 0.0
    for first, second, prob in pos.pair_policy_target_v2:
        if first == second:
            raise ReplayCodecError("pair target uses duplicate coordinates", owner="replay.codec")
        if pos.pair_known_first is not None:
            if first != pos.pair_known_first:
                raise ReplayCodecError("bad known-first pair target reference", owner="replay.codec")
            if second not in legal_set:
                raise ReplayCodecError("pair target second action is illegal", owner="replay.codec")
        elif first not in legal_set or second not in legal_set:
            raise ReplayCodecError("pair target contains illegal action", owner="replay.codec")
        if not math.isfinite(float(prob)) or float(prob) < 0.0:
            raise ReplayCodecError("pair target probability must be finite and non-negative", owner="replay.codec")
        total += float(prob)
    if not (0.999 <= total <= 1.001):
        raise ReplayCodecError(f"pair target mass must be 1.0, got {total:.6f}", owner="replay.codec")


def _assign_auxiliary_targets(record: GameRecord) -> None:
    total = len(record.positions)
    for i, pos in enumerate(record.positions):
        if bool(getattr(record, "truncated", False)) or float(record.outcome) == 0.0:
            pos.value_weight = 0.0
        opp_idx = _next_full_search_opponent_turn_start(record.positions, i)
        if opp_idx is None:
            pos.opp_policy_target = {}
            pos.opp_policy_target_v2 = []
            pos.opp_policy_legal_v2 = []
            pos.opp_policy_weight = 0.0
        else:
            opp = record.positions[opp_idx]
            pos.opp_policy_target = dict(opp.policy_target)
            pos.opp_policy_target_v2 = list(getattr(opp, "policy_target_v2", []))
            pos.opp_policy_legal_v2 = _legal_qr_from_history(opp.move_history)
            pos.opp_policy_weight = 1.0 if bool(getattr(opp, "is_full_search", False)) else 0.0
        if not pos.pair_policy_target_v2:
            pos.pair_policy_target_v2 = _second_placement_pair_target(record.positions, i)
        if pos.pair_policy_target_v2:
            legal = set(_legal_qr_from_history(pos.move_history))
            expected = len(legal) * max(len(legal) - 1, 0) // 2
            if _last_move_qr(pos.move_history) is not None and any(first == _last_move_qr(pos.move_history) for first, _second, _prob in pos.pair_policy_target_v2):
                expected = max(len(legal) - 1, 0)
            seen = {
                (tuple(first), tuple(second))
                for first, second, prob in pos.pair_policy_target_v2
                if float(prob) > 0.0
            }
            pos.pair_policy_complete = bool(expected == 0 or len(seen) >= expected)
        pos.regret_rank = _trajectory_regret(record.positions, record.outcome, i)
        pos.regret_value = pos.regret_rank
        pos.regret_weight = 0.0 if bool(getattr(record, "truncated", False)) else 1.0
        pos.moves_left = float(max(total - int(pos.turn_index), 0))
        if pos.outcome is None:
            pos.outcome = float(record.outcome)


def _trajectory_regret(positions: Sequence[PositionRecord], outcome: float, start: int) -> float:
    total = 0.0
    count = 0
    for pos in positions[start:]:
        selected = getattr(pos, "selected_action_value", None)
        if selected is None:
            return 0.0
        z = float(outcome) if int(pos.player) == 0 else -float(outcome)
        total += (float(selected) - z) ** 2
        count += 1
    return float(total / max(count, 1))


def _next_full_search_opponent_turn_start(positions: Sequence[PositionRecord], i: int) -> int | None:
    player = positions[i].player
    for j in range(i + 1, len(positions)):
        if positions[j].player == player:
            continue
        if j > 0 and positions[j - 1].player == positions[j].player:
            continue
        if positions[j].is_full_search:
            return j
    return None


def _second_placement_pair_target(
    positions: Sequence[PositionRecord],
    index: int,
) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    pos = positions[index]
    if index <= 0 or positions[index - 1].player != pos.player or not pos.policy_target_v2:
        return []
    first = _last_move_qr(pos.move_history)
    if first is None:
        return []
    legal = set(_legal_qr_from_history(pos.move_history))
    return [
        (first, (int(q), int(r)), float(prob))
        for q, r, prob in pos.policy_target_v2
        if float(prob) > 0.0 and (int(q), int(r)) in legal and (int(q), int(r)) != first
    ]


def _legal_qr_from_history(history: bytes) -> list[tuple[int, int]]:
    legal = LegalTableProvider(near_radius=8, constrain_threats=False).from_history(history)
    return [(int(q), int(r)) for q, r in legal.rows.tolist()]


def _last_move_qr(history: bytes) -> tuple[int, int] | None:
    if not history:
        return None
    rows = MoveHistory.decode(history, source=HISTORY_SOURCE_MARKER).rows
    if not rows:
        return None
    _player, q, r = rows[-1]
    return (int(q), int(r))


def _hexo_turn_start_indices(positions: Sequence[PositionRecord]) -> list[int]:
    return [i for i, pos in enumerate(positions) if i == 0 or pos.player != positions[i - 1].player]


def _value_from_source_perspective(value: float, source_player: int, target_player: int) -> float:
    return float(value) if int(source_player) == int(target_player) else -float(value)


def _position_hash(pos: ReplayPositionRecord) -> str:
    return stable_digest(
        (
            "ReplayPositionRecord",
            int(pos.schema_version),
            int(pos.game_id),
            int(pos.position_index),
            pos.history_hash,
            pos.legal_table_hash,
            pos.reconstructed_legal_table_hash,
            pos.policy_target_global_row_identity,
            tuple(pos.policy_target_v2),
            tuple(pos.pair_policy_target_v2),
            float(pos.value_target),
            float(pos.root_value),
            tuple(pos.lookahead_values),
        )
    )


def _game_hash(record: ReplayGameRecord) -> str:
    return stable_digest(
        (
            "ReplayGameRecord",
            int(record.schema_version),
            int(record.game_id),
            record.final_history_hash,
            tuple(pos.record_hash for pos in record.positions),
            record.writer_version,
            record.config_identity,
            record.checkpoint_identity,
        )
    )
