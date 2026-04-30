"""Golden replay fixtures for tests and artifact generation."""

from __future__ import annotations

import struct

from hexorl.config import Config
from hexorl.engine.legal import LegalTableProvider
from hexorl.replay.codec import ReplayGameRecord, encode_replay_game, replay_game_from_selfplay
from hexorl.selfplay.records import GameRecord, PositionRecord, dense_policy_from_v2


def golden_game_record(game_id: int = 7) -> GameRecord:
    moves: list[tuple[int, int, int]] = []
    positions: list[PositionRecord] = []
    for idx, (player, q, r) in enumerate([(0, 0, 0), (1, 1, 0), (1, 0, 1)]):
        history = _pack(moves)
        legal = [(int(a), int(b)) for a, b in LegalTableProvider(near_radius=8, constrain_threats=False).from_history(history).rows.tolist()]
        chosen = (q, r) if (q, r) in legal else legal[0]
        alternates = [item for item in legal if item != chosen][:1]
        if alternates:
            policy_v2 = [(chosen[0], chosen[1], 0.7), (alternates[0][0], alternates[0][1], 0.3)]
        else:
            policy_v2 = [(chosen[0], chosen[1], 1.0)]
        policy, outside = dense_policy_from_v2(policy_v2, -16, -16, top_k=8)
        positions.append(
            PositionRecord(
                move_history=history,
                policy_target=policy,
                policy_target_v2=policy_v2,
                root_value=0.25 - idx * 0.1,
                player=player,
                outcome=1.0,
                game_id=game_id,
                is_full_search=True,
                turn_index=idx,
                selected_action_value=0.2,
                target_policy_mass_outside_window=outside,
            )
        )
        moves.append((player, q, r))
    return GameRecord(
        positions=positions,
        outcome=1.0,
        game_id=game_id,
        game_length=len(positions),
        final_move_history=_pack(moves),
        terminal_reason="fixture",
    )


def golden_replay_game(game_id: int = 7) -> ReplayGameRecord:
    cfg = Config()
    return replay_game_from_selfplay(
        golden_game_record(game_id),
        lookahead_horizons=cfg.buffer.lookahead_horizons,
        lookahead_lambdas=cfg.buffer.lookahead_lambdas,
        config_identity="fixture-config",
        checkpoint_identity="fixture-checkpoint",
    )


def golden_replay_bytes(game_id: int = 7) -> bytes:
    return encode_replay_game(golden_replay_game(game_id))


def corrupt_replay_bytes(kind: str) -> bytes:
    good = bytearray(golden_replay_bytes())
    if kind == "bad_magic":
        good[:4] = b"BAD!"
    elif kind == "bad_version":
        good[4:6] = struct.pack("<H", 999)
    elif kind == "truncated":
        del good[-8:]
    else:
        raise ValueError(f"unknown corruption kind: {kind}")
    return bytes(good)


def _pack(moves: list[tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for player, q, r in moves:
        out.extend(struct.pack("<iii", int(player), int(q), int(r)))
    return bytes(out)
