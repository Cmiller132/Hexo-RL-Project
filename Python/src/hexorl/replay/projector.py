"""Canonical replay-record to training-batch projection."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from hexorl.contracts.candidates import CANDIDATE_FEATURES, CandidateContractBuilder
from hexorl.contracts.history import MoveHistory
from hexorl.contracts.identity import stable_digest
from hexorl.contracts.pairs import PairActionTableBuilder, PairStrategy
from hexorl.contracts.symmetry import (
    apply_tensor_symmetry,
    transform_axis_label,
    transform_history,
    transform_pair_policy_target,
    transform_policy_target,
    transform_qr,
)
from hexorl.contracts.validation import ContractValidationError
from hexorl.engine.legal import decode_legal_bytes
from hexorl.engine.rust import hex_game_class
from hexorl.graph.collate import collate_graph_batches
from hexorl.graph.tensorize import build_graph_batch_from_history, graph_batch_with_pair_table
from hexorl.replay.codec import ReplayCodecError, ReplayPositionRecord
from hexorl.selfplay.records import BOARD_AREA, BOARD_SIZE, NUM_CHANNELS, dense_policy_from_v2


@dataclass(frozen=True)
class ReplayProjectionConfig:
    use_symmetry: bool = True
    lookahead_horizons: tuple[int, ...] = ()
    include_axis_delta_norm: bool = False
    include_sparse_policy: bool = False
    include_pair_policy: bool = False
    include_graph_policy: bool = False
    candidate_budget: int = 256
    near_radius: int = 8


@dataclass(frozen=True)
class ProjectedReplayBatch:
    tensors: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    lookahead: list[np.ndarray]
    aux_targets: dict[str, np.ndarray]
    source: str = "replay/projector.py"
    schema_version: int = 1
    record_hashes: tuple[str, ...] = ()
    projection_id: str = ""
    throughput: dict[str, float] = field(default_factory=dict)

    def as_legacy_tuple(self):
        return self.tensors, self.policies, self.values, self.lookahead, self.aux_targets


class ReplayProjector:
    """Single runtime path from canonical replay records to trainable batches."""

    def __init__(self, config: ReplayProjectionConfig | None = None) -> None:
        self.config = config or ReplayProjectionConfig()

    def project(self, records: Sequence[ReplayPositionRecord]) -> ProjectedReplayBatch:
        if not records:
            raise ReplayCodecError("projector requires at least one replay record", owner="replay.projector")
        t0 = time.monotonic()
        batch_size = len(records)
        candidate_width = min(
            512,
            max(
                int(self.config.candidate_budget),
                max((len(rec.policy_target_v2) for rec in records), default=0),
                max((len(rec.opp_policy_target_v2) for rec in records), default=0),
                max((len(rec.pair_policy_target_v2) + 1 for rec in records), default=0)
                if self.config.include_pair_policy
                else 0,
                1,
            ),
        )
        tensors = np.zeros((batch_size, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        policies = np.zeros((batch_size, BOARD_AREA), dtype=np.float32)
        values = np.zeros(batch_size, dtype=np.float32)
        aux = _base_aux_targets(batch_size)
        if self.config.include_sparse_policy:
            _add_sparse_aux(aux, batch_size, candidate_width)
        if self.config.include_axis_delta_norm:
            aux["axis_delta_norm"] = np.zeros((batch_size, 6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        lookahead = [np.zeros(batch_size, dtype=np.float32) for _ in self.config.lookahead_horizons]
        graph_batches: list[Any] = []

        for row, rec in enumerate(records):
            if not isinstance(rec, ReplayPositionRecord):
                raise ReplayCodecError("projector consumes only ReplayPositionRecord", owner="replay.projector")
            sym_idx = row % 12 if self.config.use_symmetry else 0
            sample_history = transform_history(rec.move_history, sym_idx) if sym_idx else bytes(rec.move_history)
            policy_v2 = transform_policy_target(rec.policy_target_v2, sym_idx) if sym_idx else list(rec.policy_target_v2)
            opp_policy_v2 = transform_policy_target(rec.opp_policy_target_v2, sym_idx) if sym_idx else list(rec.opp_policy_target_v2)
            pair_policy_v2 = transform_pair_policy_target(rec.pair_policy_target_v2, sym_idx) if sym_idx else list(rec.pair_policy_target_v2)
            opp_legal = _unique_transformed_keys(rec.opp_policy_legal_v2, sym_idx) if sym_idx else list(rec.opp_policy_legal_v2)
            axis_label = transform_axis_label(rec.axis_label, sym_idx) if sym_idx else int(rec.axis_label)

            tensor_i, offset_q, offset_r, legal_rows = _encode_tensor_and_legal(sample_history, self.config.near_radius)
            tensors[row] = tensor_i
            policy_dict, _outside = dense_policy_from_v2(policy_v2, int(offset_q), int(offset_r), top_k=max(1, len(policy_v2)))
            policies[row] = _dense_from_sparse_policy(policy_dict)
            values[row] = float(rec.value_target)
            aux["opp_policy"][row] = _dense_from_sparse_policy(
                dense_policy_from_v2(opp_policy_v2, int(offset_q), int(offset_r), top_k=max(1, len(opp_policy_v2)))[0]
            )
            aux["opp_policy_weight"][row] = float(rec.opp_policy_weight)
            aux["regret_rank"][row] = float(rec.regret_rank)
            aux["regret_value"][row] = float(rec.regret_value)
            aux["regret_weight"][row] = float(rec.regret_weight)
            aux["axis"][row] = axis_label
            aux["moves_left"][row] = float(rec.moves_left)
            aux["value_weight"][row] = float(rec.value_weight)
            aux["policy_weight"][row] = float(rec.policy_weight if rec.is_full_search else 0.0)

            if self.config.include_sparse_policy:
                candidate = CandidateContractBuilder().build(
                    [(int(q), int(r)) for q, r in legal_rows.tolist()],
                    policy_v2,
                    offset_q=int(offset_q),
                    offset_r=int(offset_r),
                    budget=candidate_width,
                    storage_width=candidate_width,
                    critical_actions=list(policy_v2_to_qr(policy_v2)),
                    source="rust:legal",
                )
                aux["candidate_qr"][row] = candidate.rows[:candidate_width]
                aux["candidate_indices"][row] = candidate.dense_indices[:candidate_width]
                aux["candidate_features"][row] = candidate.features[:candidate_width]
                aux["candidate_mask"][row] = candidate.mask[:candidate_width]
                aux["sparse_policy_target"][row] = candidate.target[:candidate_width]
                aux["candidate_missing_mass"][row] = candidate.diagnostics.missing_mass
                aux["candidate_critical_count"][row] = float(rec.candidate_critical_count)
                aux["candidate_critical_overflow_count"][row] = float(rec.candidate_critical_overflow_count)
                for ex_idx, (q, r) in enumerate(rec.candidate_critical_overflow_examples[:8]):
                    aux["candidate_critical_overflow_examples"][row, ex_idx] = (int(q), int(r))
                if self.config.include_pair_policy:
                    _project_pair_targets(aux, row, candidate, pair_policy_v2, legal_rows, sample_history, candidate_width)

            if self.config.include_graph_policy:
                if opp_policy_v2 and not opp_legal:
                    raise ReplayCodecError("graph projection requires opponent legal rows for opponent target", owner="replay.projector")
                graph = build_graph_batch_from_history(
                    sample_history,
                    policy_target=policy_v2,
                    opp_legal_moves=opp_legal if opp_legal else None,
                    opp_policy_target=opp_policy_v2,
                    radius=self.config.near_radius,
                    include_pair_rows=False,
                )
                if self.config.include_pair_policy and int(graph.placements_remaining) >= 2 and not rec.pair_policy_complete:
                    raise ReplayCodecError(
                        "graph pair-policy projection requires complete search-observed pair targets",
                        owner="replay.projector",
                    )
                if pair_policy_v2:
                    graph = _graph_with_pair_targets(graph, pair_policy_v2, sample_history)
                graph_batches.append(graph)

            if self.config.include_axis_delta_norm:
                aux["axis_delta_norm"][row] = _axis_delta_norm(sample_history)
            for idx in range(len(lookahead)):
                lookahead[idx][row] = rec.lookahead_values[idx] if idx < len(rec.lookahead_values) else values[row]

        if self.config.include_graph_policy:
            graph = collate_graph_batches(graph_batches)
            aux.update(
                {
                    "token_features": graph.token_features,
                    "token_type": graph.token_type,
                    "token_qr": graph.token_qr,
                    "token_mask": graph.token_mask,
                    "legal_token_indices": graph.legal_token_indices,
                    "legal_qr": graph.legal_qr,
                    "legal_mask": graph.legal_mask,
                    "pair_token_indices": graph.pair_token_indices,
                    "pair_first_indices": graph.pair_first_indices,
                    "pair_second_indices": graph.pair_second_indices,
                    "relation_type": graph.relation_type,
                    "relation_bias": graph.relation_bias,
                    "policy_target": graph.policy_target,
                    "opp_legal_qr": graph.opp_legal_qr,
                    "opp_legal_mask": graph.opp_legal_mask,
                    "opp_policy_target": graph.opp_policy_target,
                    "pair_first_policy_target": graph.pair_first_policy_target,
                    "pair_policy_target": graph.pair_policy_target,
                    "tactical_target": graph.tactical_target,
                }
            )
        elapsed = max(time.monotonic() - t0, 1e-9)
        hashes = tuple(rec.record_hash for rec in records)
        return ProjectedReplayBatch(
            tensors=tensors,
            policies=policies,
            values=values,
            lookahead=lookahead,
            aux_targets=aux,
            record_hashes=hashes,
            projection_id=stable_digest(("ProjectedReplayBatch", hashes, tuple(sorted(aux.keys())))),
            throughput={
                "records": float(batch_size),
                "project_ms": elapsed * 1000.0,
                "project_samples_per_sec": float(batch_size / elapsed),
                "memory_bytes": float(tensors.nbytes + policies.nbytes + values.nbytes + sum(x.nbytes for x in lookahead)),
            },
        )


def policy_v2_to_qr(policy_v2):
    for q, r, prob in policy_v2:
        if float(prob) > 0.0:
            yield (int(q), int(r))


def _base_aux_targets(batch_size: int) -> dict[str, np.ndarray]:
    return {
        "opp_policy": np.zeros((batch_size, BOARD_AREA), dtype=np.float32),
        "regret_rank": np.zeros(batch_size, dtype=np.float32),
        "regret_value": np.zeros(batch_size, dtype=np.float32),
        "regret_weight": np.zeros(batch_size, dtype=np.float32),
        "axis": np.full(batch_size, -1, dtype=np.int64),
        "moves_left": np.zeros(batch_size, dtype=np.float32),
        "moves_left_weight": np.ones(batch_size, dtype=np.float32),
        "value_weight": np.ones(batch_size, dtype=np.float32),
        "policy_weight": np.ones(batch_size, dtype=np.float32),
        "sparse_policy_weight": np.ones(batch_size, dtype=np.float32),
        "pair_policy_weight": np.ones(batch_size, dtype=np.float32),
        "opp_policy_weight": np.zeros(batch_size, dtype=np.float32),
    }


def _add_sparse_aux(aux: dict[str, np.ndarray], batch_size: int, budget: int) -> None:
    aux["candidate_qr"] = np.zeros((batch_size, budget, 2), dtype=np.int32)
    aux["candidate_indices"] = np.full((batch_size, budget), -1, dtype=np.int64)
    aux["candidate_features"] = np.zeros((batch_size, budget, CANDIDATE_FEATURES), dtype=np.float32)
    aux["candidate_mask"] = np.zeros((batch_size, budget), dtype=np.bool_)
    aux["sparse_policy_target"] = np.zeros((batch_size, budget), dtype=np.float32)
    aux["candidate_missing_mass"] = np.zeros(batch_size, dtype=np.float32)
    aux["candidate_critical_count"] = np.zeros(batch_size, dtype=np.float32)
    aux["candidate_critical_overflow_count"] = np.zeros(batch_size, dtype=np.float32)
    aux["candidate_critical_overflow_examples"] = np.zeros((batch_size, 8, 2), dtype=np.int32)
    aux["pair_candidate_row_indices"] = np.full((batch_size, budget), -1, dtype=np.int64)
    aux["pair_candidate_features"] = np.zeros((batch_size, budget, CANDIDATE_FEATURES), dtype=np.float32)
    aux["pair_candidate_row_mask"] = np.zeros((batch_size, budget), dtype=np.bool_)
    aux["pair_candidate_indices"] = np.full((batch_size, budget, 2), -1, dtype=np.int64)
    aux["pair_candidate_mask"] = np.zeros((batch_size, budget), dtype=np.bool_)
    aux["pair_policy_target"] = np.zeros((batch_size, budget), dtype=np.float32)
    aux["pair_candidate_missing_mass"] = np.zeros(batch_size, dtype=np.float32)


def _project_pair_targets(
    aux: dict[str, np.ndarray],
    row: int,
    candidate,
    pair_policy_v2,
    legal_rows: np.ndarray,
    sample_history: bytes,
    candidate_width: int,
) -> None:
    legal_list = [(int(q), int(r)) for q, r in legal_rows.tolist()]
    known_first = _last_move_qr(sample_history)
    builder = PairActionTableBuilder()
    pair = builder.build(
        candidate,
        pair_policy_v2,
        strategy=PairStrategy(mode="capped_fill", max_pairs=candidate_width),
        legal_moves=legal_list,
        known_first=known_first,
        source="rust:legal",
    )
    aux["pair_candidate_row_indices"][row] = candidate.dense_indices[:candidate_width]
    aux["pair_candidate_features"][row] = candidate.features[:candidate_width]
    aux["pair_candidate_row_mask"][row] = candidate.mask[:candidate_width]
    width = min(candidate_width, pair.pair_indices.shape[0])
    aux["pair_candidate_indices"][row, :width] = pair.pair_indices[:width]
    aux["pair_candidate_mask"][row, :width] = pair.mask[:width]
    aux["pair_policy_target"][row, :width] = pair.target[:width]
    aux["pair_candidate_missing_mass"][row] = pair.missing_mass


def _graph_with_pair_targets(graph, pair_policy_v2, sample_history: bytes):
    graph_legal_rows = [(int(q), int(r)) for q, r in graph.legal_qr.tolist()]
    known_first = _last_move_qr(sample_history) if int(graph.placements_remaining) == 1 else None
    candidate_rows = ([known_first] if known_first is not None else []) + graph_legal_rows
    candidate = CandidateContractBuilder().build(
        candidate_rows,
        [],
        offset_q=0,
        offset_r=0,
        budget=max(1, len(candidate_rows)),
        storage_width=max(1, len(candidate_rows)),
        critical_actions=candidate_rows,
        source="rust:legal",
    )
    possible = len(graph_legal_rows) if known_first is not None else len(graph_legal_rows) * max(0, len(graph_legal_rows) - 1) // 2
    pair_table = PairActionTableBuilder().build(
        candidate,
        pair_policy_v2,
        strategy=PairStrategy(mode="full_capped", max_pairs=max(1, possible), allow_full=True),
        legal_moves=graph_legal_rows,
        known_first=known_first,
        source="rust:legal",
    )
    return graph_batch_with_pair_table(graph, pair_table)


def _encode_tensor_and_legal(history: bytes, near_radius: int) -> tuple[np.ndarray, int, int, np.ndarray]:
    game_cls = hex_game_class(required=True)
    game = game_cls()
    for player, q, r in MoveHistory.decode(history, source="rust").rows:
        if int(player) != int(game.current_player):
            raise ReplayCodecError("history player order disagrees with Rust replay", owner="replay.projector")
        game.place(int(q), int(r))
    tensor, offset_q, offset_r, legal_bytes = game.encode_board_and_legal(int(near_radius), False)
    return np.asarray(tensor, dtype=np.float32), int(offset_q), int(offset_r), decode_legal_bytes(bytes(legal_bytes))


def _dense_from_sparse_policy(policy: dict[int, float]) -> np.ndarray:
    dense = np.zeros(BOARD_AREA, dtype=np.float32)
    for idx, prob in policy.items():
        if 0 <= int(idx) < BOARD_AREA:
            dense[int(idx)] = float(prob)
    return dense


def _unique_transformed_keys(keys, sym_idx: int) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for q, r in keys:
        qr = transform_qr((int(q), int(r)), sym_idx)
        if qr not in seen:
            seen.add(qr)
            out.append(qr)
    return out


def _axis_delta_norm(history: bytes) -> np.ndarray:
    base = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    rows = MoveHistory.decode(history, source="rust").rows
    for _player, q, r in rows:
        gi = int(q) + BOARD_SIZE // 2
        gj = int(r) + BOARD_SIZE // 2
        if 0 <= gi < BOARD_SIZE and 0 <= gj < BOARD_SIZE:
            base[:, gi, gj] = 1.0
    return apply_tensor_symmetry(base, 0)


def _last_move_qr(history: bytes) -> tuple[int, int] | None:
    rows = MoveHistory.decode(history, source="rust").rows
    if not rows:
        return None
    _player, q, r = rows[-1]
    return (int(q), int(r))
