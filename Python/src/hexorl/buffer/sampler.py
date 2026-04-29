"""Batch sampler — PyTorch IterableDataset for the ring buffer.

On-the-fly decode + D6 symmetry augmentation. Runs in DataLoader workers.

§7.4 of SYSTEM_DESIGN.md.
"""

from collections import OrderedDict
import logging
import numpy as np
from typing import Dict, Iterator, Tuple, Optional, List

try:
    from torch.utils.data import IterableDataset as _IterableDataset
except ImportError:
    _IterableDataset = object  # type: ignore

from hexorl.buffer.ring import RingBuffer
from hexorl.selfplay.records import (
    BOARD_AREA,
    NUM_CHANNELS,
    BOARD_SIZE,
    PolicyTargetV2,
    action_to_board_index,
    dense_policy_from_v2,
)
from hexorl.action_contract.candidates import (
    CANDIDATE_FEATURES,
    build_candidate_batch,
    build_pair_candidate_batch,
)
from hexorl.action_contract.tactical_oracle import (
    TACTICAL_SCAN_RADIUS,
    scan_tactical_oracle_from_history,
)
from hexorl.graph.batch import (
    build_graph_batch_from_history,
    collate_graph_batches,
    graph_batch_with_reference_pair_rows,
)

logger = logging.getLogger(__name__)

try:
    from hexorl.axis_policy.core import AxisPolicyInput
    from hexorl.axis_policy.registry import get_prototype
    from hexorl.dashboard.replay import (
        encode_tensor_for_history,
        get_replay_position,
        position_payload,
    )

    HAS_AXIS_POLICY = True
except ImportError:  # pragma: no cover - optional dashboard/axis lab dependency path
    HAS_AXIS_POLICY = False

try:
    import _engine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False


def _py_encode_from_coords(
    history_bytes: bytes,
    stride: int,
    num_moves: int,
    near_radius: int,
) -> np.ndarray:
    """Encode a sequence of positions from coordinate history.

    Sets channel 0/1 (stones), channel 2 (empty mask), channel 6 (player colour),
    and channel 11 (distance from centre) based on coordinates.
    """
    half = BOARD_SIZE // 2
    positions = np.zeros(
        (num_moves + 1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE),
        dtype=np.float32,
    )
    moves: List[Tuple[int, int, int]] = []
    stones: dict[Tuple[int, int], int] = {}
    current_player = 0
    placements_remaining = 1

    qi_grid = np.arange(BOARD_SIZE)[:, None] - half
    rj_grid = np.arange(BOARD_SIZE)[None, :] - half
    dist = np.maximum(
        np.maximum(np.abs(qi_grid), np.abs(rj_grid)),
        np.abs(qi_grid + rj_grid),
    ).astype(np.float32)

    for i in range(num_moves + 1):
        _encode_position_fallback(
            positions[i],
            stones,
            moves,
            current_player,
            placements_remaining,
            dist / half,
            near_radius=near_radius,
        )
        if i >= num_moves:
            break

        offset = i * stride
        player = int.from_bytes(history_bytes[offset:offset + 4], "little", signed=True)
        q = int.from_bytes(history_bytes[offset + 4:offset + 8], "little", signed=True)
        r = int.from_bytes(history_bytes[offset + 8:offset + 12], "little", signed=True)
        if player != current_player:
            raise ValueError(
                f"Invalid compact history: move {i} stores player {player}, "
                f"expected {current_player}"
            )
        if (q, r) in stones:
            raise ValueError(f"Invalid compact history: duplicate cell ({q}, {r})")

        stones[(q, r)] = player
        moves.append((player, q, r))
        if placements_remaining > 1:
            placements_remaining -= 1
        else:
            current_player = 1 - current_player
            placements_remaining = 2

    return positions


def _encode_position_fallback(
    out: np.ndarray,
    stones: dict[Tuple[int, int], int],
    moves: List[Tuple[int, int, int]],
    current_player: int,
    placements_remaining: int,
    distance: np.ndarray,
    near_radius: int,
) -> None:
    """Encode the non-tactical feature planes used by Python fallback tests."""
    half = BOARD_SIZE // 2

    for (gqi, grj), player in stones.items():
            gi2, gj2 = gqi + half, grj + half
            if 0 <= gi2 < BOARD_SIZE and 0 <= gj2 < BOARD_SIZE:
                if player == current_player:
                    out[0, gi2, gj2] = 1.0
                else:
                    out[1, gi2, gj2] = 1.0

    out[2] = 1.0 - out[0] - out[1]
    out[11] = distance
    if current_player == 0:
        out[6].fill(1.0)

    if placements_remaining == 1 and moves:
        out[4].fill(1.0)
        _, q, r = moves[-1]
        gi, gj = q + half, r + half
        if 0 <= gi < BOARD_SIZE and 0 <= gj < BOARD_SIZE:
            out[5, gi, gj] = 1.0

    for q, r in _fallback_legal_moves(stones, near_radius):
        gi, gj = q + half, r + half
        if 0 <= gi < BOARD_SIZE and 0 <= gj < BOARD_SIZE:
            out[3, gi, gj] = 1.0

    opp = 1 - current_player
    recent_opp: List[Tuple[int, int]] = []
    for player, q, r in reversed(moves):
        if player == current_player:
            if recent_opp:
                break
            continue
        if player == opp:
            recent_opp.append((q, r))
            if len(recent_opp) == 2:
                break
    for q, r in recent_opp:
        gi, gj = q + half, r + half
        if 0 <= gi < BOARD_SIZE and 0 <= gj < BOARD_SIZE:
            out[12, gi, gj] = 1.0

    move_count = len(moves)
    for ply_idx, (player, q, r) in enumerate(moves):
        gi, gj = q + half, r + half
        if 0 <= gi < BOARD_SIZE and 0 <= gj < BOARD_SIZE:
            recency = 1.0 / (1.0 + move_count - ply_idx)
            out[7 if player == current_player else 8, gi, gj] = recency


def _fallback_legal_moves(
    stones: dict[Tuple[int, int], int],
    near_radius: int,
) -> List[Tuple[int, int]]:
    if not stones:
        return [(0, 0)]
    radius = max(0, min(int(near_radius), 8))
    legal: set[Tuple[int, int]] = set()
    for q, r in stones:
        for dq in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                if max(abs(dq), abs(dr), abs(dq + dr)) <= radius:
                    candidate = (q + dq, r + dr)
                    if candidate not in stones:
                        legal.add(candidate)
    return sorted(legal)


def _history_stones(history_bytes: bytes) -> dict[Tuple[int, int], int]:
    stones: dict[Tuple[int, int], int] = {}
    if len(history_bytes) % 12 != 0:
        return stones
    for offset in range(0, len(history_bytes), 12):
        player = int.from_bytes(history_bytes[offset:offset + 4], "little", signed=True)
        q = int.from_bytes(history_bytes[offset + 4:offset + 8], "little", signed=True)
        r = int.from_bytes(history_bytes[offset + 8:offset + 12], "little", signed=True)
        stones[(q, r)] = player
    return stones


def _critical_actions_from_tensor(
    tensor: np.ndarray,
    legal: np.ndarray,
    offset_q: int,
    offset_r: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Extract tactical action sets from encoded hot-cell planes.

    Channel 10 is current-player hot cells (win-now candidates). Channel 9 is
    opponent hot cells (forced-block / cover-set candidates).
    """
    winning: list[tuple[int, int]] = []
    forced: list[tuple[int, int]] = []
    cover: list[tuple[int, int]] = []
    if tensor.shape[0] <= 10:
        return winning, forced, cover
    for q_raw, r_raw in legal:
        q, r = int(q_raw), int(r_raw)
        flat = action_to_board_index(q, r, offset_q, offset_r)
        if flat < 0:
            continue
        gi, gj = divmod(flat, BOARD_SIZE)
        if tensor[10, gi, gj] > 0.0:
            winning.append((q, r))
        if tensor[9, gi, gj] > 0.0:
            forced.append((q, r))
            cover.append((q, r))
    return winning, forced, cover


def _py_decode_compact_record(
    history_bytes: bytes,
    near_radius: int = 8,
) -> np.ndarray:
    """Pure Python fallback for encode_compact_record.

    Replays the move history on a fresh board and encodes each position.
    Returns (N + 1, 13, 33, 33) float32 array.
    """
    if len(history_bytes) % 12 != 0:
        raise ValueError(
            f"history_bytes length {len(history_bytes)} is not a multiple of 12"
        )
    stride = 12
    num_moves = len(history_bytes) // stride
    positions = _py_encode_from_coords(
        history_bytes[:num_moves * stride],
        stride,
        num_moves,
        near_radius,
    )
    return positions


def _hex_transform(qi: int, rj: int, sym: int) -> Tuple[int, int]:
    """Apply one of 12 hex-grid D6 symmetry transforms."""
    if sym == 0:
        return (qi, rj)
    elif sym == 1:
        return (-rj, qi + rj)
    elif sym == 2:
        return (-qi - rj, qi)
    elif sym == 3:
        return (-qi, -rj)
    elif sym == 4:
        return (rj, -qi - rj)
    elif sym == 5:
        return (qi + rj, -qi)
    elif sym == 6:
        return (rj, qi)
    elif sym == 7:
        return (-qi, qi + rj)
    elif sym == 8:
        return (-qi - rj, rj)
    elif sym == 9:
        return (-rj, -qi)
    elif sym == 10:
        return (qi, -qi - rj)
    else:
        return (qi + rj, -rj)


def _transform_policy_v2(policy: PolicyTargetV2, sym_idx: int) -> PolicyTargetV2:
    """Transform global action-keyed policy entries under D6."""
    transformed: dict[tuple[int, int], float] = {}
    for q, r, prob in policy:
        tq, tr = _hex_transform(int(q), int(r), sym_idx % 12)
        transformed[(int(tq), int(tr))] = transformed.get((int(tq), int(tr)), 0.0) + float(prob)
    return [(q, r, prob) for (q, r), prob in transformed.items()]


def _transform_policy_keys(keys: List[Tuple[int, int]], sym_idx: int) -> List[Tuple[int, int]]:
    """Transform global action-key table entries under D6."""
    transformed: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for q, r in keys:
        tq, tr = _hex_transform(int(q), int(r), sym_idx % 12)
        qr = (int(tq), int(tr))
        if qr in seen:
            continue
        seen.add(qr)
        transformed.append(qr)
    return transformed


def _transform_pair_policy_v2(
    pairs: List[Tuple[Tuple[int, int], Tuple[int, int], float]],
    sym_idx: int,
) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
    """Transform pair-action policy entries under D6."""
    transformed = []
    for first, second, prob in pairs:
        q1, r1 = _hex_transform(int(first[0]), int(first[1]), sym_idx % 12)
        q2, r2 = _hex_transform(int(second[0]), int(second[1]), sym_idx % 12)
        transformed.append(((int(q1), int(r1)), (int(q2), int(r2)), float(prob)))
    return transformed


def _transform_history_bytes(history_bytes: bytes, sym_idx: int) -> bytes:
    """Transform compact move-history coordinates under D6, preserving players."""
    if sym_idx % 12 == 0:
        return history_bytes
    if len(history_bytes) % 12 != 0:
        raise ValueError(f"history_bytes length {len(history_bytes)} is not a multiple of 12")
    out = bytearray(len(history_bytes))
    for offset in range(0, len(history_bytes), 12):
        player = int.from_bytes(history_bytes[offset:offset + 4], "little", signed=True)
        q = int.from_bytes(history_bytes[offset + 4:offset + 8], "little", signed=True)
        r = int.from_bytes(history_bytes[offset + 8:offset + 12], "little", signed=True)
        tq, tr = _hex_transform(q, r, sym_idx % 12)
        out[offset:offset + 4] = int(player).to_bytes(4, "little", signed=True)
        out[offset + 4:offset + 8] = int(tq).to_bytes(4, "little", signed=True)
        out[offset + 8:offset + 12] = int(tr).to_bytes(4, "little", signed=True)
    return bytes(out)


def _last_move_qr(history_bytes: bytes) -> tuple[int, int] | None:
    if len(history_bytes) < 12:
        return None
    tail = history_bytes[-12:]
    q = int.from_bytes(tail[4:8], "little", signed=True)
    r = int.from_bytes(tail[8:12], "little", signed=True)
    return (q, r)


def _py_apply_d6_symmetry(tensor: np.ndarray, sym_idx: int) -> np.ndarray:
    """Pure Python fallback for apply_d6_symmetry.

    Applies one of 12 hex-grid transforms to a (C, 33, 33) tensor.
    Uses numpy vectorisation for speed.
    """
    sym = sym_idx % 12
    half = BOARD_SIZE // 2

    yi, xi = np.mgrid[0:BOARD_SIZE, 0:BOARD_SIZE]
    qi = yi - half
    rj = xi - half

    qi_t, rj_t = _hex_transform(qi, rj, sym)

    ti = qi_t + half
    tj = rj_t + half

    valid = (ti >= 0) & (ti < BOARD_SIZE) & (tj >= 0) & (tj < BOARD_SIZE)

    result = np.zeros_like(tensor)
    for c in range(tensor.shape[0]):
        result[c, ti[valid], tj[valid]] = tensor[c, yi[valid], xi[valid]]

    return result


def _transform_axis_label(axis_label: int, sym_idx: int) -> int:
    """Transform an unoriented axis label under a D6 symmetry."""
    if axis_label < 0:
        return axis_label
    axes = [(1, 0), (0, 1), (1, -1)]
    q, r = axes[axis_label % 3]
    tq, tr = _hex_transform(q, r, sym_idx % 12)
    for i, (aq, ar) in enumerate(axes):
        if (tq, tr) == (aq, ar) or (tq, tr) == (-aq, -ar):
            return i
    return axis_label


def _transform_dense_policy(policy: np.ndarray, sym_idx: int) -> np.ndarray:
    """Apply the same D6 transform to a dense (33*33,) policy target."""
    sym = sym_idx % 12
    half = BOARD_SIZE // 2
    result = np.zeros_like(policy)

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            value = policy[i * BOARD_SIZE + j]
            if value == 0.0:
                continue
            qi = i - half
            rj = j - half
            qi_t, rj_t = _hex_transform(qi, rj, sym)
            ti = qi_t + half
            tj = rj_t + half
            if 0 <= ti < BOARD_SIZE and 0 <= tj < BOARD_SIZE:
                result[ti * BOARD_SIZE + tj] += value

    total = result.sum()
    if total > 0:
        result /= total
    return result


def _dense_from_sparse_policy(policy: Dict[int, float]) -> np.ndarray:
    dense = np.zeros(BOARD_AREA, dtype=np.float32)
    for idx, prob in policy.items():
        if 0 <= int(idx) < BOARD_AREA:
            dense[int(idx)] = float(prob)
    return dense


def _transform_axis_maps(axis_maps: np.ndarray, sym_idx: int) -> np.ndarray:
    """Apply D6 spatial transform and axis-plane permutation to 6 axis maps.

    Planes 0..2 are own strength by unoriented axis; planes 3..5 are opponent
    strength by the same axes. A board symmetry moves both coordinates and the
    meaning of each axis plane.
    """
    if axis_maps.shape != (6, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Expected axis maps shape (6,{BOARD_SIZE},{BOARD_SIZE}), got {axis_maps.shape}")
    spatial = _py_apply_d6_symmetry(axis_maps, sym_idx)
    result = np.zeros_like(spatial)
    for src_axis in range(3):
        dst_axis = _transform_axis_label(src_axis, sym_idx)
        result[dst_axis] += spatial[src_axis]
        result[dst_axis + 3] += spatial[src_axis + 3]
    return result


class ReplayDataset(_IterableDataset):
    """Iterable dataset that samples from the ring buffer and decodes on-the-fly.

    Each DataLoader worker calls __iter__ independently. Since the buffer is
    shared, workers get different random samples automatically.
    Runs in DataLoader worker threads.
    """

    def __init__(
        self,
        buffer: RingBuffer,
        batch_size: int = 256,
        recency_decay: float = 0.99,
        pcr_weight: float = 0.25,
        use_symmetry: bool = True,
        near_radius: int = 8,
        lookahead_horizons: Optional[List[int]] = None,
        regret_fraction: float = 0.0,
        regret_temperature: float = 0.1,
        include_axis_delta_norm: bool = False,
        include_sparse_policy: bool = False,
        include_pair_policy: bool = False,
        include_graph_policy: bool = False,
        candidate_budget: int = 256,
        max_game_turns: int = 256,
    ):
        self.buffer = buffer
        self.batch_size = batch_size
        self.recency_decay = recency_decay
        self.pcr_weight = pcr_weight
        self.include_sparse_policy = bool(include_sparse_policy)
        self.include_pair_policy = bool(include_pair_policy)
        self.include_graph_policy = bool(include_graph_policy)
        self.candidate_budget = max(
            int(candidate_budget),
            int(getattr(buffer, "max_policy_v2_entries", int(candidate_budget))),
        )
        self.use_symmetry = bool(use_symmetry)
        self.near_radius = near_radius
        self.lookahead_horizons = lookahead_horizons or []
        self.regret_fraction = max(0.0, min(1.0, regret_fraction))
        self.regret_temperature = regret_temperature
        self.max_game_turns = max(1, int(max_game_turns))
        self.include_axis_delta_norm = bool(include_axis_delta_norm)
        self._axis_delta_norm_proto = (
            get_prototype("exp_delta_norm")
            if self.include_axis_delta_norm and HAS_AXIS_POLICY
            else None
        )
        self._axis_delta_norm_cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._axis_delta_norm_cache_max = 4096
        self._tensor_cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._tensor_cache_max = 4096
        self._meta_cache: OrderedDict[bytes, tuple[np.ndarray, int, int, bytes]] = OrderedDict()

        self._rng = np.random.RandomState()

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]]:
        """Yield (tensors, policies, values, lookahead_list, aux_targets) batches.

        Yields:
            tensors: (batch_size, 13, 33, 33) float32
            policies: (batch_size, BOARD_AREA) float32
            values: (batch_size,) float32
            lookahead_list: list of (batch_size,) float32 arrays, one per horizon
        """
        while True:
            batch = self._sample_batch()
            if batch is None:
                return
            yield batch

    def _sample_batch(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]]:
        """Sample one batch from the buffer. Returns None if insufficient data."""
        if len(self.buffer) < self.batch_size:
            return None

        n_regret = int(round(self.batch_size * self.regret_fraction))
        n_base = self.batch_size - n_regret
        base_indices = self.buffer.sample_indices(
            n_base,
            recency_decay=self.recency_decay,
            pcr_weight=self.pcr_weight,
        )
        regret_indices = self.buffer.sample_regret_indices(
            n_regret,
            temperature=self.regret_temperature,
        )
        indices = np.concatenate([base_indices, regret_indices])
        if len(indices) < self.batch_size:
            extra = self.buffer.sample_indices(
                self.batch_size - len(indices),
                recency_decay=self.recency_decay,
                pcr_weight=self.pcr_weight,
            )
            indices = np.concatenate([indices, extra])
        self._rng.shuffle(indices)

        records = self.buffer.get_batch(indices)
        if len(records) < self.batch_size:
            return None

        candidate_width = max(
            int(self.candidate_budget),
            max((len(getattr(rec, "policy_target_v2", [])) for rec in records), default=0),
            max((len(getattr(rec, "opp_policy_target_v2", [])) for rec in records), default=0),
            max((len(getattr(rec, "pair_policy_target_v2", [])) + 1 for rec in records), default=0)
            if self.include_pair_policy
            else 0,
            1,
        )
        candidate_width = min(candidate_width, 512)

        tensors = np.zeros(
            (self.batch_size, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype=np.float32,
        )
        policies = np.zeros((self.batch_size, BOARD_AREA), dtype=np.float32)
        values = np.zeros(self.batch_size, dtype=np.float32)
        aux_targets: Dict[str, np.ndarray] = {
            "opp_policy": np.zeros((self.batch_size, BOARD_AREA), dtype=np.float32),
            "regret_rank": np.zeros(self.batch_size, dtype=np.float32),
            "regret_value": np.zeros(self.batch_size, dtype=np.float32),
            "regret_weight": np.zeros(self.batch_size, dtype=np.float32),
            "axis": np.full(self.batch_size, -1, dtype=np.int64),
            "moves_left": np.zeros(self.batch_size, dtype=np.float32),
            "moves_left_weight": np.ones(self.batch_size, dtype=np.float32),
            "value_weight": np.ones(self.batch_size, dtype=np.float32),
            "policy_weight": np.ones(self.batch_size, dtype=np.float32),
            "sparse_policy_weight": np.ones(self.batch_size, dtype=np.float32),
            "pair_policy_weight": np.ones(self.batch_size, dtype=np.float32),
            "opp_policy_weight": np.zeros(self.batch_size, dtype=np.float32),
        }
        if self.include_sparse_policy:
            budget = candidate_width
            aux_targets["candidate_qr"] = np.zeros((self.batch_size, budget, 2), dtype=np.int32)
            aux_targets["candidate_indices"] = np.full((self.batch_size, budget), -1, dtype=np.int64)
            aux_targets["candidate_features"] = np.zeros(
                (self.batch_size, budget, CANDIDATE_FEATURES),
                dtype=np.float32,
            )
            aux_targets["candidate_mask"] = np.zeros((self.batch_size, budget), dtype=np.bool_)
            aux_targets["sparse_policy_target"] = np.zeros((self.batch_size, budget), dtype=np.float32)
            aux_targets["candidate_missing_mass"] = np.zeros(self.batch_size, dtype=np.float32)
            aux_targets["candidate_critical_count"] = np.zeros(self.batch_size, dtype=np.float32)
            aux_targets["candidate_critical_overflow_count"] = np.zeros(self.batch_size, dtype=np.float32)
            aux_targets["candidate_critical_overflow_examples"] = np.zeros(
                (self.batch_size, 8, 2),
                dtype=np.int32,
            )
            if self.include_pair_policy:
                aux_targets["pair_candidate_row_indices"] = np.full(
                    (self.batch_size, budget),
                    -1,
                    dtype=np.int64,
                )
                aux_targets["pair_candidate_features"] = np.zeros(
                    (self.batch_size, budget, CANDIDATE_FEATURES),
                    dtype=np.float32,
                )
                aux_targets["pair_candidate_row_mask"] = np.zeros((self.batch_size, budget), dtype=np.bool_)
                aux_targets["pair_candidate_indices"] = np.full(
                    (self.batch_size, budget, 2),
                    -1,
                    dtype=np.int64,
                )
                aux_targets["pair_candidate_mask"] = np.zeros((self.batch_size, budget), dtype=np.bool_)
                aux_targets["pair_policy_target"] = np.zeros((self.batch_size, budget), dtype=np.float32)
                aux_targets["pair_candidate_missing_mass"] = np.zeros(self.batch_size, dtype=np.float32)
        if self.include_axis_delta_norm:
            aux_targets["axis_delta_norm"] = np.zeros(
                (self.batch_size, 6, BOARD_SIZE, BOARD_SIZE),
                dtype=np.float32,
            )

        n_lookahead = len(self.lookahead_horizons)
        lookahead_arrays = [
            np.zeros(self.batch_size, dtype=np.float32) for _ in range(n_lookahead)
        ]

        for i, rec in enumerate(records):
            sym_idx = 0
            sample_history = rec.move_history
            policy_v2 = list(rec.policy_target_v2)
            opp_policy_v2 = list(rec.opp_policy_target_v2)
            pair_policy_v2 = list(rec.pair_policy_target_v2)
            if self.use_symmetry:
                sym_idx = self._rng.randint(0, 12)
                sample_history = _transform_history_bytes(rec.move_history, sym_idx)
                policy_v2 = _transform_policy_v2(policy_v2, sym_idx)
                opp_policy_v2 = _transform_policy_v2(opp_policy_v2, sym_idx)
                pair_policy_v2 = _transform_pair_policy_v2(pair_policy_v2, sym_idx)
                opp_legal_v2 = _transform_policy_keys(getattr(rec, "opp_policy_legal_v2", []), sym_idx)
                axis_label = _transform_axis_label(rec.axis_label, sym_idx)
            else:
                axis_label = rec.axis_label
                opp_legal_v2 = list(getattr(rec, "opp_policy_legal_v2", []))

            tensor_i, offset_q, offset_r, legal_bytes = self._encode_tensor_meta(sample_history)
            tensors[i] = tensor_i

            if policy_v2:
                policy_dict, _outside = dense_policy_from_v2(
                    policy_v2,
                    int(offset_q),
                    int(offset_r),
                    top_k=max(1, len(rec.policy_target)),
                )
                policy = _dense_from_sparse_policy(policy_dict)
            else:
                policy = rec.to_dense_policy()
                if self.use_symmetry:
                    policy = _transform_dense_policy(policy, sym_idx)

            if opp_policy_v2:
                opp_policy_dict, _outside = dense_policy_from_v2(
                    opp_policy_v2,
                    int(offset_q),
                    int(offset_r),
                    top_k=max(1, len(rec.opp_policy_target)),
                )
                opp_policy = _dense_from_sparse_policy(opp_policy_dict)
            else:
                opp_policy = rec.to_dense_opp_policy()
                if self.use_symmetry:
                    opp_policy = _transform_dense_policy(opp_policy, sym_idx)

            policies[i] = policy
            values[i] = rec.to_value_target()
            aux_targets["opp_policy"][i] = opp_policy
            aux_targets["opp_policy_weight"][i] = (
                rec.opp_policy_weight
                if rec.opp_policy_weight > 0.0
                else (1.0 if float(opp_policy.sum()) > 0.0 else 0.0)
            )
            aux_targets["regret_rank"][i] = rec.regret_rank
            aux_targets["regret_value"][i] = rec.regret_value
            aux_targets["regret_weight"][i] = rec.regret_weight
            aux_targets["axis"][i] = axis_label
            aux_targets["moves_left"][i] = np.log1p(max(float(rec.moves_left), 0.0)) / np.log1p(self.max_game_turns)
            aux_targets["moves_left_weight"][i] = 0.0 if rec.outcome == 0.0 else 1.0
            aux_targets["value_weight"][i] = rec.value_weight
            aux_targets["policy_weight"][i] = 1.0 if rec.is_full_search else 0.0
            aux_targets["sparse_policy_weight"][i] = aux_targets["policy_weight"][i]
            aux_targets["pair_policy_weight"][i] = aux_targets["policy_weight"][i]
            if int(getattr(rec, "candidate_critical_overflow_count", 0)) > 0:
                aux_targets["sparse_policy_weight"][i] = 0.0
                aux_targets["pair_policy_weight"][i] = 0.0
                aux_targets["opp_policy_weight"][i] = 0.0
            if self.include_sparse_policy:
                legal = (
                    np.frombuffer(legal_bytes, dtype=np.int32).reshape(-1, 2)
                    if legal_bytes
                    else np.empty((0, 2), dtype=np.int32)
                )
                winning_moves, forced_blocks, cover_cells = _critical_actions_from_tensor(
                    tensor_i,
                    legal,
                    int(offset_q),
                    int(offset_r),
                )
                oracle = scan_tactical_oracle_from_history(
                    sample_history,
                    [(int(q), int(r)) for q, r in legal],
                    offset_q=int(offset_q),
                    offset_r=int(offset_r),
                    near_radius=TACTICAL_SCAN_RADIUS,
                )
                cand = build_candidate_batch(
                    [(int(q), int(r)) for q, r in legal],
                    policy_v2,
                    offset_q=int(offset_q),
                    offset_r=int(offset_r),
                    budget=candidate_width,
                    storage_width=candidate_width,
                    winning_moves=list(winning_moves) + list(oracle.win_now_cells),
                    forced_block_moves=list(forced_blocks) + list(oracle.forced_block_cells),
                    cover_cells=list(cover_cells) + list(oracle.cover_cells),
                    open_four_cells=oracle.open_four_cells,
                    open_five_cells=oracle.open_five_cells,
                )
                width = min(cand.qr.shape[0], aux_targets["candidate_qr"].shape[1])
                aux_targets["candidate_qr"][i, :width] = cand.qr[:width]
                aux_targets["candidate_indices"][i, :width] = cand.indices[:width]
                aux_targets["candidate_features"][i, :width] = cand.features[:width]
                aux_targets["candidate_mask"][i, :width] = cand.mask[:width]
                critical_overflow = cand.critical_overflow_count > 0
                if not critical_overflow:
                    aux_targets["sparse_policy_target"][i, :width] = cand.target[:width]
                else:
                    logger.error(
                        "Critical candidate overflow: count=%s examples=%s history=%s",
                        cand.critical_overflow_count,
                        cand.critical_overflow_examples,
                        sample_history.hex(),
                    )
                    for ex_idx, (q, r) in enumerate(cand.critical_overflow_examples[:8]):
                        aux_targets["candidate_critical_overflow_examples"][i, ex_idx] = (q, r)
                aux_targets["candidate_critical_count"][i] = cand.critical_count
                aux_targets["candidate_critical_overflow_count"][i] = cand.critical_overflow_count
                if critical_overflow:
                    aux_targets["sparse_policy_weight"][i] = 0.0
                    aux_targets["pair_policy_weight"][i] = 0.0
                    aux_targets["opp_policy_weight"][i] = 0.0
                represented = float(aux_targets["sparse_policy_target"][i, :width].sum())
                aux_targets["candidate_missing_mass"][i] = max(
                    float(cand.missing_mass),
                    1.0 - represented,
                )
                if self.include_pair_policy and not critical_overflow:
                    legal_list = [(int(q), int(r)) for q, r in legal]
                    legal_set = set(legal_list)
                    known_first = _last_move_qr(sample_history)
                    second_targets = [
                        (second, float(prob))
                        for first, second, prob in pair_policy_v2
                        if float(prob) > 0.0
                        and known_first is not None
                        and (int(first[0]), int(first[1])) == known_first
                        and (int(first[0]), int(first[1])) not in legal_set
                        and (int(second[0]), int(second[1])) in legal_set
                    ]
                    if known_first is not None and second_targets:
                        second_rows = []
                        seen_second: set[tuple[int, int]] = set()
                        for second, _prob in second_targets:
                            qr = (int(second[0]), int(second[1]))
                            if qr not in seen_second:
                                seen_second.add(qr)
                                second_rows.append(qr)
                        for q, r in cand.qr[:width]:
                            qr = (int(q), int(r))
                            if qr in legal_set and qr not in seen_second:
                                seen_second.add(qr)
                                second_rows.append(qr)
                            if len(second_rows) >= max(0, candidate_width - 1):
                                break
                        pair_rows = [known_first] + second_rows[: max(0, candidate_width - 1)]
                        pair_cand = build_candidate_batch(
                            pair_rows,
                            [],
                            offset_q=int(offset_q),
                            offset_r=int(offset_r),
                            budget=max(1, len(pair_rows)),
                            storage_width=candidate_width,
                            critical_actions=pair_rows,
                        )
                        aux_targets["pair_candidate_row_indices"][i, :candidate_width] = pair_cand.indices[:candidate_width]
                        aux_targets["pair_candidate_features"][i, :candidate_width] = pair_cand.features[:candidate_width]
                        aux_targets["pair_candidate_row_mask"][i, :candidate_width] = pair_cand.mask[:candidate_width]
                        target_by_second: dict[tuple[int, int], float] = {}
                        total_pair_mass = 0.0
                        for second, prob in second_targets:
                            qr = (int(second[0]), int(second[1]))
                            target_by_second[qr] = target_by_second.get(qr, 0.0) + float(prob)
                            total_pair_mass += float(prob)
                        pair_indices = np.full((candidate_width, 2), -1, dtype=np.int64)
                        pair_mask = np.zeros(candidate_width, dtype=np.bool_)
                        pair_target = np.zeros(candidate_width, dtype=np.float32)
                        represented_mass = 0.0
                        row_by_qr = {
                            (int(q), int(r)): row
                            for row, (q, r) in enumerate(pair_cand.qr[:candidate_width])
                            if bool(pair_cand.mask[row])
                        }
                        first_row = row_by_qr.get(known_first, -1)
                        out_row = 0
                        for second_qr, prob in target_by_second.items():
                            second_row = row_by_qr.get(second_qr, -1)
                            if first_row < 0 or second_row < 0 or out_row >= candidate_width:
                                continue
                            pair_indices[out_row] = (first_row, second_row)
                            pair_mask[out_row] = True
                            pair_target[out_row] = float(prob)
                            represented_mass += float(prob)
                            out_row += 1
                        if represented_mass > 0.0:
                            pair_target /= represented_mass
                        pair_width = candidate_width
                        aux_targets["pair_candidate_indices"][i, :pair_width] = pair_indices[:pair_width]
                        aux_targets["pair_candidate_mask"][i, :pair_width] = pair_mask[:pair_width]
                        aux_targets["pair_policy_target"][i, :pair_width] = pair_target[:pair_width]
                        aux_targets["pair_candidate_missing_mass"][i] = max(0.0, total_pair_mass - represented_mass)
                    else:
                        pair = build_pair_candidate_batch(
                            [(int(q), int(r)) for q, r in cand.qr[:width]],
                            pair_policy_v2,
                            budget=candidate_width,
                            candidate_mask=cand.mask[:width],
                            legal_moves=legal_list,
                            known_first=known_first,
                        )
                        aux_targets["pair_candidate_row_indices"][i, :candidate_width] = aux_targets["candidate_indices"][i, :candidate_width]
                        aux_targets["pair_candidate_features"][i, :candidate_width] = aux_targets["candidate_features"][i, :candidate_width]
                        aux_targets["pair_candidate_row_mask"][i, :candidate_width] = aux_targets["candidate_mask"][i, :candidate_width]
                        pair_width = min(pair.pair_indices.shape[0], aux_targets["pair_candidate_indices"].shape[1])
                        aux_targets["pair_candidate_indices"][i, :pair_width] = pair.pair_indices[:pair_width]
                        aux_targets["pair_candidate_mask"][i, :pair_width] = pair.mask[:pair_width]
                        aux_targets["pair_policy_target"][i, :pair_width] = pair.target[:pair_width]
                        aux_targets["pair_candidate_missing_mass"][i] = pair.missing_mass
                elif self.include_pair_policy and critical_overflow:
                    aux_targets["pair_candidate_missing_mass"][i] = 1.0
            if self.include_graph_policy:
                if opp_policy_v2 and not opp_legal_v2:
                    raise ValueError("graph training requires opp_policy_legal_v2 whenever opp_policy_target_v2 is present")
                graph = build_graph_batch_from_history(
                    sample_history,
                    policy_target=policy_v2,
                    opp_legal_moves=[(int(q), int(r)) for q, r in opp_legal_v2] if opp_legal_v2 else None,
                    opp_policy_target=opp_policy_v2,
                    radius=8,
                    include_pair_rows=False,
                )
                if (
                    self.include_pair_policy
                    and int(graph.placements_remaining) >= 2
                    and not bool(getattr(rec, "pair_policy_complete", False))
                ):
                    raise ValueError(
                        "graph pair-policy training requires complete search-observed "
                        "first-placement joint pair targets; no synthetic product fallback is allowed"
                    )
                if pair_policy_v2:
                    graph = graph_batch_with_reference_pair_rows(graph, pair_policy_v2)
                graph_batch = aux_targets.setdefault("_graph_batches", [])
                graph_batch.append(graph)
            if self.include_axis_delta_norm:
                axis_delta_norm = self._compute_axis_delta_norm(sample_history)
                aux_targets["axis_delta_norm"][i] = axis_delta_norm

            for h_idx in range(n_lookahead):
                if h_idx < len(rec.lookahead_values):
                    lookahead_arrays[h_idx][i] = rec.lookahead_values[h_idx]
                else:
                    lookahead_arrays[h_idx][i] = values[i]  # fallback

        if self.include_graph_policy:
            graph_batch = collate_graph_batches(aux_targets.pop("_graph_batches"))
            aux_targets.update({
                "token_features": graph_batch.token_features,
                "token_type": graph_batch.token_type,
                "token_qr": graph_batch.token_qr,
                "token_mask": graph_batch.token_mask,
                "legal_token_indices": graph_batch.legal_token_indices,
                "legal_qr": graph_batch.legal_qr,
                "legal_mask": graph_batch.legal_mask,
                "pair_token_indices": graph_batch.pair_token_indices,
                "pair_first_indices": graph_batch.pair_first_indices,
                "pair_second_indices": graph_batch.pair_second_indices,
                "relation_type": graph_batch.relation_type,
                "relation_bias": graph_batch.relation_bias,
                "policy_target": graph_batch.policy_target,
                "opp_legal_qr": graph_batch.opp_legal_qr,
                "opp_legal_mask": graph_batch.opp_legal_mask,
                "opp_policy_target": graph_batch.opp_policy_target,
                "pair_first_policy_target": graph_batch.pair_first_policy_target,
                "pair_policy_target": graph_batch.pair_policy_target,
                "tactical_target": graph_batch.tactical_target,
            })
        return tensors, policies, values, lookahead_arrays, aux_targets

    def _encode_tensor(self, history: bytes) -> np.ndarray:
        tensor, _offset_q, _offset_r, _legal_bytes = self._encode_tensor_meta(history)
        return tensor

    def _encode_tensor_meta(self, history: bytes) -> tuple[np.ndarray, int, int, bytes]:
        cached = self._meta_cache.get(history)
        if cached is not None:
            self._meta_cache.move_to_end(history)
            return cached
        cached = self._tensor_cache.get(history)
        if cached is not None:
            self._tensor_cache.move_to_end(history)
            return cached, -16, -16, b""
        if HAS_ENGINE:
            tensor, offset_q, offset_r, legal_bytes = encode_tensor_for_history(
                history,
                near_radius=self.near_radius,
                constrain_threats=False,
            )
        else:
            decoded = _py_decode_compact_record(history, self.near_radius)
            tensor = decoded[-1] if decoded.ndim == 4 else decoded
            offset_q, offset_r = -16, -16
            legal = _fallback_legal_moves(_history_stones(history), self.near_radius)
            buf = bytearray()
            for q, r in legal:
                buf.extend(int(q).to_bytes(4, "little", signed=True))
                buf.extend(int(r).to_bytes(4, "little", signed=True))
            legal_bytes = bytes(buf)
        tensor = np.asarray(tensor, dtype=np.float32)
        self._tensor_cache[history] = tensor
        if len(self._tensor_cache) > self._tensor_cache_max:
            self._tensor_cache.popitem(last=False)
        result = (tensor, int(offset_q), int(offset_r), bytes(legal_bytes))
        self._meta_cache[history] = result
        if len(self._meta_cache) > self._tensor_cache_max:
            self._meta_cache.popitem(last=False)
        return result

    def _compute_axis_delta_norm(self, history: bytes) -> np.ndarray:
        if self._axis_delta_norm_proto is None:
            return np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        cached = self._axis_delta_norm_cache.get(history)
        if cached is not None:
            self._axis_delta_norm_cache.move_to_end(history)
            return cached
        pos = position_payload(
            get_replay_position(
                history,
                near_radius=self.near_radius,
                constrain_threats=False,
            )
        )
        axis_input = AxisPolicyInput(
            stones=pos["stones"],
            legal_moves=pos["legal_moves"],
            current_player=int(pos["current_player"]),
            offset_q=int(pos["encoding"].get("offset_q", -16)),
            offset_r=int(pos["encoding"].get("offset_r", -16)),
        )
        target = self._axis_delta_norm_proto.compute(axis_input).axis_maps.astype(np.float32)
        self._axis_delta_norm_cache[history] = target
        if len(self._axis_delta_norm_cache) > self._axis_delta_norm_cache_max:
            self._axis_delta_norm_cache.popitem(last=False)
        return target

    def __len__(self) -> int:
        """Number of batches available."""
        return max(0, len(self.buffer) // self.batch_size)
