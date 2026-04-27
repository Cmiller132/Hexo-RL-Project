"""Batch sampler — PyTorch IterableDataset for the ring buffer.

On-the-fly decode + D6 symmetry augmentation. Runs in DataLoader workers.

§7.4 of SYSTEM_DESIGN.md.
"""

from collections import OrderedDict
import numpy as np
from typing import Dict, Iterator, Tuple, Optional, List

try:
    from torch.utils.data import IterableDataset as _IterableDataset
except ImportError:
    _IterableDataset = object  # type: ignore

from hexorl.buffer.ring import RingBuffer
from hexorl.selfplay.records import BOARD_AREA, NUM_CHANNELS, BOARD_SIZE

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
    ):
        self.buffer = buffer
        self.batch_size = batch_size
        self.recency_decay = recency_decay
        self.pcr_weight = pcr_weight
        self.use_symmetry = use_symmetry
        self.near_radius = near_radius
        self.lookahead_horizons = lookahead_horizons or []
        self.regret_fraction = max(0.0, min(1.0, regret_fraction))
        self.regret_temperature = regret_temperature
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
            "axis": np.full(self.batch_size, -1, dtype=np.int64),
            "moves_left": np.zeros(self.batch_size, dtype=np.float32),
            "value_weight": np.ones(self.batch_size, dtype=np.float32),
            "policy_weight": np.ones(self.batch_size, dtype=np.float32),
        }
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
            policy = rec.to_dense_policy()
            opp_policy = rec.to_dense_opp_policy()
            sym_idx = 0
            tensors[i] = self._encode_tensor(rec.move_history)

            if self.use_symmetry:
                sym_idx = self._rng.randint(0, 12)
                if HAS_ENGINE and hasattr(_engine, 'apply_d6_symmetry'):
                    tensors[i] = np.array(
                        _engine.apply_d6_symmetry(tensors[i], sym_idx),
                        dtype=np.float32,
                    )
                else:
                    tensors[i] = _py_apply_d6_symmetry(tensors[i], sym_idx)
                policy = _transform_dense_policy(policy, sym_idx)
                opp_policy = _transform_dense_policy(opp_policy, sym_idx)
                axis_label = _transform_axis_label(rec.axis_label, sym_idx)
            else:
                axis_label = rec.axis_label

            policies[i] = policy
            values[i] = rec.to_value_target()
            aux_targets["opp_policy"][i] = opp_policy
            aux_targets["regret_rank"][i] = rec.regret_rank
            aux_targets["regret_value"][i] = rec.regret_value
            aux_targets["axis"][i] = axis_label
            aux_targets["moves_left"][i] = rec.moves_left
            aux_targets["value_weight"][i] = rec.value_weight
            aux_targets["policy_weight"][i] = 1.0 if rec.is_full_search else 0.0
            if self.include_axis_delta_norm:
                axis_delta_norm = self._compute_axis_delta_norm(rec)
                if self.use_symmetry:
                    axis_delta_norm = _transform_axis_maps(axis_delta_norm, sym_idx)
                aux_targets["axis_delta_norm"][i] = axis_delta_norm

            for h_idx in range(n_lookahead):
                if h_idx < len(rec.lookahead_values):
                    lookahead_arrays[h_idx][i] = rec.lookahead_values[h_idx]
                else:
                    lookahead_arrays[h_idx][i] = values[i]  # fallback

        return tensors, policies, values, lookahead_arrays, aux_targets

    def _encode_tensor(self, history: bytes) -> np.ndarray:
        cached = self._tensor_cache.get(history)
        if cached is not None:
            self._tensor_cache.move_to_end(history)
            return cached
        if HAS_ENGINE:
            tensor, _offset_q, _offset_r, _legal_bytes = encode_tensor_for_history(
                history,
                near_radius=self.near_radius,
                constrain_threats=False,
            )
        else:
            decoded = _py_decode_compact_record(history, self.near_radius)
            tensor = decoded[-1] if decoded.ndim == 4 else decoded
        tensor = np.asarray(tensor, dtype=np.float32)
        self._tensor_cache[history] = tensor
        if len(self._tensor_cache) > self._tensor_cache_max:
            self._tensor_cache.popitem(last=False)
        return tensor

    def _compute_axis_delta_norm(self, rec) -> np.ndarray:
        if self._axis_delta_norm_proto is None:
            return np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        cached = self._axis_delta_norm_cache.get(rec.move_history)
        if cached is not None:
            self._axis_delta_norm_cache.move_to_end(rec.move_history)
            return cached
        pos = position_payload(
            get_replay_position(
                rec.move_history,
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
        self._axis_delta_norm_cache[rec.move_history] = target
        if len(self._axis_delta_norm_cache) > self._axis_delta_norm_cache_max:
            self._axis_delta_norm_cache.popitem(last=False)
        return target

    def __len__(self) -> int:
        """Number of batches available."""
        return max(0, len(self.buffer) // self.batch_size)
