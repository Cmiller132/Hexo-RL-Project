"""Batch sampler — PyTorch IterableDataset for the ring buffer.

On-the-fly decode + D6 symmetry augmentation. Runs in DataLoader workers.

§7.4 of SYSTEM_DESIGN.md.
"""

import numpy as np
from typing import Dict, Iterator, Tuple, Optional, List

try:
    from torch.utils.data import IterableDataset as _IterableDataset
except ImportError:
    _IterableDataset = object  # type: ignore

from hexorl.buffer.ring import RingBuffer
from hexorl.selfplay.records import PositionRecord, BOARD_AREA, NUM_CHANNELS, BOARD_SIZE

try:
    import _engine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False


def _py_encode_from_coords(
    history_bytes: bytes,
    stride: int,
    num_moves: int,
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
    cells_p0: List[Tuple[int, int, int]] = []
    cells_p1: List[Tuple[int, int, int]] = []

    dist = np.sqrt(
        (np.arange(BOARD_SIZE)[:, None] - half) ** 2 +
        (np.arange(BOARD_SIZE)[None, :] - half) ** 2
    )

    for i in range(num_moves + 1):
        for (gqi, grj, p) in cells_p0 + cells_p1:
            gi2, gj2 = gqi + half, grj + half
            if 0 <= gi2 < BOARD_SIZE and 0 <= gj2 < BOARD_SIZE:
                if p == 0:
                    positions[i, 0, gi2, gj2] = 1.0
                else:
                    positions[i, 1, gi2, gj2] = 1.0
        positions[i, 2] = 1.0 - positions[i, 0] - positions[i, 1]
        positions[i, 11] = dist / half
        if i >= num_moves:
            break

        offset = i * stride
        player = int.from_bytes(history_bytes[offset:offset+4], 'little', signed=True)
        q = int.from_bytes(history_bytes[offset+4:offset+8], 'little', signed=True)
        r = int.from_bytes(history_bytes[offset+8:offset+12], 'little', signed=True)

        gi = q + half
        gj = r + half

        if 0 <= gi < BOARD_SIZE and 0 <= gj < BOARD_SIZE:
            if player == 0:
                cells_p0.append((q, r, 0))
            else:
                cells_p1.append((q, r, 1))

    return positions


def _py_decode_compact_record(
    history_bytes: bytes,
    near_radius: int = 8,
) -> np.ndarray:
    """Pure Python fallback for encode_compact_record.

    Replays the move history on a fresh board and encodes each position.
    Returns (N + 1, 13, 33, 33) float32 array.
    """
    if not history_bytes or len(history_bytes) < 12:
        return np.zeros((1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    stride = 12
    num_moves = len(history_bytes) // stride
    if num_moves == 0:
        return np.zeros((1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    positions = _py_encode_from_coords(history_bytes, stride, num_moves)
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
        return (-qi, qi + rj)
    elif sym == 7:
        return (-qi - rj, -qi)
    elif sym == 8:
        return (-rj, -qi - rj)
    elif sym == 9:
        return (qi, -qi - rj)
    elif sym == 10:
        return (qi + rj, rj)
    else:
        return (rj, qi)


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
        result[c][valid] = tensor[c, ti[valid], tj[valid]]

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
    ):
        self.buffer = buffer
        self.batch_size = batch_size
        self.recency_decay = recency_decay
        self.pcr_weight = pcr_weight
        self.use_symmetry = use_symmetry
        self.near_radius = near_radius
        self.lookahead_horizons = lookahead_horizons or []

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

        indices = self.buffer.sample_indices(
            self.batch_size,
            recency_decay=self.recency_decay,
            pcr_weight=self.pcr_weight,
        )

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
        }

        n_lookahead = len(self.lookahead_horizons)
        lookahead_arrays = [
            np.zeros(self.batch_size, dtype=np.float32) for _ in range(n_lookahead)
        ]

        for i, rec in enumerate(records):
            if HAS_ENGINE and hasattr(_engine, 'encode_compact_record'):
                tensor = np.array(
                    _engine.encode_compact_record(rec.move_history, self.near_radius),
                    dtype=np.float32,
                )
                if tensor.ndim == 4:
                    tensor = tensor[-1]
                tensors[i] = tensor
            else:
                decoded = _py_decode_compact_record(rec.move_history, self.near_radius)
                if decoded.ndim == 4:
                    tensors[i] = decoded[-1]
                else:
                    tensors[i] = decoded

            if self.use_symmetry:
                sym_idx = self._rng.randint(0, 12)
                if HAS_ENGINE and hasattr(_engine, 'apply_d6_symmetry'):
                    tensors[i] = np.array(
                        _engine.apply_d6_symmetry(tensors[i], sym_idx),
                        dtype=np.float32,
                    )
                else:
                    tensors[i] = _py_apply_d6_symmetry(tensors[i], sym_idx)

            policies[i] = rec.to_dense_policy()
            values[i] = rec.to_value_target()
            aux_targets["opp_policy"][i] = rec.to_dense_opp_policy()
            aux_targets["regret_rank"][i] = rec.regret_rank
            aux_targets["regret_value"][i] = rec.regret_value
            aux_targets["axis"][i] = rec.axis_label
            aux_targets["moves_left"][i] = rec.moves_left

            for h_idx in range(n_lookahead):
                if h_idx < len(rec.lookahead_values):
                    lookahead_arrays[h_idx][i] = rec.lookahead_values[h_idx]
                else:
                    lookahead_arrays[h_idx][i] = values[i]  # fallback

        return tensors, policies, values, lookahead_arrays, aux_targets

    def __len__(self) -> int:
        """Number of batches available."""
        return max(0, len(self.buffer) // self.batch_size)
