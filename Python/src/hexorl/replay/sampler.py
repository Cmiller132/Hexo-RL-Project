"""New-record-only replay sampler."""

from __future__ import annotations

from typing import Iterator

try:
    from torch.utils.data import IterableDataset as _IterableDataset
except ImportError:
    _IterableDataset = object  # type: ignore

from hexorl.replay.codec import ReplayCodecError, ReplayPositionRecord
from hexorl.replay.projector import ProjectedReplayBatch, ReplayProjectionConfig, ReplayProjector
from hexorl.replay.storage import ReplayStorage


class ReplayDataset(_IterableDataset):
    """Iterable dataset that samples canonical replay records and projects them."""

    def __init__(
        self,
        storage: ReplayStorage,
        *,
        batch_size: int,
        recency_decay: float = 0.99,
        pcr_weight: float = 0.25,
        use_symmetry: bool = True,
        lookahead_horizons: list[int] | None = None,
        regret_fraction: float = 0.0,
        include_axis_delta_norm: bool = False,
        include_sparse_policy: bool = False,
        include_pair_policy: bool = False,
        include_graph_policy: bool = False,
        candidate_budget: int = 256,
        max_game_turns: int = 256,
    ) -> None:
        if not isinstance(storage, ReplayStorage):
            raise ReplayCodecError("ReplayDataset reads only ReplayStorage", owner="replay.sampler")
        self.storage = storage
        self.batch_size = int(batch_size)
        self.recency_decay = float(recency_decay)
        self.pcr_weight = float(pcr_weight)
        self.regret_fraction = float(regret_fraction)
        self.max_game_turns = int(max_game_turns)
        self.projector = ReplayProjector(
            ReplayProjectionConfig(
                use_symmetry=bool(use_symmetry),
                lookahead_horizons=tuple(lookahead_horizons or ()),
                include_axis_delta_norm=bool(include_axis_delta_norm),
                include_sparse_policy=bool(include_sparse_policy),
                include_pair_policy=bool(include_pair_policy),
                include_graph_policy=bool(include_graph_policy),
                candidate_budget=int(candidate_budget),
            )
        )

    def __iter__(self) -> Iterator[ProjectedReplayBatch]:
        while True:
            records = self.storage.sample_records(
                self.batch_size,
                recency_decay=self.recency_decay,
                pcr_weight=self.pcr_weight,
            )
            if len(records) < self.batch_size:
                return
            for rec in records:
                if not isinstance(rec, ReplayPositionRecord):
                    raise ReplayCodecError("sampler encountered non-canonical replay record", owner="replay.sampler")
            yield self.projector.project(records)
