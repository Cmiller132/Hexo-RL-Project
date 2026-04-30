"""Server-side shared-memory slot collation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from hexorl.config import Config
from hexorl.inference.shm_queue import (
    CANDIDATE_FEATURES,
    GRAPH_SCHEMA_VERSION,
    MAX_GRAPH_ACTIONS,
    MAX_GRAPH_PAIRS,
    MAX_GRAPH_TOKENS,
    MAX_PAIR_CANDIDATES,
    RELATION_SCHEMA_VERSION,
)


@dataclass(frozen=True)
class CollatedBatch:
    ready_workers: list[int]
    per_worker_counts: list[int]
    total_count: int
    dense_tensor: torch.Tensor | None = None
    sparse_inputs: dict[str, torch.Tensor] | None = None
    graph_inputs: dict[str, torch.Tensor] | None = None

    @property
    def is_graph(self) -> bool:
        return self.graph_inputs is not None


class ServerCollator:
    def __init__(self, *, cfg: Config, queue, device: torch.device, max_batch: int, global_graph_kind: bool):
        self.cfg = cfg
        self.queue = queue
        self.device = device
        self.max_batch = int(max_batch)
        self.global_graph_kind = bool(global_graph_kind)
        self.host_batch_tensor: torch.Tensor | None = None
        self.host_batch_np: np.ndarray | None = None
        self.prepare_host_batch()

    def prepare_host_batch(self) -> None:
        pin = self.device.type == "cuda"
        try:
            self.host_batch_tensor = torch.empty(
                (self.max_batch, 13, 33, 33),
                dtype=torch.float32,
                pin_memory=pin,
            )
            self.host_batch_np = self.host_batch_tensor.numpy()
        except Exception:
            self.host_batch_np = np.empty((self.max_batch, 13, 33, 33), dtype=np.float32)
            self.host_batch_tensor = torch.from_numpy(self.host_batch_np)

    def collate_dense(self, ready_workers: list[int]) -> CollatedBatch:
        batch_tensor, counts, total = self._build_dense_tensor(ready_workers)
        sparse_inputs = self._build_sparse_inputs(ready_workers, counts, total)
        return CollatedBatch(
            ready_workers=ready_workers,
            per_worker_counts=counts,
            total_count=total,
            dense_tensor=batch_tensor,
            sparse_inputs=sparse_inputs,
        )

    def collate_graph(self, ready_workers: list[int]) -> CollatedBatch:
        graph_inputs, counts, total = self._build_graph_inputs(ready_workers)
        return CollatedBatch(
            ready_workers=ready_workers,
            per_worker_counts=counts,
            total_count=total,
            graph_inputs=graph_inputs,
        )

    def _build_dense_tensor(self, ready_workers: list[int]) -> tuple[torch.Tensor, list[int], int]:
        counts = []
        total = 0
        for worker_id in ready_workers:
            slot = self.queue.get_slot(worker_id)
            c = int(slot.req_count[0])
            if c > 0:
                if self.host_batch_np is None or self.host_batch_tensor is None:
                    raise RuntimeError("host batch buffer was not initialized")
                np.copyto(self.host_batch_np[total:total + c], slot.req_tensor[:c])
                total += c
                counts.append(c)

        if total == 0:
            return torch.empty(0), [], 0

        if self.host_batch_tensor is None:
            raise RuntimeError("host batch tensor was not initialized")
        batch_tensor = self.host_batch_tensor[:total].to(self.device, non_blocking=True)
        if self.device.type == "cuda" and getattr(self.cfg.runtime, "channels_last", True):
            batch_tensor = batch_tensor.contiguous(memory_format=torch.channels_last)
        return batch_tensor, counts, total

    def _build_graph_inputs(self, ready_workers: list[int]) -> tuple[dict[str, torch.Tensor], list[int], int]:
        if not self.global_graph_kind:
            raise RuntimeError("graph IPC request received by a non-global-graph inference server")
        metas = []
        for worker_id in ready_workers:
            slot = self.queue.get_slot(worker_id)
            if int(slot.req_count[0]) != 1:
                raise ValueError("graph IPC supports exactly one graph position per worker request")
            meta = np.array(slot.req_graph_meta, copy=True)
            if int(meta[0]) != GRAPH_SCHEMA_VERSION or int(meta[1]) != RELATION_SCHEMA_VERSION:
                raise ValueError(
                    "graph IPC schema mismatch: "
                    f"got ({int(meta[0])}, {int(meta[1])}), "
                    f"expected ({GRAPH_SCHEMA_VERSION}, {RELATION_SCHEMA_VERSION})"
                )
            token_count, legal_count, opp_count, pair_count = map(int, meta[2:6])
            if token_count > MAX_GRAPH_TOKENS or legal_count > MAX_GRAPH_ACTIONS:
                raise ValueError("graph request exceeds shared-memory token/legal capacity")
            if opp_count > MAX_GRAPH_ACTIONS or pair_count > MAX_GRAPH_PAIRS:
                raise ValueError("graph request exceeds shared-memory opponent/pair capacity")
            metas.append(meta)
        total = len(ready_workers)
        if total == 0:
            return {}, [], 0
        max_t = max(int(meta[2]) for meta in metas)
        max_a = max(int(meta[3]) for meta in metas)
        max_o = max(int(meta[4]) for meta in metas)
        max_p = max(int(meta[5]) for meta in metas)

        token_features = np.zeros((total, max_t, self.queue.get_slot(ready_workers[0]).req_graph_token_features.shape[1]), dtype=np.float32)
        token_type = np.zeros((total, max_t), dtype=np.int64)
        token_qr = np.zeros((total, max_t, 2), dtype=np.int32)
        token_mask = np.zeros((total, max_t), dtype=np.bool_)
        legal_token_indices = np.full((total, max_a), -1, dtype=np.int64)
        legal_mask = np.zeros((total, max_a), dtype=np.bool_)
        opp_legal_qr = np.zeros((total, max_o, 2), dtype=np.int32)
        opp_legal_mask = np.zeros((total, max_o), dtype=np.bool_)
        pair_token_indices = np.full((total, max_p), -1, dtype=np.int64)
        pair_first_indices = np.full((total, max_p), -1, dtype=np.int64)
        pair_second_indices = np.full((total, max_p), -1, dtype=np.int64)
        relation_type = np.zeros((total, max_t, max_t), dtype=np.int64)
        relation_bias = np.zeros((total, 1, max_t, max_t), dtype=np.float32)

        for row, worker_id in enumerate(ready_workers):
            slot = self.queue.get_slot(worker_id)
            t, a, o, p = map(int, metas[row][2:6])
            token_features[row, :t] = slot.req_graph_token_features[:t]
            token_type[row, :t] = slot.req_graph_token_type[:t].astype(np.int64)
            token_qr[row, :t] = slot.req_graph_token_qr[:t]
            token_mask[row, :t] = slot.req_graph_token_mask[:t].astype(bool)
            legal_token_indices[row, :a] = slot.req_graph_legal_token_indices[:a]
            legal_mask[row, :a] = slot.req_graph_legal_mask[:a].astype(bool)
            if o:
                opp_legal_qr[row, :o] = slot.req_graph_opp_legal_qr[:o]
                opp_legal_mask[row, :o] = slot.req_graph_opp_legal_mask[:o].astype(bool)
            if p:
                pair_token_indices[row, :p] = slot.req_graph_pair_token_indices[:p]
                pair_first_indices[row, :p] = slot.req_graph_pair_first_indices[:p]
                pair_second_indices[row, :p] = slot.req_graph_pair_second_indices[:p]
            relation_type[row, :t, :t] = slot.req_graph_relation_type[:t, :t].astype(np.int64)
            relation_bias[row, :, :t, :t] = slot.req_graph_relation_bias[:, :t, :t]

        return (
            {
                "token_features": torch.from_numpy(token_features).to(self.device, non_blocking=True),
                "token_type": torch.from_numpy(token_type).to(self.device, non_blocking=True),
                "token_qr": torch.from_numpy(token_qr).to(self.device, non_blocking=True),
                "token_mask": torch.from_numpy(token_mask).to(self.device, non_blocking=True),
                "legal_token_indices": torch.from_numpy(legal_token_indices).to(self.device, non_blocking=True),
                "legal_mask": torch.from_numpy(legal_mask).to(self.device, non_blocking=True),
                "opp_legal_qr": torch.from_numpy(opp_legal_qr).to(self.device, non_blocking=True),
                "opp_legal_mask": torch.from_numpy(opp_legal_mask).to(self.device, non_blocking=True),
                "pair_token_indices": torch.from_numpy(pair_token_indices).to(self.device, non_blocking=True),
                "pair_first_indices": torch.from_numpy(pair_first_indices).to(self.device, non_blocking=True),
                "pair_second_indices": torch.from_numpy(pair_second_indices).to(self.device, non_blocking=True),
                "relation_type": torch.from_numpy(relation_type).to(self.device, non_blocking=True),
                "relation_bias": torch.from_numpy(relation_bias).to(self.device, non_blocking=True),
            },
            [1 for _ in ready_workers],
            total,
        )

    def _build_sparse_inputs(
        self,
        ready_workers: list[int],
        per_worker_counts: list[int],
        total_count: int,
    ) -> Optional[dict[str, torch.Tensor]]:
        if total_count <= 0:
            return None
        max_k = 0
        for worker_id, count in zip(ready_workers, per_worker_counts):
            slot = self.queue.get_slot(worker_id)
            if count > 0:
                counts = slot.req_candidate_count[:count]
                max_k = max(max_k, int(counts.max()) if counts.size else 0)
        max_p = 0
        for worker_id, count in zip(ready_workers, per_worker_counts):
            slot = self.queue.get_slot(worker_id)
            if count > 0 and getattr(slot, "req_pair_count", None) is not None:
                counts = slot.req_pair_count[:count]
                max_p = max(max_p, int(counts.max()) if counts.size else 0)
        if max_k <= 0 and max_p <= 0:
            return None

        max_k = max(max_k, 1)
        indices = np.full((total_count, max_k), -1, dtype=np.int64)
        features = np.zeros((total_count, max_k, CANDIDATE_FEATURES), dtype=np.float32)
        mask = np.zeros((total_count, max_k), dtype=np.bool_)
        max_p = max(max_p, 0)
        pair_indices = np.full((total_count, max_p, 2), -1, dtype=np.int64) if max_p > 0 else None
        pair_mask = np.zeros((total_count, max_p), dtype=np.bool_) if max_p > 0 else None
        offset = 0
        for worker_id, count in zip(ready_workers, per_worker_counts):
            slot = self.queue.get_slot(worker_id)
            for row in range(count):
                k = int(slot.req_candidate_count[row])
                if k > 0:
                    kk = min(k, max_k)
                    indices[offset + row, :kk] = slot.req_candidate_indices[row, :kk]
                    features[offset + row, :kk] = slot.req_candidate_features[row, :kk]
                    mask[offset + row, :kk] = slot.req_candidate_mask[row, :kk].astype(bool)
                if pair_indices is not None and pair_mask is not None:
                    p = int(slot.req_pair_count[row])
                    if p > 0:
                        pp = min(p, max_p, MAX_PAIR_CANDIDATES)
                        pair_indices[offset + row, :pp] = slot.req_pair_indices[row, :pp]
                        pair_mask[offset + row, :pp] = slot.req_pair_mask[row, :pp].astype(bool)
            offset += count
        out = {
            "candidate_indices": torch.from_numpy(indices).to(self.device, non_blocking=True),
            "candidate_features": torch.from_numpy(features).to(self.device, non_blocking=True),
            "candidate_mask": torch.from_numpy(mask).to(self.device, non_blocking=True),
        }
        if pair_indices is not None and pair_mask is not None:
            out["pair_candidate_indices"] = torch.from_numpy(pair_indices).to(self.device, non_blocking=True)
            out["pair_candidate_mask"] = torch.from_numpy(pair_mask).to(self.device, non_blocking=True)
        return out


__all__ = ["CollatedBatch", "ServerCollator"]
