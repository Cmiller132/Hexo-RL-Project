"""Server-side shared-memory response scattering."""

from __future__ import annotations

from hexorl.inference.server.outputs import DenseForwardOutputs, GraphForwardOutputs
from hexorl.inference.shm_queue import (
    GRAPH_SCHEMA_VERSION,
    MAX_GRAPH_ACTIONS,
    MAX_GRAPH_TOKENS,
    RELATION_SCHEMA_VERSION,
)


class ServerScatterer:
    def __init__(self, *, queue):
        self.queue = queue

    def scatter_dense(
        self,
        *,
        ready_workers: list[int],
        per_worker_counts: list[int],
        outputs: DenseForwardOutputs,
    ) -> None:
        offset = 0
        for worker_id, count in zip(ready_workers, per_worker_counts):
            slot = self.queue.get_slot(worker_id)
            slot.res_policy[:count] = outputs.policies[offset:offset + count]
            slot.res_value[:count] = outputs.values[offset:offset + count]
            if getattr(slot, "res_regret_rank", None) is not None:
                slot.res_regret_rank[:count] = 0.0
                if outputs.regret_rank is not None:
                    slot.res_regret_rank[:count] = outputs.regret_rank[offset:offset + count]
            if getattr(slot, "res_regret_value", None) is not None:
                slot.res_regret_value[:count] = 0.0
                if outputs.regret_value is not None:
                    slot.res_regret_value[:count] = outputs.regret_value[offset:offset + count]
            if outputs.sparse_logits is not None:
                k = outputs.sparse_logits.shape[1]
                slot.res_sparse_logits[:count, :k] = outputs.sparse_logits[offset:offset + count]
            if outputs.pair_logits is not None:
                p = outputs.pair_logits.shape[1]
                slot.res_pair_logits[:count, :p] = outputs.pair_logits[offset:offset + count]
            slot.req_kind[0] = 0
            offset += count

    def scatter_graph(
        self,
        *,
        ready_workers: list[int],
        outputs: GraphForwardOutputs,
    ) -> None:
        for row, worker_id in enumerate(ready_workers):
            slot = self.queue.get_slot(worker_id)
            token_count, legal_count, opp_count, pair_count = map(int, slot.req_graph_meta[2:6])
            if legal_count > outputs.place_logits.shape[1]:
                raise ValueError("graph place logits shorter than legal row table")
            if pair_count and (outputs.pair_joint_logits is None or outputs.pair_second_logits is None):
                raise ValueError("graph model did not return joint/second pair logits for a pair request")
            slot.res_value[0] = float(outputs.values[row])
            slot.res_graph_meta[:] = (
                GRAPH_SCHEMA_VERSION,
                RELATION_SCHEMA_VERSION,
                legal_count,
                opp_count,
                pair_count,
                token_count,
                MAX_GRAPH_TOKENS,
                MAX_GRAPH_ACTIONS,
            )
            slot.res_graph_place_logits.fill(0.0)
            slot.res_graph_opp_logits.fill(0.0)
            slot.res_graph_pair_first_logits.fill(0.0)
            slot.res_graph_pair_logits.fill(0.0)
            slot.res_graph_pair_second_logits.fill(0.0)
            if getattr(slot, "res_graph_regret_rank", None) is not None:
                slot.res_graph_regret_rank[0] = (
                    float(outputs.regret_rank[row])
                    if outputs.regret_rank is not None and row < len(outputs.regret_rank)
                    else 0.0
                )
            if getattr(slot, "res_graph_regret_value", None) is not None:
                slot.res_graph_regret_value[0] = (
                    float(outputs.regret_value[row])
                    if outputs.regret_value is not None and row < len(outputs.regret_value)
                    else 0.0
                )
            slot.res_graph_place_logits[:legal_count] = outputs.place_logits[row, :legal_count]
            if opp_count and outputs.opp_logits is not None:
                slot.res_graph_opp_logits[:opp_count] = outputs.opp_logits[row, :opp_count]
            if legal_count and outputs.pair_first_logits is not None:
                slot.res_graph_pair_first_logits[:legal_count] = outputs.pair_first_logits[row, :legal_count]
            if pair_count and outputs.pair_joint_logits is not None:
                slot.res_graph_pair_logits[:pair_count] = outputs.pair_joint_logits[row, :pair_count]
            if pair_count and outputs.pair_second_logits is not None:
                slot.res_graph_pair_second_logits[:pair_count] = outputs.pair_second_logits[row, :pair_count]
            slot.req_kind[0] = 0


__all__ = ["ServerScatterer"]
