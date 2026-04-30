"""Graph tensor batch collation."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from hexorl.graph.semantic_builder import GRAPH_FEATURE_DIM
from hexorl.graph.tensorize import GraphBatch


def collate_graph_batches(batches: Sequence[GraphBatch]) -> GraphBatch:
    if not batches:
        raise ValueError("cannot collate an empty graph batch list")
    max_t = max(b.token_features.shape[0] for b in batches)
    max_a = max(b.legal_qr.shape[0] for b in batches)
    max_o = max(b.opp_legal_qr.shape[0] for b in batches)
    max_p = max(b.pair_token_indices.shape[0] for b in batches)
    bsz = len(batches)

    def pad(shape, dtype, fill=0):
        return np.full(shape, fill, dtype=dtype)

    token_features = pad((bsz, max_t, GRAPH_FEATURE_DIM), np.float32)
    token_type = pad((bsz, max_t), np.int64)
    token_qr = pad((bsz, max_t, 2), np.int32)
    token_mask = pad((bsz, max_t), np.bool_)
    relation_type = pad((bsz, max_t, max_t), np.int64)
    relation_bias = pad((bsz, 1, max_t, max_t), np.float32)
    legal_token_indices = pad((bsz, max_a), np.int64, -1)
    legal_qr = pad((bsz, max_a, 2), np.int32)
    legal_mask = pad((bsz, max_a), np.bool_)
    policy_target = pad((bsz, max_a), np.float32)
    opp_legal_qr = pad((bsz, max_o, 2), np.int32)
    opp_legal_mask = pad((bsz, max_o), np.bool_)
    opp_policy_target = pad((bsz, max_o), np.float32)
    pair_first_policy_target = pad((bsz, max_a), np.float32)
    pair_token_indices = pad((bsz, max_p), np.int64, -1)
    pair_first_indices = pad((bsz, max_p), np.int64, -1)
    pair_second_indices = pad((bsz, max_p), np.int64, -1)
    pair_rows = pad((bsz, max_p, 4), np.int32)
    pair_table_mask = pad((bsz, max_p), np.bool_)
    pair_phase = pad((bsz,), np.int64)
    pair_known_first = pad((bsz, 2), np.int32)
    pair_known_first_mask = pad((bsz,), np.bool_)
    pair_policy_target = pad((bsz, max_p), np.float32)
    tactical_target = pad((bsz, 4), np.float32)

    for row, batch in enumerate(batches):
        t = batch.token_features.shape[0]
        a = batch.legal_qr.shape[0]
        o = batch.opp_legal_qr.shape[0]
        p = batch.pair_token_indices.shape[0]
        token_features[row, :t] = batch.token_features
        token_type[row, :t] = batch.token_type
        token_qr[row, :t] = batch.token_qr
        token_mask[row, :t] = True
        relation_type[row, :t, :t] = batch.relation_type
        relation_bias[row, :, :t, :t] = batch.relation_bias
        legal_token_indices[row, :a] = batch.legal_token_indices
        legal_qr[row, :a] = batch.legal_qr
        legal_mask[row, :a] = True
        policy_target[row, :a] = batch.policy_target
        opp_legal_qr[row, :o] = batch.opp_legal_qr
        opp_legal_mask[row, :o] = True
        opp_policy_target[row, :o] = batch.opp_policy_target
        pair_first_policy_target[row, :a] = batch.pair_first_policy_target
        pair_token_indices[row, :p] = batch.pair_token_indices
        pair_first_indices[row, :p] = batch.pair_first_indices
        pair_second_indices[row, :p] = batch.pair_second_indices
        if batch.pair_rows is not None:
            pair_rows[row, :p] = np.asarray(batch.pair_rows, dtype=np.int32).reshape(-1, 4)[:p]
        if batch.pair_table_mask is not None:
            pair_table_mask[row, :p] = np.asarray(batch.pair_table_mask, dtype=np.bool_).reshape(-1)[:p]
        if batch.pair_phase is not None:
            pair_phase[row] = int(np.asarray(batch.pair_phase).reshape(()))
        if batch.pair_known_first is not None:
            pair_known_first[row] = np.asarray(batch.pair_known_first, dtype=np.int32).reshape(2)
        if batch.pair_known_first_mask is not None:
            pair_known_first_mask[row] = bool(np.asarray(batch.pair_known_first_mask).reshape(()))
        pair_policy_target[row, :p] = batch.pair_policy_target
        tactical_target[row] = batch.tactical_target

    return GraphBatch(
        token_features=token_features,
        token_type=token_type,
        token_qr=token_qr,
        token_mask=token_mask,
        legal_token_indices=legal_token_indices,
        legal_qr=legal_qr,
        legal_mask=legal_mask,
        pair_token_indices=pair_token_indices,
        pair_first_indices=pair_first_indices,
        pair_second_indices=pair_second_indices,
        relation_bias=relation_bias,
        relation_type=relation_type,
        policy_target=policy_target,
        opp_legal_qr=opp_legal_qr,
        opp_legal_mask=opp_legal_mask,
        opp_policy_target=opp_policy_target,
        pair_first_policy_target=pair_first_policy_target,
        pair_policy_target=pair_policy_target,
        tactical_target=tactical_target,
        placements_remaining=-1,
        current_player=-1,
        pair_rows=pair_rows,
        pair_table_mask=pair_table_mask,
        pair_phase=pair_phase,
        pair_known_first=pair_known_first,
        pair_known_first_mask=pair_known_first_mask,
    )
