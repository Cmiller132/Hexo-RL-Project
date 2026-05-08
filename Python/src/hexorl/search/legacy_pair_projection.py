"""Legacy pair-to-single projection helpers for non-V1 baseline modes."""

from __future__ import annotations

import numpy as np


def pair_logits_to_action_logits(
    pair_qr: np.ndarray,
    pair_logits: np.ndarray,
    legal: np.ndarray,
) -> np.ndarray:
    legal = np.asarray(legal, dtype=np.int32).reshape(-1, 2)
    out = np.full(legal.shape[0], -80.0, dtype=np.float32)
    if legal.shape[0] == 0 or pair_qr.size == 0 or pair_logits.size == 0:
        return out
    legal_index = {(int(q), int(r)): idx for idx, (q, r) in enumerate(legal.tolist())}
    logits = np.asarray(pair_logits, dtype=np.float32).reshape(-1)
    rows = np.asarray(pair_qr, dtype=np.int32).reshape(-1, 4)
    if logits.shape[0] < rows.shape[0]:
        raise ValueError(
            f"legacy pair projection logits have {logits.shape[0]} rows for {rows.shape[0]} pair rows"
        )
    logits = logits[: rows.shape[0]]
    if not np.all(np.isfinite(logits)):
        raise ValueError("legacy pair projection logits must be finite")
    exp = np.exp(logits - np.max(logits))
    denom = max(float(exp.sum()), 1e-12)
    mass = np.zeros(legal.shape[0], dtype=np.float32)
    for row, prob in zip(rows, exp / denom):
        first = legal_index.get((int(row[0]), int(row[1])))
        second = legal_index.get((int(row[2]), int(row[3])))
        if first is not None:
            mass[first] += float(prob)
        if second is not None:
            mass[second] += float(prob)
    total = float(mass.sum())
    if total > 0.0:
        mass /= total
        out = np.log(np.maximum(mass, 1e-12)).astype(np.float32)
    return out
