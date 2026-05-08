"""Explicit pair-prior strategies for search runtime behavior."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from hexorl.action_contract.candidates import build_candidate_batch
from hexorl.graph.batch import GraphBatch, GraphTokenType
from hexorl.inference.shm_queue import MAX_CANDIDATES, MAX_GRAPH_PAIRS, MAX_PAIR_CANDIDATES


PAIR_STRATEGY_NONE = "none"
PAIR_STRATEGY_ROOT_PAIR_MCTS = "root_pair_mcts"
PAIR_STRATEGY_FULL_PAIR_MCTS = "full_pair_mcts"
PAIR_STRATEGY_SAMPLED_JOINT_PAIR_V1 = "sampled_joint_pair_v1"
PAIR_STRATEGY_MODES = (
    PAIR_STRATEGY_NONE,
    PAIR_STRATEGY_ROOT_PAIR_MCTS,
    PAIR_STRATEGY_FULL_PAIR_MCTS,
    PAIR_STRATEGY_SAMPLED_JOINT_PAIR_V1,
)


@dataclass(frozen=True)
class PairStrategyConfig:
    name: str = PAIR_STRATEGY_NONE
    max_pairs: int = 0
    prior_mix: float = 0.0


@dataclass(frozen=True)
class PairStrategy:
    """Declared pair behavior used by search providers and self-play."""

    config: PairStrategyConfig
    required_output_contracts: tuple[str, ...] = ()
    pair_rows_owned: bool = False
    leaf_pair_scoring_enabled: bool = False

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def enabled(self) -> bool:
        return self.name != PAIR_STRATEGY_NONE

    @property
    def max_pairs(self) -> int:
        return int(self.config.max_pairs)

    @property
    def prior_mix(self) -> float:
        return float(self.config.prior_mix)

    def require_enabled(self, *, context: str) -> None:
        if not self.enabled:
            raise ValueError(f"{context}: pair behavior requires an explicit pair strategy")
        if self.max_pairs <= 0:
            raise ValueError(f"{context}: pair_strategy_max_pairs must be > 0")
        if self.prior_mix <= 0.0:
            raise ValueError(f"{context}: pair_prior_mix must be > 0")

    def require_leaf_pair_scoring(self, *, context: str) -> None:
        self.require_enabled(context=context)
        if not self.leaf_pair_scoring_enabled:
            raise ValueError(f"{context}: pair strategy {self.name!r} is root-only")

    def require_pair_phase(self, *, second_placement: bool, known_first: bool, context: str) -> None:
        self.require_enabled(context=context)
        if second_placement and not known_first:
            raise ValueError(f"{context}: second-placement pair strategy requires known first action")

    def graph_batch_with_pair_rows(
        self,
        graph_batch: GraphBatch,
        pair_first_indices: np.ndarray,
        pair_second_indices: np.ndarray,
    ) -> GraphBatch:
        self.require_enabled(context="graph pair row generation")
        pair_count = int(np.asarray(pair_first_indices).shape[0])
        return replace(
            graph_batch,
            pair_token_indices=np.full(pair_count, -1, dtype=np.int64),
            pair_first_indices=np.asarray(pair_first_indices, dtype=np.int64),
            pair_second_indices=np.asarray(pair_second_indices, dtype=np.int64),
            pair_policy_target=np.zeros(pair_count, dtype=np.float32),
            pair_second_policy_target=np.zeros(pair_count, dtype=np.float32),
            pair_features=None,
        )

    def score_graph_pair_chunks(
        self,
        client: Any,
        graph_batch: GraphBatch,
        *,
        second_placement: bool,
        first_qr: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score graph pair rows through bounded IPC chunks owned by the strategy."""
        self.require_pair_phase(
            second_placement=bool(second_placement),
            known_first=first_qr is not None,
            context="graph pair scoring",
        )
        legal = np.asarray(graph_batch.legal_qr, dtype=np.int32)
        legal_tokens = np.asarray(graph_batch.legal_token_indices, dtype=np.int64)
        if legal.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.int32), np.zeros(0, dtype=np.float32)

        pair_qr_chunks: list[np.ndarray] = []
        logit_chunks: list[np.ndarray] = []

        if second_placement:
            first_token = _graph_stone_token_for_qr(graph_batch, first_qr)
            first = np.asarray(first_qr, dtype=np.int32)
            scored = 0
            for start in range(0, legal.shape[0], MAX_GRAPH_PAIRS):
                if scored >= self.max_pairs:
                    break
                stop = min(start + MAX_GRAPH_PAIRS, legal.shape[0], start + (self.max_pairs - scored))
                width = stop - start
                if width <= 0:
                    break
                chunk = self.graph_batch_with_pair_rows(
                    graph_batch,
                    np.full(width, first_token, dtype=np.int64),
                    legal_tokens[start:stop],
                )
                out = client.submit_graph(chunk)
                logits = self.graph_pair_second_logits(out, width)
                pair_qr = np.column_stack([
                    np.full(width, int(first[0]), dtype=np.int32),
                    np.full(width, int(first[1]), dtype=np.int32),
                    legal[start:stop, 0],
                    legal[start:stop, 1],
                ])
                pair_qr_chunks.append(pair_qr)
                logit_chunks.append(logits)
                scored += width
        else:
            first_rows: list[int] = []
            second_rows: list[int] = []
            qr_rows: list[tuple[int, int, int, int]] = []
            scored = 0
            for a_idx in range(legal.shape[0]):
                if scored >= self.max_pairs:
                    break
                for b_idx in range(a_idx + 1, legal.shape[0]):
                    if scored >= self.max_pairs:
                        break
                    first_rows.append(int(legal_tokens[a_idx]))
                    second_rows.append(int(legal_tokens[b_idx]))
                    qr_rows.append((
                        int(legal[a_idx, 0]),
                        int(legal[a_idx, 1]),
                        int(legal[b_idx, 0]),
                        int(legal[b_idx, 1]),
                    ))
                    scored += 1
                    if len(first_rows) == MAX_GRAPH_PAIRS:
                        chunk = self.graph_batch_with_pair_rows(
                            graph_batch,
                            np.asarray(first_rows, dtype=np.int64),
                            np.asarray(second_rows, dtype=np.int64),
                        )
                        out = client.submit_graph(chunk)
                        logit_chunks.append(self.graph_pair_joint_logits(out, len(first_rows)))
                        pair_qr_chunks.append(np.asarray(qr_rows, dtype=np.int32))
                        first_rows.clear()
                        second_rows.clear()
                        qr_rows.clear()
            if first_rows:
                chunk = self.graph_batch_with_pair_rows(
                    graph_batch,
                    np.asarray(first_rows, dtype=np.int64),
                    np.asarray(second_rows, dtype=np.int64),
                )
                out = client.submit_graph(chunk)
                logit_chunks.append(self.graph_pair_joint_logits(out, len(first_rows)))
                pair_qr_chunks.append(np.asarray(qr_rows, dtype=np.int32))

        if not pair_qr_chunks:
            return np.zeros((0, 4), dtype=np.int32), np.zeros(0, dtype=np.float32)
        return np.concatenate(pair_qr_chunks, axis=0), np.concatenate(logit_chunks, axis=0)

    def score_crop_pair_chunks(
        self,
        client: Any,
        root_tensor: np.ndarray,
        legal: np.ndarray,
        *,
        offset_q: int,
        offset_r: int,
        second_placement: bool,
        first_qr: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score crop pair rows through bounded IPC chunks owned by the strategy."""
        self.require_pair_phase(
            second_placement=bool(second_placement),
            known_first=first_qr is not None,
            context="crop pair scoring",
        )
        legal = np.asarray(legal, dtype=np.int32).reshape(-1, 2)
        if legal.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.int32), np.zeros(0, dtype=np.float32)
        rows_per_chunk = max(1, min(MAX_CANDIDATES - 1, MAX_PAIR_CANDIDATES))
        pair_qr_chunks: list[np.ndarray] = []
        logit_chunks: list[np.ndarray] = []

        if second_placement:
            first = (int(first_qr[0]), int(first_qr[1]))
            scored = 0
            for start in range(0, legal.shape[0], rows_per_chunk):
                if scored >= self.max_pairs:
                    break
                stop = min(start + rows_per_chunk, legal.shape[0], start + (self.max_pairs - scored))
                seconds = [(int(q), int(r)) for q, r in legal[start:stop].tolist()]
                if not seconds:
                    break
                pair_qr, pair_logits = self._submit_crop_pair_chunk(
                    client,
                    root_tensor,
                    [first] + seconds,
                    [(first[0], first[1], second[0], second[1]) for second in seconds],
                    offset_q=offset_q,
                    offset_r=offset_r,
                )
                pair_qr_chunks.append(pair_qr)
                logit_chunks.append(pair_logits)
                scored += len(seconds)
        else:
            scored = 0
            for anchor_idx in range(max(0, legal.shape[0] - 1)):
                if scored >= self.max_pairs:
                    break
                first = (int(legal[anchor_idx, 0]), int(legal[anchor_idx, 1]))
                for start in range(anchor_idx + 1, legal.shape[0], rows_per_chunk):
                    if scored >= self.max_pairs:
                        break
                    stop = min(start + rows_per_chunk, legal.shape[0], start + (self.max_pairs - scored))
                    seconds = [(int(q), int(r)) for q, r in legal[start:stop].tolist()]
                    if not seconds:
                        break
                    pair_qr, pair_logits = self._submit_crop_pair_chunk(
                        client,
                        root_tensor,
                        [first] + seconds,
                        [(first[0], first[1], second[0], second[1]) for second in seconds],
                        offset_q=offset_q,
                        offset_r=offset_r,
                    )
                    pair_qr_chunks.append(pair_qr)
                    logit_chunks.append(pair_logits)
                    scored += len(seconds)

        if not pair_qr_chunks:
            return np.zeros((0, 4), dtype=np.int32), np.zeros(0, dtype=np.float32)
        return np.concatenate(pair_qr_chunks, axis=0), np.concatenate(logit_chunks, axis=0)

    def _submit_crop_pair_chunk(
        self,
        client: Any,
        root_tensor: np.ndarray,
        candidate_rows: list[tuple[int, int]],
        pair_rows: list[tuple[int, int, int, int]],
        *,
        offset_q: int,
        offset_r: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        indices, features, mask = _candidate_forward_rows(
            candidate_rows,
            offset_q=offset_q,
            offset_r=offset_r,
        )
        pair_indices = np.asarray([[0, idx] for idx in range(1, len(candidate_rows))], dtype=np.int64)
        pair_mask = np.ones(pair_indices.shape[0], dtype=np.bool_)
        _policy, _value, _sparse, pair_logits = client.submit_sparse_pair(
            root_tensor,
            1,
            indices.reshape(1, -1),
            features.reshape(1, features.shape[0], features.shape[1]),
            mask.reshape(1, -1),
            pair_indices.reshape(1, pair_indices.shape[0], 2),
            pair_mask.reshape(1, -1),
        )
        rows = np.asarray(pair_rows, dtype=np.int32)
        logits = _validate_logits(
            pair_logits[0, : pair_indices.shape[0]],
            rows.shape[0],
            context="crop pair scoring",
        )
        return rows, logits

    def filter_root_pair_rows(
        self,
        pair_qr: np.ndarray,
        pair_logits: np.ndarray,
        root_child_qr: np.ndarray,
        *,
        second_placement: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        pair_rows = np.asarray(pair_qr, dtype=np.int32).reshape(-1, 4)
        logits = _validate_logits(pair_logits, pair_rows.shape[0], context="root pair row filtering")
        if pair_rows.shape[0] == 0:
            return pair_rows, logits
        child_set = {(int(q), int(r)) for q, r in np.asarray(root_child_qr, dtype=np.int32).reshape(-1, 2)}
        keep = []
        for idx, row in enumerate(pair_rows):
            first = (int(row[0]), int(row[1]))
            second = (int(row[2]), int(row[3]))
            if second_placement:
                if second in child_set:
                    keep.append(idx)
            elif first in child_set and second in child_set:
                keep.append(idx)
        if not keep:
            return np.zeros((0, 4), dtype=np.int32), np.zeros(0, dtype=np.float32)
        keep_idx = np.asarray(keep, dtype=np.int64)
        return pair_rows[keep_idx], logits[keep_idx]

    def blend_action_logits(self, base_logits: np.ndarray, aux_logits: np.ndarray) -> np.ndarray:
        """Blend keyed action-logit vectors in probability space."""
        base = np.asarray(base_logits, dtype=np.float32)
        aux = np.asarray(aux_logits, dtype=np.float32)
        if base.shape != aux.shape:
            raise ValueError(f"pair logit blend shape mismatch: base={base.shape}, aux={aux.shape}")
        if base.size == 0:
            return base
        _require_finite(base, context="base action logits for pair blend")
        _require_finite(aux, context="pair action logits for pair blend")
        mix = float(np.clip(self.prior_mix, 0.0, 1.0))
        if mix <= 0.0:
            return base
        base_probs = _softmax(base)
        aux_probs = _softmax(aux)
        blended = (1.0 - mix) * base_probs + mix * aux_probs
        blended /= max(float(blended.sum()), 1e-12)
        return np.log(np.maximum(blended, 1e-12)).astype(np.float32)

    def apply_root_pair_first_priors(
        self,
        engine: Any,
        pair_first_logits: np.ndarray,
    ) -> None:
        self.require_enabled(context="root pair-first priors")
        logits = _validate_logits(
            pair_first_logits,
            np.asarray(pair_first_logits, dtype=np.float32).reshape(-1).shape[0],
            context="root pair-first priors",
        )
        engine.apply_root_pair_first_priors(
            logits,
            self.prior_mix,
        )

    def apply_root_pair_rows(
        self,
        engine: Any,
        pair_qr: np.ndarray,
        pair_logits: np.ndarray,
        *,
        placements_remaining: int,
    ) -> None:
        self.require_pair_phase(
            second_placement=int(placements_remaining) == 1,
            known_first=True,
            context="root pair priors",
        )
        rows = np.asarray(pair_qr, dtype=np.int32).reshape(-1, 4)
        logits = _validate_logits(pair_logits, rows.shape[0], context="root pair priors")
        if rows.shape[0] == 0:
            return
        if int(placements_remaining) == 1:
            engine.apply_root_pair_second_priors(rows, logits, self.prior_mix)
        elif int(placements_remaining) >= 2:
            engine.apply_root_pair_priors(rows, logits, self.prior_mix)
        else:
            raise ValueError("root pair priors require placements_remaining >= 1")

    def summary(self) -> dict[str, object]:
        return {
            "pair_strategy": self.name,
            "pair_prior_mix": self.prior_mix,
            "pair_strategy_max_pairs": self.max_pairs,
            "required_output_contracts": list(self.required_output_contracts),
            "pair_rows_owned": bool(self.pair_rows_owned),
            "leaf_pair_scoring_enabled": bool(self.leaf_pair_scoring_enabled),
        }

    @staticmethod
    def has_graph_pair_first(outputs: dict[str, object]) -> bool:
        return "policy_pair_first" in outputs

    @staticmethod
    def graph_pair_first_logits(outputs: dict[str, object], width: int) -> np.ndarray:
        if "policy_pair_first" not in outputs:
            raise ValueError("graph pair strategy requires policy_pair_first output")
        return _validate_logits(outputs["policy_pair_first"], int(width), context="graph pair-first logits")

    @staticmethod
    def graph_pair_joint_logits(outputs: dict[str, object], width: int) -> np.ndarray:
        key = "policy_pair_joint" if "policy_pair_joint" in outputs else "pair_joint_logits"
        if key not in outputs:
            raise ValueError("graph pair strategy requires pair-joint output")
        return _validate_logits(outputs[key], int(width), context="graph pair-joint logits")

    @staticmethod
    def graph_pair_second_logits(outputs: dict[str, object], width: int) -> np.ndarray:
        key = "policy_pair_second" if "policy_pair_second" in outputs else "pair_completion_logits"
        if key not in outputs:
            raise ValueError("graph pair strategy requires pair-second output")
        return _validate_logits(outputs[key], int(width), context="graph pair-second logits")


def build_pair_strategy(
    name: str,
    *,
    max_pairs: int,
    prior_mix: float,
    allow_legacy_baseline: bool = False,
) -> PairStrategy:
    normalized = str(name)
    if normalized == PAIR_STRATEGY_NONE:
        if int(max_pairs) != 0:
            raise ValueError("pair_strategy_max_pairs must be 0 when pair_strategy='none'")
        return PairStrategy(PairStrategyConfig(normalized, 0, 0.0))
    if normalized in {PAIR_STRATEGY_ROOT_PAIR_MCTS, PAIR_STRATEGY_FULL_PAIR_MCTS}:
        if not allow_legacy_baseline:
            raise ValueError(
                f"{normalized} is quarantined as a legacy/offline baseline; "
                "use sampled_joint_pair_v1 for V1 runtime self-play"
            )
        strategy = PairStrategy(
            PairStrategyConfig(normalized, int(max_pairs), float(prior_mix)),
            required_output_contracts=(
                "pair_policy",
                "policy_pair_first",
                "policy_pair_joint",
                "policy_pair_second",
            ),
            pair_rows_owned=True,
            leaf_pair_scoring_enabled=normalized == PAIR_STRATEGY_FULL_PAIR_MCTS,
        )
        strategy.require_enabled(context=normalized)
        return strategy
    if normalized == PAIR_STRATEGY_SAMPLED_JOINT_PAIR_V1:
        strategy = PairStrategy(
            PairStrategyConfig(normalized, int(max_pairs), float(prior_mix)),
            required_output_contracts=(
                "cell_marginal_logits",
                "pair_completion_logits",
                "pair_proposal_score",
                "pair_joint_logits",
                "terminal_tactical_v1",
            ),
            pair_rows_owned=True,
            leaf_pair_scoring_enabled=True,
        )
        strategy.require_enabled(context=normalized)
        return strategy
    raise ValueError(
        f"model.pair_strategy must be one of "
        f"{list(PAIR_STRATEGY_MODES)}"
    )


def build_legacy_pair_baseline_strategy(
    name: str,
    *,
    max_pairs: int,
    prior_mix: float,
) -> PairStrategy:
    """Build quarantined pre-V1 pair-prior baselines for explicit offline evaluation."""

    return build_pair_strategy(
        name,
        max_pairs=max_pairs,
        prior_mix=prior_mix,
        allow_legacy_baseline=True,
    )


def _graph_stone_token_for_qr(graph_batch: GraphBatch, qr: tuple[int, int] | None) -> int:
    if qr is None:
        raise ValueError("second-placement graph pair scoring requires first_qr")
    q, r = int(qr[0]), int(qr[1])
    for idx, (token_q, token_r) in enumerate(np.asarray(graph_batch.token_qr, dtype=np.int32).tolist()):
        if (
            int(token_q) == q
            and int(token_r) == r
            and int(graph_batch.token_type[idx]) == int(GraphTokenType.STONE)
        ):
            return idx
    raise ValueError(f"first placement {qr} is not present as a graph STONE token")


def _candidate_forward_rows(
    rows: list[tuple[int, int]],
    *,
    offset_q: int,
    offset_r: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cand = build_candidate_batch(
        rows,
        [],
        offset_q=int(offset_q),
        offset_r=int(offset_r),
        budget=max(1, len(rows)),
        storage_width=max(1, len(rows)),
        critical_actions=rows,
    )
    return cand.indices, cand.features, cand.mask


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    _require_finite(x, context="softmax logits")
    x -= np.max(x)
    exp = np.exp(x)
    return exp / max(float(exp.sum()), 1e-12)


def _validate_logits(values: object, width: int, *, context: str) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float32).reshape(-1)
    expected = int(width)
    if expected < 0:
        raise ValueError(f"{context}: expected logit width must be non-negative")
    if logits.shape[0] < expected:
        raise ValueError(f"{context}: logits have {logits.shape[0]} rows for {expected} pair rows")
    out = logits[:expected]
    _require_finite(out, context=context)
    return out


def _require_finite(values: np.ndarray, *, context: str) -> None:
    if not bool(np.all(np.isfinite(np.asarray(values)))):
        raise ValueError(f"{context}: logits must be finite")
