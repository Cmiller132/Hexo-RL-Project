import numpy as np
import pytest
import torch

from hexorl.config import Config
from hexorl.graph.batch import (
    GRAPH_FEATURE_DIM,
    GRAPH_FEATURE_PLACEMENTS_REMAINING,
    GRAPH_SCHEMA_VERSION,
    RELATION_SCHEMA_VERSION,
    GraphBatch,
    GraphTokenType,
    collate_graph_batches,
    graph_batch_with_admitted_pair_rows,
)
from hexorl.models.families.global_graph import GlobalHexGraphNet
from hexorl.replay.training_batch import graph_batch_training_targets
from hexorl.selfplay.records import (
    V1CandidatePair,
    V1CandidateSourceContribution,
    V1ProposalCorrectionParameters,
    V1ProposalPropensityMetadata,
    V1SearchPairMetadata,
)
from hexorl.train.loss_plan import LossContractError, build_loss_plan
from hexorl.train.losses import compute_losses
from hexorl.train.trainer import Trainer
from hexorl.train.v1_pair_targets import (
    build_v1_pair_training_targets,
    collate_v1_pair_training_targets,
)


V1_OUTPUTS = (
    "cell_marginal_logits",
    "pair_completion_logits",
    "pair_proposal_score",
    "pair_joint_logits",
    "value",
    "terminal_tactical_v1",
)


def _manual_graph() -> GraphBatch:
    token_features = np.zeros((5, GRAPH_FEATURE_DIM), dtype=np.float32)
    token_features[:, GRAPH_FEATURE_PLACEMENTS_REMAINING] = 1.0
    token_type = np.asarray(
        [
            int(GraphTokenType.STATE),
            int(GraphTokenType.TURN),
            int(GraphTokenType.LEGAL),
            int(GraphTokenType.LEGAL),
            int(GraphTokenType.LEGAL),
        ],
        dtype=np.int64,
    )
    token_qr = np.asarray([[0, 0], [0, 0], [0, 0], [1, 0], [0, 1]], dtype=np.int32)
    return GraphBatch(
        token_features=token_features,
        token_type=token_type,
        token_qr=token_qr,
        token_mask=np.ones(5, dtype=np.bool_),
        legal_token_indices=np.asarray([2, 3, 4], dtype=np.int64),
        legal_qr=token_qr[2:5].copy(),
        legal_mask=np.ones(3, dtype=np.bool_),
        pair_token_indices=np.zeros(0, dtype=np.int64),
        pair_first_indices=np.zeros(0, dtype=np.int64),
        pair_second_indices=np.zeros(0, dtype=np.int64),
        relation_bias=np.zeros((1, 5, 5), dtype=np.float32),
        relation_type=np.zeros((5, 5), dtype=np.int16),
        policy_target=np.zeros(3, dtype=np.float32),
        opp_legal_qr=np.zeros((0, 2), dtype=np.int32),
        opp_legal_mask=np.zeros(0, dtype=np.bool_),
        opp_policy_target=np.zeros(0, dtype=np.float32),
        pair_first_policy_target=np.zeros(3, dtype=np.float32),
        pair_policy_target=np.zeros(0, dtype=np.float32),
        pair_second_policy_target=np.zeros(0, dtype=np.float32),
        tactical_target=np.zeros(4, dtype=np.float32),
        placements_remaining=2,
        current_player=0,
        schema_version=GRAPH_SCHEMA_VERSION,
        relation_schema_version=RELATION_SCHEMA_VERSION,
    )


def _source(rank: int, source_type: str = "direct_pair_retrieval") -> V1CandidateSourceContribution:
    return V1CandidateSourceContribution(
        source_type=source_type,
        source_rank=rank,
        source_weight=1.0,
        local_probability_or_score=1.0 / float(rank + 1),
        quota_id=source_type,
        inclusion_kind="deterministic_top_k",
        exact_inclusion_probability=1.0,
        correction_mode="exact_importance",
    )


def _proposal(prob: float = 0.5) -> V1ProposalPropensityMetadata:
    return V1ProposalPropensityMetadata(
        proposal_policy="pair_candidate_selector_v1",
        correction_mode="exact_importance",
        total_proposal_probability=prob,
        log_proposal_probability=float(np.log(prob)),
        sampling_without_replacement=True,
    )


def _candidate(idx: int, first, second, flags=("admitted",)) -> V1CandidatePair:
    return V1CandidatePair(
        candidate_id=f"cand-{idx}",
        pair_key=(first, second),
        first_legal_row_id=idx,
        second_legal_row_id=idx + 1,
        row_table_schema_version=1,
        source_contributions=(_source(idx),),
        proposal_propensity_metadata=_proposal(),
        forced_exploration_flag="forced" in flags,
        terminal_exact_flag="terminal_exact" in flags,
        terminal_equivalence_flag="terminal_equivalent" in flags,
        target_support_flags=tuple(flags),
        admission_generation=0,
        root_or_interior="root",
    )


def _metadata() -> V1SearchPairMetadata:
    candidates = (
        _candidate(0, (0, 0), (1, 0), ("admitted",)),
        _candidate(1, (0, 0), (0, 1), ("admitted", "terminal_equivalent")),
        _candidate(2, (1, 0), (0, 1), ("explicit_negative", "sampled_negative")),
    )
    return V1SearchPairMetadata(
        candidate_selector_version="pair_candidate_selector_v1",
        support_type="admitted_candidate_set_with_explicit_negatives",
        legal_pair_count=3,
        legal_row_schema_version=1,
        pair_row_schema_version=1,
        candidate_pairs=candidates,
        proposal_correction_parameters=V1ProposalCorrectionParameters(
            correction_mode="exact_importance",
            min_log=-4.0,
            max_log=4.0,
            prior_temperature=1.0,
        ),
        root_gumbel_values=(0.2, 0.1, 0.0),
        root_admission_order=(0, 1, 2),
        root_simulation_allocation=(8, 4, 1),
        visit_counts=(6, 2, 0),
        q_values=(0.3, 0.7, -0.5),
        completed_q_values=(0.4, 0.8, -0.6),
        selected_pair=((0, 0), (1, 0)),
        target_support_flags=tuple(candidate.target_support_flags for candidate in candidates),
        terminal_equivalence_flags=tuple(candidate.terminal_equivalence_flag for candidate in candidates),
        neural_calls_per_expanded_full_turn_node=1.0,
    )


def _v1_targets_for_graph(graph: GraphBatch):
    legal_index = {
        (int(q), int(r)): row
        for row, (q, r) in enumerate(np.asarray(graph.legal_qr, dtype=np.int32).tolist())
    }
    return build_v1_pair_training_targets(
        _metadata(),
        legal_row_count=int(graph.legal_qr.shape[0]),
        legal_row_index_by_qr=legal_index,
    )


def test_v1_loss_plan_trains_all_pair_heads_and_blocks_unsampled_negative_masks():
    graph = graph_batch_with_admitted_pair_rows(_manual_graph(), _v1_targets_for_graph(_manual_graph()).candidate_pair_qr)
    batch = collate_graph_batches([graph])
    target = _v1_targets_for_graph(graph)
    targets = graph_batch_training_targets(batch)
    targets.update(
        collate_v1_pair_training_targets(
            [target],
            legal_width=int(batch.legal_qr.shape[1]),
            pair_width=int(batch.pair_first_indices.shape[1]),
        )
    )
    targets["value"] = torch.zeros(1)
    targets["value_weight"] = torch.ones(1)
    targets["policy_weight"] = torch.ones(1)
    targets["v1_pair_weight"] = torch.ones(1)
    targets["pair_row_mask"] = (torch.from_numpy(batch.pair_first_indices) >= 0) & (
        torch.from_numpy(batch.pair_second_indices) >= 0
    )

    predictions = {
        "cell_marginal_logits": torch.zeros(1, 3, requires_grad=True),
        "pair_completion_logits": torch.zeros(1, 3, requires_grad=True),
        "pair_proposal_score": torch.zeros(1, 3, requires_grad=True),
        "pair_joint_logits": torch.zeros(1, 3, requires_grad=True),
        "value": torch.zeros(1, 65, requires_grad=True),
        "terminal_tactical_v1": torch.zeros(1, 8, requires_grad=True),
    }
    weights = {name: 1.0 for name in V1_OUTPUTS}

    total, per_head = compute_losses(
        predictions,
        targets,
        loss_weights=weights,
        loss_plan=build_loss_plan(tuple(predictions), weights),
    )

    assert torch.isfinite(total)
    assert set(per_head) == set(V1_OUTPUTS)
    total.backward()
    assert predictions["pair_joint_logits"].grad is not None

    entropy_weights = {**weights, "entropy": 0.01}
    entropy_total, entropy_heads = compute_losses(
        predictions,
        targets,
        loss_weights=entropy_weights,
        loss_plan=build_loss_plan(tuple(predictions), entropy_weights),
    )
    assert torch.isfinite(entropy_total)
    assert "entropy" in entropy_heads

    bad_targets = dict(targets)
    bad_targets["v1_pair_ranking_mask"] = bad_targets["v1_pair_ranking_mask"].copy()
    bad_targets["v1_unsampled_pair_mask"] = bad_targets["v1_unsampled_pair_mask"].copy()
    bad_targets["v1_unsampled_pair_mask"][0, 2] = True
    bad_targets["v1_pair_ranking_mask"][0, 2] = True
    with pytest.raises(LossContractError, match="unsampled legal pairs"):
        compute_losses(
            predictions,
            bad_targets,
            loss_weights=weights,
            loss_plan=build_loss_plan(tuple(predictions), weights),
        )


def test_global_pair_biaffine_v1_training_smoke_runs_all_v1_losses():
    base = _manual_graph()
    target = _v1_targets_for_graph(base)
    graph = graph_batch_with_admitted_pair_rows(base, target.candidate_pair_qr)
    graph = GraphBatch(
        **{
            **graph.__dict__,
            "pair_policy_target": target.pair_joint_target.copy(),
            "pair_second_policy_target": target.pair_completion_target.copy(),
        }
    )
    batch = collate_graph_batches([graph])
    aux = graph_batch_training_targets(batch)
    aux.update(
        collate_v1_pair_training_targets(
            [target],
            legal_width=int(batch.legal_qr.shape[1]),
            pair_width=int(batch.pair_first_indices.shape[1]),
        )
    )
    aux["policy_weight"] = np.ones(1, dtype=np.float32)
    aux["value_weight"] = np.ones(1, dtype=np.float32)
    aux["v1_pair_weight"] = np.ones(1, dtype=np.float32)
    cfg = Config.model_validate(
        {
            "model": {
                "architecture": "global_pair_biaffine_0",
                "channels": 16,
                "attention_heads": 4,
                "graph_layers": 1,
                "heads": list(V1_OUTPUTS),
                "pair_strategy": "sampled_joint_pair_v1",
                "pair_strategy_max_pairs": 16,
                "pair_prior_mix": 0.35,
            },
            "selfplay": {
                "legal_row_mode": "full_rust_legal",
                "tactical_mode": "proposal_and_label",
                "constrain_threats": False,
            },
            "train": {
                "batches_per_epoch": 1,
                "graph_microbatch_size": 1,
                "loss_weights": {name: 1.0 for name in V1_OUTPUTS},
            },
            "inference": {"fp16": False},
        }
    )
    model = GlobalHexGraphNet(
        channels=16,
        heads=4,
        layers=1,
        architecture="global_pair_biaffine_0",
        output_heads=V1_OUTPUTS,
    )
    trainer = Trainer(model, cfg, dataloader=[], device=torch.device("cpu"))

    losses = trainer._train_step(
        (
            torch.zeros(1, 13, 33, 33),
            torch.zeros(1, 1089),
            torch.zeros(1),
            [],
            aux,
        ),
        0,
    )

    assert np.isfinite(losses["total"])
    for name in V1_OUTPUTS:
        assert name in losses
        assert losses[name] >= 0.0
