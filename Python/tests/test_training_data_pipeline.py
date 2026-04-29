import struct

import numpy as np
import pytest
import torch

from hexorl.buffer.ring import RingBuffer
from hexorl.buffer.regret_buffer import compute_regret
from hexorl.buffer.sampler import (
    _py_apply_d6_symmetry,
    _py_decode_compact_record,
    _hex_transform,
    _transform_pair_policy_v2,
    _transform_axis_maps,
    _transform_axis_label,
    _transform_dense_policy,
    ReplayDataset,
)
from hexorl.buffer.targets import (
    _turn_boundary_indices,
    compute_ema_lookahead,
    hexo_turn_start_indices,
    process_game_record,
    value_from_source_perspective,
)
from hexorl.config import Config
from hexorl.epoch import pipeline
from hexorl.selfplay.orchestrator import SelfPlayOrchestrator
from hexorl.selfplay.records import (
    GameRecord,
    PositionRecord,
    BOARD_SIZE,
    action_to_board_index,
    dense_policy_from_v2,
    pair_policy_v2_from_place_target,
    policy_v2_from_visits,
    sparsify_policy,
)
from hexorl.train.losses import binned_value_loss, compute_losses, policy_loss, sparse_policy_loss
from hexorl.model.network import HexNet


def _move(player: int, q: int, r: int) -> bytes:
    return struct.pack("<iii", player, q, r)


class _FixedSymmetryRng:
    def __init__(self, sym_idx: int):
        self.sym_idx = sym_idx

    def randint(self, _low, _high=None):
        return self.sym_idx

    def shuffle(self, values):
        return None


def test_python_decoder_returns_final_position_for_history():
    history = _move(0, 0, 0)
    decoded = _py_decode_compact_record(history)

    assert decoded.shape == (2, 13, BOARD_SIZE, BOARD_SIZE)
    assert decoded[0, 0].sum() == 0.0
    assert decoded[0, 2].sum() == BOARD_SIZE * BOARD_SIZE
    assert decoded[0, 3, BOARD_SIZE // 2, BOARD_SIZE // 2] == 1.0
    assert decoded[0, 6].sum() == BOARD_SIZE * BOARD_SIZE
    assert decoded[-1, 1, BOARD_SIZE // 2, BOARD_SIZE // 2] == 1.0
    assert decoded[-1, 6].sum() == 0.0


def test_policy_symmetry_transform_tracks_dense_target():
    policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    src_i = BOARD_SIZE // 2 + 1
    src_j = BOARD_SIZE // 2
    policy[src_i * BOARD_SIZE + src_j] = 1.0

    transformed = _transform_dense_policy(policy, sym_idx=3)

    dst_i = BOARD_SIZE // 2 - 1
    dst_j = BOARD_SIZE // 2
    assert transformed[dst_i * BOARD_SIZE + dst_j] == 1.0
    assert transformed.sum() == 1.0


def test_tensor_and_policy_symmetry_match_for_all_transforms():
    src_i = BOARD_SIZE // 2 + 2
    src_j = BOARD_SIZE // 2 - 1
    tensor = np.zeros((13, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    tensor[0, src_i, src_j] = 1.0
    policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    policy[src_i * BOARD_SIZE + src_j] = 1.0

    for sym_idx in range(12):
        transformed_tensor = _py_apply_d6_symmetry(tensor, sym_idx)
        transformed_policy = _transform_dense_policy(policy, sym_idx)
        tensor_idx = int(transformed_tensor[0].argmax())
        policy_idx = int(transformed_policy.argmax())
        assert tensor_idx == policy_idx


def test_axis_label_symmetry_transform_remains_valid():
    for axis in range(3):
        for sym_idx in range(12):
            assert _transform_axis_label(axis, sym_idx) in {0, 1, 2}


def test_each_symmetry_permutates_axes_one_to_one():
    for sym_idx in range(12):
        mapped = [_transform_axis_label(axis, sym_idx) for axis in range(3)]
        assert sorted(mapped) == [0, 1, 2]


def test_axis_delta_maps_symmetry_transforms_space_and_axis_planes():
    maps = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    src_i = BOARD_SIZE // 2 + 1
    src_j = BOARD_SIZE // 2
    maps[0, src_i, src_j] = 2.0
    maps[3, src_i, src_j] = 3.0

    transformed = _transform_axis_maps(maps, sym_idx=1)

    dst_axis = _transform_axis_label(0, 1)
    dst_i = BOARD_SIZE // 2
    dst_j = BOARD_SIZE // 2 + 1
    assert transformed[dst_axis, dst_i, dst_j] == 2.0
    assert transformed[dst_axis + 3, dst_i, dst_j] == 3.0
    assert transformed.sum() == 5.0


def test_hexo_turn_boundaries_follow_player_runs():
    positions = [
        PositionRecord(b"", {1: 1.0}, 0.1, player=0),
        PositionRecord(b"", {2: 1.0}, 0.2, player=1),
        PositionRecord(b"", {3: 1.0}, 0.3, player=1),
        PositionRecord(b"", {4: 1.0}, 0.4, player=0),
        PositionRecord(b"", {5: 1.0}, 0.5, player=0),
    ]

    assert _turn_boundary_indices(positions) == [0, 1, 3]
    assert hexo_turn_start_indices(positions) == [0, 1, 3]


def test_random_histories_have_stable_hexo_turn_starts():
    rng = np.random.default_rng(20260428)
    for _ in range(32):
        length = int(rng.integers(1, 24))
        players = []
        current_player = 0
        placements_remaining = 1
        for _move_idx in range(length):
            players.append(current_player)
            if placements_remaining > 1:
                placements_remaining -= 1
            else:
                current_player = 1 - current_player
                placements_remaining = 2

        positions = [
            PositionRecord(b"", {idx: 1.0}, 0.0, player=player)
            for idx, player in enumerate(players)
        ]
        starts = hexo_turn_start_indices(positions)
        run_lengths = [
            (starts[i + 1] if i + 1 < len(starts) else len(players)) - start
            for i, start in enumerate(starts)
        ]

        assert starts[0] == 0
        assert run_lengths[0] == 1
        assert all(run_len in (1, 2) for run_len in run_lengths[1:])
        assert all(
            positions[start].player != positions[starts[i - 1]].player
            for i, start in enumerate(starts[1:], start=1)
        )


def test_value_from_source_perspective_flips_opponent_values():
    assert value_from_source_perspective(0.75, source_player=0, target_player=0) == pytest.approx(0.75)
    assert value_from_source_perspective(0.75, source_player=0, target_player=1) == pytest.approx(-0.75)


def test_lookahead_flips_future_player_perspective():
    positions = [
        PositionRecord(b"", {1: 1.0}, 0.2, player=0),
        PositionRecord(_move(0, 0, 0), {2: 1.0}, 0.6, player=1),
    ]

    lookahead = compute_ema_lookahead(positions, horizon=1, lambda_=1.0)

    assert lookahead[0] == pytest.approx(-0.6)
    assert lookahead[1] == pytest.approx(0.6)


def test_lookahead_keeps_same_player_perspective():
    positions = [
        PositionRecord(b"", {1: 1.0}, 0.2, player=0),
        PositionRecord(b"", {2: 1.0}, 0.6, player=1),
        PositionRecord(b"", {3: 1.0}, 0.1, player=1),
        PositionRecord(b"", {4: 1.0}, 0.7, player=0),
    ]

    lookahead = compute_ema_lookahead(positions, horizon=2, lambda_=1.0)

    assert lookahead[0] == pytest.approx(0.7)


def test_ema_lookahead_uses_source_perspective_for_every_future_term():
    positions = [
        PositionRecord(b"", {1: 1.0}, 0.0, player=0),
        PositionRecord(b"", {2: 1.0}, 0.2, player=1),
        PositionRecord(b"", {3: 1.0}, 0.4, player=1),
        PositionRecord(b"", {4: 1.0}, 0.8, player=0),
        PositionRecord(b"", {5: 1.0}, 0.6, player=0),
        PositionRecord(b"", {6: 1.0}, -0.5, player=1),
    ]

    lookahead = compute_ema_lookahead(positions, horizon=1, lambda_=1.0)

    assert lookahead.tolist() == pytest.approx([0.5, -0.5, -0.5, 0.5, 0.5, -0.5])


def test_mid_turn_lookahead_targets_next_turn_start():
    positions = [
        PositionRecord(b"", {1: 1.0}, 0.1, player=0),
        PositionRecord(b"", {2: 1.0}, 0.2, player=1),
        PositionRecord(b"", {3: 1.0}, 0.3, player=1),
        PositionRecord(b"", {4: 1.0}, 0.7, player=0),
    ]

    lookahead = compute_ema_lookahead(positions, horizon=1, lambda_=1.0)

    assert lookahead[2] == pytest.approx(-0.7)


def test_opponent_policy_uses_next_full_search_opponent_turn_start():
    game = GameRecord(
        positions=[
            PositionRecord(b"", {1: 1.0}, 0.0, player=0, is_full_search=False),
            PositionRecord(b"", {2: 1.0}, 0.0, player=1, is_full_search=False),
            PositionRecord(b"", {3: 1.0}, 0.0, player=1, is_full_search=False),
            PositionRecord(b"", {4: 1.0}, 0.0, player=0, is_full_search=True),
            PositionRecord(b"", {5: 1.0}, 0.0, player=0, is_full_search=True),
            PositionRecord(b"", {6: 1.0}, 0.0, player=1, is_full_search=True),
        ],
        outcome=1.0,
    )

    process_game_record(game)

    assert game.positions[1].opp_policy_target == {4: 1.0}
    assert game.positions[1].opp_policy_weight == pytest.approx(1.0)
    assert game.positions[0].opp_policy_target == {6: 1.0}
    assert game.positions[3].opp_policy_target == {6: 1.0}


def test_opponent_policy_ignores_low_pcr_opponent_turn():
    game = GameRecord(
        positions=[
            PositionRecord(b"", {1: 1.0}, 0.0, player=0, is_full_search=True),
            PositionRecord(b"", {2: 1.0}, 0.0, player=1, is_full_search=False),
            PositionRecord(b"", {3: 1.0}, 0.0, player=1, is_full_search=False),
            PositionRecord(b"", {4: 1.0}, 0.0, player=0, is_full_search=True),
        ],
        outcome=1.0,
    )

    process_game_record(game)

    assert game.positions[0].opp_policy_target == {}
    assert game.positions[0].opp_policy_weight == pytest.approx(0.0)
    assert game.positions[1].opp_policy_target == {4: 1.0}
    assert game.positions[1].opp_policy_weight == pytest.approx(1.0)


def test_opponent_policy_end_of_game_without_future_turn_zeroes_weight():
    game = GameRecord(
        positions=[
            PositionRecord(b"", {1: 1.0}, 0.0, player=0, is_full_search=True),
        ],
        outcome=1.0,
    )

    process_game_record(game)

    assert game.positions[0].opp_policy_target == {}
    assert game.positions[0].opp_policy_weight == pytest.approx(0.0)


def test_regret_uses_selected_action_value_and_raw_scale():
    game = GameRecord(
        positions=[
            PositionRecord(b"", {1: 1.0}, 0.9, player=0, selected_action_value=-1.0),
            PositionRecord(b"", {2: 1.0}, 0.2, player=1, selected_action_value=1.0),
        ],
        outcome=1.0,
    )

    process_game_record(game)

    assert game.positions[0].regret_rank == pytest.approx(4.0)
    assert game.positions[0].regret_value == pytest.approx(4.0)
    assert game.positions[0].regret_weight == pytest.approx(1.0)


def test_regret_weight_zero_when_selected_action_value_missing():
    game = GameRecord(
        positions=[
            PositionRecord(b"", {1: 1.0}, 0.9, player=0, selected_action_value=None),
            PositionRecord(b"", {2: 1.0}, 0.2, player=1, selected_action_value=1.0),
        ],
        outcome=1.0,
    )

    process_game_record(game)

    assert game.positions[0].regret_rank == pytest.approx(0.0)
    assert game.positions[0].regret_value == pytest.approx(0.0)
    assert game.positions[0].regret_weight == pytest.approx(0.0)
    assert game.positions[1].regret_weight == pytest.approx(1.0)


def test_regret_suffix_average_matches_paper_equation_2():
    game = GameRecord(
        positions=[
            PositionRecord(b"", {1: 1.0}, 0.0, player=0, selected_action_value=0.0),
            PositionRecord(b"", {2: 1.0}, 0.0, player=1, selected_action_value=1.0),
            PositionRecord(b"", {3: 1.0}, 0.0, player=0, selected_action_value=-1.0),
        ],
        outcome=1.0,
    )

    process_game_record(game)

    assert [pos.regret_rank for pos in game.positions] == pytest.approx([3.0, 4.0, 4.0])
    assert compute_regret(game.positions, game.outcome) == pytest.approx([3.0, 4.0, 4.0])


def test_compute_regret_requires_selected_action_value_by_default():
    positions = [PositionRecord(b"", {1: 1.0}, 0.25, player=0)]

    with pytest.raises(ValueError, match="selected_action_value"):
        compute_regret(positions, outcome=1.0)


def test_truncated_games_zero_regret_weight():
    game = GameRecord(
        positions=[
            PositionRecord(b"", {1: 1.0}, 0.9, player=0, selected_action_value=-1.0),
            PositionRecord(b"", {2: 1.0}, 0.2, player=1, selected_action_value=1.0),
        ],
        outcome=1.0,
        truncated=True,
    )

    process_game_record(game)

    assert game.positions[0].regret_weight == pytest.approx(0.0)
    assert game.positions[1].regret_weight == pytest.approx(0.0)


def test_truncated_games_keep_policy_targets_but_zero_value_weight():
    game = GameRecord(
        positions=[
            PositionRecord(
                b"",
                {action_to_board_index(0, 0): 1.0},
                0.9,
                player=0,
                selected_action_value=-1.0,
            ),
        ],
        outcome=1.0,
        truncated=True,
    )

    process_game_record(game)

    assert game.positions[0].policy_target
    assert game.positions[0].value_weight == pytest.approx(0.0)
    assert game.positions[0].regret_weight == pytest.approx(0.0)


def test_draw_games_keep_policy_targets_but_zero_value_weight():
    game = GameRecord(
        positions=[
            PositionRecord(
                b"",
                {action_to_board_index(0, 0): 1.0},
                0.0,
                player=0,
                selected_action_value=0.0,
            ),
        ],
        outcome=0.0,
        truncated=False,
    )

    process_game_record(game)

    assert game.positions[0].policy_target
    assert game.positions[0].value_weight == pytest.approx(0.0)


def test_process_game_record_populates_auxiliary_targets():
    game = GameRecord(
        positions=[
            PositionRecord(b"", {1: 1.0}, 0.2, player=0, turn_index=0),
            PositionRecord(_move(0, 0, 0), {2: 1.0}, -0.1, player=1, turn_index=1),
        ],
        outcome=1.0,
        final_move_history=_move(0, 0, 0) + _move(1, 1, 0),
    )

    process_game_record(game, lookahead_horizons=[1], lookahead_lambdas=[0.5])

    assert game.positions[0].opp_policy_target == {2: 1.0}
    assert game.positions[0].moves_left == 2.0
    assert game.positions[1].moves_left == 1.0
    assert game.positions[0].regret_rank >= 0.0
    assert len(game.positions[0].lookahead_values) == 1


def test_ring_buffer_preserves_auxiliary_targets():
    rec = PositionRecord(
        move_history=b"",
        policy_target={action_to_board_index(0, 0): 1.0},
        root_value=0.0,
        selected_action_value=0.4,
        player=0,
        outcome=1.0,
        opp_policy_target={action_to_board_index(1, 0): 1.0},
        opp_policy_weight=1.0,
        regret_rank=0.25,
        regret_value=-0.5,
        regret_weight=0.75,
        sparse_prior_stage=2,
        sparse_prior_root_candidate_count=7,
        sparse_prior_leaf_candidate_count=5.5,
        sparse_prior_root_hit_frac=0.25,
        sparse_prior_leaf_hit_frac=0.5,
        fallback_prior_use=0.125,
        fallback_prior_use_on_mcts_top4=0.25,
        pair_prior_candidate_count=3,
        pair_prior_hit_frac=0.5,
        pair_fallback_prior_use=1.0,
        pair_fallback_prior_use_on_mcts_top1=0.0,
        pair_fallback_prior_use_on_mcts_top8=0.25,
        axis_label=2,
        moves_left=7.0,
        value_weight=0.0,
    )
    buffer = RingBuffer(capacity=4)
    buffer.append(rec)

    out = buffer[0]
    assert out is not None
    assert out.opp_policy_target == rec.opp_policy_target
    assert out.opp_policy_weight == pytest.approx(1.0)
    assert out.selected_action_value == pytest.approx(0.4)
    assert out.regret_rank == rec.regret_rank
    assert out.regret_weight == pytest.approx(0.75)
    assert out.sparse_prior_stage == 2
    assert out.sparse_prior_root_candidate_count == 7
    assert out.sparse_prior_leaf_candidate_count == pytest.approx(5.5)
    assert out.sparse_prior_root_hit_frac == pytest.approx(0.25)
    assert out.fallback_prior_use == pytest.approx(0.125)
    assert out.fallback_prior_use_on_mcts_top4 == pytest.approx(0.25)
    assert out.pair_prior_candidate_count == 3
    assert out.pair_prior_hit_frac == pytest.approx(0.5)
    assert out.pair_fallback_prior_use == pytest.approx(1.0)
    assert out.pair_fallback_prior_use_on_mcts_top1 == pytest.approx(0.0)
    assert out.pair_fallback_prior_use_on_mcts_top8 == pytest.approx(0.25)
    assert buffer.stats["fallback_prior_use_on_mcts_topk"] == pytest.approx(0.25)
    assert buffer.stats["pair_prior_hit_frac"] == pytest.approx(0.5)
    assert buffer.stats["pair_prior_candidate_count"] == pytest.approx(3.0)
    assert buffer.stats["pair_fallback_prior_use_on_mcts_top1"] == pytest.approx(0.0)
    assert buffer.stats["pair_fallback_prior_use_on_mcts_top8"] == pytest.approx(0.25)
    assert out.axis_label == rec.axis_label
    assert out.moves_left == rec.moves_left
    assert out.value_weight == rec.value_weight


def test_ring_buffer_preserves_missing_selected_action_and_regret_weight():
    rec = PositionRecord(
        move_history=b"",
        policy_target={action_to_board_index(0, 0): 1.0},
        root_value=0.5,
        selected_action_value=None,
        player=0,
        outcome=1.0,
        regret_rank=3.0,
        regret_value=3.0,
        regret_weight=0.0,
    )
    buffer = RingBuffer(capacity=2)
    buffer.append(rec)

    out = buffer[0]

    assert out is not None
    assert out.selected_action_value is None
    assert out.regret_weight == pytest.approx(0.0)


def test_policy_target_v2_preserves_outside_window_mass():
    target_v2 = policy_v2_from_visits([0, 50], [0, 50], [3, 7], top_k=8)
    policy, outside = dense_policy_from_v2(target_v2, -16, -16, top_k=8)

    assert sum(prob for _q, _r, prob in target_v2) == pytest.approx(1.0)
    assert outside == pytest.approx(0.7)
    assert policy[action_to_board_index(0, 0)] == pytest.approx(1.0)


def test_compact_record_v2_roundtrip_preserves_global_targets():
    pair_target = [((0, 0), (-40, 10), 1.0)]
    rec = PositionRecord(
        move_history=_move(0, 0, 0),
        policy_target={action_to_board_index(0, 0): 1.0},
        policy_target_v2=[(0, 0, 0.4), (-40, 10, 0.6)],
        pair_policy_target_v2=pair_target,
        target_policy_mass_outside_window=0.6,
        missing_target_policy_mass=0.0,
        candidate_recall_mcts_top4=0.5,
        candidate_recall_winning_move=1.0,
        candidate_recall_forced_block=0.75,
        candidate_recall_two_placement_cover=0.5,
        root_value=0.25,
        selected_action_value=-0.5,
        player=1,
        outcome=-1.0,
        opp_policy_target_v2=[(1, 0, 1.0)],
        opp_policy_weight=1.0,
        regret_weight=0.25,
        value_weight=0.0,
        sparse_prior_stage=2,
        sparse_prior_root_candidate_count=9,
        sparse_prior_leaf_candidate_count=3.5,
        sparse_prior_root_hit_frac=0.5,
        sparse_prior_leaf_hit_frac=0.25,
        fallback_prior_use=0.125,
        fallback_prior_use_on_mcts_top1=1.0,
        fallback_prior_use_on_mcts_top4=0.5,
        fallback_prior_use_on_mcts_top8=0.25,
        sparse_vs_dense_disagreement=1.0,
        sparse_prior_forward_ms=2.0,
        sparse_prior_candidate_build_ms=3.0,
        pair_prior_candidate_count=4,
        pair_prior_hit_frac=0.0,
        pair_fallback_prior_use=1.0,
        pair_fallback_prior_use_on_mcts_top1=1.0,
        pair_fallback_prior_use_on_mcts_top4=1.0,
        pair_fallback_prior_use_on_mcts_top8=1.0,
    )
    game = GameRecord(positions=[rec], outcome=-1.0, game_id=9, final_move_history=_move(0, 0, 0))

    out = GameRecord.from_compact_bytes(game.to_compact_bytes())

    assert out.game_id == 9
    assert [(q, r) for q, r, _ in out.positions[0].policy_target_v2] == [(0, 0), (-40, 10)]
    assert [prob for _q, _r, prob in out.positions[0].policy_target_v2] == pytest.approx([0.4, 0.6])
    assert [(q, r) for q, r, _ in out.positions[0].opp_policy_target_v2] == [(1, 0)]
    assert [prob for _q, _r, prob in out.positions[0].opp_policy_target_v2] == pytest.approx([1.0])
    assert out.positions[0].pair_policy_target_v2 == pair_target
    assert out.positions[0].selected_action_value == pytest.approx(-0.5)
    assert out.positions[0].opp_policy_weight == pytest.approx(1.0)
    assert out.positions[0].regret_weight == pytest.approx(0.25)
    assert out.positions[0].value_weight == pytest.approx(0.0)
    assert out.positions[0].sparse_prior_stage == 2
    assert out.positions[0].sparse_prior_root_candidate_count == 9
    assert out.positions[0].sparse_prior_leaf_candidate_count == pytest.approx(3.5)
    assert out.positions[0].fallback_prior_use_on_mcts_top1 == pytest.approx(1.0)
    assert out.positions[0].sparse_vs_dense_disagreement == pytest.approx(1.0)
    assert out.positions[0].pair_fallback_prior_use_on_mcts_top8 == pytest.approx(1.0)
    assert out.positions[0].target_policy_mass_outside_window == pytest.approx(0.6)
    assert out.positions[0].candidate_recall_mcts_top4 == pytest.approx(0.5)
    assert out.positions[0].candidate_recall_winning_move == pytest.approx(1.0)
    assert out.positions[0].candidate_recall_forced_block == pytest.approx(0.75)
    assert out.positions[0].candidate_recall_two_placement_cover == pytest.approx(0.5)


def test_compact_record_preserves_missing_selected_action_as_invalid_regret_target():
    rec = PositionRecord(
        move_history=b"",
        policy_target={action_to_board_index(0, 0): 1.0},
        root_value=0.25,
        selected_action_value=None,
        player=0,
        outcome=1.0,
        regret_rank=0.0,
        regret_value=0.0,
        regret_weight=0.0,
    )
    game = GameRecord(positions=[rec], outcome=1.0, game_id=11)

    out = GameRecord.from_compact_bytes(game.to_compact_bytes())

    assert out.positions[0].selected_action_value is None
    assert out.positions[0].regret_weight == pytest.approx(0.0)


def test_ring_buffer_preserves_policy_target_v2():
    rec = PositionRecord(
        move_history=b"",
        policy_target={action_to_board_index(0, 0): 1.0},
        policy_target_v2=[(0, 0, 0.5), (30, -12, 0.5)],
        pair_policy_target_v2=[((0, 0), (30, -12), 1.0)],
        target_policy_mass_outside_window=0.5,
        candidate_recall_winning_move=1.0,
        root_value=0.0,
        player=0,
        outcome=1.0,
    )
    buffer = RingBuffer(capacity=4, max_policy_v2_entries=2)
    buffer.append(rec)

    out = buffer[0]
    assert out is not None
    assert out.policy_target_v2 == rec.policy_target_v2
    assert out.pair_policy_target_v2 == rec.pair_policy_target_v2
    assert out.target_policy_mass_outside_window == pytest.approx(0.5)
    assert out.candidate_recall_winning_move == pytest.approx(1.0)


def test_dense_projection_uses_all_v2_visits_before_topk():
    moves_q = [100, 0, 1]
    moves_r = [100, 0, 0]
    visits = [100, 20, 10]
    target_v2 = policy_v2_from_visits(moves_q, moves_r, visits)
    policy, outside = dense_policy_from_v2(target_v2, -16, -16, top_k=2)

    assert outside == pytest.approx(100 / 130)
    assert set(policy) == {action_to_board_index(0, 0), action_to_board_index(1, 0)}
    assert policy[action_to_board_index(0, 0)] == pytest.approx(2 / 3)
    assert policy[action_to_board_index(1, 0)] == pytest.approx(1 / 3)


def test_ring_buffer_reports_v2_truncation_as_missing_mass():
    rec = PositionRecord(
        move_history=b"",
        policy_target={action_to_board_index(0, 0): 1.0},
        policy_target_v2=[(0, 0, 0.5), (1, 0, 0.3), (2, 0, 0.2)],
        root_value=0.0,
        player=0,
        outcome=1.0,
    )
    buffer = RingBuffer(capacity=2, max_policy_v2_entries=2)
    buffer.append(rec)

    out = buffer[0]
    assert out is not None
    assert len(out.policy_target_v2) == 2
    assert out.missing_target_policy_mass == pytest.approx(0.2)


def test_sparse_sampler_outputs_candidate_targets():
    rec = PositionRecord(
        move_history=b"",
        policy_target={action_to_board_index(0, 0): 1.0},
        policy_target_v2=[(0, 0, 0.75), (1, 0, 0.25)],
        root_value=0.0,
        player=0,
        outcome=1.0,
    )
    buffer = RingBuffer(capacity=4, max_policy_v2_entries=8)
    buffer.append(rec)
    dataset = ReplayDataset(
        buffer,
        batch_size=1,
        use_symmetry=True,
        include_sparse_policy=True,
        candidate_budget=4,
    )

    _tensors, _policies, _values, _lookahead, aux = next(iter(dataset))

    assert aux["candidate_qr"].shape == (1, 8, 2)
    assert aux["candidate_features"].shape[2] == 12
    assert aux["candidate_mask"][0].any()
    assert aux["sparse_policy_target"][0].sum() == pytest.approx(1.0)


def test_replay_dataset_emits_regret_weight_for_loss_masking():
    rec = PositionRecord(
        move_history=b"",
        policy_target={action_to_board_index(0, 0): 1.0},
        root_value=0.0,
        selected_action_value=None,
        player=0,
        outcome=1.0,
        regret_rank=4.0,
        regret_value=4.0,
        regret_weight=0.0,
    )
    buffer = RingBuffer(capacity=4)
    buffer.append(rec)
    dataset = ReplayDataset(buffer, batch_size=1, use_symmetry=False)

    _tensors, _policies, _values, _lookahead, aux = next(iter(dataset))

    assert aux["regret_weight"][0] == pytest.approx(0.0)
    assert aux["regret_rank"][0] == pytest.approx(4.0)


def test_regret_biased_sampling_ignores_zero_weight_regret_rows():
    invalid = PositionRecord(
        b"",
        {action_to_board_index(0, 0): 1.0},
        0.0,
        player=0,
        outcome=1.0,
        selected_action_value=None,
        regret_rank=100.0,
        regret_weight=0.0,
    )
    valid = PositionRecord(
        b"",
        {action_to_board_index(1, 0): 1.0},
        0.0,
        player=0,
        outcome=1.0,
        selected_action_value=0.0,
        regret_rank=0.1,
        regret_weight=1.0,
    )
    buffer = RingBuffer(capacity=4)
    buffer.append(invalid)
    buffer.append(valid)

    indices = buffer.sample_regret_indices(16, temperature=0.1)

    assert set(indices.tolist()) == {1}


def test_candidate_feature_names_match_tensor_width():
    from hexorl.action_contract.candidates import (
        CANDIDATE_FEATURE_NAMES,
        CANDIDATE_FEATURE_VERSION,
        CANDIDATE_FEATURES,
    )

    assert CANDIDATE_FEATURE_VERSION == 2
    assert len(CANDIDATE_FEATURE_NAMES) == CANDIDATE_FEATURES
    assert CANDIDATE_FEATURES == 12


def test_checkpoint_reports_candidate_feature_version(tmp_path):
    from hexorl.action_contract.candidates import CANDIDATE_FEATURE_VERSION
    from hexorl.train.trainer import Trainer

    cfg = Config.model_validate(
        {
            "model": {"channels": 4, "blocks": 1},
            "train": {"batches_per_epoch": 1},
            "inference": {"fp16": False},
        }
    )
    model = HexNet(channels=4, blocks=1)
    trainer = Trainer(model, cfg, dataloader=[], device=torch.device("cpu"))
    path = tmp_path / "ckpt.pt"

    trainer.save_checkpoint(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    assert checkpoint["action_contract_metadata"]["candidate_feature_version"] == CANDIDATE_FEATURE_VERSION
    assert checkpoint["model_metadata"]["candidate_feature_version"] == CANDIDATE_FEATURE_VERSION


def test_sparse_sampler_keeps_d6_enabled_and_transforms_candidates():
    pytest.importorskip("_engine")
    history = _move(0, 0, 0)
    target = (1, 0)
    target_t = _hex_transform(*target, 1)
    rec = PositionRecord(
        move_history=history,
        policy_target={action_to_board_index(*target): 1.0},
        policy_target_v2=[(target[0], target[1], 1.0)],
        root_value=0.0,
        player=1,
        outcome=1.0,
    )
    buffer = RingBuffer(capacity=4, max_policy_v2_entries=8)
    buffer.append(rec)
    dataset = ReplayDataset(
        buffer,
        batch_size=1,
        use_symmetry=True,
        include_sparse_policy=True,
        candidate_budget=8,
    )
    dataset._rng = _FixedSymmetryRng(1)

    _tensors, policies, _values, _lookahead, aux = next(iter(dataset))

    represented = {tuple(qr) for qr in aux["candidate_qr"][0][aux["candidate_mask"][0]]}
    assert tuple(target_t) in represented
    row = np.where((aux["candidate_qr"][0] == np.array(target_t)).all(axis=1))[0][0]
    idx = int(aux["candidate_indices"][0, row])
    assert idx >= 0
    assert policies[0, idx] == pytest.approx(1.0)
    assert aux["sparse_policy_target"][0, row] == pytest.approx(1.0)
    assert dataset.use_symmetry is True


@pytest.mark.parametrize("architecture", ["cnn", "restnet", "graph_hybrid_0"])
def test_sparse_d6_batch_trains_for_all_model_architectures(architecture):
    pytest.importorskip("_engine")
    rec = PositionRecord(
        move_history=_move(0, 0, 0),
        policy_target={action_to_board_index(1, 0): 1.0},
        policy_target_v2=[(1, 0, 1.0)],
        root_value=0.0,
        player=1,
        outcome=1.0,
    )
    buffer = RingBuffer(capacity=4, max_policy_v2_entries=8)
    buffer.append(rec)
    dataset = ReplayDataset(
        buffer,
        batch_size=1,
        use_symmetry=True,
        include_sparse_policy=True,
        candidate_budget=8,
    )
    dataset._rng = _FixedSymmetryRng(1)
    tensors, policies, values, _lookahead, aux = next(iter(dataset))

    model = HexNet(
        channels=8,
        blocks=2,
        heads=["policy", "value", "sparse_policy"],
        architecture=architecture,
        attention_heads=4,
        graph_token_budget=32,
        graph_layers=1,
        sparse_policy=True,
    )
    out = model(
        torch.from_numpy(tensors),
        candidate_indices=torch.from_numpy(aux["candidate_indices"]),
        candidate_features=torch.from_numpy(aux["candidate_features"]),
        candidate_mask=torch.from_numpy(aux["candidate_mask"]),
    )
    targets = {
        "policy": torch.from_numpy(policies),
        "value": torch.from_numpy(values),
        "sparse_policy_target": torch.from_numpy(aux["sparse_policy_target"]),
        "candidate_mask": torch.from_numpy(aux["candidate_mask"]),
        "policy_weight": torch.from_numpy(aux["policy_weight"]),
    }

    total, per_head = compute_losses(
        out,
        targets,
        {"policy": 1.0, "value": 1.0, "sparse_policy": 1.0},
    )

    assert torch.isfinite(total)
    assert "sparse_policy" in per_head


def test_replay_dataset_can_emit_pair_policy_target():
    policy_v2 = [(1, 0, 0.75), (0, 1, 0.25)]
    rec = PositionRecord(
        move_history=_move(0, 0, 0),
        policy_target={action_to_board_index(1, 0): 1.0},
        policy_target_v2=policy_v2,
        pair_policy_target_v2=pair_policy_v2_from_place_target(policy_v2, top_k=4),
        root_value=0.0,
        player=1,
        outcome=1.0,
    )
    buffer = RingBuffer(capacity=4, max_policy_v2_entries=8)
    buffer.append(rec)
    dataset = ReplayDataset(
        buffer,
        batch_size=1,
        use_symmetry=True,
        include_sparse_policy=True,
        include_pair_policy=True,
        candidate_budget=4,
    )

    _tensors, _policies, _values, _lookahead, aux = next(iter(dataset))

    assert aux["pair_candidate_indices"].shape == (1, 8, 2)
    assert aux["pair_candidate_mask"][0].any()
    assert aux["pair_policy_target"][0].sum() == pytest.approx(1.0)


def test_pair_policy_targets_use_full_policy_v2_by_default():
    policy_v2 = [(i, 0, float(6 - i)) for i in range(5)]

    pair_target = pair_policy_v2_from_place_target(policy_v2)

    represented = {tuple(sorted((first, second))) for first, second, _prob in pair_target}
    assert len(pair_target) == 10
    assert ((0, 0), (4, 0)) in represented
    assert sum(prob for _first, _second, prob in pair_target) == pytest.approx(1.0)


def test_pair_policy_d6_bijection_preserves_pair_identity():
    from hexorl.action_contract.candidates import build_pair_candidate_batch

    base_candidates = [(1, 0), (0, 1), (2, 0)]
    base_pair = [((1, 0), (0, 1), 1.0)]

    for sym_idx in range(12):
        candidates = [_hex_transform(q, r, sym_idx) for q, r in base_candidates]
        target = _transform_pair_policy_v2(base_pair, sym_idx)
        pair = build_pair_candidate_batch(
            candidates,
            target,
            budget=4,
            legal_moves=candidates,
        )
        row = int(np.where(pair.mask)[0][0])
        first_idx, second_idx = pair.pair_indices[row]
        represented = {tuple(candidates[int(first_idx)]), tuple(candidates[int(second_idx)])}
        expected = {tuple(target[0][0]), tuple(target[0][1])}
        assert represented == expected
        assert pair.target[row] == pytest.approx(1.0)


def test_pair_policy_rejects_duplicate_and_illegal_pairs():
    from hexorl.action_contract.candidates import build_pair_candidate_batch

    with pytest.raises(ValueError, match="duplicate coordinates"):
        build_pair_candidate_batch(
            [(0, 0), (1, 0)],
            [((0, 0), (0, 0), 1.0)],
            budget=2,
            legal_moves=[(0, 0), (1, 0)],
        )

    with pytest.raises(ValueError, match="illegal action pair"):
        build_pair_candidate_batch(
            [(0, 0), (1, 0)],
            [((0, 0), (9, 0), 1.0)],
            budget=2,
            legal_moves=[(0, 0), (1, 0)],
        )

    with pytest.raises(ValueError, match="duplicate active candidate row"):
        build_pair_candidate_batch(
            [(0, 0), (0, 0), (1, 0)],
            [((0, 0), (1, 0), 1.0)],
            budget=2,
            candidate_mask=[True, True, True],
            legal_moves=[(0, 0), (1, 0)],
        )


def test_pair_candidate_builder_ignores_padded_candidate_rows():
    from hexorl.action_contract.candidates import build_pair_candidate_batch

    pair = build_pair_candidate_batch(
        [(1, 0), (0, 0), (0, 0)],
        [((1, 0), (0, 0), 1.0)],
        budget=3,
        candidate_mask=[True, False, False],
    )

    assert not pair.mask.any()
    assert pair.target.sum() == pytest.approx(0.0)


def test_candidate_builder_accepts_list_legal_moves():
    from hexorl.action_contract.candidates import build_candidate_batch

    cand = build_candidate_batch(
        [[0, 0], [1, 0]],
        [(0, 0, 1.0)],
        offset_q=-16,
        offset_r=-16,
        budget=4,
    )

    assert cand.mask.sum() == 2
    assert cand.target.sum() == pytest.approx(1.0)


def test_candidate_builder_keeps_critical_actions_past_budget():
    from hexorl.action_contract.candidates import build_candidate_batch

    cand = build_candidate_batch(
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 0, 1.0)],
        offset_q=-16,
        offset_r=-16,
        budget=1,
        winning_moves=[(3, 0)],
        forced_block_moves=[(2, 0)],
        cover_cells=[(1, 0)],
    )

    represented = {tuple(qr) for qr in cand.qr[cand.mask]}
    assert {(0, 0), (1, 0), (2, 0), (3, 0)} <= represented
    assert cand.recall_winning_move == pytest.approx(1.0)
    assert cand.recall_forced_block == pytest.approx(1.0)
    assert cand.recall_two_placement_cover == pytest.approx(1.0)


def test_critical_actions_are_inserted_before_heuristic_candidates():
    from hexorl.action_contract.candidates import build_candidate_batch

    cand = build_candidate_batch(
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [],
        offset_q=-16,
        offset_r=-16,
        budget=2,
        winning_moves=[(3, 0)],
        forced_block_moves=[(2, 0)],
    )

    assert [tuple(qr) for qr in cand.qr[:2]] == [(3, 0), (2, 0)]


def test_candidate_recall_reports_protected_and_discovery_modes():
    from hexorl.action_contract.candidates import build_candidate_batch

    cand = build_candidate_batch(
        [(0, 0), (1, 0), (2, 0)],
        [(2, 0, 1.0)],
        offset_q=-16,
        offset_r=-16,
        budget=1,
    )

    assert cand.recall_top1 == pytest.approx(1.0)
    assert cand.discovery_top1 == pytest.approx(0.0)


def test_discovery_recall_does_not_include_target_only_actions():
    from hexorl.action_contract.candidates import build_candidate_batch

    cand = build_candidate_batch(
        [(0, 0), (1, 0), (9, 0)],
        [(9, 0, 1.0)],
        offset_q=-16,
        offset_r=-16,
        budget=2,
        mode="protected",
    )

    represented = {tuple(qr) for qr in cand.qr[cand.mask]}
    assert (9, 0) in represented
    assert cand.discovery_top1 == pytest.approx(0.0)


def test_candidate_features_do_not_include_policy_target_labels():
    from hexorl.action_contract.candidates import build_candidate_batch

    kwargs = {
        "legal_moves": [(0, 0), (1, 0)],
        "offset_q": -16,
        "offset_r": -16,
        "budget": 4,
    }
    with_target = build_candidate_batch(policy_target_v2=[(1, 0, 1.0)], **kwargs)
    without_target = build_candidate_batch(policy_target_v2=[], **kwargs)

    row_with = np.where((with_target.qr == np.array([1, 0])).all(axis=1))[0][0]
    row_without = np.where((without_target.qr == np.array([1, 0])).all(axis=1))[0][0]
    assert with_target.features[row_with].tolist() == pytest.approx(
        without_target.features[row_without].tolist()
    )
    assert with_target.target[row_with] == pytest.approx(1.0)


def test_critical_overflow_zeroes_sparse_and_pair_signal(caplog):
    history = b"".join(
        [
            _move(0, 0, 0),
            _move(1, 0, 5),
            _move(1, 1, 5),
            _move(0, 1, 0),
            _move(0, 2, 0),
            _move(1, 0, 7),
            _move(1, 1, 7),
            _move(0, 3, 0),
            _move(0, 4, 0),
            _move(1, 0, 9),
            _move(1, 1, 9),
        ]
    )
    rec = PositionRecord(
        move_history=history,
        policy_target={action_to_board_index(-1, 0): 1.0},
        policy_target_v2=[(-1, 0, 0.5), (5, 0, 0.5)],
        pair_policy_target_v2=[((-1, 0), (5, 0), 1.0)],
        root_value=0.0,
        player=0,
        outcome=1.0,
    )
    buffer = RingBuffer(capacity=4, max_policy_v2_entries=2)
    buffer.append(rec)
    dataset = ReplayDataset(
        buffer,
        batch_size=1,
        use_symmetry=False,
        include_sparse_policy=True,
        include_pair_policy=True,
        candidate_budget=2,
    )

    with caplog.at_level("ERROR"):
        _tensors, policies, _values, _lookahead, aux = next(iter(dataset))

    assert aux["candidate_critical_overflow_count"][0] > 0
    assert aux["sparse_policy_target"][0].sum() == pytest.approx(0.0)
    assert aux["pair_policy_target"][0].sum() == pytest.approx(0.0)
    assert policies[0].sum() == pytest.approx(1.0)
    assert "Critical candidate overflow" in caplog.text


def test_sparse_sampler_reports_missing_mass_if_protected_candidates_overflow_width():
    targets = [(i, 0, 0.2) for i in range(5)]
    rec = PositionRecord(
        move_history=b"",
        policy_target={action_to_board_index(0, 0): 1.0},
        policy_target_v2=targets,
        root_value=0.0,
        player=0,
        outcome=1.0,
    )
    buffer = RingBuffer(capacity=4, max_policy_v2_entries=2)
    buffer.append(rec)
    dataset = ReplayDataset(
        buffer,
        batch_size=1,
        use_symmetry=False,
        include_sparse_policy=True,
        candidate_budget=2,
    )

    *_prefix, aux = next(iter(dataset))

    assert aux["sparse_policy_target"].shape[1] == 2
    assert aux["sparse_policy_target"][0].sum() == pytest.approx(1.0)
    assert aux["candidate_missing_mass"][0] == pytest.approx(0.6)


def test_sparse_policy_loss_masks_invalid_candidates():
    logits = torch.tensor([[0.0, 2.0, -5.0], [1.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False], [False, False, False]])

    loss = sparse_policy_loss(logits, target, mask)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_sparse_policy_loss_accepts_half_logits_float_targets():
    logits = torch.tensor([[0.0, 2.0, -5.0]], dtype=torch.float16)
    target = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[True, True, False]])

    loss = sparse_policy_loss(logits, target, mask)

    assert torch.isfinite(loss)
    assert loss.dtype == torch.float32


def test_policy_and_value_losses_accept_half_logits():
    policy = policy_loss(
        torch.tensor([[0.0, 2.0, -5.0]], dtype=torch.float16),
        torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
    )
    value = binned_value_loss(
        torch.zeros((1, 65), dtype=torch.float16),
        torch.tensor([0.0], dtype=torch.float32),
    )

    assert torch.isfinite(policy)
    assert torch.isfinite(value)
    assert policy.dtype == torch.float32
    assert value.dtype == torch.float32


def test_policy_target_top64_is_preserved_when_configured():
    dense = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    dense[:80] = np.arange(80, 0, -1, dtype=np.float32)
    policy = sparsify_policy(dense, top_k=64)
    rec = PositionRecord(
        move_history=b"",
        policy_target=policy,
        root_value=0.0,
        player=0,
    )
    buffer = RingBuffer(capacity=2, max_policy_entries=64)
    buffer.append(rec)

    out = buffer[0]
    assert out is not None
    assert len(out.policy_target) == 64
    assert abs(sum(out.policy_target.values()) - 1.0) < 1e-6


def test_run_epoch_appends_selfplay_to_existing_replay(monkeypatch, tmp_path):
    generated = RingBuffer(capacity=8)
    generated.append(
        PositionRecord(
            move_history=_move(0, 0, 0),
            policy_target={action_to_board_index(1, 0): 1.0},
            root_value=0.0,
            player=1,
            game_id=1 << 24,
        )
    )
    generated.append(
        PositionRecord(
            move_history=_move(0, 0, 0) + _move(1, 1, 0),
            policy_target={action_to_board_index(2, 0): 1.0},
            root_value=0.0,
            player=0,
            game_id=1 << 24,
        )
    )

    class FakeOrchestrator:
        buffer = generated
        stats = {"games_done": 1, "positions_done": 1}

    monkeypatch.setattr(pipeline, "run_orchestrator", lambda *args, **kwargs: FakeOrchestrator())
    cfg = Config()
    cfg.model.channels = 4
    cfg.model.blocks = 1

    empty_existing = RingBuffer(capacity=8)
    empty_result = pipeline.run_epoch(
        cfg,
        buffer=empty_existing,
        output_dir=tmp_path / "empty",
        use_selfplay=True,
        train=False,
    )

    assert empty_result.buffer_stats["size"] == 2
    assert len(empty_existing) == 2
    assert [record.game_id for record in empty_existing.records()] == [0, 0]

    existing = RingBuffer(capacity=8)
    existing.append(
        PositionRecord(
            move_history=b"",
            policy_target={action_to_board_index(0, 0): 1.0},
            root_value=0.0,
            player=0,
            game_id=4,
        )
    )
    result = pipeline.run_epoch(
        cfg,
        buffer=existing,
        output_dir=tmp_path,
        use_selfplay=True,
        train=False,
    )

    assert result.buffer_stats["size"] == 3
    assert [record.game_id for record in existing.records()] == [4, 5, 5]


def test_selfplay_epoch_completion_requires_games_and_states():
    cfg = Config()
    cfg.selfplay.games_per_epoch = 2
    cfg.selfplay.states_per_epoch = 10
    orchestrator = SelfPlayOrchestrator(cfg, buffer_capacity=16)

    orchestrator._games_done = 2
    orchestrator._positions_done = 9
    assert not orchestrator.epoch_complete
    assert orchestrator.progress == pytest.approx(0.9)

    orchestrator._positions_done = 10
    assert orchestrator.epoch_complete
    assert orchestrator.progress == 1.0


def test_orchestrator_masks_truncated_game_value_targets(tmp_path):
    from hexorl.dashboard.recorder import RunRecorder
    from hexorl.dashboard.db import DashboardStore

    cfg = Config()
    cfg.selfplay.train_on_truncated_games = False
    store = DashboardStore(tmp_path / "dashboard.sqlite3")
    recorder = RunRecorder(store, "trunc-test")
    orchestrator = SelfPlayOrchestrator(cfg, buffer_capacity=16, recorder=recorder)
    game = GameRecord(
        positions=[
            PositionRecord(
                move_history=b"",
                policy_target={action_to_board_index(0, 0): 1.0},
                root_value=0.0,
                player=0,
                outcome=0.0,
            )
        ],
        outcome=0.0,
        game_id=9,
        game_length=1,
        final_move_history=b"",
        truncated=True,
        terminal_reason="max_game_moves",
    )

    orchestrator._ingest_game(game)

    assert len(orchestrator.buffer) == 1
    assert orchestrator.buffer[0].value_weight == 0.0
    assert orchestrator.stats["positions_done"] == 1
    rows = store.rows("SELECT payload_json FROM games")
    assert rows[0]["payload_json"]["truncated"] is True
    assert rows[0]["payload_json"]["terminal_reason"] == "max_game_moves"


def test_compute_losses_skips_missing_targets_and_handles_batch_one():
    predictions = {
        "policy": torch.zeros(1, 1089),
        "value": torch.zeros(1, 65),
        "regret_rank": torch.zeros(1, 1),
        "axis": torch.zeros(1, 3),
        "axis_delta_norm": torch.zeros(1, 6, 33, 33),
        "moves_left": torch.ones(1, 1),
    }
    targets = {
        "policy": torch.nn.functional.one_hot(torch.tensor([0]), 1089).float(),
        "value": torch.tensor([1.0]),
        "axis": torch.tensor([-1]),
        "axis_delta_norm": torch.ones(1, 6, 33, 33),
    }

    total, per_head = compute_losses(
        predictions,
        targets,
        {
            "policy": 1.0,
            "value": 1.0,
            "regret_rank": 1.0,
            "axis": 1.0,
            "axis_delta_norm": 1.0,
            "moves_left": 1.0,
        },
    )

    assert torch.isfinite(total)
    assert "regret_rank" not in per_head
    assert "moves_left" not in per_head
    assert per_head["axis"].item() == 0.0
    assert per_head["axis_delta_norm"].item() > 0.0


def test_policy_loss_can_be_masked_to_full_search_samples():
    predictions = {
        "policy": torch.zeros(2, 1089, requires_grad=True),
    }
    targets = {
        "policy": torch.nn.functional.one_hot(torch.tensor([0, 1]), 1089).float(),
        "policy_weight": torch.tensor([1.0, 0.0]),
    }

    total, per_head = compute_losses(predictions, targets, {"policy": 1.0})

    expected = torch.log(torch.tensor(1089.0))
    assert torch.allclose(total.detach(), expected, atol=1e-5)
    assert torch.allclose(per_head["policy"].detach(), expected, atol=1e-5)


def test_opp_policy_loss_uses_opponent_policy_weight():
    predictions = {
        "opp_policy": torch.zeros(2, 1089, requires_grad=True),
    }
    targets = {
        "opp_policy": torch.nn.functional.one_hot(torch.tensor([0, 1]), 1089).float(),
        "opp_policy_weight": torch.tensor([1.0, 0.0]),
    }

    total, per_head = compute_losses(predictions, targets, {"opp_policy": 1.0})

    expected = torch.log(torch.tensor(1089.0))
    assert torch.allclose(total.detach(), expected, atol=1e-5)
    assert torch.allclose(per_head["opp_policy"].detach(), expected, atol=1e-5)


def test_opp_policy_loss_skips_empty_targets():
    predictions = {"opp_policy": torch.zeros(2, 1089, requires_grad=True)}
    targets = {
        "opp_policy": torch.stack(
            [
                torch.nn.functional.one_hot(torch.tensor(0), 1089).float(),
                torch.zeros(1089),
            ]
        ),
        "opp_policy_weight": torch.ones(2),
    }

    total, per_head = compute_losses(predictions, targets, {"opp_policy": 1.0})

    expected = torch.log(torch.tensor(1089.0))
    assert torch.allclose(total.detach(), expected, atol=1e-5)
    assert torch.allclose(per_head["opp_policy"].detach(), expected, atol=1e-5)


def test_value_loss_can_be_masked_for_truncated_games():
    predictions = {"value": torch.zeros(2, 65, requires_grad=True)}
    targets = {
        "value": torch.tensor([1.0, -1.0]),
        "value_weight": torch.tensor([0.0, 0.0]),
    }

    total, per_head = compute_losses(predictions, targets, {"value": 1.0})

    assert total.item() == 0.0
    assert per_head["value"].item() == 0.0


def test_regret_losses_can_be_masked_by_regret_weight():
    predictions = {
        "regret_rank": torch.zeros(2, 1, requires_grad=True),
        "regret_value": torch.zeros(2, 65, requires_grad=True),
    }
    targets = {
        "regret_rank": torch.tensor([1.0, 4.0]),
        "regret_value": torch.tensor([1.0, 4.0]),
        "regret_weight": torch.tensor([0.0, 0.0]),
    }

    total, per_head = compute_losses(
        predictions,
        targets,
        {"regret_rank": 1.0, "regret_value": 1.0},
    )

    assert total.item() == 0.0
    assert per_head["regret_rank"].item() == 0.0
    assert per_head["regret_value"].item() == 0.0


def test_axis_delta_norm_head_shape():
    model = HexNet(channels=4, blocks=1, heads=["axis_delta_norm"])
    out = model(torch.zeros(2, 13, 33, 33))
    assert out["axis_delta_norm"].shape == (2, 6, 33, 33)


def test_replay_dataset_can_emit_axis_delta_norm_target():
    pytest.importorskip("_engine")
    buffer = RingBuffer(capacity=8)
    buffer.append(
        PositionRecord(
            move_history=_move(0, 0, 0),
            policy_target={action_to_board_index(1, 0): 1.0},
            root_value=0.0,
            player=1,
        )
    )
    dataset = ReplayDataset(
        buffer,
        batch_size=1,
        use_symmetry=False,
        include_axis_delta_norm=True,
    )

    *_rest, aux_targets = next(iter(dataset))

    assert aux_targets["axis_delta_norm"].shape == (1, 6, 33, 33)
    assert aux_targets["axis_delta_norm"].sum() > 0.0


def test_replay_dataset_marks_low_sim_policy_weight_zero():
    buffer = RingBuffer(capacity=2)
    buffer.append(
        PositionRecord(
            move_history=b"",
            policy_target={action_to_board_index(0, 0): 1.0},
            root_value=0.0,
            player=0,
            is_full_search=False,
        )
    )
    dataset = ReplayDataset(
        buffer,
        batch_size=1,
        use_symmetry=False,
    )
    *_rest, aux_targets = next(iter(dataset))

    assert aux_targets["policy_weight"].shape == (1,)
    assert aux_targets["policy_weight"][0].item() == 0.0


def test_bootstrap_games_are_diverse_and_legal():
    cfg = Config()
    cfg.run.seed = 123
    cfg.selfplay.max_game_moves = 24

    games = pipeline._make_bootstrap_game_records(cfg, 8)
    histories = {game.final_move_history for game in games}

    assert len(histories) > 1
    assert all(game.positions for game in games)
    assert all(pos.policy_target for game in games for pos in game.positions)
    assert all(len(pos.move_history) % 12 == 0 for game in games for pos in game.positions)
