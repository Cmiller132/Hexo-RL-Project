import numpy as np

from hexorl.selfplay.records import GameRecord, PositionRecord
from hexorl.selfplay.rgsc import RGSCRestartService, encode_move_history


class _RGSCFakeClient:
    def evaluate_regret_heads(self, tensor, count):
        markers = np.asarray(tensor, dtype=np.float32)[:count, 0, 0, 0]
        rank = markers * 10.0
        regret_value = markers + 0.25
        return rank.astype(np.float32), regret_value.astype(np.float32)


class _RGSCFakePolicyProvider:
    def __init__(self):
        self.client = _RGSCFakeClient()


class _RGSCFakeGame:
    def __init__(self):
        self.current_player = 0
        self.placements_remaining = 1
        self.is_over = False
        self._stones = []

    def place(self, q, r):
        self._stones.append((int(q), int(r)))
        self.placements_remaining -= 1
        if self.placements_remaining == 0:
            self.current_player = 1 - self.current_player
            self.placements_remaining = 2

    def encode_board_and_legal(self, near_radius, constrain_threats):
        tensor = np.zeros((13, 33, 33), dtype=np.float32)
        tensor[0, 0, 0] = float(len(self._stones))
        return tensor, -16, -16, np.asarray([[0, 0]], dtype=np.int32).tobytes()


def test_rgsc_runner_selects_tree_candidate_by_rank_and_value_head(runner_factory):
    runner, _telemetry, _writer = runner_factory()
    runner.policy_provider = _RGSCFakePolicyProvider()
    runner.game_factory = _RGSCFakeGame
    runner.rgsc = RGSCRestartService(beta=1.0, capacity=4, seed=23, enabled=True)
    trajectory_history = encode_move_history([(0, 0, 0)])
    tree_history = encode_move_history([(0, 0, 0), (1, 1, 0)])
    record = GameRecord(
        positions=[
            PositionRecord(
                trajectory_history,
                {0: 1.0},
                0.0,
                player=1,
                selected_action_value=0.0,
                outcome=1.0,
                game_id=5,
            )
        ],
        outcome=1.0,
        game_id=5,
        game_length=1,
    )

    runner._attach_rgsc_ranked_candidates(record, tree_histories=[tree_history])

    assert record.positions[0].regret_value == 1.0
    assert len(record.rgsc_ranked_candidates) == 1
    selected = record.rgsc_ranked_candidates[0]
    assert selected.move_history == tree_history
    assert selected.rank_score == 20.0
    assert selected.regret == 2.25
    assert selected.source == "mcts_tree_node_regret_value_estimate"
