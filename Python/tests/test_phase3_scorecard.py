import pytest

from hexorl.eval.scorecard import (
    compute_phase3_scorecard,
    final_score_from_league_lcb,
    milestone_k_hard_gate_failures,
    should_prune_phase3_trial,
)
from hexorl.eval.tactical_suite import (
    evaluate_tactical_suite,
    phase3_tactical_suite_positions,
    replay_position,
)


def test_scorecard_uses_documented_weights():
    metrics = {
        "league_lcb": 2.0,
        "outside_window_robustness": 3.0,
        "tactical_suite_score": 5.0,
        "classical_survival_score": 7.0,
        "value_calibration_score": 11.0,
        "policy_target_quality": 13.0,
        "epoch_seconds": 17.0,
        "truncation_rate": 19.0,
        "illegal_or_crash_rate": 0.0,
    }
    result = compute_phase3_scorecard(metrics, epoch=12)
    expected_strength = (
        0.40 * 2.0
        + 0.20 * 3.0
        + 0.15 * 5.0
        + 0.10 * 7.0
        + 0.10 * 11.0
        + 0.05 * 13.0
    )
    expected = expected_strength - 0.10 * 17.0 - 0.10 * 19.0
    assert result.mode == "scheduler_strength"
    assert result.base_score == pytest.approx(expected_strength)
    assert result.score == pytest.approx(expected)


def test_scorecard_uses_health_mode_before_epoch_8():
    metrics = {
        "policy_target_quality": 10.0,
        "value_calibration_score": 2.0,
        "outside_window_robustness": 3.0,
        "league_lcb": 1000.0,
        "classical_survival_score": 1000.0,
    }
    result = compute_phase3_scorecard(metrics, epoch=7)
    assert result.mode == "health_warmup"
    assert result.score == pytest.approx(0.45 * 10.0 + 0.35 * 2.0 + 0.20 * 3.0)


def test_scorecard_uses_pre_classical_mode_before_epoch_12():
    metrics = {
        "tactical_suite_score": 4.0,
        "outside_window_robustness": 6.0,
        "policy_target_quality": 8.0,
        "value_calibration_score": 10.0,
        "league_lcb": 1000.0,
        "classical_survival_score": 1000.0,
    }
    result = compute_phase3_scorecard(metrics, epoch=11)
    assert result.mode == "pre_classical_strategy"
    assert result.score == 0.30 * 4.0 + 0.25 * 6.0 + 0.25 * 8.0 + 0.20 * 10.0


def test_scorecard_ignores_classical_survival_before_epoch_12():
    base = {
        "tactical_suite_score": 1.0,
        "outside_window_robustness": 1.0,
        "policy_target_quality": 1.0,
        "value_calibration_score": 1.0,
    }
    low = compute_phase3_scorecard({**base, "classical_survival_score": -999.0}, epoch=10)
    high = compute_phase3_scorecard({**base, "classical_survival_score": 999.0}, epoch=10)
    assert low.score == high.score


def test_scorecard_applies_candidate_hard_gates():
    metrics = {
        "candidate_discovery_winning_move": 0.994,
        "candidate_discovery_forced_block": 0.995,
        "candidate_discovery_two_placement_cover": 0.990,
        "missing_target_policy_mass": 0.005,
        "critical_overflow_count": 0.0,
    }
    result = compute_phase3_scorecard(metrics, epoch=12, candidate_model=True)
    assert not result.hard_pass
    assert "candidate_discovery_winning_move" in result.hard_failures


def test_scorecard_penalizes_illegal_crash_and_truncation_rates():
    metrics = {
        "league_lcb": 0.0,
        "outside_window_robustness": 0.0,
        "tactical_suite_score": 0.0,
        "classical_survival_score": 0.0,
        "value_calibration_score": 0.0,
        "policy_target_quality": 0.0,
        "epoch_seconds": 0.0,
        "truncation_rate": 2.0,
        "illegal_or_crash_rate": 3.0,
    }
    result = compute_phase3_scorecard(metrics, epoch=12)
    assert result.score == -0.10 * 2.0 - 0.20 * 3.0
    assert not result.hard_pass


def test_scorecard_enforces_milestone_k_bug_sentinels():
    failures = milestone_k_hard_gate_failures(
        {
            "illegal_move_rate": 0.0,
            "post_terminal_move_attempts": 1.0,
            "replay_mismatch_rate": 0.0,
            "d6_mismatch_rate": 0.0,
            "legal_mask_mismatch_rate": 0.0,
            "oracle_threat_mismatch_rate": 0.0,
            "missing_legal_action_rows": 0.0,
            "pair_mask_violation_rate": 0.0,
            "target_leakage_check_status": "fail",
        }
    )
    assert "post_terminal_move_attempts" in failures
    assert "target_leakage_check_status" in failures

    result = compute_phase3_scorecard(
        {
            "post_terminal_move_attempts": 1.0,
            "target_leakage_check_status": "fail",
        },
        epoch=12,
    )
    assert not result.hard_pass


def test_final_score_uses_league_lcb():
    assert final_score_from_league_lcb({"rating_mean": 2000.0, "lcb": 1750.0}) == 1750.0


def test_short_health_rung_prunes_only_hard_failures():
    noisy_low = compute_phase3_scorecard(
        {
            "policy_target_quality": -100.0,
            "value_calibration_score": -100.0,
            "outside_window_robustness": -100.0,
        },
        epoch=7,
    )
    hard_fail = compute_phase3_scorecard({"illegal_or_crash_rate": 1.0}, epoch=7)
    assert not should_prune_phase3_trial(noisy_low, epoch=7)
    assert should_prune_phase3_trial(hard_fail, epoch=7)


def test_tactical_suite_positions_are_replayable():
    positions = phase3_tactical_suite_positions()
    assert {position.suite for position in positions} >= {
        "win-now",
        "forced-block",
        "open-four",
        "open-five",
        "two-placement cover",
        "outside-window win",
        "outside-window block",
        "separated-cluster long-span",
        "late-game high-legal-count",
    }
    for position in positions:
        stones, _current = replay_position(position)
        assert stones
        assert position.expected_action_set


def test_tactical_suite_positions_replay_in_engine_when_available():
    engine = pytest.importorskip("_engine")
    game_cls = getattr(engine, "HexGame", None) or getattr(engine, "PyHexGame")
    for position in phase3_tactical_suite_positions():
        game = game_cls()
        for _player, q, r in position.move_history:
            game.place(int(q), int(r))


def test_tactical_suite_expected_actions_are_legal():
    for position in phase3_tactical_suite_positions():
        stones, _current = replay_position(position)
        legal = set()
        for q, r in stones:
            for dq in range(-8, 9):
                for dr in range(-8, 9):
                    if max(abs(dq), abs(dr), abs(dq + dr)) <= 8:
                        cell = (q + dq, r + dr)
                        if cell not in stones:
                            legal.add(cell)
        assert set(position.expected_action_set) <= legal


def test_outside_window_suite_contains_actions_outside_33_crop():
    outside_positions = [
        position
        for position in phase3_tactical_suite_positions()
        if position.suite.startswith("outside-window")
    ]
    assert outside_positions
    for position in outside_positions:
        assert any(q < -16 or q > 16 or r < -16 or r > 16 for q, r in position.expected_action_set)


def test_tactical_suite_evaluates_expected_cover_pairs():
    position = next(
        position
        for position in phase3_tactical_suite_positions()
        if position.suite == "two-placement cover"
    )
    first, second = position.expected_pair_set[0]

    def player(move_history, _time_ms, _player):
        return second if len(move_history) > len(position.move_history) else first

    result = evaluate_tactical_suite(player, positions=[position])
    assert result.score == 1.0
    assert result.positions[0]["selected_pair"] == (first, second)
