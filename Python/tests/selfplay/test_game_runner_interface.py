from hexorl.selfplay.game_runner import GameRunRequest, GameRunner


def test_same_game_runner_constructor_drives_dense_graph_and_global(runner_factory):
    for model_family, is_global in (
        ("dense_cnn", False),
        ("graph_hybrid", False),
        ("global_xattn", True),
    ):
        runner, telemetry, writer = runner_factory(model_family=model_family, is_global_graph=is_global)
        assert isinstance(runner, GameRunner)

        result = runner.run_game(GameRunRequest(run_id="r", game_id=1, game_index=0, seed=11))

        assert result.ok is True
        assert result.positions_written == 1
        assert writer.records[0].positions[0].policy_target_v2[0][:2] == (0, 0)
        assert any(event["event"] == "policy_eval_timing" for event in telemetry.events)
        assert any(event["event"] == "selfplay_game_summary" for event in telemetry.events)


def test_runner_uses_explicit_dependencies_not_worker_runtime(runner_factory):
    runner, _telemetry, writer = runner_factory()
    result = runner.run_game(GameRunRequest(run_id="r", game_id=3, game_index=0, seed=5))

    assert result.ok is True
    assert writer.records[0].game_id == 3
    assert result.record_write.record_hash == writer.records[0].game_hash


def test_runner_pair_strategy_consumes_pair_scoring_provider(runner_factory, fake_pair_scorer):
    runner, telemetry, writer = runner_factory(
        pair_strategy_name="two_stage_root_only",
        pair_strategy_max_pairs=1,
        pair_scorer=fake_pair_scorer,
    )
    result = runner.run_game(GameRunRequest(run_id="r", game_id=5, game_index=0, seed=5))

    assert result.ok is True
    assert fake_pair_scorer.calls == 1
    assert result.game_record.positions[0].pair_prior_candidate_count == 1
    assert any(
        event["event"] == "pair_strategy_summary" and event["scored_rows"] == 1
        for event in telemetry.events
    )
